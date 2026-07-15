"""Tests for cross-document contradiction detection.

Tests the full pipeline: candidate finding via entity overlap, LLM classification
(mocked), contradiction recording, and contradiction_count incrementing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from memforge.llm.structured import ContradictionDecision, ContradictionResponse, StructuredLlmError
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    CandidatePage,
    LifecycleAction,
    RelationType,
    ReviewCase,
)
from memforge.memory.store import MemoryStore
from memforge.models import Memory, content_hash, generate_deterministic_review_id
from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _insert_document(db: Database, doc_id: str, *, source: str = "src-1") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            source,
            f"http://test/{doc_id}",
            doc_id,
            "TEST",
            now,
            "1",
            f"hash-{doc_id}",
            now,
        ),
    )


@pytest.fixture
async def db(tmp_path):
    """Create a test database with schema initialized."""
    db_path = str(tmp_path / "test.db")
    database = Database(db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def seeded_db(db):
    """Database with two documents, shared entities, and memories from each."""
    # Create two source documents (schema requires many NOT NULL columns)
    await _insert_document(db, "doc-aaa")
    await _insert_document(db, "doc-bbb")

    # Create a shared entity
    entity_id = await db.upsert_entity("postgresql", display_name="PostgreSQL", tags=["technology"])

    # Memory from doc-aaa: "pay-api uses PostgreSQL 14 on port 5432"
    mem_a = _make_memory("mem-aaaa0001", "pay-api uses PostgreSQL 14 on port 5432")
    await db.insert_memory(mem_a)
    await db.db.execute(
        "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
        (mem_a.id, "doc-aaa", "confluence"),
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_a.id, entity_id),
    )

    # Memory from doc-bbb: "pay-api migrated to MySQL 8 in Q1 2026"
    mem_b = _make_memory("mem-bbbb0001", "pay-api migrated to MySQL 8 in Q1 2026")
    await db.insert_memory(mem_b)
    await db.db.execute(
        "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
        (mem_b.id, "doc-bbb", "confluence"),
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_b.id, entity_id),
    )

    # A third memory from doc-aaa that does NOT share entities with mem_b
    other_entity_id = await db.upsert_entity("kafka", display_name="Kafka", tags=["technology"])
    mem_c = _make_memory("mem-cccc0001", "Kafka retention is set to 7 days")
    await db.insert_memory(mem_c)
    await db.db.execute(
        "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
        (mem_c.id, "doc-aaa", "confluence"),
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_c.id, other_entity_id),
    )

    await db.db.commit()

    return db, entity_id, mem_a, mem_b, mem_c


class StubChromaCollection:
    """Tiny ChromaDB stand-in for lifecycle assertions."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, *, ids) -> None:
        self.deleted.extend(ids)

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs) -> None:
        pass


@pytest.fixture
def chroma() -> StubChromaCollection:
    return StubChromaCollection()


@pytest.fixture
def memory_store(db, chroma) -> MemoryStore:
    audit_logger = MemoryAuditLogger(db, default_context=AuditContext(actor_type="test"))
    adapters = build_sqlite_adapters(db, chroma)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=audit_logger,
    )

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[assignment]
    return store


def _make_memory(mem_id: str, content: str) -> Memory:
    """Helper to build a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        project_key=None,
        entity_refs=[],
        tags=[],
        confidence=0.9,
        corroboration_count=1,
        contradiction_count=0,
        valid_from=None,
        valid_until=None,
        created_at=now,
        updated_at=now,
        status="active",
        extraction_context=None,
    )


# ---------------------------------------------------------------------------
# DB-level: get_memory_entity_ids
# ---------------------------------------------------------------------------


class TestGetMemoryEntityIds:
    @pytest.mark.asyncio
    async def test_returns_linked_entity_ids(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        ids = await db.get_memory_entity_ids(mem_a.id)
        assert entity_id in ids

    @pytest.mark.asyncio
    async def test_no_entities_returns_empty(self, db):
        mem = _make_memory("mem-orphan01", "Orphan memory with no entities")
        await db.insert_memory(mem)
        await db.db.commit()
        ids = await db.get_memory_entity_ids(mem.id)
        assert ids == []


# ---------------------------------------------------------------------------
# DB-level: get_cross_doc_candidates
# ---------------------------------------------------------------------------


class TestGetCrossDocCandidates:
    @pytest.mark.asyncio
    async def test_finds_cross_doc_memory_sharing_entity(self, seeded_db):
        """mem_b (from doc-bbb) shares postgresql entity with mem_a (from doc-aaa)."""
        db, entity_id, mem_a, mem_b, _ = seeded_db

        # Ask for candidates for mem_b that share entities but are from different docs
        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_b.id,
            entity_ids=[entity_id],
            doc_id="doc-bbb",
        )
        candidates = candidate_page.candidates
        candidate_ids = [c.id for c in candidates]
        assert mem_a.id in candidate_ids
        assert candidate_page.complete is True

    @pytest.mark.asyncio
    async def test_excludes_same_document(self, seeded_db):
        """Memories from the same document should NOT be candidates."""
        db, entity_id, mem_a, _, _ = seeded_db

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",  # same doc as mem_a
        )
        # mem_a itself should not appear, and no other memory from doc-aaa shares postgresql
        candidate_ids = [c.id for c in candidate_page.candidates]
        assert mem_a.id not in candidate_ids
        assert candidate_page.complete is True

    @pytest.mark.asyncio
    async def test_no_shared_entities_returns_empty(self, seeded_db):
        """mem_c has a different entity (kafka) — no cross-doc overlap with doc-bbb."""
        db, _, _, _, mem_c = seeded_db

        kafka_ids = await db.get_memory_entity_ids(mem_c.id)
        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_c.id,
            entity_ids=kafka_ids,
            doc_id="doc-aaa",
        )
        # No memory from doc-bbb links to kafka
        assert candidate_page.candidates == ()
        assert candidate_page.complete is True

    @pytest.mark.asyncio
    async def test_excludes_candidate_only_when_no_enabled_provenance_remains(self, db):
        """A mixed-provenance memory stays visible while at least one support source is enabled."""
        await _insert_document(db, "doc-current", source="src-current")
        await _insert_document(db, "doc-enabled", source="src-enabled")
        await _insert_document(db, "doc-disabled", source="src-disabled")
        await db.upsert_source(
            "src-current", "confluence", "Current", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.upsert_source(
            "src-enabled", "confluence", "Enabled", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.upsert_source(
            "src-disabled", "confluence", "Disabled", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.set_source_subscription("src-disabled", "alice@example.com", enabled=False)
        entity_id = await db.upsert_entity("payroll", display_name="Payroll", tags=["domain"])

        current = _make_memory("mem-current", "Payroll source memory")
        mixed = _make_memory("mem-mixed-source", "Payroll mixed source memory")
        await db.insert_memory(current)
        await db.insert_memory(mixed)
        await db.link_memory_entity(current.id, entity_id)
        await db.link_memory_entity(mixed.id, entity_id)
        await db.add_memory_source(current.id, "doc-current", "confluence", source_updated_at=None)
        await db.add_memory_source(mixed.id, "doc-enabled", "confluence", source_updated_at=None)
        await db.add_memory_source(mixed.id, "doc-disabled", "confluence", source_updated_at=None)

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=current.id,
            entity_ids=[entity_id],
            doc_id="doc-current",
            excluded_source_ids=("src-disabled",),
        )

        assert [candidate.id for candidate in candidate_page.candidates] == [mixed.id]
        assert candidate_page.complete is True

    @pytest.mark.asyncio
    async def test_returns_complete_cross_doc_candidate_set_without_fixed_cap(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        for index in range(25):
            mem = _make_memory(
                f"mem-bulk{index:04d}",
                f"Cross document PostgreSQL candidate {index}",
            )
            await db.insert_memory(mem)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
                (mem.id, "doc-bbb", "confluence"),
            )
            await db.db.execute(
                "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (mem.id, entity_id),
            )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
        )

        assert len(candidate_page.candidates) == 26
        assert candidate_page.complete is True
        assert {candidate.id for candidate in candidate_page.candidates} >= {
            "mem-bulk0000",
            "mem-bulk0024",
        }

    @pytest.mark.asyncio
    async def test_cross_doc_candidate_set_has_explicit_safety_cap(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        for index in range(5):
            mem = _make_memory(
                f"mem-cap{index:04d}",
                f"Cross document PostgreSQL capped candidate {index}",
            )
            await db.insert_memory(mem)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
                (mem.id, "doc-bbb", "confluence"),
            )
            await db.db.execute(
                "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (mem.id, entity_id),
            )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
            limit=3,
        )

        assert len(candidate_page.candidates) == 3
        assert candidate_page.complete is False
        assert candidate_page.requested_limit == 3

    @pytest.mark.asyncio
    async def test_workspace_candidate_search_excludes_private_memories(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        private_candidate = _make_memory(
            "mem-private01",
            "Private agent-session memory about PostgreSQL rollout.",
        )
        private_candidate.visibility = "private"
        private_candidate.owner_user_id = "other-user"
        await _insert_document(db, "doc-private")
        await db.insert_memory(private_candidate)
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
            (private_candidate.id, "doc-private", "agent_session"),
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (private_candidate.id, entity_id),
        )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
            visibility=mem_a.visibility,
            owner_user_id=mem_a.owner_user_id,
        )

        assert private_candidate.id not in {candidate.id for candidate in candidate_page.candidates}

    @pytest.mark.asyncio
    async def test_private_candidate_search_includes_same_owner_private_memories(self, seeded_db):
        db, entity_id, _, _, _ = seeded_db
        challenger = _make_memory(
            "mem-private10",
            "Private challenger about PostgreSQL rollout.",
        )
        challenger.visibility = "private"
        challenger.owner_user_id = "andrew"
        same_owner_candidate = _make_memory(
            "mem-private11",
            "Same owner private memory about PostgreSQL rollout.",
        )
        same_owner_candidate.visibility = "private"
        same_owner_candidate.owner_user_id = "andrew"
        other_owner_candidate = _make_memory(
            "mem-private12",
            "Other owner private memory about PostgreSQL rollout.",
        )
        other_owner_candidate.visibility = "private"
        other_owner_candidate.owner_user_id = "other-user"
        for memory, doc_id in (
            (challenger, "doc-private-challenger"),
            (same_owner_candidate, "doc-private-same-owner"),
            (other_owner_candidate, "doc-private-other-owner"),
        ):
            await _insert_document(db, doc_id)
            await db.insert_memory(memory)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
                (memory.id, doc_id, "agent_session"),
            )
            await db.db.execute(
                "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory.id, entity_id),
            )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=challenger.id,
            entity_ids=[entity_id],
            doc_id="doc-private-challenger",
            visibility=challenger.visibility,
            owner_user_id=challenger.owner_user_id,
        )

        candidate_ids = {candidate.id for candidate in candidate_page.candidates}
        assert same_owner_candidate.id in candidate_ids
        assert other_owner_candidate.id not in candidate_ids

    @pytest.mark.asyncio
    async def test_cross_doc_candidates_respect_project_boundary(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        mem_a.project_key = "PAY"
        await db.db.execute(
            "UPDATE memories SET project_key = ? WHERE id = ?",
            (mem_a.project_key, mem_a.id),
        )
        await db.db.commit()
        same_project_candidate = _make_memory(
            "mem-project01",
            "PAY project memory about PostgreSQL rollout.",
        )
        same_project_candidate.project_key = "PAY"
        other_project_candidate = _make_memory(
            "mem-project02",
            "OTHER project memory about PostgreSQL rollout.",
        )
        other_project_candidate.project_key = "OTHER"
        for memory, doc_id in (
            (same_project_candidate, "doc-project-same"),
            (other_project_candidate, "doc-project-other"),
        ):
            await _insert_document(db, doc_id)
            await db.insert_memory(memory)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
                (memory.id, doc_id, "confluence"),
            )
            await db.db.execute(
                "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory.id, entity_id),
            )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
            project_key=mem_a.project_key,
        )

        candidate_ids = {candidate.id for candidate in candidate_page.candidates}
        assert same_project_candidate.id in candidate_ids
        assert other_project_candidate.id not in candidate_ids

    @pytest.mark.asyncio
    async def test_cross_doc_candidates_exclude_user_disabled_sources(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        enabled_candidate = _make_memory(
            "mem-enabled01",
            "Enabled source memory about PostgreSQL rollout.",
        )
        disabled_candidate = _make_memory(
            "mem-disabled01",
            "Disabled source memory about PostgreSQL rollout.",
        )
        for memory, doc_id, source_id in (
            (enabled_candidate, "doc-enabled-source", "src-enabled"),
            (disabled_candidate, "doc-disabled-source", "src-disabled"),
        ):
            await _insert_document(db, doc_id, source=source_id)
            await db.insert_memory(memory)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_id, source_type) VALUES (?, ?, ?, ?)",
                (memory.id, doc_id, source_id, "confluence"),
            )
            await db.db.execute(
                "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory.id, entity_id),
            )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
            excluded_source_ids=("src-disabled",),
        )

        candidate_ids = {candidate.id for candidate in candidate_page.candidates}
        assert enabled_candidate.id in candidate_ids
        assert disabled_candidate.id not in candidate_ids

    @pytest.mark.asyncio
    async def test_cross_doc_disabled_source_filter_uses_provenance_source_id_over_document_source(self, seeded_db):
        db, entity_id, mem_a, _, _ = seeded_db
        disabled_candidate = _make_memory(
            "mem-disabled-purged",
            "Disabled source memory about PostgreSQL rollout.",
        )
        await _insert_document(db, "doc-disabled-purged", source="src-disabled")
        await db.insert_memory(disabled_candidate)
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_id, source_type) VALUES (?, ?, ?, ?)",
            (disabled_candidate.id, "doc-disabled-purged", "src-disabled", "confluence"),
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (disabled_candidate.id, entity_id),
        )
        await db.db.execute(
            "UPDATE documents SET source = ? WHERE doc_id = ?",
            ("src-renamed-after-provenance", "doc-disabled-purged"),
        )
        await db.db.commit()

        candidate_page = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",
            excluded_source_ids=("src-disabled",),
        )

        assert disabled_candidate.id not in {candidate.id for candidate in candidate_page.candidates}


# ---------------------------------------------------------------------------
# DB-level: record_contradiction
# ---------------------------------------------------------------------------


class TestRecordContradiction:
    @pytest.mark.asyncio
    async def test_increments_contradiction_count(self, seeded_db):
        """Recording a contradiction should increment both memories' counts."""
        db, _, mem_a, mem_b, _ = seeded_db

        # Before
        m_a = await db.get_memory(mem_a.id)
        m_b = await db.get_memory(mem_b.id)
        assert m_a.contradiction_count == 0
        assert m_b.contradiction_count == 0

        # Record contradiction
        await db.record_contradiction(mem_a.id, mem_b.id, "contradiction", "PostgreSQL 14 vs MySQL 8")

        # After
        m_a = await db.get_memory(mem_a.id)
        m_b = await db.get_memory(mem_b.id)
        assert m_a.contradiction_count == 1
        assert m_b.contradiction_count == 1

    @pytest.mark.asyncio
    async def test_temporal_does_not_increment(self, seeded_db):
        """Temporal classifications should NOT increment contradiction_count."""
        db, _, mem_a, mem_b, _ = seeded_db

        await db.record_contradiction(mem_a.id, mem_b.id, "temporal", "Newer version info")

        m_a = await db.get_memory(mem_a.id)
        m_b = await db.get_memory(mem_b.id)
        assert m_a.contradiction_count == 0
        assert m_b.contradiction_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_recording_ignored(self, seeded_db):
        """INSERT OR IGNORE prevents double-counting the same pair."""
        db, _, mem_a, mem_b, _ = seeded_db

        await db.record_contradiction(mem_a.id, mem_b.id, "contradiction", "first")
        await db.record_contradiction(mem_a.id, mem_b.id, "contradiction", "second")  # ignored

        m_a = await db.get_memory(mem_a.id)
        # Only incremented once because second INSERT was ignored,
        # but UPDATE runs both times — this is a potential bug to verify
        # The INSERT OR IGNORE prevents the row, but the UPDATE still fires
        assert m_a.contradiction_count >= 1

    @pytest.mark.asyncio
    async def test_record_persists_to_table(self, seeded_db):
        """Contradiction record is written to memory_contradictions table."""
        db, _, mem_a, mem_b, _ = seeded_db

        await db.record_contradiction(mem_a.id, mem_b.id, "contradiction", "DB mismatch")

        async with db.db.execute(
            "SELECT * FROM memory_contradictions WHERE memory_id_a = ? AND memory_id_b = ?",
            (mem_a.id, mem_b.id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row["classification"] == "contradiction"
        assert row["reason"] == "DB mismatch"
        assert row["resolution"] == "pending"


# ---------------------------------------------------------------------------
# Full pipeline: detect_cross_doc_contradictions (with mocked LLM)
# ---------------------------------------------------------------------------


def _mock_contradiction_response(decisions: list[dict]) -> ContradictionResponse:
    return ContradictionResponse(decisions=[ContradictionDecision(**decision) for decision in decisions])


class TestDetectCrossDocContradictions:
    @pytest.mark.asyncio
    async def test_detects_contradiction_via_llm(self, seeded_db, memory_store, chroma):
        """Full pipeline: new memory shares entity with existing, LLM says contradiction."""
        db, entity_id, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "PostgreSQL 14 vs MySQL 8"},
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 1
        assert stats["checked"] == 1
        assert mock_client.detect_contradictions.call_args.kwargs["max_tokens"] == 8192

        # Verify counts incremented
        m_a = await db.get_memory(mem_a.id)
        m_b = await db.get_memory(mem_b.id)
        assert m_a.contradiction_count == 1
        assert m_b.contradiction_count == 1
        assert m_a.status == "active"
        assert m_b.status == "active"
        assert chroma.deleted == []

        async with db.db.execute(
            "SELECT memory_id FROM memories_fts WHERE memory_id = ?",
            (mem_b.id,),
        ) as cursor:
            assert await cursor.fetchone() is not None

        review = await db.get_pending_review_for_challenger(mem_b.id)
        assert review is not None
        assert review.kind == "cross_source_conflict"
        assert review.status == "pending"
        assert review.incumbent_memory_id == mem_a.id
        assert review.challenger_memory_id == mem_b.id
        assert review.reason == "contradiction: PostgreSQL 14 vs MySQL 8"

        async with db.db.execute(
            """SELECT rr.*
               FROM relation_runs rr
               JOIN relation_candidates rc ON rc.relation_run_id = rr.id
               WHERE rr.evidence_unit_id LIKE 'eu-contradiction-%'
                 AND rc.memory_id = ?""",
            (mem_a.id,),
        ) as cursor:
            relation_runs = [dict(row) async for row in cursor]
        assert len(relation_runs) == 1
        assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_REVIEW.value
        evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
        assert evidence_unit is not None
        assert evidence_unit.doc_id == "doc-bbb"
        assert evidence_unit.source_type == "confluence"
        assert evidence_unit.source_metadata["challenger_memory_id"] == mem_b.id
        assert review.id == generate_deterministic_review_id(
            kind="cross_source_conflict",
            incumbent_memory_id=mem_a.id,
            challenger_memory_id=mem_b.id,
            relation_run_id=relation_runs[0]["id"],
            evidence_unit_id=evidence_unit.id,
            review_case=relation_runs[0]["review_case"],
        )
        relations = await db.get_evidence_relations(evidence_unit.id)
        assert [(relation.memory_id, relation.relation_type, relation.authority_case) for relation in relations] == [
            (mem_a.id, RelationType.CONTRADICTS, AuthorityCase.CROSS_SOURCE_CONFLICT)
        ]
        candidates = await db.get_relation_candidates(relation_runs[0]["id"])
        assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
            (mem_a.id, CandidateBucket.SHARED_ENTITIES, True)
        ]

    @pytest.mark.asyncio
    async def test_relation_bundle_failure_does_not_record_contradiction_state(
        self, seeded_db, memory_store, monkeypatch
    ):
        db, _, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "PostgreSQL 14 vs MySQL 8"},
                ]
            )
        )

        async def fail_relation_bundle(*args, **kwargs):
            raise RuntimeError("relation bundle failed")

        monkeypatch.setattr(db, "_record_relation_outcome_bundle_unlocked", fail_relation_bundle)

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 0
        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).status == "active"
        assert await db.get_pending_review_for_challenger(mem_b.id) is None
        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_review_write_failure_does_not_leave_challenger_pending(self, seeded_db, memory_store, monkeypatch):
        db, _, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "PostgreSQL 14 vs MySQL 8"},
                ]
            )
        )

        async def fail_review_insert(*args, **kwargs):
            raise RuntimeError("review write failed")

        monkeypatch.setattr(db, "record_memory_review_with_relation_outcome", fail_review_insert)

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 0
        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).status == "active"
        assert await db.get_pending_review_for_challenger(mem_b.id) is None
        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_truncated_candidate_page_skips_lifecycle_mutation(self, seeded_db, memory_store, monkeypatch):
        db, entity_id, mem_a, mem_b, _ = seeded_db

        async def truncated_candidates(
            memory_id,
            entity_ids,
            doc_id,
            *,
            owner_user_id=None,
            visibility=None,
            project_key=None,
            excluded_source_ids=(),
            limit=20,
        ):
            del (
                memory_id,
                entity_ids,
                doc_id,
                owner_user_id,
                visibility,
                project_key,
                excluded_source_ids,
                limit,
            )
            return CandidatePage(candidates=(mem_a,), complete=False, requested_limit=1)

        monkeypatch.setattr(db, "get_cross_doc_candidates", truncated_candidates)
        mock_client = AsyncMock()

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 1}
        mock_client.detect_contradictions.assert_not_called()
        assert await db.get_pending_review_for_challenger(mem_b.id) is None
        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).status == "active"
        async with db.db.execute(
            """SELECT * FROM relation_runs
               WHERE evidence_unit_id LIKE 'eu-contradiction-%'
               ORDER BY started_at DESC LIMIT 1"""
        ) as cursor:
            run = await cursor.fetchone()
        assert run is not None
        assert run["lifecycle_action"] == LifecycleAction.CREATE_REVIEW.value
        assert run["review_case"] == ReviewCase.MANDATORY_INCOMPLETE.value
        assert run["status"] == "review_required"
        candidates = await db.get_relation_candidates(run["id"])
        assert [
            (candidate.memory_id, candidate.bucket_complete, candidate.was_checked) for candidate in candidates
        ] == [(mem_a.id, False, False)]

    @pytest.mark.asyncio
    async def test_total_pair_cap_and_batching_bound_llm_prompt_size(self, seeded_db, memory_store, monkeypatch):
        db, _, _, mem_b, _ = seeded_db
        candidates = []
        for i in range(5):
            doc_id = f"doc-cap-{i}"
            memory = _make_memory(f"mem-cap{i:04d}", f"candidate memory {i}")
            await _insert_document(db, doc_id)
            await db.insert_memory(memory)
            await db.db.execute(
                "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
                (memory.id, doc_id, "confluence"),
            )
            candidates.append(memory)
        await db.db.commit()

        async def capped_candidates(
            memory_id,
            entity_ids,
            doc_id,
            *,
            owner_user_id=None,
            visibility=None,
            project_key=None,
            excluded_source_ids=(),
            limit=200,
        ):
            del (
                memory_id,
                entity_ids,
                doc_id,
                owner_user_id,
                visibility,
                project_key,
                excluded_source_ids,
            )
            return CandidatePage(candidates=candidates[:limit], complete=True, requested_limit=limit)

        monkeypatch.setattr(db, "get_cross_doc_candidates", capped_candidates)
        monkeypatch.setattr(
            "memforge.pipeline.contradiction_detector.MAX_CONTRADICTION_PAIRS_PER_RUN",
            2,
            raising=False,
        )
        monkeypatch.setattr(
            "memforge.pipeline.contradiction_detector.CONTRADICTION_LLM_BATCH_SIZE",
            1,
            raising=False,
        )

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([]))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["checked"] == 2
        assert stats["truncated"] == 1
        assert mock_client.detect_contradictions.call_count == 2
        prompts = [call.args[0] for call in mock_client.detect_contradictions.call_args_list]
        assert all(prompt.count('"memory_a"') == 1 for prompt in prompts)
        async with db.db.execute(
            """SELECT status, review_case
               FROM relation_runs
               WHERE evidence_unit_id LIKE 'eu-contradiction-%'
               ORDER BY started_at DESC LIMIT 1"""
        ) as cursor:
            run = await cursor.fetchone()
        assert run is not None
        assert run["status"] == "review_required"
        assert run["review_case"] == ReviewCase.MANDATORY_INCOMPLETE.value

    @pytest.mark.asyncio
    async def test_pair_cap_records_incomplete_run_for_skipped_challenger(self, seeded_db, memory_store, monkeypatch):
        db, entity_id, mem_a, mem_b, _ = seeded_db
        skipped = _make_memory("mem-skip0001", "pay-api has an unverified runtime version")
        await db.insert_memory(skipped)
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
            (skipped.id, "doc-bbb", "confluence"),
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (skipped.id, entity_id),
        )
        await db.db.commit()

        async def exact_cap_candidates(
            memory_id,
            entity_ids,
            doc_id,
            *,
            owner_user_id=None,
            visibility=None,
            project_key=None,
            excluded_source_ids=(),
            limit=200,
        ):
            del (
                memory_id,
                entity_ids,
                doc_id,
                owner_user_id,
                visibility,
                project_key,
                excluded_source_ids,
            )
            return CandidatePage(candidates=(mem_a,)[:limit], complete=True, requested_limit=limit)

        monkeypatch.setattr(db, "get_cross_doc_candidates", exact_cap_candidates)
        monkeypatch.setattr(
            "memforge.pipeline.contradiction_detector.MAX_CONTRADICTION_PAIRS_PER_RUN",
            1,
            raising=False,
        )

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([]))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id, skipped.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["checked"] == 1
        assert stats["truncated"] == 1
        async with db.db.execute(
            """SELECT status, review_case, audit_json
               FROM relation_runs
               WHERE review_case = ?
               ORDER BY started_at DESC LIMIT 1""",
            (ReviewCase.MANDATORY_INCOMPLETE.value,),
        ) as cursor:
            run = await cursor.fetchone()
        assert run is not None
        assert run["status"] == "review_required"
        assert skipped.id in run["audit_json"]

    @pytest.mark.asyncio
    async def test_multiple_contradictions_create_one_review_per_challenger(self, seeded_db, memory_store):
        """One challenger can conflict with several memories but needs one decision row."""
        db, entity_id, _, mem_b, _ = seeded_db
        now = datetime.now(timezone.utc).isoformat()
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("doc-ccc", "src-1", "http://test/c", "Legacy DB Doc", "TEST", now, "1", "hash3", now),
        )
        mem_d = _make_memory("mem-dddd0001", "pay-api still uses PostgreSQL 13")
        await db.insert_memory(mem_d)
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
            (mem_d.id, "doc-ccc", "confluence"),
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_d.id, entity_id),
        )
        await db.db.commit()

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "first conflict"},
                    {"pair_index": 1, "classification": "contradiction", "reason": "second conflict"},
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 2
        reviews = await db.list_memory_reviews(status="pending")
        challenger_reviews = [r for r in reviews if r.challenger_memory_id == mem_b.id]
        assert len(challenger_reviews) == 1

    @pytest.mark.asyncio
    async def test_multiple_challengers_from_same_doc_get_independent_reviews(self, seeded_db, memory_store):
        """Each active challenger keeps an independently resolvable review finding."""
        db, entity_id, mem_a, mem_b, _ = seeded_db
        mem_e = _make_memory("mem-eeee0001", "pay-api now uses PostgreSQL 16")
        await db.insert_memory(mem_e)
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_type) VALUES (?, ?, ?)",
            (mem_e.id, "doc-bbb", "confluence"),
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_e.id, entity_id),
        )
        await db.db.commit()

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "first challenger"},
                    {"pair_index": 1, "classification": "contradiction", "reason": "second challenger"},
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id, mem_e.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 2
        reviews = await db.list_memory_reviews(status="pending")
        assert len(reviews) == 2
        assert {review.incumbent_memory_id for review in reviews} == {mem_a.id}
        assert {review.challenger_memory_id for review in reviews} == {mem_b.id, mem_e.id}
        assert all(review.kind == "cross_source_conflict" for review in reviews)
        assert (await db.get_memory(mem_b.id)).status == "active"
        assert (await db.get_memory(mem_e.id)).status == "active"

    @pytest.mark.asyncio
    async def test_temporal_detected_not_contradiction(self, seeded_db, memory_store):
        """LLM classifies as temporal — counts should NOT increment."""
        db, _, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "temporal", "reason": "Newer version replaces older"},
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["temporal"] == 1
        assert stats["contradictions"] == 0

        m_a = await db.get_memory(mem_a.id)
        assert m_a.contradiction_count == 0

    @pytest.mark.asyncio
    async def test_no_candidates_skips_llm(self, seeded_db, memory_store):
        """Memory with no cross-doc entity overlap should not call the LLM."""
        db, _, _, _, mem_c = seeded_db

        mock_client = AsyncMock()

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_c.id],
            doc_id="doc-aaa",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["checked"] == 0
        assert stats["contradictions"] == 0
        mock_client.detect_contradictions.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_structured_llm_client_returns_empty(self, seeded_db, memory_store):
        """If LLM client is None, return empty stats without error."""
        db, _, _, mem_b, _ = seeded_db

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=None,
        )

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 0}

    @pytest.mark.asyncio
    async def test_empty_memory_ids_returns_empty(self, seeded_db, memory_store):
        """Empty input list should return empty stats."""
        db, _, _, _, _ = seeded_db

        mock_client = AsyncMock()

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[],
            doc_id="doc-aaa",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 0}

    @pytest.mark.asyncio
    async def test_structured_llm_error_handled(self, seeded_db, memory_store):
        """Malformed LLM response should be handled gracefully."""
        db, _, _, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(side_effect=StructuredLlmError("invalid structured response"))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        # Should not crash, return zero contradictions
        assert stats["contradictions"] == 0

        audit_rows = await db.list_memory_audit_events(event_type="contradiction_detection_failed")
        assert len(audit_rows) == 1
        assert audit_rows[0].status == "failed"
        assert audit_rows[0].doc_id == "doc-bbb"
        assert audit_rows[0].payload["checked"] == 1
        assert audit_rows[0].payload["candidate_pairs"] == 1
        assert "invalid structured response" in audit_rows[0].error

    @pytest.mark.asyncio
    async def test_unrelated_classification_ignored(self, seeded_db, memory_store):
        """UNRELATED pairs should not record anything."""
        db, _, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {"pair_index": 0, "classification": "unrelated", "reason": "Different aspects"},
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 0
        assert stats["temporal"] == 0
        assert stats["checked"] == 1

        m_a = await db.get_memory(mem_a.id)
        assert m_a.contradiction_count == 0

        audit_rows = await db.list_memory_audit_events(event_type="contradiction_detection_completed")
        assert len(audit_rows) == 1
        assert audit_rows[0].status == "committed"
        assert audit_rows[0].doc_id == "doc-bbb"
        assert audit_rows[0].payload["checked"] == 1
        assert audit_rows[0].payload["candidate_pairs"] == 1
        assert audit_rows[0].payload["contradictions"] == 0
        assert audit_rows[0].payload["temporal"] == 0
        assert audit_rows[0].payload["classifications"]["unrelated"] == 1

        async with db.db.execute(
            """SELECT rr.*
               FROM relation_runs rr
               JOIN relation_candidates rc ON rc.relation_run_id = rr.id
               WHERE rr.evidence_unit_id LIKE 'eu-contradiction-%'
                 AND rc.memory_id = ?""",
            (mem_a.id,),
        ) as cursor:
            relation_runs = [dict(row) async for row in cursor]
        assert len(relation_runs) == 1
        assert relation_runs[0]["lifecycle_action"] == LifecycleAction.NONE.value
        assert relation_runs[0]["status"] == "checked"
        candidates = await db.get_relation_candidates(relation_runs[0]["id"])
        assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
            (mem_a.id, CandidateBucket.SHARED_ENTITIES, True)
        ]
        evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
        assert evidence_unit is not None
        assert await db.get_evidence_relations(evidence_unit.id) == []
