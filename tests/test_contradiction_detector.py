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
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    LifecycleAction,
    MemorySupportAssertion,
    RelationType,
    ReviewCase,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateRetriever,
    CrossDocumentCandidateSelection,
    RetrievedRelationCandidate,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    Memory,
    NormalizedContent,
    RawContent,
    content_hash,
    generate_deterministic_review_id,
)
from memforge.pipeline.contradiction_detector import (
    CONTRADICTION_PROMPT,
    _build_cross_doc_relation_outcome_bundles,
    _iter_prompt_sized_pair_batches,
    detect_cross_doc_contradictions,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import AnchorKind, SourceAnchor
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


async def _seed_projected_support(
    db: Database,
    *,
    memory: Memory,
    doc_id: str,
    source_id: str,
) -> None:
    """Persist the current Source Projection evidence behind one Memory."""
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name=source_id,
        config_json="{}",
        access_policy="workspace",
        owner_user_id="test-owner",
    )
    item = ContentItem(
        item_id=doc_id,
        title=doc_id,
        source_url=f"http://test/{doc_id}",
        last_modified=datetime.now(timezone.utc),
        version="1",
        extra={"page_id": doc_id, "space_key": "TEST"},
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id=f"projection-{memory.id}",
        item=item,
        raw=RawContent(
            item=item,
            body=memory.content.encode(),
            content_type="text/plain",
        ),
        normalized=NormalizedContent(item=item, markdown_body=memory.content),
    )
    await db.record_source_projection(projection)
    observation = projection.observations[0]
    revision = next(
        item
        for item in projection.observation_revisions
        if item.observation_id == observation.id
    )
    unit = EvidenceUnit(
        id=f"eu-{memory.id}",
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key=memory.project_key,
        visibility=memory.visibility,
        owner_user_id=memory.owner_user_id,
        repo_identifier=memory.repo_identifier,
        content=revision.content,
        excerpt=memory.content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-test",
    )
    await db.upsert_evidence_unit(unit)
    reference = (
        await db.record_evidence_references(
            unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=observation.id,
                        observation_revision_id=revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{memory.id}",
            memory_id=memory.id,
            evidence_reference_id=reference.id or "",
            source_id=source_id,
            access_context_hash="workspace-test",
        )
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
    await _insert_document(db, "doc-aaa", source="src-a")
    await _insert_document(db, "doc-bbb", source="src-b")

    # Create a shared entity
    entity_id = await db.upsert_entity("postgresql", display_name="PostgreSQL", tags=["technology"])

    # Memory from doc-aaa: "pay-api uses PostgreSQL 14 on port 5432"
    mem_a = _make_memory("mem-aaaa0001", "pay-api uses PostgreSQL 14 on port 5432")
    await db.insert_memory(mem_a)
    await db.add_memory_source(
        mem_a.id,
        "doc-aaa",
        "confluence",
        mem_a.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_a.id, entity_id),
    )

    # Memory from doc-bbb: "pay-api migrated to MySQL 8 in Q1 2026"
    mem_b = _make_memory("mem-bbbb0001", "pay-api migrated to MySQL 8 in Q1 2026")
    await db.insert_memory(mem_b)
    await db.add_memory_source(
        mem_b.id,
        "doc-bbb",
        "confluence",
        mem_b.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_b.id, entity_id),
    )

    # A third memory from doc-aaa that does NOT share entities with mem_b
    other_entity_id = await db.upsert_entity("kafka", display_name="Kafka", tags=["technology"])
    mem_c = _make_memory("mem-cccc0001", "Kafka retention is set to 7 days")
    await db.insert_memory(mem_c)
    await db.add_memory_source(
        mem_c.id,
        "doc-aaa",
        "confluence",
        mem_c.content,
        source_updated_at=None,
    )
    await db.db.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (mem_c.id, other_entity_id),
    )

    await db.db.commit()
    await _seed_projected_support(
        db,
        memory=mem_a,
        doc_id="doc-aaa",
        source_id="src-a",
    )
    await _seed_projected_support(
        db,
        memory=mem_b,
        doc_id="doc-bbb",
        source_id="src-b",
    )
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


def _candidate_retriever(memory_store: MemoryStore) -> CrossDocumentCandidateRetriever:
    return CrossDocumentCandidateRetriever(
        relational=memory_store.relational,
        keyword=memory_store.keyword,
        vector=memory_store.vector,
    )


def _make_memory(mem_id: str, content: str) -> Memory:
    """Helper to build a Memory object for testing."""
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        project_key="TEST",
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
        assert m_a.contradiction_count == 1

    @pytest.mark.asyncio
    async def test_reverse_pair_retry_is_idempotent(self, seeded_db):
        db, _, mem_a, mem_b, _ = seeded_db

        await db.record_contradiction(
            mem_a.id,
            mem_b.id,
            "contradiction",
            "first",
        )
        await db.record_contradiction(
            mem_b.id,
            mem_a.id,
            "contradiction",
            "reverse retry",
        )

        assert (await db.get_memory(mem_a.id)).contradiction_count == 1
        assert (await db.get_memory(mem_b.id)).contradiction_count == 1
        async with db.db.execute(
            "SELECT COUNT(*) FROM memory_contradictions"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1

    @pytest.mark.asyncio
    async def test_temporal_pair_upgrades_to_contradiction_once(self, seeded_db):
        db, _, mem_a, mem_b, _ = seeded_db

        await db.record_contradiction(
            mem_b.id,
            mem_a.id,
            "temporal",
            "initial temporal relation",
        )
        await db.record_contradiction(
            mem_a.id,
            mem_b.id,
            "contradiction",
            "confirmed conflict",
        )
        await db.record_contradiction(
            mem_b.id,
            mem_a.id,
            "temporal",
            "late downgrade retry",
        )

        assert (await db.get_memory(mem_a.id)).contradiction_count == 1
        assert (await db.get_memory(mem_b.id)).contradiction_count == 1
        async with db.db.execute(
            """SELECT classification, reason
                 FROM memory_contradictions"""
        ) as cursor:
            row = await cursor.fetchone()
        assert tuple(row) == ("contradiction", "confirmed conflict")

    @pytest.mark.asyncio
    async def test_self_contradiction_is_rejected(self, seeded_db):
        db, _, mem_a, _, _ = seeded_db

        with pytest.raises(
            ValueError,
            match="contradiction requires two distinct Memories",
        ):
            await db.record_contradiction(
                mem_a.id,
                mem_a.id,
                "contradiction",
                "invalid self pair",
            )

        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        async with db.db.execute(
            "SELECT COUNT(*) FROM memory_contradictions"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0

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


class TestContradictionSummaryMigration:
    @pytest.mark.asyncio
    async def test_legacy_directional_rows_are_reset_once(self, seeded_db):
        db, _, mem_a, mem_b, _ = seeded_db

        await db.db.executemany(
            """INSERT INTO memory_contradictions (
                   memory_id_a, memory_id_b, classification, reason
               ) VALUES (?, ?, ?, ?)""",
            [
                (mem_a.id, mem_b.id, "contradiction", "legacy forward"),
                (mem_b.id, mem_a.id, "contradiction", "legacy reverse"),
            ],
        )
        await db.db.execute(
            "UPDATE memories SET contradiction_count = 2 WHERE id IN (?, ?)",
            (mem_a.id, mem_b.id),
        )
        await db.db.execute("DELETE FROM schema_migrations WHERE version = 62")
        await db.db.commit()

        await db._run_migrations()
        await db._run_migrations()

        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            assert (await cursor.fetchone())[0] == 0
        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).contradiction_count == 0
        async with db.db.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE version = 62"
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1


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
            candidate_retriever=_candidate_retriever(memory_store),
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
               WHERE rr.evidence_unit_id = ?
                 AND rc.memory_id = ?""",
            (f"eu-{mem_b.id}", mem_a.id),
        ) as cursor:
            relation_runs = [dict(row) async for row in cursor]
        assert len(relation_runs) == 1
        assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_REVIEW.value
        evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
        assert evidence_unit is not None
        assert evidence_unit.doc_id == "doc-bbb"
        assert evidence_unit.source_type == "confluence"
        assert evidence_unit.source_id == "src-b"
        assert evidence_unit.source_lineage_id
        assert evidence_unit.access_context_hash == "workspace-test"
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
            (mem_a.id, CandidateBucket.HYBRID_DISCOVERY, True)
        ]

    @pytest.mark.asyncio
    async def test_each_conflicting_incumbent_gets_a_deterministic_review(
        self,
        seeded_db,
        memory_store,
    ):
        db, entity_id, mem_a, mem_b, _ = seeded_db
        await _insert_document(db, "doc-ddd", source="src-d")
        mem_d = _make_memory(
            "mem-dddd0001",
            "pay-api still uses PostgreSQL 13",
        )
        await db.insert_memory(mem_d)
        await db.add_memory_source(
            mem_d.id,
            "doc-ddd",
            "confluence",
            mem_d.content,
            source_updated_at=None,
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_d.id, entity_id),
        )
        await db.db.commit()
        await _seed_projected_support(
            db,
            memory=mem_d,
            doc_id="doc-ddd",
            source_id="src-d",
        )

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {
                        "pair_index": 0,
                        "classification": "contradiction",
                        "reason": "first conflict",
                    },
                    {
                        "pair_index": 1,
                        "classification": "contradiction",
                        "reason": "second conflict",
                    },
                ]
            )
        )

        for _ in range(2):
            stats = await detect_cross_doc_contradictions(
                new_memory_ids=[mem_b.id],
                doc_id="doc-bbb",
                db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=mock_client,
            )
            assert stats["contradictions"] == 2

        async with db.db.execute(
            """SELECT id, incumbent_memory_id
                 FROM memory_reviews
                WHERE challenger_memory_id = ?
                ORDER BY incumbent_memory_id""",
            (mem_b.id,),
        ) as cursor:
            reviews = [dict(row) async for row in cursor]
        assert [review["incumbent_memory_id"] for review in reviews] == [
            mem_a.id,
            mem_d.id,
        ]

        async with db.db.execute(
            """SELECT id
                 FROM relation_runs
                WHERE evidence_unit_id = ?
                  AND review_case = ?""",
            (f"eu-{mem_b.id}", ReviewCase.CROSS_SOURCE_CONFLICT.value),
        ) as cursor:
            relation_runs = [dict(row) async for row in cursor]
        assert len(relation_runs) == 1
        relation_run_id = relation_runs[0]["id"]
        evidence_unit = await db.get_evidence_unit(f"eu-{mem_b.id}")
        assert evidence_unit is not None
        assert {
            review["id"]
            for review in reviews
        } == {
            generate_deterministic_review_id(
                kind="cross_source_conflict",
                incumbent_memory_id=incumbent_id,
                challenger_memory_id=mem_b.id,
                relation_run_id=relation_run_id,
                evidence_unit_id=evidence_unit.id,
                review_case=ReviewCase.CROSS_SOURCE_CONFLICT.value,
            )
            for incumbent_id in (mem_a.id, mem_d.id)
        }
        relations = await db.get_evidence_relations(evidence_unit.id)
        assert {
            relation.memory_id
            for relation in relations
            if relation.relation_type is RelationType.CONTRADICTS
        } == {mem_a.id, mem_d.id}
        assert (await db.get_memory(mem_a.id)).contradiction_count == 1
        assert (await db.get_memory(mem_b.id)).contradiction_count == 2
        assert (await db.get_memory(mem_d.id)).contradiction_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("decisions", "error_fragment"),
        [
            (
                [
                    {"pair_index": 0, "classification": "contradiction", "reason": "first conflict"},
                    {"pair_index": 0, "classification": "unrelated", "reason": "duplicate overwrite"},
                    {"pair_index": 1, "classification": "contradiction", "reason": "second conflict"},
                ],
                "duplicate_count=1",
            ),
            (
                [
                    {"pair_index": 0, "classification": "unrelated", "reason": "one decision missing"},
                ],
                "missing_count=1",
            ),
        ],
    )
    async def test_invalid_pair_decision_coverage_fails_closed_before_relation_or_review_writes(
        self, seeded_db, memory_store, decisions, error_fragment
    ):
        """Ambiguous structured coverage cannot split a Review from its relation snapshot."""
        db, entity_id, mem_a, mem_b, _ = seeded_db
        await _insert_document(db, "doc-ccc")
        mem_d = _make_memory("mem-dddd0001", "pay-api still uses PostgreSQL 13")
        await db.insert_memory(mem_d)
        await db.add_memory_source(
            mem_d.id,
            "doc-ccc",
            "confluence",
            mem_d.content,
            source_updated_at=None,
        )
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (mem_d.id, entity_id),
        )
        await db.db.commit()
        await _seed_projected_support(
            db,
            memory=mem_d,
            doc_id="doc-ccc",
            source_id="src-1",
        )

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response(decisions))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=mock_client,
        )

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 2, "truncated": 0}
        assert await db.get_pending_review_for_challenger(mem_b.id) is None
        async with db.db.execute(
            "SELECT COUNT(*) FROM relation_runs WHERE evidence_unit_id = ?",
            (f"eu-{mem_b.id}",),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with db.db.execute("SELECT COUNT(*) FROM memory_contradictions") as cursor:
            assert (await cursor.fetchone())[0] == 0
        assert (await db.get_memory(mem_a.id)).contradiction_count == 0
        assert (await db.get_memory(mem_b.id)).contradiction_count == 0
        assert (await db.get_memory(mem_d.id)).contradiction_count == 0

        audit_rows = await db.list_memory_audit_events(event_type="contradiction_detection_failed")
        assert len(audit_rows) == 1
        assert audit_rows[0].reason == "structured_output_failure"
        assert error_fragment in audit_rows[0].error

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
            candidate_retriever=_candidate_retriever(memory_store),
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
            candidate_retriever=_candidate_retriever(memory_store),
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


    def test_prompt_batches_are_bounded_without_mocking_llm_judgment(self, monkeypatch):
        monkeypatch.setattr(
            "memforge.pipeline.contradiction_detector.CONTRADICTION_LLM_BATCH_SIZE",
            2,
        )
        challenger = _make_memory("mem-batch-new", "challenger " + "x" * 2_000)
        pairs = [
            (challenger, _make_memory(f"mem-batch-{index}", "candidate " + "y" * 2_000))
            for index in range(3)
        ]

        batches = _iter_prompt_sized_pair_batches(pairs)

        assert [start for start, _ in batches] == [0, 2]
        assert [len(batch) for _, batch in batches] == [2, 1]

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
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=mock_client,
        )

        assert stats["temporal"] == 1
        assert stats["contradictions"] == 0

        m_a = await db.get_memory(mem_a.id)
        assert m_a.contradiction_count == 0
        assert await db.get_pending_review_for_challenger(mem_b.id) is None
        async with db.db.execute(
            """SELECT id, lifecycle_action, review_case
                 FROM relation_runs
                WHERE evidence_unit_id = ?
                ORDER BY started_at DESC LIMIT 1""",
            (f"eu-{mem_b.id}",),
        ) as cursor:
            run = await cursor.fetchone()
        assert run is not None
        assert run["lifecycle_action"] == LifecycleAction.NONE.value
        assert run["review_case"] is None
        relations = await db.get_evidence_relations(f"eu-{mem_b.id}")
        assert [
            (relation.memory_id, relation.relation_type, relation.authority_case)
            for relation in relations
        ] == [
            (
                mem_a.id,
                RelationType.REFINES,
                AuthorityCase.INDEPENDENT_REFINEMENT,
            )
        ]

    @pytest.mark.asyncio
    async def test_clarification_deterministically_maps_to_independent_refinement(
        self,
        seeded_db,
    ):
        db, _, mem_a, mem_b, _ = seeded_db
        selection = CrossDocumentCandidateSelection(
            discovery=(
                RetrievedRelationCandidate(
                    memory=CandidateMemory(
                        memory_id=mem_a.id,
                        source_id="src-a",
                        doc_id="doc-aaa",
                        source_lineage_id="doc-aaa",
                        visibility=mem_a.visibility,
                        owner_user_id=mem_a.owner_user_id,
                        repo_identifier=mem_a.repo_identifier,
                    ),
                    score=1.0,
                    channels=(CandidateBucket.SHARED_ENTITIES.value,),
                ),
            ),
            audit={"candidate_count_kind": "windowed"},
        )

        bundles = await _build_cross_doc_relation_outcome_bundles(
            pairs=[(mem_b, mem_a)],
            decisions_by_pair={
                0: {
                    "classification": "clarification",
                    "reason": "The challenger adds a compatible operational detail.",
                }
            },
            doc_id="doc-bbb",
            db=db,
            candidate_selection_by_challenger={mem_b.id: selection},
        )

        bundle = bundles[mem_b.id]
        assert bundle.relation_run.lifecycle_action is LifecycleAction.NONE
        assert bundle.relation_run.review_case is None
        assert [
            (relation.memory_id, relation.relation_type, relation.authority_case)
            for relation in bundle.relations
        ] == [
            (
                mem_a.id,
                RelationType.REFINES,
                AuthorityCase.INDEPENDENT_REFINEMENT,
            )
        ]

    def test_relation_prompt_does_not_assign_temporal_authority(self):
        assert "newer one supersedes" not in CONTRADICTION_PROMPT
        assert "does not decide which source is authoritative" in CONTRADICTION_PROMPT
        assert "does not supersede either Memory" in CONTRADICTION_PROMPT

    @pytest.mark.asyncio
    async def test_repository_incompatible_candidates_are_filtered_before_llm(
        self,
        seeded_db,
        memory_store,
    ):
        db, _, mem_a, mem_b, _ = seeded_db
        mem_a.repo_identifier = "repo-a"
        mem_b.repo_identifier = "repo-b"
        await db.db.executemany(
            "UPDATE memories SET repo_identifier = ? WHERE id = ?",
            (
                (mem_a.repo_identifier, mem_a.id),
                (mem_b.repo_identifier, mem_b.id),
            ),
        )
        await db.db.commit()
        client = AsyncMock()
        client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {
                        "pair_index": 0,
                        "classification": "unrelated",
                        "reason": "must not be reached",
                    }
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=client,
        )

        assert stats == {
            "contradictions": 0,
            "temporal": 0,
            "checked": 0,
            "truncated": 0,
        }
        client.detect_contradictions.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_primary_support_fails_closed_before_relation_write(
        self,
        seeded_db,
        memory_store,
    ):
        db, _, _, mem_b, _ = seeded_db
        item = ContentItem(
            item_id="doc-bbb",
            title="doc-bbb",
            source_url="http://test/doc-bbb",
            last_modified=datetime.now(timezone.utc),
            version="2",
            extra={"page_id": "doc-bbb", "space_key": "TEST"},
        )
        await db.record_source_projection(
            project_source_item(
                source_id="src-b",
                source_type="confluence",
                run_id="projection-doc-bbb-new-revision",
                item=item,
                raw=RawContent(
                    item=item,
                    body=b"Current source revision no longer carries the old claim.",
                    content_type="text/plain",
                ),
                normalized=NormalizedContent(
                    item=item,
                    markdown_body="Current source revision no longer carries the old claim.",
                ),
            )
        )
        client = AsyncMock()
        client.detect_contradictions = AsyncMock(
            return_value=_mock_contradiction_response(
                [
                    {
                        "pair_index": 0,
                        "classification": "temporal",
                        "reason": "stale evidence must not be reused",
                    }
                ]
            )
        )

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=client,
        )

        assert stats == {
            "contradictions": 0,
            "temporal": 0,
            "checked": 0,
            "truncated": 0,
        }
        client.detect_contradictions.assert_not_called()
        async with db.db.execute(
            "SELECT COUNT(*) FROM relation_runs WHERE evidence_unit_id = ?",
            (f"eu-{mem_b.id}",),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        failures = await db.list_memory_audit_events(
            event_type="contradiction_detection_failed"
        )
        assert len(failures) == 1
        assert failures[0].reason == "evidence_preflight_failure"

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
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=mock_client,
        )

        assert stats["checked"] == 0
        assert stats["contradictions"] == 0
        mock_client.detect_contradictions.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_discovery_channel_failures_are_preserved_in_completion_audit(
        self,
        seeded_db,
        memory_store,
    ):
        db, _, _, _, mem_c = seeded_db
        selection = CrossDocumentCandidateSelection(
            discovery=(),
            audit={
                "candidate_count_kind": "windowed",
                "rank_window_size": 128,
                "selected_discovery_count": 0,
                "mandatory_candidate_count": 0,
            },
            telemetry={
                "channel_errors": [
                    CandidateBucket.SHARED_ENTITIES.value,
                    CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS.value,
                    CandidateBucket.LEXICAL_BM25.value,
                ]
            },
        )
        candidate_retriever = AsyncMock(spec=CrossDocumentCandidateRetriever)
        candidate_retriever.retrieve.return_value = selection
        candidate_retriever.load_selected_memories.return_value = (selection, {})
        llm_client = AsyncMock()

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_c.id],
            doc_id="doc-aaa",
            db=db,
            memory_store=memory_store,
            candidate_retriever=candidate_retriever,
            structured_llm_client=llm_client,
        )

        assert stats["checked"] == 0
        llm_client.detect_contradictions.assert_not_called()
        audit_rows = await db.list_memory_audit_events(
            event_type="contradiction_detection_completed"
        )
        assert len(audit_rows) == 1
        assert audit_rows[0].reason == "no_cross_doc_candidates"
        assert audit_rows[0].payload["retrieval"]["channel_errors"] == [
            CandidateBucket.LEXICAL_BM25.value,
            CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS.value,
            CandidateBucket.SHARED_ENTITIES.value,
        ]

    @pytest.mark.asyncio
    async def test_no_structured_llm_client_returns_empty(self, seeded_db, memory_store):
        """If LLM client is None, return empty stats without error."""
        db, _, _, mem_b, _ = seeded_db

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=None,
        )

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0, "truncated": 0}

    @pytest.mark.asyncio
    async def test_access_transition_before_relation_write_fails_closed(
        self,
        seeded_db,
        memory_store,
    ):
        db, _, mem_a, mem_b, _ = seeded_db

        async def transition_access_before_response(*args, **kwargs):
            del args, kwargs
            await db.db.execute(
                "UPDATE memories SET visibility = 'private', owner_user_id = ? WHERE id = ?",
                ("other-owner", mem_a.id),
            )
            await db.db.commit()
            return _mock_contradiction_response(
                [
                    {
                        "pair_index": 0,
                        "classification": "unrelated",
                        "reason": "not material to the contract test",
                    }
                ]
            )

        llm_client = AsyncMock()
        llm_client.detect_contradictions.side_effect = transition_access_before_response

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            candidate_retriever=_candidate_retriever(memory_store),
            structured_llm_client=llm_client,
        )

        assert stats["checked"] == 1
        async with db.db.execute(
            "SELECT COUNT(*) FROM relation_runs WHERE evidence_unit_id = ?",
            (f"eu-{mem_b.id}",),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        failures = await db.list_memory_audit_events(
            event_type="contradiction_detection_failed"
        )
        assert len(failures) == 1
        assert failures[0].reason == "candidate_selection_stale"

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
            candidate_retriever=_candidate_retriever(memory_store),
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
            candidate_retriever=_candidate_retriever(memory_store),
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
            candidate_retriever=_candidate_retriever(memory_store),
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
        assert audit_rows[0].payload["llm_calls"] == 1
        assert audit_rows[0].payload["retrieval"]["full_memory_rows_loaded"] == 1
        assert audit_rows[0].payload["retrieval"]["rank_window_size"] == 128
        assert audit_rows[0].payload["retrieval"]["selected_candidate_count"] == 1
        assert audit_rows[0].payload["retrieval"]["fused_candidate_count"] == 1
        assert audit_rows[0].payload["classifications"]["unrelated"] == 1

        async with db.db.execute(
            """SELECT rr.*
               FROM relation_runs rr
               JOIN relation_candidates rc ON rc.relation_run_id = rr.id
               WHERE rr.evidence_unit_id = ?
                 AND rc.memory_id = ?""",
            (f"eu-{mem_b.id}", mem_a.id),
        ) as cursor:
            relation_runs = [dict(row) async for row in cursor]
        assert len(relation_runs) == 1
        assert relation_runs[0]["lifecycle_action"] == LifecycleAction.NONE.value
        assert relation_runs[0]["status"] == "checked"
        candidates = await db.get_relation_candidates(relation_runs[0]["id"])
        assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
            (mem_a.id, CandidateBucket.HYBRID_DISCOVERY, True)
        ]
        evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
        assert evidence_unit is not None
        assert await db.get_evidence_relations(evidence_unit.id) == []
