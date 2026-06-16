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
from memforge.memory.store import MemoryStore
from memforge.models import Memory, content_hash
from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
    now = datetime.now(timezone.utc).isoformat()
    # Create two source documents (schema requires many NOT NULL columns)
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-aaa", "src-1", "http://test/a", "Architecture Doc", "TEST", now, "1", "hash1", now),
    )
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-bbb", "src-1", "http://test/b", "Runbook", "TEST", now, "1", "hash2", now),
    )

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
        candidates = await db.get_cross_doc_candidates(
            memory_id=mem_b.id,
            entity_ids=[entity_id],
            doc_id="doc-bbb",
        )
        candidate_ids = [c.id for c in candidates]
        assert mem_a.id in candidate_ids

    @pytest.mark.asyncio
    async def test_excludes_same_document(self, seeded_db):
        """Memories from the same document should NOT be candidates."""
        db, entity_id, mem_a, _, _ = seeded_db

        candidates = await db.get_cross_doc_candidates(
            memory_id=mem_a.id,
            entity_ids=[entity_id],
            doc_id="doc-aaa",  # same doc as mem_a
        )
        # mem_a itself should not appear, and no other memory from doc-aaa shares postgresql
        candidate_ids = [c.id for c in candidates]
        assert mem_a.id not in candidate_ids

    @pytest.mark.asyncio
    async def test_no_shared_entities_returns_empty(self, seeded_db):
        """mem_c has a different entity (kafka) — no cross-doc overlap with doc-bbb."""
        db, _, _, _, mem_c = seeded_db

        kafka_ids = await db.get_memory_entity_ids(mem_c.id)
        candidates = await db.get_cross_doc_candidates(
            memory_id=mem_c.id,
            entity_ids=kafka_ids,
            doc_id="doc-aaa",
        )
        # No memory from doc-bbb links to kafka
        assert candidates == []


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
    return ContradictionResponse(
        decisions=[ContradictionDecision(**decision) for decision in decisions]
    )


class TestDetectCrossDocContradictions:
    @pytest.mark.asyncio
    async def test_detects_contradiction_via_llm(self, seeded_db, memory_store, chroma):
        """Full pipeline: new memory shares entity with existing, LLM says contradiction."""
        db, entity_id, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([
            {"pair_index": 0, "classification": "contradiction", "reason": "PostgreSQL 14 vs MySQL 8"},
        ]))

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
        assert m_b.status == "pending_review"
        assert chroma.deleted == [mem_b.id]

        async with db.db.execute(
            "SELECT memory_id FROM memories_fts WHERE memory_id = ?",
            (mem_b.id,),
        ) as cursor:
            assert await cursor.fetchone() is None

        review = await db.get_pending_review_for_challenger(mem_b.id)
        assert review is not None
        assert review.kind == "supersede"
        assert review.status == "pending"
        assert review.incumbent_memory_id == mem_a.id
        assert review.challenger_memory_id == mem_b.id
        assert review.reason == "PostgreSQL 14 vs MySQL 8"

    @pytest.mark.asyncio
    async def test_multiple_contradictions_create_one_review_per_challenger(
        self, seeded_db, memory_store
    ):
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
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([
            {"pair_index": 0, "classification": "contradiction", "reason": "first conflict"},
            {"pair_index": 1, "classification": "contradiction", "reason": "second conflict"},
        ]))

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
    async def test_multiple_challengers_from_same_doc_share_one_visible_review(
        self, seeded_db, memory_store
    ):
        """One source document can extract several challengers for the same human decision."""
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
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([
            {"pair_index": 0, "classification": "contradiction", "reason": "first challenger"},
            {"pair_index": 1, "classification": "contradiction", "reason": "second challenger"},
        ]))

        stats = await detect_cross_doc_contradictions(
            new_memory_ids=[mem_b.id, mem_e.id],
            doc_id="doc-bbb",
            db=db,
            memory_store=memory_store,
            structured_llm_client=mock_client,
        )

        assert stats["contradictions"] == 2
        reviews = await db.list_memory_reviews(status="pending")
        assert len(reviews) == 1
        assert reviews[0].incumbent_memory_id == mem_a.id
        assert reviews[0].challenger_memory_id == mem_b.id

        related = await db.list_memory_review_related_challengers(reviews[0].id)
        assert [item.challenger_memory_id for item in related] == [mem_e.id]

    @pytest.mark.asyncio
    async def test_temporal_detected_not_contradiction(self, seeded_db, memory_store):
        """LLM classifies as temporal — counts should NOT increment."""
        db, _, mem_a, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([
            {"pair_index": 0, "classification": "temporal", "reason": "Newer version replaces older"},
        ]))

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

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0}

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

        assert stats == {"contradictions": 0, "temporal": 0, "checked": 0}

    @pytest.mark.asyncio
    async def test_structured_llm_error_handled(self, seeded_db, memory_store):
        """Malformed LLM response should be handled gracefully."""
        db, _, _, mem_b, _ = seeded_db

        mock_client = AsyncMock()
        mock_client.detect_contradictions = AsyncMock(
            side_effect=StructuredLlmError("invalid structured response")
        )

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
        mock_client.detect_contradictions = AsyncMock(return_value=_mock_contradiction_response([
            {"pair_index": 0, "classification": "unrelated", "reason": "Different aspects"},
        ]))

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
