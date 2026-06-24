"""Tests for human-intervention gates in reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
)
from memforge.memory.store import MemoryStore
from memforge.llm.structured import ReconciliationDecision, ReconciliationResponse, StructuredLlmError
from memforge.models import Memory, MemoryReview, RawMemory, ReconcileAction, ReconcileOperation, content_hash
from memforge.memory.review_service import ReviewKind, ReviewStatus
from memforge.pipeline.reconciler import _parse_decisions, reconcile_memories
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class FakeCollection:
    def __init__(self) -> None:
        self.upserted: list[str] = []
        self.deleted: list[str] = []

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, ids, embeddings=None, metadatas=None):
        self.upserted.extend(ids)

    def delete(self, ids):
        self.deleted.extend(ids)

    def get(self, ids=None, include=None):
        return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "reconciliation.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str, *, corroboration_count: int = 1) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        corroboration_count=corroboration_count,
        created_at=now,
        updated_at=now,
        status="active",
    )


async def _insert_doc(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "src-1", f"http://test/{doc_id}", doc_id, "TEST", now, "1", f"hash-{doc_id}", now),
    )
    await db.db.commit()


def test_parse_decisions_preserves_flag_for_review():
    raw = RawMemory(content="PostgreSQL version is 16", memory_type="fact")
    existing = [_memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=3)]

    [op] = _parse_decisions(
        [
            {
                "index": 0,
                "action": "SUPERSEDE",
                "memory_id": existing[0].id,
                "reason": "Version changed",
                "flag_for_review": True,
            }
        ],
        [raw],
        existing,
    )

    assert op.action == ReconcileAction.SUPERSEDE
    assert op.flag_for_review is True


def test_parse_decisions_can_remove_existing_memory_without_new_extraction():
    existing = [_memory("mem-old0001", "PostgreSQL version is 14")]

    [op] = _parse_decisions(
        [
            {
                "action": "DELETE",
                "memory_id": existing[0].id,
                "reason": "The updated document no longer supports this memory",
            }
        ],
        [],
        existing,
    )

    assert op.action == ReconcileAction.DELETE
    assert op.memory_id == existing[0].id
    assert op.memory is None


def test_parse_decisions_treats_null_index_as_existing_memory_audit():
    existing = [_memory("mem-old0001", "PostgreSQL version is 14")]

    [op] = _parse_decisions(
        [
            {
                "index": None,
                "action": "DELETE",
                "memory_id": existing[0].id,
                "reason": "The updated document no longer supports this memory",
            }
        ],
        [],
        existing,
    )

    assert op.action == ReconcileAction.DELETE
    assert op.memory_id == existing[0].id
    assert op.memory is None


@pytest.mark.asyncio
async def test_reconciliation_prompt_requires_canonical_replacement_content():
    class Client:
        def __init__(self) -> None:
            self.prompt = ""

        async def reconcile_memories(self, prompt: str, **kwargs):
            self.prompt = prompt
            return ReconciliationResponse(decisions=[ReconciliationDecision(index=0, action="NOOP", reason="ok")])

    client = Client()

    await reconcile_memories(
        new_extractions=[
            RawMemory(
                content="Option A is standalone and depends on the prospective slot builder.",
                memory_type="decision",
            )
        ],
        existing_memories=[
            _memory(
                "mem-old0001",
                "Option A should be dependent on the design of OD assignment validation.",
            )
        ],
        doc_type="design",
        structured_llm_client=client,
        updated_document="### Option A: Reuse Prospective Slot Building",
        update_mode="diff_guided",
        changed_hunks="-### Option A: Reuse Prospective Slot Building (Should be dependent on the design of OD assignment validation)\n+### Option A: Reuse Prospective Slot Building",
        update_plan_stats={"reason": "small_diff"},
    )

    assert "replacement memory content must state the current durable fact" in client.prompt
    assert "Do not write replacement content as edit history" in client.prompt
    assert "DELETE or SUPERSEDE an existing memory only when <changed_hunks>" in client.prompt
    assert "Do not DELETE solely because support is absent from unrelated context" in client.prompt


@pytest.mark.asyncio
async def test_reconciliation_llm_failure_with_existing_memories_fails_closed():
    class FailingClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            raise StructuredLlmError("structured unavailable")

    operations = await reconcile_memories(
        new_extractions=[RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")],
        existing_memories=[_memory("mem-existing", "Service uses PostgreSQL 15.")],
        doc_type="design",
        structured_llm_client=FailingClient(),
        updated_document="# Design\n\nService uses PostgreSQL 16.",
        update_mode="full_document",
    )

    assert operations == []


@pytest.mark.asyncio
async def test_reconciliation_llm_failure_is_audited_and_fails_closed(db):
    class FailingClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            raise StructuredLlmError("structured unavailable")

    await _insert_doc(db, "doc-runbook")
    old = _memory("mem-existing", "Service uses PostgreSQL 15.")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", support_kind="extracted", source_updated_at=None)

    adapters = build_sqlite_adapters(db, FakeCollection())
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=FailingClient(),
    )

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")],
        source_type="confluence",
        doc_type="design",
        document_content="# Design\n\nService uses PostgreSQL 16.",
        update_mode="full_document",
        source_updated_at=None,
    )

    rows = await db.list_memory_audit_events(event_type="reconciliation_failed")
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].doc_id == "doc-runbook"
    assert rows[0].decision == "skip_mutations"
    assert rows[0].reason == "structured_llm_error"
    assert rows[0].error == "structured unavailable"
    assert rows[0].payload["new_extraction_count"] == 1
    assert rows[0].payload["existing_memory_count"] == 1
    assert stats["added"] == 0
    assert stats["updated"] == 0
    assert (await db.get_memory(old.id)).content == "Service uses PostgreSQL 15."


@pytest.mark.asyncio
async def test_flagged_supersede_inserts_challenger_pending_review_and_keeps_incumbent_active(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    old = _memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=3)
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    challenger = RawMemory(content="PostgreSQL version is 16", memory_type="fact", confidence=0.9)

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory_id=old.id,
                memory=challenger,
                reason="Version changed",
                flag_for_review=True,
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[challenger],
        source_type="confluence",
        doc_type="runbook",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    pending = await db.list_memories(status="pending_review")
    assert stats["pending_review"] == 1
    assert stored_old.status == "active"
    assert len(pending) == 1
    assert pending[0].content == "PostgreSQL version is 16"
    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (pending[0].id,),
    ) as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_REVIEW.value
    assert relation_runs[0]["status"] == "review"
    assert relation_runs[0]["result_memory_id"] == pending[0].id
    candidates = await db.get_relation_candidates(relation_runs[0]["id"])
    assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
        (old.id, CandidateBucket.SAME_DOC_LINEAGE, True)
    ]


@pytest.mark.asyncio
async def test_pending_review_challenger_insert_rolls_back_relation_when_review_insert_fails(db):
    await _insert_doc(db, "doc-runbook")
    challenger = _memory("mem-review-chal", "PostgreSQL version is 16")
    challenger.status = "pending_review"
    review = MemoryReview(
        id="review-fails",
        kind=ReviewKind.SUPERSEDE.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id="mem-old0001",
        challenger_memory_id=challenger.id,
        reason="Version changed",
        created_at=datetime(2026, 6, 22, 12, 0, tzinfo=timezone.utc),
    )
    unit = EvidenceUnit(
        id="eu-review-fails",
        source_id="src-1",
        doc_id="doc-runbook",
        doc_revision_id="rev-1",
        source_type="confluence",
        source_anchor="doc-runbook",
        source_lineage_id="doc-runbook",
        source_metadata={},
        project_key="UNSORTED",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="PostgreSQL version is 16",
        excerpt="PostgreSQL version is 16",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )
    relation_outcome = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=RelationRunRecord(
            id="rel-run-review-fails",
            evidence_unit_id=unit.id,
            access_context_hash=None,
            candidate_count=1,
            mandatory_candidate_count=1,
            checked_candidate_count=1,
            incomplete_mandatory_buckets=(),
            classifier_version="memory-engine-v1",
            lifecycle_action=LifecycleAction.CREATE_REVIEW,
            review_case=ReviewCase.MANUAL_REVIEW_GATE,
            status="review",
            result_memory_id=challenger.id,
        ),
        relations=[
            EvidenceRelationRecord(
                evidence_unit_id=unit.id,
                memory_id=challenger.id,
                relation_type=RelationType.CONTRADICTS,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                is_authoritative_support=True,
                source_lineage_id=unit.source_lineage_id,
                confidence=0.9,
                reason="Version changed",
                classifier_version="memory-engine-v1",
                relation_run_id="rel-run-review-fails",
            )
        ],
    )
    await db.db.execute(
        """CREATE TRIGGER fail_memory_review_insert
           BEFORE INSERT ON memory_reviews
           BEGIN
             SELECT RAISE(ABORT, 'review insert failed');
           END"""
    )
    await db.db.commit()

    with pytest.raises(Exception, match="review insert failed"):
        await db.insert_memory_with_source_and_relation(
            challenger,
            doc_id="doc-runbook",
            source_type="confluence",
            excerpt="PostgreSQL version is 16",
            relation_outcome=relation_outcome,
            review=review,
            source_updated_at=None,
        )

    assert await db.get_memory(challenger.id) is None
    assert await db.get_memory_review(review.id) is None
    async with db.db.execute(
        "SELECT COUNT(*) AS total FROM relation_runs WHERE id = ?", ("rel-run-review-fails",)
    ) as cursor:
        row = await cursor.fetchone()
    assert row["total"] == 0


@pytest.mark.asyncio
async def test_reconcile_delete_removes_only_updated_document_support(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    await _insert_doc(db, "doc-other")
    old = _memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=2)
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-other", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="The updated document no longer supports this memory",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The updated runbook no longer mentions PostgreSQL 14.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    sources = await db.get_memory_sources(old.id)
    assert stats["deleted"] == 1
    assert stored_old.status == "active"
    assert [source.doc_id for source in sources] == ["doc-other"]


@pytest.mark.asyncio
async def test_reconcile_delete_retires_memory_when_current_doc_is_only_support(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    old = _memory("mem-old0001", "PostgreSQL version is 14")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", support_kind="extracted", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="The updated document no longer supports this memory",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The updated runbook no longer mentions PostgreSQL 14.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    sources = await db.get_memory_sources(old.id)
    audit_rows = await db.list_memory_audit_events(event_type="source_support_removal_retired_memory")
    assert stats["deleted"] == 1
    assert stored_old.status == "retired"
    assert sources == []
    assert old.id in collection.deleted
    assert audit_rows[0].memory_id == old.id


@pytest.mark.asyncio
async def test_high_corroboration_delete_removes_current_support_when_other_support_remains(
    db,
    monkeypatch,
):
    await _insert_doc(db, "doc-runbook")
    await _insert_doc(db, "doc-other")
    await _insert_doc(db, "doc-third")
    old = _memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=3)
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-other", "confluence", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-third", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="The updated document no longer supports this memory",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The updated runbook no longer mentions PostgreSQL 14.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    sources = await db.get_memory_sources(old.id)
    review_rows = await db.list_memories(status="pending_review")
    assert stats["deleted"] == 1
    assert stats["pending_review"] == 0
    assert stored_old.status == "active"
    assert [source.doc_id for source in sources] == ["doc-other", "doc-third"]
    assert review_rows == []


@pytest.mark.asyncio
async def test_high_corroboration_delete_review_records_relation_audit(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    old = _memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=3)
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="The updated document no longer supports this memory",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The updated runbook no longer mentions PostgreSQL 14.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    assert stats["pending_review"] == 1
    assert stored_old.status == "pending_review"
    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (old.id,),
    ) as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_REVIEW.value
    assert relation_runs[0]["status"] == "review"
    assert relation_runs[0]["result_memory_id"] == old.id
    assert relation_runs[0]["review_case"] == ReviewCase.MANUAL_REVIEW_GATE.value
    relations = await db.get_evidence_relations(relation_runs[0]["evidence_unit_id"])
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (old.id, RelationType.CONTRADICTS)
    ]
    unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert unit is not None
    assert unit.content == "The updated document no longer supports this memory"
    assert unit.content != old.content
    assert unit.evidence_provenance == EvidenceContentProvenance.NO_EXCERPT
    candidates = await db.get_relation_candidates(relation_runs[0]["id"])
    assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
        (old.id, CandidateBucket.SAME_DOC_LINEAGE, True)
    ]


@pytest.mark.asyncio
async def test_diff_guided_reconciliation_context_is_passed_to_reconciler(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    old = _memory("mem-old0001", "PostgreSQL version is 14")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )
    seen_kwargs: dict = {}

    async def fake_reconcile_memories(**kwargs):
        seen_kwargs.update(kwargs)
        return [
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=old.id,
                reason="Still supported",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The service now uses PostgreSQL 15.",
        update_mode="diff_guided",
        changed_hunks="-The service uses PostgreSQL 14.\n+The service uses PostgreSQL 15.",
        update_plan_stats={"diff_line_count": 2, "added_lines": 1, "removed_lines": 1},
        source_updated_at=None,
    )

    assert stats["noop"] == 1
    assert seen_kwargs["update_mode"] == "diff_guided"
    assert "PostgreSQL 15" in seen_kwargs["changed_hunks"]
    assert seen_kwargs["update_plan_stats"]["diff_line_count"] == 2


@pytest.mark.asyncio
async def test_reconciliation_decisions_are_audited_before_mutation(db, monkeypatch):
    await _insert_doc(db, "doc-runbook")
    await _insert_doc(db, "doc-other")
    old = _memory("mem-old0001", "PostgreSQL version is 14")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-runbook", "confluence", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-other", "confluence", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="The updated document no longer supports this memory",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    await engine.reconcile_and_persist(
        doc_id="doc-runbook",
        raw_memories=[],
        source_type="confluence",
        doc_type="runbook",
        document_content="The updated runbook no longer mentions PostgreSQL 14.",
        update_mode="diff_guided",
        changed_hunks="-The service uses PostgreSQL 14.",
        update_plan_stats={"diff_line_count": 1},
        source_updated_at=None,
    )

    rows = await db.list_memory_audit_events(event_type="reconciliation_decision_returned")
    assert len(rows) == 1
    assert rows[0].doc_id == "doc-runbook"
    assert rows[0].memory_id == old.id
    assert rows[0].decision == "DELETE"
    assert rows[0].reason == "The updated document no longer supports this memory"
    assert rows[0].payload["update_mode"] == "diff_guided"


@pytest.mark.parametrize("source_type", ["jira", "confluence", "teams", "github_pages", "local_markdown"])
@pytest.mark.asyncio
async def test_reconciliation_update_replaces_memory_lifecycle_for_document_sources(db, monkeypatch, source_type):
    await _insert_doc(db, "doc-current")
    old = _memory("mem-update-lifecycle", "Service A uses PostgreSQL 15.")
    await db.insert_memory(old)
    await db.add_memory_source(
        old.id,
        "doc-current",
        source_type,
        excerpt="Old extracted source.",
        support_kind="extracted",
        source_updated_at=None,
    )

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id=old.id,
                memory=RawMemory(
                    content="Service A uses PostgreSQL 16.",
                    memory_type="fact",
                    confidence=0.9,
                    tags=["database"],
                    extraction_context="The current design says Service A uses PostgreSQL 16.",
                ),
                reason="Current document refined the PostgreSQL version claim",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-current",
        raw_memories=[],
        source_type=source_type,
        doc_type="design-doc",
        document_content="Service A uses PostgreSQL 16.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    active = [memory for memory in await db.list_memories(limit=10) if memory.status == "active"]
    assert stats["updated"] == 1
    assert stats["superseded"] == 0
    assert stored_old is not None
    assert stored_old.status == "superseded"
    assert stored_old.superseded_by is not None
    assert stored_old.replacement_reason == "Current document refined the PostgreSQL version claim"
    assert stored_old.replacement_kind == "revision"

    assert len(active) == 1
    replacement = active[0]
    assert replacement.id == stored_old.superseded_by
    assert replacement.content == "Service A uses PostgreSQL 16."
    assert replacement.extraction_context == "The current design says Service A uses PostgreSQL 16."
    assert replacement.tags == ["database"]
    assert collection.deleted == [old.id]
    assert replacement.id in collection.upserted

    sources = await db.get_memory_sources(replacement.id)
    assert [(source.doc_id, source.source_type, source.support_kind, source.excerpt) for source in sources] == [
        ("doc-current", source_type, "extracted", "The current design says Service A uses PostgreSQL 16.")
    ]

    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (replacement.id,),
    ) as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.SUPERSEDE_MEMORY.value
    evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.source_type == source_type
    assert evidence_unit.doc_id == "doc-current"
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (replacement.id, RelationType.SUPPORTS)
    ]
    candidates = await db.get_relation_candidates(relation_runs[0]["id"])
    assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
        (old.id, CandidateBucket.SAME_DOC_LINEAGE, True)
    ]


@pytest.mark.asyncio
async def test_reconciliation_rejects_update_without_current_doc_extracted_support(db, monkeypatch):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-owner")
    target = _memory("mem-target1", "Service A uses PostgreSQL 15.")
    anchor = _memory("mem-anchor1", "Current doc extracted memory.")
    await db.insert_memory(target)
    await db.insert_memory(anchor)
    await db.add_memory_source(
        target.id, "doc-current", "confluence", support_kind="corroborated", source_updated_at=None
    )
    await db.add_memory_source(target.id, "doc-owner", "confluence", support_kind="extracted", source_updated_at=None)
    # The anchor makes reconciliation run while the target remains outside current-doc extracted authority.
    await db.add_memory_source(
        anchor.id, "doc-current", "confluence", support_kind="extracted", source_updated_at=None
    )

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id=target.id,
                memory=RawMemory(
                    content="Service A uses PostgreSQL 16.",
                    memory_type="fact",
                    confidence=0.9,
                ),
                reason="Model attempted to update a corroborated-only support edge",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-current",
        raw_memories=[],
        source_type="confluence",
        doc_type="design-doc",
        document_content="Current doc supports only as corroboration.",
        source_updated_at=None,
    )

    stored = await db.get_memory(target.id)
    rows = await db.list_memory_audit_events(event_type="reconciliation_authority_rejected")
    assert stats["skipped"] == 1
    assert stored.content == "Service A uses PostgreSQL 15."
    assert rows[0].memory_id == target.id
    assert rows[0].decision == "UPDATE"


@pytest.mark.asyncio
async def test_reconciliation_rejects_delete_for_corroborated_only_current_doc_support(db, monkeypatch):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-owner")
    target = _memory("mem-target2", "Service A uses PostgreSQL 15.")
    anchor = _memory("mem-anchor2", "Current doc extracted memory.")
    await db.insert_memory(target)
    await db.insert_memory(anchor)
    await db.add_memory_source(
        target.id, "doc-current", "confluence", support_kind="corroborated", source_updated_at=None
    )
    await db.add_memory_source(target.id, "doc-owner", "confluence", support_kind="extracted", source_updated_at=None)
    # The anchor makes reconciliation run while the target remains outside current-doc extracted authority.
    await db.add_memory_source(
        anchor.id, "doc-current", "confluence", support_kind="extracted", source_updated_at=None
    )

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=target.id,
                reason="Model attempted to delete via corroborated support",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-current",
        raw_memories=[],
        source_type="confluence",
        doc_type="design-doc",
        document_content="Current doc no longer supports the target memory.",
        source_updated_at=None,
    )

    sources = await db.get_memory_sources(target.id)
    rows = await db.list_memory_audit_events(event_type="reconciliation_authority_rejected")
    assert stats["skipped"] == 1
    assert {(source.doc_id, source.support_kind) for source in sources} == {
        ("doc-current", "corroborated"),
        ("doc-owner", "extracted"),
    }
    assert rows[0].memory_id == target.id
    assert rows[0].decision == "DELETE"


@pytest.mark.asyncio
async def test_reconciliation_routes_supersede_with_other_support_to_review(db, monkeypatch):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-support")
    old = _memory("mem-shared1", "Service A uses PostgreSQL 15.")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-current", "confluence", support_kind="extracted", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-support", "jira", support_kind="corroborated", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory_id=old.id,
                memory=RawMemory(
                    content="Service A uses PostgreSQL 16.",
                    memory_type="fact",
                    confidence=0.9,
                ),
                reason="Current document changed PostgreSQL 15 to 16",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-current",
        raw_memories=[],
        source_type="confluence",
        doc_type="design-doc",
        document_content="Service A uses PostgreSQL 16.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    challengers = await db.list_memories(status="pending_review")
    reviews = await db.list_memory_reviews(status="pending")
    assert stats["pending_review"] == 1
    assert stored_old.status == "active"
    assert len(challengers) == 1
    assert challengers[0].content == "Service A uses PostgreSQL 16."
    assert len(reviews) == 1
    assert reviews[0].incumbent_memory_id == old.id
    assert reviews[0].challenger_memory_id == challengers[0].id


@pytest.mark.asyncio
async def test_reconciliation_routes_update_with_other_extracted_support_to_review(db, monkeypatch):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-support")
    old = _memory("mem-shared2", "Service A uses PostgreSQL 15.")
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-current", "confluence", support_kind="extracted", source_updated_at=None)
    await db.add_memory_source(old.id, "doc-support", "jira", support_kind="extracted", source_updated_at=None)

    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )
    store._embed = AsyncMock(return_value=[0.1])
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id=old.id,
                memory=RawMemory(
                    content="Service A uses PostgreSQL 16.",
                    memory_type="fact",
                    confidence=0.9,
                ),
                reason="Current document changed PostgreSQL 15 to 16",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id="doc-current",
        raw_memories=[],
        source_type="confluence",
        doc_type="design-doc",
        document_content="Service A uses PostgreSQL 16.",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    challengers = await db.list_memories(status="pending_review")
    reviews = await db.list_memory_reviews(status="pending")
    assert stats["pending_review"] == 1
    assert stored_old.status == "active"
    assert stored_old.content == "Service A uses PostgreSQL 15."
    assert len(challengers) == 1
    assert challengers[0].content == "Service A uses PostgreSQL 16."
    assert len(reviews) == 1
    assert reviews[0].incumbent_memory_id == old.id
    assert reviews[0].challenger_memory_id == challengers[0].id
