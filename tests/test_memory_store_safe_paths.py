"""Safe memory write path tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memforge.memory.audit import AuditContext, MemoryAuditEvent, MemoryAuditLogger
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    RelationCandidateRecord,
    RelationDirection,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
)
from memforge.memory.lifecycle_plan import (
    LifecycleVectorOperation,
    LifecycleVectorTask,
    LifecycleVectorTaskStatus,
)
from memforge.models import DocumentRecord, Memory, MemoryReview, MemorySource, content_hash
from memforge.memory.review_service import ReviewKind, ReviewStatus
from memforge.memory.store import MemoryStore
from memforge.retrieval.document_index import DocumentVectorIndex
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class RecordingCollection:
    def __init__(self, query_ids: list[str] | None = None, distances: list[float] | None = None) -> None:
        self.query_ids = query_ids or []
        self.distances = distances or [0.01 for _ in self.query_ids]
        self.upserted: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.documents: dict[str, str] = {}
        self.deleted: list[str] = []

    def query(self, **kwargs):
        return {"ids": [self.query_ids], "distances": [self.distances]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = dict(metadatas[index] if metadatas else {})
            if embeddings:
                self.embeddings[record_id] = embeddings[index]
                self.upserted[record_id]["embedding"] = embeddings[index]
            if documents:
                self.documents[record_id] = documents[index]
                self.upserted[record_id]["document"] = documents[index]

    def delete(self, *, ids) -> None:
        self.deleted.extend(ids)
        for record_id in ids:
            self.upserted.pop(record_id, None)
            self.embeddings.pop(record_id, None)
            self.documents.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        selected_ids = [record_id for record_id in (ids or list(self.upserted)) if record_id in self.upserted]
        result: dict[str, Any] = {"ids": selected_ids}
        include = include or []
        if "metadatas" in include:
            result["metadatas"] = [self.upserted[record_id] for record_id in selected_ids]
        if "embeddings" in include:
            result["embeddings"] = [self.embeddings.get(record_id) for record_id in selected_ids]
        if "documents" in include:
            result["documents"] = [self.documents.get(record_id) for record_id in selected_ids]
        return result


class AmbiguousSequence:
    def __init__(self, values) -> None:
        self.values = values

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an array with more than one element is ambiguous")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class ArrayLikeEmbeddingCollection(RecordingCollection):
    def get(self, *, ids=None, include=None):
        result = super().get(ids=ids, include=include)
        if "embeddings" in result:
            result["embeddings"] = AmbiguousSequence(result["embeddings"])
        return result


class FailingDeleteCollection(RecordingCollection):
    def delete(self, *, ids) -> None:
        raise RuntimeError("delete failed")


class FailingSpecificDeleteCollection(RecordingCollection):
    def __init__(self, failing_id: str) -> None:
        super().__init__()
        self.failing_id = failing_id

    def delete(self, *, ids) -> None:
        if self.failing_id in ids:
            raise RuntimeError("delete failed")
        super().delete(ids=ids)


class MutatingFailingDeleteCollection(RecordingCollection):
    def seed(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def delete(self, *, ids) -> None:
        super().delete(ids=ids)
        raise RuntimeError("delete failed after mutation")


class InsertThenFailingDeleteCollection(RecordingCollection):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def delete(self, *, ids) -> None:
        if self.failures_remaining:
            self.failures_remaining -= 1
            for record_id in ids:
                self.upsert(
                    ids=[record_id],
                    embeddings=[[9.0, 9.0, 9.0]],
                    documents=[f"unexpected {record_id}"],
                    metadatas=[{"content_hash": f"unexpected-{record_id}", "version": "bad"}],
                )
            raise RuntimeError("delete failed after mutation")
        super().delete(ids=ids)


class FailingUpsertCollection(RecordingCollection):
    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        raise RuntimeError("upsert failed")


class FailingQueryCollection(RecordingCollection):
    def query(self, **kwargs):
        raise RuntimeError("query failed")


class FailingSpecificUpsertCollection(RecordingCollection):
    def __init__(self, failing_id: str) -> None:
        super().__init__()
        self.failing_id = failing_id

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        if self.failing_id in ids:
            raise RuntimeError("upsert failed")
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


class MutatingFailingUpsertCollection(RecordingCollection):
    def seed(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        raise RuntimeError("upsert failed after mutation")


class MutatingFailingUpsertAndDeleteCollection(MutatingFailingUpsertCollection):
    def delete(self, *, ids) -> None:
        raise RuntimeError("delete failed during rollback")


class MutatingFailingSpecificUpsertAndDeleteCollection(RecordingCollection):
    def __init__(self, failing_id: str) -> None:
        super().__init__()
        self.failing_id = failing_id

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        if self.failing_id in ids:
            raise RuntimeError("upsert failed after mutation")

    def delete(self, *, ids) -> None:
        if self.failing_id in ids:
            raise RuntimeError("delete failed")
        super().delete(ids=ids)


class FailingSourceInsertDatabase:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add_memory_source(self, *args, **kwargs):
        raise RuntimeError("source insert failed")

    async def insert_memory_with_source_and_relation(self, *args, **kwargs):
        raise RuntimeError("source insert failed")

    async def supersede_memory_with_source_and_relation(self, *args, **kwargs):
        raise RuntimeError("source insert failed")

    def __getattr__(self, name: str):
        return getattr(self._db, name)


class FailingSecondDeleteCollection(RecordingCollection):
    def __init__(self) -> None:
        super().__init__()
        self.delete_calls = 0

    def delete(self, *, ids) -> None:
        self.delete_calls += 1
        if self.delete_calls > 1:
            raise RuntimeError("second delete failed")
        super().delete(ids=ids)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "safe-paths.db"))
    await database.connect()
    yield database
    await database.close()


async def _insert_doc(db: Database, doc_id: str = "doc-1", source: str = "src-1") -> None:
    if await db.get_source(source) is None:
        await db.upsert_source(
            id=source,
            type="confluence",
            name=source,
            config_json="{}",
            access_policy="workspace",
            owner_user_id="owner-1",
        )
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, source, f"http://test/{doc_id}", doc_id, "TEST", now, "1", f"hash-{doc_id}", now),
    )
    await db.db.commit()


async def _insert_doc_side_tables(db: Database, doc_id: str, source: str = "src-1") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO document_metadata
           (doc_id, summary, tags, entities, doc_type, complexity, enriched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "Summary", '["tag"]', '[{"name":"Entity","tags":[]}]', "doc", "low", now),
    )
    await db.db.execute(
        """INSERT INTO document_relationships
           (source_doc_id, target_doc_id, target_title, relation_type, confidence, link_source)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (doc_id, "target-doc", "Target", "mentions", 0.8, "enrichment"),
    )
    await db.db.execute(
        """INSERT INTO changelog
           (doc_id, change_type, previous_version, current_version, ai_change_summary, detected_at, title, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "created", None, "1", "created", now, doc_id, source),
    )
    await db.db.execute(
        """INSERT INTO agent_session_receipts
           (doc_id, source_id, client, session_id, trigger, workspace, repo, branch, commit_sha,
            history_window_kind, history_window_start, history_window_end, submitted_at,
            document_hash, source_kind, document_uri, metadata, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            source,
            "codex",
            "session-1",
            "manual",
            "/repo",
            "repo",
            "main",
            "abc",
            "recent",
            None,
            None,
            now,
            f"hash-{doc_id}",
            "agent_session",
            f"file:///{doc_id}.md",
            "{}",
            now,
        ),
    )
    await db.db.commit()


async def _doc_side_counts(db: Database, doc_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    queries = {
        "metadata": ("SELECT COUNT(*) FROM document_metadata WHERE doc_id = ?", (doc_id,)),
        "relationships": (
            "SELECT COUNT(*) FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
            (doc_id, doc_id),
        ),
        "changelog": ("SELECT COUNT(*) FROM changelog WHERE doc_id = ?", (doc_id,)),
        "receipts": ("SELECT COUNT(*) FROM agent_session_receipts WHERE doc_id = ?", (doc_id,)),
    }
    for key, (sql, params) in queries.items():
        async with db.db.execute(sql, params) as cursor:
            counts[key] = (await cursor.fetchone())[0]
    return counts


async def _source_bookkeeping_counts(db: Database, source: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    queries = {
        "sync_state": ("SELECT COUNT(*) FROM sync_state WHERE source = ?", (source,)),
        "sync_history": ("SELECT COUNT(*) FROM sync_history WHERE source = ?", (source,)),
    }
    for key, (sql, params) in queries.items():
        async with db.db.execute(sql, params) as cursor:
            counts[key] = (await cursor.fetchone())[0]
    return counts


def _memory(mem_id: str, content: str, *, status: str = "active") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
    )


def _relation_outcome_bundle(
    *,
    unit_id: str,
    run_id: str,
    doc_id: str,
    memory_id: str,
    relation_type: RelationType = RelationType.SUPPORTS,
    action: LifecycleAction = LifecycleAction.ATTACH_SUPPORT,
) -> RelationOutcomeBundle:
    unit = EvidenceUnit(
        id=unit_id,
        source_id="src-1",
        doc_id=doc_id,
        doc_revision_id="1",
        source_type="confluence",
        source_anchor=f"{doc_id}#claim",
        source_lineage_id=doc_id,
        project_key="TEST",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="Evidence excerpt",
        excerpt="Evidence excerpt",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )
    run = RelationRunRecord(
        id=run_id,
        evidence_unit_id=unit.id,
        access_context_hash=unit.access_context_hash,
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=action,
        review_case=None,
        status="applied",
        audit={"source": "test"},
    )
    candidate = RelationCandidateRecord(
        relation_run_id=run.id,
        evidence_unit_id=unit.id,
        memory_id=memory_id,
        bucket=CandidateBucket.SAME_DOC_LINEAGE,
        bucket_rank=0,
        candidate_rank=0,
        score=1.0,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="same doc lineage",
    )
    relation = EvidenceRelationRecord(
        evidence_unit_id=unit.id,
        memory_id=memory_id,
        relation_type=relation_type,
        direction=(
            RelationDirection.CHALLENGER_TO_CANDIDATE
            if relation_type is RelationType.REFINES
            else RelationDirection.SYMMETRIC
        ),
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        is_authoritative_support=True,
        source_lineage_id=unit.source_lineage_id,
        confidence=0.9,
        reason="same claim",
        proposed_memory_content="Evidence excerpt",
        excerpt=unit.excerpt,
        classifier_version="test-v1",
        relation_run_id=run.id,
    )
    return RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=run,
        candidates=(candidate,),
        relations=(relation,),
    )


def _store(
    db: Database,
    collection: RecordingCollection,
    document_collection: RecordingCollection | None = None,
) -> MemoryStore:
    logger = MemoryAuditLogger(db, default_context=AuditContext(actor_type="test", run_id="run-1"))
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        document_index=DocumentVectorIndex(document_collection),
        embed_cfg={},
        audit_logger=logger,
    )

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[assignment]
    return store


@pytest.mark.asyncio
async def test_dedup_ignores_stale_chroma_candidate(db: Database):
    await _insert_doc(db)
    retired = _memory("mem-stale1", "Old inactive fact", status="retired")
    await db.insert_memory(retired)
    candidate = _memory("mem-new001", "New active fact")
    collection = RecordingCollection(query_ids=[retired.id], distances=[0.01])
    store = _store(db, collection)

    result = await store.deduplicate_and_insert(candidate, "doc-1", "confluence", source_updated_at=None)

    stored_candidate = await db.get_memory(candidate.id)
    sources = await db.get_memory_sources(retired.id)
    audit_rows = await db.list_memory_audit_events(event_type="stale_chroma_candidate_detected")
    assert result == "inserted"
    assert stored_candidate is not None
    assert sources == []
    assert [row.memory_id for row in audit_rows] == [retired.id]


@pytest.mark.asyncio
async def test_dedup_corrobates_active_chroma_candidate(db: Database):
    await _insert_doc(db)
    active = _memory("mem-active1", "Existing active fact")
    await db.insert_memory(active)
    candidate = _memory("mem-new002", "Equivalent active fact")
    collection = RecordingCollection(query_ids=[active.id], distances=[0.01])
    store = _store(db, collection)
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    result = await store.deduplicate_and_insert(
        candidate,
        "doc-1",
        "confluence",
        excerpt="same fact",
        source_updated_at=source_updated_at,
    )

    stored_candidate = await db.get_memory(candidate.id)
    sources = await db.get_memory_sources(active.id)
    assert result == "corroborated"
    assert stored_candidate is None
    assert [(source.doc_id, source.support_kind) for source in sources] == [("doc-1", "extracted")]
    assert sources[0].source_updated_at == source_updated_at

    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (active.id,),
    ) as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.ATTACH_SUPPORT.value
    assert relation_runs[0]["result_memory_id"] == active.id
    evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.doc_id == "doc-1"
    assert evidence_unit.source_type == "confluence"
    assert evidence_unit.excerpt == "same fact"
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (active.id, RelationType.SUPPORTS)
    ]
    candidates = await db.get_relation_candidates(relation_runs[0]["id"])
    assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
        (active.id, CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS, True)
    ]


@pytest.mark.asyncio
async def test_agent_claim_retry_with_unknown_source_timestamp_clears_stale_value(db: Database):
    await db.upsert_source(
        "agent-source-codex",
        "agent_session",
        "Codex Session",
        "{}",
        "private",
        "andrew.sun01@sap.com",
        created_by_user_id="andrew.sun01@sap.com",
    )
    await _insert_doc(db, source="agent-source-codex")
    observed_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)
    source_updated_at = datetime(2026, 6, 19, 8, 30, tzinfo=timezone.utc)
    memory = _memory("mem-agent-claim-source-updated", "Agent claim source timestamp")

    await db.insert_memory_and_upsert_agent_claim(
        memory,
        doc_id="doc-1",
        source_type="agent_session",
        excerpt="initial excerpt",
        relation_outcome=None,
        claim_id="claim-source-updated",
        concept_id="concept-source-updated",
        display_anchor="source-updated",
        claim_text="Agent claim source timestamp",
        memory_type="fact",
        tags=["agent-session"],
        confidence=0.9,
        observed_at=observed_at,
        source_updated_at=source_updated_at,
        concept_projection={
            "concept_id": "concept-source-updated",
            "source_id": "agent-source-codex",
            "owner_user_id": "andrew.sun01@sap.com",
            "workspace": "/workspace",
            "repo_identifier": "github.com/shno-labs/mem-forge",
            "concept_type": "topic",
            "concept_path": "concepts/source-updated.md",
            "title": "Source updated",
            "markdown_body": "# Source updated\n",
            "frontmatter": {},
        },
    )
    assert (await db.get_memory_sources(memory.id))[0].source_updated_at == source_updated_at

    await db.insert_memory_and_upsert_agent_claim(
        memory,
        doc_id="doc-1",
        source_type="agent_session",
        excerpt="retry excerpt",
        relation_outcome=None,
        claim_id="claim-source-updated",
        concept_id="concept-source-updated",
        display_anchor="source-updated",
        claim_text="Agent claim source timestamp",
        memory_type="fact",
        tags=["agent-session"],
        confidence=0.9,
        observed_at=observed_at,
        source_updated_at=None,
        concept_projection={
            "concept_id": "concept-source-updated",
            "source_id": "agent-source-codex",
            "owner_user_id": "andrew.sun01@sap.com",
            "workspace": "/workspace",
            "repo_identifier": "github.com/shno-labs/mem-forge",
            "concept_type": "topic",
            "concept_path": "concepts/source-updated.md",
            "title": "Source updated",
            "markdown_body": "# Source updated\n",
            "frontmatter": {},
        },
    )

    sources = await db.get_memory_sources(memory.id)
    assert len(sources) == 1
    assert sources[0].excerpt == "retry excerpt"
    assert sources[0].source_updated_at is None


@pytest.mark.asyncio
async def test_restore_memory_source_snapshot_rejects_naive_source_updated_at(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-naive-source-updated", "Naive source timestamp")
    await db.insert_memory(memory)

    with pytest.raises(ValueError, match="timezone"):
        await db.restore_memory_source_snapshot(
            MemorySource(
                memory_id=memory.id,
                doc_id="doc-1",
                source_type="agent_session",
                source_id="src-1",
                excerpt="source excerpt",
                support_kind="extracted",
                added_at=datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc),
                source_updated_at=datetime(2026, 6, 19, 8, 30),
            )
        )

    assert await db.get_memory_sources(memory.id) == []


@pytest.mark.asyncio
async def test_dedup_query_failure_aborts_and_records_failed_index_event(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-query-fail", "Should not insert without dedup")
    store = _store(db, FailingQueryCollection())

    with pytest.raises(RuntimeError, match="query failed"):
        await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert await db.get_memory(memory.id) is None
    assert "index_operation_failed" in {row.event_type for row in audit_rows}
    assert "memory_insert_committed" not in {row.event_type for row in audit_rows}


@pytest.mark.asyncio
async def test_insert_audit_events_share_one_operation_id(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-auditop", "Grouped audit fact")
    store = _store(db, RecordingCollection())

    await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    event_types = {row.event_type for row in audit_rows}
    operation_ids = {row.operation_id for row in audit_rows}
    assert {
        "fts_upsert_committed",
        "chroma_upsert_attempted",
        "chroma_upsert_committed",
        "memory_insert_committed",
    }.issubset(event_types)
    assert len(operation_ids) == 1


@pytest.mark.asyncio
async def test_add_source_support_records_audit_event(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-support1", "Supported fact")
    await db.insert_memory(memory)
    store = _store(db, RecordingCollection())

    result = await store.add_source_support(
        memory.id,
        "doc-1",
        "jira",
        excerpt="Supported fact",
        support_kind="corroborated",
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(event_type="source_support_added")
    assert result == "inserted"
    assert [(row.memory_id, row.support_kind) for row in audit_rows] == [(memory.id, "corroborated")]


@pytest.mark.asyncio
async def test_add_source_support_records_support_and_relation_atomically(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-support-relation", "Supported fact")
    await db.insert_memory(memory)
    store = _store(db, RecordingCollection())

    result = await store.add_source_support(
        memory.id,
        "doc-1",
        "jira",
        excerpt="Supported fact",
        support_kind="corroborated",
        source_updated_at=None,
        relation_outcome=_relation_outcome_bundle(
            unit_id="eu-support-relation",
            run_id="relrun-support-relation",
            doc_id="doc-1",
            memory_id=memory.id,
        ),
    )

    sources = await db.get_memory_sources(memory.id)
    relation_run = await db.get_relation_run("relrun-support-relation")
    relations = await db.get_evidence_relations("eu-support-relation")
    assert result == "inserted"
    assert [(source.memory_id, source.doc_id, source.support_kind) for source in sources] == [
        (memory.id, "doc-1", "corroborated")
    ]
    assert relation_run is not None
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (memory.id, RelationType.SUPPORTS)
    ]


@pytest.mark.asyncio
async def test_add_source_support_rolls_back_when_relation_bundle_fails(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
):
    await _insert_doc(db)
    memory = _memory("mem-support-relation-fail", "Supported fact")
    await db.insert_memory(memory)
    store = _store(db, RecordingCollection())

    async def fail_relation_bundle(*args, **kwargs):
        raise RuntimeError("relation bundle failed")

    monkeypatch.setattr(db, "_record_relation_outcome_bundle_unlocked", fail_relation_bundle)

    with pytest.raises(RuntimeError, match="relation bundle failed"):
        await store.add_source_support(
            memory.id,
            "doc-1",
            "jira",
            excerpt="Supported fact",
            support_kind="corroborated",
            source_updated_at=None,
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-support-relation-fail",
                run_id="relrun-support-relation-fail",
                doc_id="doc-1",
                memory_id=memory.id,
            ),
        )

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    assert stored is not None
    assert stored.corroboration_count == memory.corroboration_count
    assert sources == []


@pytest.mark.asyncio
async def test_update_memory_refreshes_chroma_embedding(db: Database):
    memory = _memory("mem-update1", "Old content")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.update_memory(memory.id, "New content", new_confidence=0.8)

    stored = await db.get_memory(memory.id)
    audit_rows = await db.list_memory_audit_events(event_type="memory_update_committed")
    assert stored.content == "New content"
    assert collection.upserted[memory.id]["confidence"] == 0.8
    assert [row.memory_id for row in audit_rows] == [memory.id]


@pytest.mark.asyncio
async def test_retire_expired_memories_removes_search_indexes(db: Database):
    expired = _memory("mem-expired", "Temporary fact")
    expired.valid_until = datetime.now(timezone.utc) - timedelta(days=1)
    await db.insert_memory(expired)
    collection = RecordingCollection()
    store = _store(db, collection)

    retired_count = await store.retire_expired_memories()

    stored = await db.get_memory(expired.id)
    audit_rows = await db.list_memory_audit_events(event_type="memory_retire_committed")
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (expired.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert retired_count == 1
    assert stored.status == "retired"
    assert fts_count == 0
    assert collection.deleted == [expired.id]
    assert [row.memory_id for row in audit_rows] == [expired.id]


@pytest.mark.asyncio
async def test_mark_pending_review_removes_indexes_and_records_audit(db: Database):
    memory = _memory("mem-pending", "Needs review")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.mark_pending_review(memory.id, reason="conflict")

    stored = await db.get_memory(memory.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert stored.status == "pending_review"
    assert fts_count == 0
    assert collection.deleted == [memory.id]
    assert "memory_pending_review_committed" in {row.event_type for row in audit_rows}
    assert len({row.operation_id for row in audit_rows}) == 1


@pytest.mark.asyncio
async def test_mark_pending_review_rolls_back_when_relation_bundle_fails(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
):
    await _insert_doc(db)
    memory = _memory("mem-pending-relation-fail", "Needs review")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    async def fail_status_with_relation(*args, **kwargs):
        raise RuntimeError("relation bundle failed")

    monkeypatch.setattr(db, "update_memory_status_with_relation_outcome", fail_status_with_relation)

    with pytest.raises(RuntimeError, match="relation bundle failed"):
        await store.mark_pending_review(
            memory.id,
            reason="conflict",
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-pending-relation-fail",
                run_id="relrun-pending-relation-fail",
                doc_id="doc-1",
                memory_id=memory.id,
                relation_type=RelationType.CONTRADICTS,
                action=LifecycleAction.CREATE_REVIEW,
            ),
        )

    stored = await db.get_memory(memory.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert stored is not None
    assert stored.status == "active"
    assert fts_count == 1


@pytest.mark.asyncio
async def test_mark_pending_review_with_relation_rejects_superseded_memory(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-pending-superseded", "Superseded memory")
    await db.insert_memory(memory)
    replacement = _memory("mem-pending-replacement", "Replacement memory")
    await db.supersede_memory(
        memory.id,
        replacement,
        replacement_reason="newer evidence",
        replacement_kind="revision",
    )

    store = _store(db, RecordingCollection())

    with pytest.raises(RuntimeError, match="cannot transition to pending_review"):
        await store.mark_pending_review(
            memory.id,
            reason="conflict after supersede",
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-pending-superseded",
                run_id="relrun-pending-superseded",
                doc_id="doc-1",
                memory_id=memory.id,
                relation_type=RelationType.CONTRADICTS,
                action=LifecycleAction.CREATE_REVIEW,
            ),
        )

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert stored.status == "superseded"
    async with db.db.execute(
        "SELECT COUNT(*) FROM relation_runs WHERE id = ?",
        ("relrun-pending-superseded",),
    ) as cursor:
        row = await cursor.fetchone()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_mark_pending_review_with_case_rejects_resolved_review_collision_and_restores_indexes(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-pending-review-collision", "Needs review")
    await db.insert_memory(memory)
    review = MemoryReview(
        id="review-resolved-collision",
        kind=ReviewKind.SUPERSEDE.value,
        status=ReviewStatus.APPROVED.value,
        incumbent_memory_id=memory.id,
        challenger_memory_id=memory.id,
        reason="already resolved",
        created_at=datetime.now(timezone.utc),
        resolved_at=datetime.now(timezone.utc),
    )
    await db.insert_memory_review(review)
    collection = RecordingCollection()
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="already exists with status approved"):
        await store.mark_pending_review_with_case(
            memory.id,
            reason="conflict after resolved review",
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-pending-review-collision",
                run_id="relrun-pending-review-collision",
                doc_id="doc-1",
                memory_id=memory.id,
                relation_type=RelationType.CONTRADICTS,
                action=LifecycleAction.CREATE_REVIEW,
            ),
            review=MemoryReview(
                id=review.id,
                kind=ReviewKind.SUPERSEDE.value,
                status=ReviewStatus.PENDING.value,
                incumbent_memory_id=memory.id,
                challenger_memory_id=memory.id,
                reason="new review",
                created_at=datetime.now(timezone.utc),
            ),
        )

    stored = await db.get_memory(memory.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert stored is not None
    assert stored.status == "active"
    assert fts_count == 1
    async with db.db.execute(
        "SELECT COUNT(*) FROM relation_runs WHERE id = ?",
        ("relrun-pending-review-collision",),
    ) as cursor:
        relation_count = (await cursor.fetchone())[0]
    assert relation_count == 0


@pytest.mark.asyncio
async def test_supersede_memory_records_old_and_new_index_audit(db: Database):
    await _insert_doc(db)
    old = _memory("mem-oldsup", "Old superseded fact")
    await db.insert_memory(old)
    new = _memory("mem-newsup", "New superseding fact")
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.supersede_memory(
        old.id,
        new,
        "doc-1",
        "confluence",
        replacement_kind="supersession",
        replacement_reason="newer source",
        source_updated_at=None,
    )

    old_rows = await db.list_memory_audit_events(memory_id=old.id)
    new_rows = await db.list_memory_audit_events(memory_id=new.id)
    all_rows = old_rows + new_rows
    event_types = {row.event_type for row in all_rows}
    assert {
        "memory_supersede_attempted",
        "fts_delete_committed",
        "chroma_delete_committed",
        "chroma_upsert_committed",
        "memory_supersede_committed",
    }.issubset(event_types)
    assert len({row.operation_id for row in all_rows}) == 1
    assert old.id in collection.deleted
    assert new.id in collection.upserted


@pytest.mark.asyncio
async def test_supersede_memory_rolls_back_when_relation_bundle_fails(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
):
    await _insert_doc(db)
    old = _memory("mem-old-relation-fail", "Old fact")
    await db.insert_memory(old)
    new = _memory("mem-new-relation-fail", "New fact")
    collection = RecordingCollection()
    store = _store(db, collection)

    async def fail_relation_bundle(*args, **kwargs):
        raise RuntimeError("relation bundle failed")

    monkeypatch.setattr(db, "_record_relation_outcome_bundle_unlocked", fail_relation_bundle)

    with pytest.raises(RuntimeError, match="relation bundle failed"):
        await store.supersede_memory(
            old.id,
            new,
            "doc-1",
            "confluence",
            replacement_kind="supersession",
            replacement_reason="newer source",
            source_updated_at=None,
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-supersede-relation-fail",
                run_id="relrun-supersede-relation-fail",
                doc_id="doc-1",
                memory_id=new.id,
                relation_type=RelationType.REFINES,
                action=LifecycleAction.SUPERSEDE_MEMORY,
            ),
        )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (old.id,)) as cursor:
        old_fts_count = (await cursor.fetchone())[0]
    assert stored_old is not None
    assert stored_old.status == "active"
    assert stored_new is None
    assert old_fts_count == 1


@pytest.mark.asyncio
async def test_supersede_memory_does_not_blindly_carry_support_to_revision(db: Database):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-support")
    old = _memory("mem-old-support", "Old supported fact")
    await db.insert_memory(old)
    current_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)
    support_updated_at = datetime(2026, 6, 21, 4, 23, 51, tzinfo=timezone.utc)
    await db.add_memory_source(
        old.id,
        "doc-current",
        "confluence",
        "Old excerpt",
        support_kind="extracted",
        source_updated_at=current_updated_at,
    )
    await db.add_memory_source(
        old.id,
        "doc-support",
        "jira",
        "Still valid support",
        support_kind="corroborated",
        source_updated_at=support_updated_at,
    )
    new = _memory("mem-new-support", "New supported fact")
    collection = RecordingCollection()
    store = _store(db, collection)
    replacement_updated_at = datetime(2026, 6, 22, 4, 23, 51, tzinfo=timezone.utc)

    await store.supersede_memory(
        old.id,
        new,
        "doc-current",
        "confluence",
        excerpt="New excerpt",
        replacement_reason="source revision",
        replacement_kind="revision",
        source_updated_at=replacement_updated_at,
    )

    old_sources = await db.get_memory_sources(old.id)
    new_sources = await db.get_memory_sources(new.id)
    stored_old = await db.get_memory(old.id)
    assert stored_old.replacement_kind == "revision"
    assert sorted(
        (source.doc_id, source.source_type, source.support_kind, source.excerpt, source.source_updated_at)
        for source in old_sources
    ) == [
        ("doc-current", "confluence", "extracted", "Old excerpt", current_updated_at),
        ("doc-support", "jira", "corroborated", "Still valid support", support_updated_at),
    ]
    assert sorted(
        (source.doc_id, source.source_type, source.support_kind, source.excerpt, source.source_updated_at)
        for source in new_sources
    ) == [
        ("doc-current", "confluence", "extracted", "New excerpt", replacement_updated_at),
        ("doc-support", "jira", "corroborated", "Still valid support", support_updated_at),
    ]


@pytest.mark.asyncio
async def test_supersession_does_not_carry_old_support_to_replacement(db: Database):
    await _insert_doc(db, "doc-current")
    await _insert_doc(db, "doc-support")
    old = _memory("mem-old-supersession-support", "Old obsolete fact")
    await db.insert_memory(old)
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)
    await db.add_memory_source(
        old.id,
        "doc-current",
        "confluence",
        "Old excerpt",
        support_kind="extracted",
        source_updated_at=source_updated_at,
    )
    await db.add_memory_source(
        old.id,
        "doc-support",
        "jira",
        "Old corroboration",
        support_kind="corroborated",
        source_updated_at=None,
    )
    new = _memory("mem-new-supersession-support", "Replacement fact with different meaning")
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.supersede_memory(
        old.id,
        new,
        "doc-current",
        "confluence",
        excerpt="Replacement excerpt",
        replacement_reason="old claim invalidated",
        replacement_kind="supersession",
        source_updated_at=source_updated_at,
    )

    old_sources = await db.get_memory_sources(old.id)
    new_sources = await db.get_memory_sources(new.id)
    stored_old = await db.get_memory(old.id)
    assert stored_old.replacement_kind == "supersession"
    assert sorted(
        (source.doc_id, source.source_type, source.support_kind, source.excerpt) for source in old_sources
    ) == [
        ("doc-current", "confluence", "extracted", "Old excerpt"),
        ("doc-support", "jira", "corroborated", "Old corroboration"),
    ]
    assert [
        (source.doc_id, source.source_type, source.support_kind, source.excerpt, source.source_updated_at)
        for source in new_sources
    ] == [("doc-current", "confluence", "extracted", "Replacement excerpt", source_updated_at)]


@pytest.mark.asyncio
async def test_supersede_memory_snapshots_array_like_chroma_embeddings(db: Database):
    await _insert_doc(db)
    old = _memory("mem-array-old", "Old array-like snapshot fact")
    await db.insert_memory(old)
    new = _memory("mem-array-new", "New array-like snapshot fact")
    collection = ArrayLikeEmbeddingCollection()
    collection.upsert(
        ids=[old.id],
        embeddings=[[0.4, 0.5, 0.6]],
        documents=["old semantic text"],
        metadatas=[{"content_hash": old.content_hash}],
    )
    store = _store(db, collection)

    await store.supersede_memory(
        old.id,
        new,
        "doc-1",
        "confluence",
        replacement_kind="supersession",
        replacement_reason="newer source",
        source_updated_at=None,
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    assert stored_old.status == "superseded"
    assert stored_new is not None
    assert new.id in collection.upserted


@pytest.mark.asyncio
async def test_supersede_audit_uses_old_memory_as_subject_and_new_memory_as_candidate(db: Database):
    await _insert_doc(db)
    old = _memory("mem-oldsem", "Old fact")
    await db.insert_memory(old)
    new = _memory("mem-newsem", "New fact")
    store = _store(db, RecordingCollection())

    await store.supersede_memory(
        old.id,
        new,
        "doc-1",
        "confluence",
        replacement_kind="supersession",
        replacement_reason="newer source",
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    assert [(row.memory_id, row.candidate_id, row.payload) for row in audit_rows] == [
        (old.id, new.id, {"old_memory_id": old.id, "new_memory_id": new.id})
    ]


@pytest.mark.asyncio
async def test_supersede_audit_uses_stable_old_new_identity(db: Database):
    await _insert_doc(db)
    old = _memory("mem-oldidentity", "Old identity fact")
    await db.insert_memory(old)
    new = _memory("mem-newidentity", "New identity fact")
    store = _store(db, RecordingCollection())

    await store.supersede_memory(
        old.id,
        new,
        "doc-1",
        "confluence",
        replacement_kind="supersession",
        replacement_reason="newer source",
        source_updated_at=None,
    )

    supersede_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    assert [(row.memory_id, row.candidate_id, row.payload) for row in supersede_rows] == [
        (
            old.id,
            new.id,
            {"old_memory_id": old.id, "new_memory_id": new.id},
        )
    ]


@pytest.mark.asyncio
async def test_promote_quarantined_challenger_routes_indexes_and_audit(db: Database):
    incumbent = _memory("mem-incsafe", "Old approved fact")
    challenger = _memory("mem-chalsafe", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.promote_quarantined_challenger(
        incumbent=incumbent,
        challenger=challenger,
        replacement_kind="supersession",
        replacement_reason="review approved",
        review_id="rev-safe",
    )

    stored_incumbent = await db.get_memory(incumbent.id)
    stored_challenger = await db.get_memory(challenger.id)
    audit_rows = await db.list_memory_audit_events(
        operation_id=(await db.list_memory_audit_events(memory_id=challenger.id))[0].operation_id
    )
    assert stored_incumbent.status == "superseded"
    assert stored_challenger.status == "active"
    assert incumbent.id in collection.deleted
    assert challenger.id in collection.upserted
    supersede_rows = [row for row in audit_rows if row.event_type == "memory_supersede_committed"]
    assert [(row.memory_id, row.candidate_id) for row in supersede_rows] == [(incumbent.id, challenger.id)]
    assert len({row.operation_id for row in audit_rows}) == 1


@pytest.mark.asyncio
async def test_promoted_challenger_preserves_staged_source_updated_at(db: Database):
    await _insert_doc(db, doc_id="doc-review")
    incumbent = _memory("mem-incobserved", "Old approved fact")
    challenger = _memory("mem-chalobserved", "New approved fact", status="pending_review")
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)
    await db.insert_memory(incumbent)
    store = _store(db, RecordingCollection())

    await store.insert_memory(
        memory=challenger,
        doc_id="doc-review",
        source_type="agent_session",
        excerpt="Review challenger excerpt",
        source_updated_at=source_updated_at,
    )
    await store.promote_quarantined_challenger(
        incumbent=incumbent,
        challenger=challenger,
        replacement_kind="supersession",
        replacement_reason="review approved",
        review_id="rev-observed",
    )

    sources = await db.get_memory_sources(challenger.id)
    assert [(source.doc_id, source.source_type, source.source_updated_at) for source in sources] == [
        ("doc-review", "agent_session", source_updated_at)
    ]


@pytest.mark.asyncio
async def test_promote_audit_uses_incumbent_as_subject_and_challenger_as_candidate(db: Database):
    incumbent = _memory("mem-incsem", "Old approved fact")
    challenger = _memory("mem-chalsem", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    store = _store(db, RecordingCollection())

    await store.promote_quarantined_challenger(
        incumbent=incumbent,
        challenger=challenger,
        replacement_kind="supersession",
        replacement_reason="review approved",
        review_id="rev-semantics",
    )

    audit_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    assert [(row.memory_id, row.candidate_id, row.review_id) for row in audit_rows] == [
        (incumbent.id, challenger.id, "rev-semantics")
    ]


@pytest.mark.asyncio
async def test_promote_audit_uses_stable_old_new_identity(db: Database):
    incumbent = _memory("mem-incidentity", "Old approved fact")
    challenger = _memory("mem-chalidentity", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    store = _store(db, RecordingCollection())

    await store.promote_quarantined_challenger(
        incumbent=incumbent,
        challenger=challenger,
        replacement_kind="supersession",
        replacement_reason="review approved",
        review_id="rev-identity",
    )

    supersede_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    assert [(row.memory_id, row.candidate_id, row.review_id, row.payload) for row in supersede_rows] == [
        (
            incumbent.id,
            challenger.id,
            "rev-identity",
            {"old_memory_id": incumbent.id, "new_memory_id": challenger.id},
        )
    ]


@pytest.mark.asyncio
async def test_purge_memory_records_audit_without_payload_snapshots(db: Database):
    memory = _memory("mem-purge", "Private fact")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    purged = await store.purge_memory(memory.id)

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert purged is True
    assert await db.get_memory(memory.id) is None
    assert memory.id in collection.deleted
    assert {"memory_purge_attempted", "memory_purge_committed"}.issubset({row.event_type for row in audit_rows})
    assert all(row.before_snapshot is None and row.after_snapshot is None for row in audit_rows)
    assert len({row.operation_id for row in audit_rows}) == 1


@pytest.mark.asyncio
async def test_purge_memory_redacts_existing_audit_payloads(db: Database):
    memory = _memory("mem-purge-redact", "Private fact")
    await db.insert_memory(memory)
    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_id="evt-sensitive",
            operation_id="op-sensitive",
            event_type="memory_insert_committed",
            status="committed",
            memory_id=memory.id,
            payload={"private": "secret"},
            before_snapshot={"content": "secret"},
            evidence_refs=[{"excerpt": "secret"}],
            error="secret error",
        )
    )
    store = _store(db, RecordingCollection())

    await store.purge_memory(memory.id)

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    redacted = [row for row in audit_rows if row.event_id == "evt-sensitive"][0]
    assert redacted.payload == {"redacted": True}
    assert redacted.before_snapshot is None
    assert redacted.evidence_refs == []
    assert redacted.error is None


@pytest.mark.asyncio
async def test_delete_document_removes_document_chroma_vector(db: Database):
    await _insert_doc(db, "doc-delete")
    memory = _memory("mem-docdelete", "Document supported fact")
    await db.insert_memory(memory)
    await db.add_memory_source(memory.id, "doc-delete", "confluence", source_updated_at=None)
    doc_collection = RecordingCollection()
    store = _store(db, RecordingCollection(), document_collection=doc_collection)

    await store.delete_document("doc-delete")

    assert doc_collection.deleted == ["doc-delete"]


@pytest.mark.asyncio
async def test_delete_document_audit_records_source_absence_context(db: Database):
    await _insert_doc(db, "doc-source-absence")
    memory = _memory("mem-source-absence", "Only supported by source absence doc")
    await db.insert_memory(memory)
    await db.add_memory_source(memory.id, "doc-source-absence", "jira", source_updated_at=None)
    store = _store(db, RecordingCollection(), document_collection=RecordingCollection())

    await store.delete_document(
        "doc-source-absence",
        deletion_context={
            "deletion_kind": "source_absence",
            "reason": "not_returned_by_latest_successful_crawl",
            "source_filter_summary": "updated >= -90d",
        },
    )

    audit_rows = await db.list_memory_audit_events(event_type="document_delete_committed")
    assert len(audit_rows) == 1
    assert audit_rows[0].payload == {
        "deletion_kind": "source_absence",
        "reason": "not_returned_by_latest_successful_crawl",
        "source_filter_summary": "updated >= -90d",
        "retired_memory_ids": [memory.id],
    }


@pytest.mark.asyncio
async def test_delete_document_fails_when_document_chroma_delete_fails(db: Database):
    await _insert_doc(db, "doc-delete-fail")
    store = _store(db, RecordingCollection(), document_collection=FailingDeleteCollection())

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.delete_document("doc-delete-fail")

    audit_rows = await db.list_memory_audit_events(event_type="index_operation_failed")
    assert [(row.doc_id, row.payload["index"], row.payload["operation"]) for row in audit_rows] == [
        ("doc-delete-fail", "document_chroma", "delete"),
        ("doc-delete-fail", "document_chroma", "restore"),
    ]


@pytest.mark.asyncio
async def test_delete_document_restores_document_vector_when_chroma_delete_mutates_then_fails(db: Database):
    await _insert_doc(db, "doc-delete-mutating-fail")
    doc_collection = MutatingFailingDeleteCollection()
    doc_collection.seed(
        ids=["doc-delete-mutating-fail"],
        embeddings=[[0.4, 0.5, 0.6]],
        documents=["original mutating delete document text"],
        metadatas=[{"content_hash": "hash-doc-delete-mutating-fail", "version": "1"}],
    )
    store = _store(db, RecordingCollection(), document_collection=doc_collection)

    with pytest.raises(RuntimeError, match="delete failed after mutation"):
        await store.delete_document("doc-delete-mutating-fail")

    stored_doc = await db.get_document("doc-delete-mutating-fail")
    assert stored_doc is not None
    assert doc_collection.upserted["doc-delete-mutating-fail"]["document"] == ("original mutating delete document text")
    assert doc_collection.upserted["doc-delete-mutating-fail"]["embedding"] == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_delete_document_restores_document_vector_when_db_delete_fails(db: Database, monkeypatch):
    await _insert_doc(db, "doc-db-delete-fail")
    doc_collection = RecordingCollection()
    doc_collection.upsert(
        ids=["doc-db-delete-fail"],
        embeddings=[[0.8, 0.7, 0.6]],
        documents=["original semantic document text"],
        metadatas=[{"content_hash": "hash-doc-db-delete-fail", "version": "1"}],
    )
    store = _store(db, RecordingCollection(), document_collection=doc_collection)

    async def fail_delete_document(doc_id: str):
        raise RuntimeError("db delete failed")

    monkeypatch.setattr(db, "delete_document", fail_delete_document)

    with pytest.raises(RuntimeError, match="db delete failed"):
        await store.delete_document("doc-db-delete-fail")

    assert doc_collection.deleted == ["doc-db-delete-fail"]
    assert doc_collection.upserted["doc-db-delete-fail"]["document"] == "original semantic document text"
    assert doc_collection.upserted["doc-db-delete-fail"]["embedding"] == [0.8, 0.7, 0.6]


@pytest.mark.asyncio
async def test_remove_source_support_restores_provenance_when_index_delete_fails(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-source-rollback", "Last sourced fact")
    await db.insert_memory(memory)
    await db.add_memory_source(memory.id, "doc-1", "confluence", "source excerpt", source_updated_at=None)
    store = _store(db, FailingDeleteCollection())

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.remove_source_support(memory.id, "doc-1", reason="no_support")

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert stored.status == "active"
    assert [(source.doc_id, source.excerpt) for source in sources] == [("doc-1", "source excerpt")]
    assert fts_count == 1


@pytest.mark.asyncio
async def test_delete_document_restores_db_when_retired_memory_index_delete_fails(db: Database):
    await _insert_doc(db, "doc-delete-rollback")
    await _insert_doc_side_tables(db, "doc-delete-rollback")
    memory = _memory("mem-doc-rollback", "Only supported by deleted doc")
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "doc-delete-rollback",
        "confluence",
        "source excerpt",
        source_updated_at=None,
    )
    doc_collection = RecordingCollection()
    doc_collection.upsert(
        ids=["doc-delete-rollback"],
        embeddings=[[0.1, 0.1, 0.1]],
        documents=["original delete rollback document"],
        metadatas=[{"content_hash": "hash-doc-delete-rollback", "version": "1"}],
    )
    store = _store(db, FailingDeleteCollection(), document_collection=doc_collection)

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.delete_document("doc-delete-rollback")

    stored_doc = await db.get_document("doc-delete-rollback")
    stored_memory = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    assert stored_doc is not None
    assert stored_memory.status == "active"
    assert [(source.doc_id, source.excerpt) for source in sources] == [("doc-delete-rollback", "source excerpt")]
    assert await _doc_side_counts(db, "doc-delete-rollback") == {
        "metadata": 1,
        "relationships": 1,
        "changelog": 1,
        "receipts": 1,
    }
    assert doc_collection.upserted["doc-delete-rollback"]["content_hash"] == "hash-doc-delete-rollback"
    assert doc_collection.upserted["doc-delete-rollback"]["document"] == "original delete rollback document"


@pytest.mark.asyncio
async def test_delete_virtual_document_restores_without_configured_source_on_index_failure(
    db: Database,
):
    now = datetime.now(timezone.utc)
    document = DocumentRecord(
        doc_id="user-memory-rollback",
        source="user_memory",
        source_url="memforge://user-memory/user-memory-rollback",
        title="User memory",
        space_or_project="UNSORTED",
        author="owner-1",
        last_modified=now,
        labels=["user_memory"],
        version="1",
        content_hash="user-memory-hash",
        token_count=3,
        raw_content_uri=None,
        raw_content_type=None,
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )
    await db.upsert_document(document)
    memory = _memory("mem-user-rollback", "User supplied fact")
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        document.doc_id,
        "user_memory",
        "User supplied fact",
        source_updated_at=now,
    )
    store = _store(db, FailingDeleteCollection())

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.delete_document(document.doc_id)

    assert await db.get_document(document.doc_id) is not None
    restored = await db.get_memory(memory.id)
    assert restored is not None and restored.status == "active"
    assert [source.doc_id for source in await db.get_memory_sources(memory.id)] == [
        document.doc_id
    ]


@pytest.mark.asyncio
async def test_delete_document_restores_missing_document_vector_when_delete_mutates_then_fails(db: Database):
    await _insert_doc(db, "doc-missing-vector-rollback")
    doc_collection = InsertThenFailingDeleteCollection()
    store = _store(db, RecordingCollection(), document_collection=doc_collection)

    with pytest.raises(RuntimeError, match="delete failed after mutation"):
        await store.delete_document("doc-missing-vector-rollback")

    assert await db.get_document("doc-missing-vector-rollback") is not None
    assert "doc-missing-vector-rollback" not in doc_collection.upserted


@pytest.mark.asyncio
async def test_delete_document_rolls_back_sqlite_when_db_delete_fails_mid_transaction(db: Database, monkeypatch):
    await _insert_doc(db, "doc-mid-db-fail")
    await _insert_doc_side_tables(db, "doc-mid-db-fail")
    memory = _memory("mem-mid-db-fail", "Only supported by failing doc")
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "doc-mid-db-fail",
        "confluence",
        "source excerpt",
        source_updated_at=None,
    )
    store = _store(db, RecordingCollection(), document_collection=RecordingCollection())

    async def fail_refresh(memory_ids, *, retire_reason="source_deleted"):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(db, "_refresh_support_after_source_removal_unlocked", fail_refresh)

    with pytest.raises(RuntimeError, match="refresh failed"):
        await store.delete_document("doc-mid-db-fail")

    stored_doc = await db.get_document("doc-mid-db-fail")
    sources = await db.get_memory_sources(memory.id)
    assert stored_doc is not None
    assert [(source.doc_id, source.excerpt) for source in sources] == [("doc-mid-db-fail", "source excerpt")]
    assert await _doc_side_counts(db, "doc-mid-db-fail") == {
        "metadata": 1,
        "relationships": 1,
        "changelog": 1,
        "receipts": 1,
    }


@pytest.mark.asyncio
async def test_lifecycle_vector_delivery_attempt_is_bounded() -> None:
    class Relational:
        def __init__(self) -> None:
            self.completed: list[str] = []

        async def list_lifecycle_vector_tasks(self, **kwargs):
            assert kwargs["limit"] == 101
            return [
                LifecycleVectorTask(
                    id=f"task-{index}",
                    lifecycle_plan_id="plan-1",
                    memory_id=f"memory-{index}",
                    operation=LifecycleVectorOperation.DELETE,
                    status=LifecycleVectorTaskStatus.PENDING,
                )
                for index in range(101)
            ]

        async def complete_lifecycle_vector_task(self, task_id: str) -> None:
            self.completed.append(task_id)

        async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None:
            raise AssertionError((task_id, error))

    class Vector:
        async def delete(self, memory_ids: list[str]) -> None:
            assert len(memory_ids) == 1

    relational = Relational()
    store = object.__new__(MemoryStore)
    store.relational = relational
    store.vector = Vector()

    result = await store.attempt_lifecycle_vector_delivery(lifecycle_plan_id="plan-1")

    assert result.pending
    assert result.attempted_tasks == 100
    assert result.delivered_tasks == 100
    assert result.failed_tasks == 0
    assert len(relational.completed) == 100


@pytest.mark.asyncio
async def test_lifecycle_vector_delivery_rechecks_after_concurrent_completion() -> None:
    task = LifecycleVectorTask(
        id="task-raced",
        lifecycle_plan_id="plan-raced",
        memory_id="memory-raced",
        operation=LifecycleVectorOperation.DELETE,
        status=LifecycleVectorTaskStatus.PENDING,
    )

    class Relational:
        def __init__(self) -> None:
            self.list_calls = 0

        async def list_lifecycle_vector_tasks(self, **kwargs):
            self.list_calls += 1
            return [task] if self.list_calls == 1 else []

        async def complete_lifecycle_vector_task(self, task_id: str) -> None:
            assert task_id == task.id
            raise ValueError("lifecycle vector task is not pending")

        async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None:
            assert task_id == task.id
            raise ValueError("lifecycle vector task is not pending")

    class Vector:
        async def delete(self, memory_ids: list[str]) -> None:
            assert memory_ids == [task.memory_id]

    relational = Relational()
    store = object.__new__(MemoryStore)
    store.relational = relational
    store.vector = Vector()

    result = await store.attempt_lifecycle_vector_delivery(source_id="src-raced")

    assert not result.pending
    assert result.attempted_tasks == 1
    assert result.delivered_tasks == 0
    assert result.failed_tasks == 0
    assert result.error_types == ()
    assert relational.list_calls == 2


@pytest.mark.asyncio
async def test_lifecycle_vector_delivery_retries_historical_failed_task() -> None:
    task = LifecycleVectorTask(
        id="task-retry",
        lifecycle_plan_id="plan-retry",
        memory_id="memory-retry",
        operation=LifecycleVectorOperation.DELETE,
        status=LifecycleVectorTaskStatus.FAILED,
        attempts=1,
        error="provider unavailable",
    )

    class Relational:
        def __init__(self) -> None:
            self.completed: list[str] = []

        async def list_lifecycle_vector_tasks(self, **kwargs):
            return [task]

        async def complete_lifecycle_vector_task(self, task_id: str) -> None:
            self.completed.append(task_id)

        async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None:
            raise AssertionError((task_id, error))

    class Vector:
        async def delete(self, memory_ids: list[str]) -> None:
            assert memory_ids == [task.memory_id]

    relational = Relational()
    store = object.__new__(MemoryStore)
    store.relational = relational
    store.vector = Vector()

    result = await store.attempt_lifecycle_vector_delivery(source_id="src-retry")

    assert not result.pending
    assert result.delivered_tasks == 1
    assert result.failed_tasks == 0
    assert relational.completed == [task.id]


@pytest.mark.asyncio
async def test_delete_source_cascade_keeps_durable_cleanup_when_memory_vector_delete_fails(db: Database):
    await db.upsert_source(
        "src-rollback", "confluence", "Rollback Source", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await db.db.execute(
        """UPDATE sources
           SET status = ?, last_sync = ?, doc_count = ?, created_at = ?
           WHERE id = ?""",
        ("paused", "2026-05-01T00:00:00+00:00", 7, "2026-04-01T00:00:00+00:00", "src-rollback"),
    )
    await db.db.commit()
    await _insert_doc(db, "doc-source-rollback", source="src-rollback")
    await _insert_doc_side_tables(db, "doc-source-rollback", source="src-rollback")
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO sync_state
           (source, last_sync_at, last_sync_status, docs_processed, docs_updated, error_message)
           VALUES (?, ?, ?, ?, ?, ?)""",
        ("src-rollback", now, "success", 1, 1, None),
    )
    await db.db.execute(
        """INSERT INTO sync_history
           (source, status, docs_processed, docs_updated, docs_failed, memories_extracted,
            started_at, finished_at, run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("src-rollback", "success", 1, 1, 0, 1, now, now, "run-1"),
    )
    await db.db.commit()
    memory = _memory("mem-sourcecascade-rollback", "Only supported by source doc")
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "doc-source-rollback",
        "confluence",
        "source excerpt",
        source_updated_at=None,
    )
    doc_collection = RecordingCollection()
    doc_collection.upsert(
        ids=["doc-source-rollback"],
        embeddings=[[0.2, 0.2, 0.2]],
        documents=["original source rollback document"],
        metadatas=[{"content_hash": "hash-doc-source-rollback", "version": "1"}],
    )
    store = _store(db, FailingDeleteCollection(), document_collection=doc_collection)

    retired = await store.delete_source_cascade("src-rollback")

    stored_source = await db.get_source("src-rollback")
    stored_doc = await db.get_document("doc-source-rollback")
    stored_memory = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    assert retired == [memory.id]
    assert stored_source is None
    assert stored_doc is None
    assert stored_memory.status == "retired"
    assert sources == []
    assert await _doc_side_counts(db, "doc-source-rollback") == {
        "metadata": 0,
        "relationships": 0,
        "changelog": 0,
        "receipts": 0,
    }
    assert await _source_bookkeeping_counts(db, "src-rollback") == {
        "sync_state": 0,
        "sync_history": 0,
    }
    [cleanup] = await db.list_lifecycle_vector_tasks(source_id="src-rollback")
    assert cleanup.memory_id == memory.id
    assert cleanup.status is LifecycleVectorTaskStatus.FAILED

    retry_store = _store(db, RecordingCollection())
    delivery = await retry_store.attempt_lifecycle_vector_delivery(source_id="src-rollback")
    assert not delivery.pending
    assert await db.list_lifecycle_vector_tasks(source_id="src-rollback") == []
    async with db.db.execute(
        "SELECT status, error FROM source_deletion_vector_outbox WHERE id = ?",
        (cleanup.id,),
    ) as cursor:
        recovered = await cursor.fetchone()
    assert recovered is not None
    assert (recovered["status"], recovered["error"]) == (
        "completed",
        cleanup.error,
    )


@pytest.mark.asyncio
async def test_delete_source_cascade_restores_document_vectors_when_later_delete_fails(db: Database):
    await db.upsert_source(
        "src-partial-doc-delete",
        "confluence",
        "Partial Delete Source",
        "{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    await _insert_doc(db, "doc-partial-1", source="src-partial-doc-delete")
    await _insert_doc(db, "doc-partial-2", source="src-partial-doc-delete")
    doc_collection = FailingSecondDeleteCollection()
    for index, doc_id in enumerate(["doc-partial-1", "doc-partial-2"]):
        doc_collection.upsert(
            ids=[doc_id],
            embeddings=[[float(index), float(index), float(index)]],
            documents=[f"original {doc_id}"],
            metadatas=[{"content_hash": f"hash-{doc_id}", "version": "1"}],
        )
    store = _store(db, RecordingCollection(), document_collection=doc_collection)

    with pytest.raises(RuntimeError, match="second delete failed"):
        await store.delete_source_cascade("src-partial-doc-delete")

    assert doc_collection.upserted["doc-partial-1"]["document"] == "original doc-partial-1"
    assert doc_collection.upserted["doc-partial-1"]["embedding"] == [0.0, 0.0, 0.0]
    assert doc_collection.upserted["doc-partial-2"]["document"] == "original doc-partial-2"
    assert doc_collection.upserted["doc-partial-2"]["embedding"] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_purge_memory_restores_chroma_when_sqlite_purge_fails(db: Database, monkeypatch):
    memory = _memory("mem-purge-db-fail", "Purge DB failure")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    async def fail_purge(memory_id: str) -> bool:
        raise RuntimeError("purge failed")

    monkeypatch.setattr(db, "purge_memory", fail_purge)

    with pytest.raises(RuntimeError, match="purge failed"):
        await store.purge_memory(memory.id)

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert collection.deleted == [memory.id]
    assert memory.id in collection.upserted


@pytest.mark.asyncio
async def test_purge_memory_restores_chroma_when_delete_mutates_then_fails(db: Database):
    memory = _memory("mem-purge-delete-mutates", "Purge delete mutates")
    await db.insert_memory(memory)
    collection = MutatingFailingDeleteCollection()
    collection.seed(
        ids=[memory.id],
        embeddings=[[0.8, 0.8, 0.8]],
        metadatas=[{"status": "active", "content_hash": memory.content_hash}],
    )
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="delete failed after mutation"):
        await store.purge_memory(memory.id)

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert collection.upserted[memory.id]["content_hash"] == memory.content_hash


@pytest.mark.asyncio
async def test_insert_memory_raises_and_avoids_committed_event_when_chroma_upsert_fails(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-upsert-fail", "Cannot index")
    store = _store(db, FailingUpsertCollection())

    with pytest.raises(RuntimeError, match="upsert failed"):
        await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert "index_operation_failed" in {row.event_type for row in audit_rows}
    assert "memory_insert_committed" not in {row.event_type for row in audit_rows}
    assert await db.get_memory(memory.id) is None


@pytest.mark.asyncio
async def test_insert_memory_cleans_chroma_when_upsert_mutates_then_fails(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-insert-mutating-fail", "Cannot index cleanly")
    collection = MutatingFailingUpsertCollection()
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="upsert failed after mutation"):
        await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    assert await db.get_memory(memory.id) is None
    assert memory.id not in collection.upserted


@pytest.mark.asyncio
async def test_insert_memory_purges_sqlite_when_chroma_cleanup_fails(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-insert-cleanup-fail", "Cleanup cannot finish")
    collection = MutatingFailingUpsertAndDeleteCollection()
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="delete failed during rollback"):
        await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    assert await db.get_memory(memory.id) is None


@pytest.mark.asyncio
async def test_insert_memory_rebuilds_fts_from_canonical_entity_links(db: Database):
    await _insert_doc(db)
    alpha_id = await db.upsert_entity("alpha", display_name="Alpha", tags=["team"])
    beta_id = await db.upsert_entity("beta", display_name="Beta", tags=["team"])
    memory = _memory("mem-canonical-fts", "Canonical entity search text")
    memory.entity_refs = ["beta", "alpha"]
    collection = RecordingCollection()
    store = _store(db, collection)

    await store.deduplicate_and_insert(
        memory,
        "doc-1",
        "confluence",
        source_updated_at=None,
        entity_ids=[alpha_id, beta_id],
    )

    async with db.db.execute(
        "SELECT entities_text FROM memories_fts WHERE memory_id = ?",
        (memory.id,),
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row["entities_text"] == "alpha beta"
    assert memory.id in collection.upserted


@pytest.mark.asyncio
async def test_supersede_memory_rebuilds_fts_from_canonical_entity_links(db: Database):
    await _insert_doc(db)
    old = _memory("mem-canonical-supersede-old", "Old entity search text")
    await db.insert_memory(old)
    alpha_id = await db.upsert_entity("alpha", display_name="Alpha", tags=["team"])
    beta_id = await db.upsert_entity("beta", display_name="Beta", tags=["team"])
    new = _memory("mem-canonical-supersede-new", "New entity search text")
    new.entity_refs = ["beta", "alpha"]
    store = _store(db, RecordingCollection())

    await store.supersede_memory(
        old.id,
        new,
        "doc-1",
        "confluence",
        replacement_kind="supersession",
        source_updated_at=None,
        entity_ids=[alpha_id, beta_id],
        replacement_reason="newer source",
    )

    async with db.db.execute(
        "SELECT entities_text FROM memories_fts WHERE memory_id = ?",
        (new.id,),
    ) as cursor:
        row = await cursor.fetchone()

    assert row is not None
    assert row["entities_text"] == "alpha beta"


@pytest.mark.asyncio
async def test_insert_memory_rolls_back_sqlite_when_source_link_fails(db: Database):
    await _insert_doc(db)
    memory = _memory("mem-source-link-fail", "Cannot link source")
    collection = RecordingCollection()
    store = _store(FailingSourceInsertDatabase(db), collection)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="source insert failed"):
        await store.deduplicate_and_insert(memory, "doc-1", "confluence", source_updated_at=None)

    assert await db.get_memory(memory.id) is None
    assert memory.id not in collection.upserted


@pytest.mark.asyncio
async def test_insert_memory_rolls_back_when_relation_bundle_fails(db: Database, monkeypatch):
    await _insert_doc(db)
    memory = _memory("mem-create-relation-fail", "Create relation failure")
    collection = RecordingCollection()
    store = _store(db, collection)

    async def fail_relation_bundle(*args, **kwargs):
        raise RuntimeError("relation bundle failed")

    monkeypatch.setattr(db, "_record_relation_outcome_bundle_unlocked", fail_relation_bundle)

    with pytest.raises(RuntimeError, match="relation bundle failed"):
        await store.insert_memory(
            memory,
            "doc-1",
            "confluence",
            source_updated_at=None,
            relation_outcome=_relation_outcome_bundle(
                unit_id="eu-create-relation-fail",
                run_id="relrun-create-relation-fail",
                doc_id="doc-1",
                memory_id=memory.id,
                action=LifecycleAction.CREATE_MEMORY,
            ),
        )

    assert await db.get_memory(memory.id) is None
    assert memory.id not in collection.upserted


@pytest.mark.asyncio
async def test_retire_memory_raises_and_avoids_committed_event_when_chroma_delete_fails(db: Database):
    memory = _memory("mem-delete-fail", "Cannot remove from index")
    await db.insert_memory(memory)
    store = _store(db, FailingDeleteCollection())

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.retire_memory(memory.id, reason="admin_hidden")

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    stored = await db.get_memory(memory.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert "index_operation_failed" in {row.event_type for row in audit_rows}
    assert "memory_retire_committed" not in {row.event_type for row in audit_rows}
    assert stored.status == "active"
    assert fts_count == 1


@pytest.mark.asyncio
async def test_update_memory_restores_sqlite_when_chroma_upsert_fails(db: Database):
    memory = _memory("mem-update-fail", "Old content")
    await db.insert_memory(memory)
    store = _store(db, FailingUpsertCollection())

    with pytest.raises(RuntimeError, match="upsert failed"):
        await store.update_memory(memory.id, "New content", new_confidence=0.4)

    stored = await db.get_memory(memory.id)
    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert stored.content == "Old content"
    assert stored.confidence == 0.9
    assert stored.content_hash == memory.content_hash
    assert "memory_update_committed" not in {row.event_type for row in audit_rows}


@pytest.mark.asyncio
async def test_update_memory_restores_chroma_when_upsert_mutates_then_fails(db: Database):
    memory = _memory("mem-update-mutating-fail", "Old content")
    await db.insert_memory(memory)
    collection = MutatingFailingUpsertCollection()
    collection.seed(
        ids=[memory.id],
        embeddings=[[0.9, 0.9, 0.9]],
        metadatas=[{"status": "active", "content_hash": memory.content_hash}],
    )
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="upsert failed after mutation"):
        await store.update_memory(memory.id, "New content", new_confidence=0.4)

    stored = await db.get_memory(memory.id)
    assert stored.content == "Old content"
    assert collection.upserted[memory.id]["content_hash"] == memory.content_hash


@pytest.mark.asyncio
async def test_update_memory_restores_sqlite_when_reembedding_fails(db: Database):
    memory = _memory("mem-update-embed-fail", "Old content")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)

    async def fail_embed(text: str) -> list[float]:
        raise RuntimeError("embedding failed")

    store._embed = fail_embed  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="embedding failed"):
        await store.update_memory(memory.id, "New content", new_confidence=0.4)

    stored = await db.get_memory(memory.id)
    assert stored.content == "Old content"
    assert stored.confidence == 0.9
    assert memory.id not in collection.upserted


@pytest.mark.asyncio
async def test_supersede_memory_restores_sqlite_when_new_chroma_upsert_fails(db: Database):
    await _insert_doc(db)
    old = _memory("mem-sup-fail-old", "Old fact")
    await db.insert_memory(old)
    unit = EvidenceUnit(
        id="eu-sup-fail-old",
        source_id="src-1",
        doc_id="doc-1",
        doc_revision_id="rev-1",
        source_type="confluence",
        source_anchor="page-1#old",
        source_lineage_id="doc-1",
        project_key="TEST",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="Old fact",
        excerpt="Old fact excerpt",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )
    await db.upsert_evidence_unit(unit)
    relation = EvidenceRelationRecord(
        evidence_unit_id=unit.id,
        memory_id=old.id,
        relation_type=RelationType.SUPPORTS,
        authority_case=AuthorityCase.SAME_DOCUMENT_REVISION,
        is_authoritative_support=True,
        source_lineage_id="doc-1",
        confidence=0.93,
        reason="original relation",
        excerpt="Old fact excerpt",
        classifier_version="test-v1",
        relation_run_id="rel-run-sup-fail",
        created_at="2026-06-22T12:00:00+00:00",
    )
    await db.replace_evidence_relations(unit.id, [relation])
    new = _memory("mem-sup-fail-new", "New fact")
    store = _store(db, FailingSpecificUpsertCollection(new.id))

    with pytest.raises(RuntimeError, match="upsert failed"):
        await store.supersede_memory(
            old.id,
            new,
            "doc-1",
            "confluence",
            replacement_kind="supersession",
            source_updated_at=None,
        )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    audit_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (old.id,)) as cursor:
        old_fts_count = (await cursor.fetchone())[0]
    restored_relations = await db.get_evidence_relations(unit.id)
    assert stored_old.status == "active"
    assert stored_new is None
    assert old_fts_count == 1
    assert restored_relations == [relation]
    assert audit_rows == []


@pytest.mark.asyncio
async def test_supersede_memory_cleans_new_chroma_when_source_link_fails(db: Database):
    await _insert_doc(db)
    old = _memory("mem-sup-source-fail-old", "Old fact")
    await db.insert_memory(old)
    new = _memory("mem-sup-source-fail-new", "New fact")
    collection = RecordingCollection()
    store = _store(FailingSourceInsertDatabase(db), collection)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="source insert failed"):
        await store.supersede_memory(
            old.id,
            new,
            "doc-1",
            "confluence",
            replacement_kind="supersession",
            source_updated_at=None,
        )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    assert stored_old.status == "active"
    assert stored_new is None
    assert old.id in collection.upserted
    assert new.id not in collection.upserted


@pytest.mark.asyncio
async def test_supersede_memory_restores_sqlite_when_new_chroma_cleanup_fails(db: Database):
    await _insert_doc(db)
    old = _memory("mem-sup-cleanup-fail-old", "Old fact")
    await db.insert_memory(old)
    new = _memory("mem-sup-cleanup-fail-new", "New fact")
    collection = MutatingFailingSpecificUpsertAndDeleteCollection(new.id)
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.supersede_memory(
            old.id,
            new,
            "doc-1",
            "confluence",
            replacement_kind="supersession",
            source_updated_at=None,
        )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    assert stored_old.status == "active"
    assert stored_new is None
    assert new.id in collection.upserted


@pytest.mark.asyncio
async def test_promote_challenger_restores_sqlite_when_challenger_chroma_upsert_fails(db: Database):
    incumbent = _memory("mem-promote-fail-inc", "Old approved fact")
    challenger = _memory("mem-promote-fail-chal", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    store = _store(db, FailingSpecificUpsertCollection(challenger.id))

    with pytest.raises(RuntimeError, match="upsert failed"):
        await store.promote_quarantined_challenger(
            incumbent=incumbent,
            challenger=challenger,
            replacement_kind="supersession",
            replacement_reason="review approved",
            review_id="rev-failing",
        )

    stored_incumbent = await db.get_memory(incumbent.id)
    stored_challenger = await db.get_memory(challenger.id)
    audit_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (incumbent.id,)) as cursor:
        incumbent_fts_count = (await cursor.fetchone())[0]
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (challenger.id,)) as cursor:
        challenger_fts_count = (await cursor.fetchone())[0]
    assert stored_incumbent.status == "active"
    assert stored_challenger.status == "pending_review"
    assert incumbent_fts_count == 1
    assert challenger_fts_count == 0
    assert audit_rows == []


@pytest.mark.asyncio
async def test_promote_challenger_cleans_challenger_when_upsert_mutates_then_fails(db: Database):
    incumbent = _memory("mem-inc-mutating-fail", "Old approved fact")
    challenger = _memory("mem-chal-mutating-fail", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    collection = MutatingFailingUpsertCollection()
    collection.seed(
        ids=[incumbent.id],
        embeddings=[[0.9, 0.9, 0.9]],
        metadatas=[{"status": "active", "content_hash": incumbent.content_hash}],
    )
    store = _store(db, collection)

    with pytest.raises(RuntimeError, match="upsert failed after mutation"):
        await store.promote_quarantined_challenger(
            incumbent=incumbent,
            challenger=challenger,
            replacement_kind="supersession",
            replacement_reason="review approved",
            review_id="rev-mutating-fail",
        )

    stored_incumbent = await db.get_memory(incumbent.id)
    stored_challenger = await db.get_memory(challenger.id)
    assert stored_incumbent.status == "active"
    assert stored_challenger.status == "pending_review"
    assert incumbent.id in collection.upserted
    assert challenger.id not in collection.upserted


@pytest.mark.asyncio
async def test_promote_challenger_restores_sqlite_when_incumbent_chroma_delete_fails(db: Database):
    incumbent = _memory("mem-promote-delete-inc", "Old approved fact")
    challenger = _memory("mem-promote-delete-chal", "New approved fact", status="pending_review")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    store = _store(db, FailingDeleteCollection())

    with pytest.raises(RuntimeError, match="delete failed"):
        await store.promote_quarantined_challenger(
            incumbent=incumbent,
            challenger=challenger,
            replacement_kind="supersession",
            replacement_reason="review approved",
            review_id="rev-delete-failing",
        )

    stored_incumbent = await db.get_memory(incumbent.id)
    stored_challenger = await db.get_memory(challenger.id)
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (incumbent.id,)) as cursor:
        incumbent_fts_count = (await cursor.fetchone())[0]
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (challenger.id,)) as cursor:
        challenger_fts_count = (await cursor.fetchone())[0]
    audit_rows = await db.list_memory_audit_events(event_type="memory_supersede_committed")
    assert stored_incumbent.status == "active"
    assert stored_challenger.status == "pending_review"
    assert incumbent_fts_count == 1
    assert challenger_fts_count == 0
    assert audit_rows == []
