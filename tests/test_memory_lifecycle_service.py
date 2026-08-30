"""Tests for user-facing memory lifecycle actions exposed to MCP/API clients."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memforge.agent_knowledge import AgentKnowledgeBundleService, AgentKnowledgePatchProposal
from memforge.config import AppConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.memory.lifecycle_service import MemoryLifecycleConflict, MemoryLifecycleService
from memforge.memory.review_decision import memory_review_decision_fingerprint
from memforge.memory.review_service import ReviewService, ReviewStaleConflict
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    Visibility,
    content_hash,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.retrieval.search import SearchEngine
from memforge.server.source_admin_service import can_manage_source
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class RecordingCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "embedding": embeddings[index] if embeddings else None,
                "metadata": metadatas[index] if metadatas else {},
                "document": documents[index] if documents else None,
            }

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.deleted.append(record_id)
            self.records.pop(record_id, None)

    def query(self, **_params):
        return {"ids": [[]], "distances": [[]]}

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or self.records) if record_id in self.records]
        include = include or []
        result: dict[str, Any] = {"ids": selected}
        if "metadatas" in include:
            result["metadatas"] = [self.records[record_id]["metadata"] for record_id in selected]
        if "embeddings" in include:
            result["embeddings"] = [self.records[record_id]["embedding"] for record_id in selected]
        if "documents" in include:
            result["documents"] = [self.records[record_id]["document"] for record_id in selected]
        return result


class _TestCorrectionAuthority:
    def __init__(self, actor_user_id: str, workspace_role: str) -> None:
        self.actor_user_id = actor_user_id
        self.workspace_role = workspace_role

    def can_manage_source(self, source) -> bool:
        return can_manage_source(
            dict(source),
            viewer_id=self.actor_user_id,
            viewer_role=self.workspace_role,
        )

    def can_manage_workspace_memory(self) -> bool:
        return self.workspace_role in {"owner", "workspace_admin"}


def _authority(actor_user_id: str, workspace_role: str) -> _TestCorrectionAuthority:
    return _TestCorrectionAuthority(actor_user_id, workspace_role)


def _api_config(tmp_path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return config


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "lifecycle.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.91,
        created_at=now,
        updated_at=now,
    )


def _store(db: Database, collection: RecordingCollection) -> MemoryStore:
    audit_logger = MemoryAuditLogger(db, default_context=AuditContext(actor_type="test", run_id="lifecycle-test"))
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=audit_logger,
    )

    async def fake_embed(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[method-assign]
    return store


async def _source_backed_memory(
    db: Database,
    *,
    suffix: str,
    source_owner: str = "source-owner",
) -> Memory:
    source_id = f"src-{suffix}"
    doc_id = f"doc-{suffix}"
    old = _memory(f"mem-{suffix}", f"The {suffix} source-backed rule is current.")
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name=f"Managed source {suffix}",
        config_json="{}",
        access_policy="workspace",
        owner_user_id=source_owner,
    )
    observed = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"https://example.test/{doc_id}",
            title=doc_id,
            space_or_project="ENG",
            author=None,
            last_modified=observed,
            labels=[],
            version="1",
            content_hash=f"{suffix}-doc-hash",
            token_count=8,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=observed,
        )
    )
    await db.insert_memory(old)
    await db.add_memory_source(
        old.id,
        doc_id,
        "confluence",
        old.content,
        source_updated_at=observed,
    )
    item = ContentItem(
        item_id=doc_id,
        title=doc_id,
        source_url=f"https://example.test/{doc_id}",
        last_modified=observed,
        version="1",
        extra={"page_id": suffix, "space_key": "ENG"},
    )
    projection = project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id=f"projection-{suffix}",
        item=item,
        raw=RawContent(item=item, body=old.content.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=old.content),
    )
    await db.record_source_projection(projection)
    observation = projection.observations[0]
    observation_revision = projection.observation_revisions[0]
    unit = EvidenceUnit(
        id=f"eu-{suffix}",
        source_id=source_id,
        doc_id=doc_id,
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=old.content,
        excerpt=old.content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(unit)
    reference_id = f"eref-{suffix}"
    await db.db.execute(
        """INSERT INTO evidence_references (
               id, evidence_unit_id, role, anchor_kind, observation_id,
               observation_revision_id, fragment_id, range_start, range_end, created_at
           ) VALUES (?, ?, 'primary', 'whole_observation', ?, ?, NULL, NULL, NULL, ?)""",
        (
            reference_id,
            unit.id,
            observation.id,
            observation_revision.id,
            observed.isoformat(),
        ),
    )
    await db.db.commit()
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{suffix}",
            memory_id=old.id,
            evidence_reference_id=reference_id,
            source_id=source_id,
            access_context_hash="workspace-eng",
        )
    )
    return old


async def _move_source_support(db: Database, *, from_memory: Memory, to_memory: Memory) -> None:
    sources = await db.get_memory_sources(from_memory.id)
    for source in sources:
        await db.add_memory_source(
            to_memory.id,
            source.doc_id,
            source.source_type,
            source.excerpt,
            support_kind=source.support_kind,
            source_updated_at=source.source_updated_at,
        )
    await db.db.execute(
        "UPDATE memory_support_assertions SET memory_id = ? WHERE memory_id = ?",
        (to_memory.id, from_memory.id),
    )
    await db.db.commit()
    await db.purge_memory(from_memory.id)


@pytest.mark.asyncio
async def test_create_memory_writes_private_user_memory_with_provenance(db: Database):
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    result = await service.create_memory(
        content="Use Status and FollowUpStepStatus when polling PayrollProcessingTriggerViews.",
        provenance=(
            "During the xall-004 smoke test, PayrollProcessingTriggerViews polling only "
            "matched after using Status and FollowUpStepStatus."
        ),
        memory_type="fact",
        owner_user_id="andrew.sun01@sap.com",
        client="codex",
        repo_identifier="github.com/shno-labs/mem-forge",
    )

    stored = await db.get_memory(result.memory_id)
    sources = await db.get_memory_sources(result.memory_id)
    assert result.status == "inserted"
    assert stored is not None
    assert stored.content == "Use Status and FollowUpStepStatus when polling PayrollProcessingTriggerViews."
    assert stored.extraction_context == (
        "During the xall-004 smoke test, PayrollProcessingTriggerViews polling only "
        "matched after using Status and FollowUpStepStatus."
    )
    assert stored.visibility == Visibility.PRIVATE.value
    assert stored.owner_user_id == "andrew.sun01@sap.com"
    assert stored.project_key == "UNSORTED"
    assert stored.repo_identifier == "github.com/shno-labs/mem-forge"
    assert [(source.doc_id, source.source_type) for source in sources] == [
        (f"user-memory-{result.memory_id}", "user_memory")
    ]
    assert sources[0].excerpt == (
        "During the xall-004 smoke test, PayrollProcessingTriggerViews polling only "
        "matched after using Status and FollowUpStepStatus."
    )
    assert sources[0].excerpt == stored.extraction_context
    document = await db.get_document(f"user-memory-{result.memory_id}")
    assert document is not None
    assert document.source == "user_memory"
    assert document.client == "codex"
    assert document.normalized_content_uri is None


@pytest.mark.asyncio
async def test_retire_memory_uses_expected_hash_guard(db: Database):
    memory = _memory("mem-retire-tool", "Old fact")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    with pytest.raises(MemoryLifecycleConflict, match="content_hash_mismatch"):
        await service.retire_memory(
            memory.id,
            reason="User says this is stale",
            expected_content_hash="wrong",
        )

    await service.retire_memory(
        memory.id,
        reason="User says this is stale",
        expected_content_hash=memory.content_hash,
    )

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert stored.status == "retired"
    assert stored.retirement_reason == "User says this is stale"


@pytest.mark.asyncio
async def test_user_lifecycle_cannot_bypass_active_projected_source_support(db: Database):
    old = _memory("mem-source-managed", "The source-owned rule is current.")
    await db.upsert_source(
        id="src-managed",
        type="confluence",
        name="Managed source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    observed = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="doc-managed",
            source="src-managed",
            source_url="https://example.test/doc-managed",
            title="Managed document",
            space_or_project="ENG",
            author=None,
            last_modified=observed,
            labels=[],
            version="1",
            content_hash="managed-doc-hash",
            token_count=8,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=observed,
        )
    )
    await db.insert_memory(old)
    await db.add_memory_source(
        old.id,
        "doc-managed",
        "confluence",
        old.content,
        source_updated_at=observed,
    )
    item = ContentItem(
        item_id="doc-managed",
        title="Managed document",
        source_url="https://example.test/doc-managed",
        last_modified=observed,
        version="1",
        extra={"page_id": "managed", "space_key": "ENG"},
    )
    projection = project_source_item(
        source_id="src-managed",
        source_type="confluence",
        run_id="projection-managed",
        item=item,
        raw=RawContent(item=item, body=old.content.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=old.content),
    )
    await db.record_source_projection(projection)
    observation = projection.observations[0]
    observation_revision = projection.observation_revisions[0]
    unit = EvidenceUnit(
        id="eu-source-managed",
        source_id="src-managed",
        doc_id="doc-managed",
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=observation.id,
        source_lineage_id=projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=old.content,
        excerpt=old.content,
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(unit)
    await db.db.execute(
        """INSERT INTO evidence_references (
               id, evidence_unit_id, role, anchor_kind, observation_id,
               observation_revision_id, fragment_id, range_start, range_end, created_at
           ) VALUES (?, ?, 'primary', 'whole_observation', ?, ?, NULL, NULL, NULL, ?)""",
        (
            "eref-source-managed",
            unit.id,
            observation.id,
            observation_revision.id,
            observed.isoformat(),
        ),
    )
    await db.db.commit()
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-source-managed",
            memory_id=old.id,
            evidence_reference_id="eref-source-managed",
            source_id="src-managed",
            access_context_hash="workspace-eng",
        )
    )
    service = MemoryLifecycleService(
        db=db,
        memory_store=_store(db, RecordingCollection()),
    )

    with pytest.raises(
        MemoryLifecycleConflict,
        match="source_backed_memory_requires_lifecycle_review",
    ):
        await service.retire_memory(
            old.id,
            reason="manual retirement",
            expected_content_hash=old.content_hash,
        )
    assert (await db.get_memory(old.id)).status == "active"
    assert await db.count_documents("user_correction") == 0


@pytest.mark.asyncio
async def test_propose_memory_correction_applies_for_private_memory_owner(db: Database):
    store = _store(db, RecordingCollection())
    service = MemoryLifecycleService(db=db, memory_store=store)
    created = await service.create_memory(
        content="The original private claim.",
        provenance="The user recorded the original claim.",
        owner_user_id="alice",
        client="codex",
    )
    old = await db.get_memory(created.memory_id)
    assert old is not None

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The corrected private claim.",
        provenance="Alice supplied the correction.",
        reason="Alice corrected her private memory.",
        expected_content_hash=old.content_hash,
        authority=_authority("alice", "member"),
        replacement_kind="revision",
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    review = await db.get_memory_review(result.review_id)
    assert result.outcome == "applied"
    assert stored_old is not None and stored_old.status == "superseded"
    assert stored_new is not None and stored_new.status == "active"
    assert stored_new.visibility == "private"
    assert stored_new.owner_user_id == "alice"
    assert review is not None and review.status == "approved"
    assert review.expected_support_set_hash is None


@pytest.mark.asyncio
async def test_propose_memory_correction_applies_for_complete_source_authority(db: Database):
    old = await _source_backed_memory(db, suffix="authorized", source_owner="alice")
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))
    support_hash = await db.get_memory_support_set_hash(old.id)

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The authorized corrected source-backed rule.",
        provenance="The Source owner supplied the correction.",
        reason="The Source owner corrected the rule.",
        expected_content_hash=old.content_hash,
        authority=_authority("alice", "member"),
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    review = await db.get_memory_review(result.review_id)
    assert result.outcome == "applied"
    assert stored_old is not None and stored_old.status == "superseded"
    assert stored_new is not None and stored_new.status == "active"
    assert stored_new.visibility == "workspace"
    assert stored_new.owner_user_id is None
    assert await db.get_active_memory_support_reference_ids(old.id) == ()
    assert review is not None and review.status == "approved"
    assert review.expected_support_set_hash == support_hash
    old_sources = await db.get_memory_sources(old.id)
    assert any(source.source_type == "confluence" for source in old_sources)


@pytest.mark.asyncio
async def test_legacy_limited_correction_stages_review_without_direct_apply(
    db: Database,
):
    old = _memory("mem-legacy-limited", "The legacy source-backed rule is stale.")
    observed = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-legacy-limited",
        type="confluence",
        name="Legacy limited source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="source-owner",
    )
    await db.upsert_document(
        DocumentRecord(
            doc_id="doc-legacy-limited",
            source="src-legacy-limited",
            source_url="https://example.test/doc-legacy-limited",
            title="Legacy limited document",
            space_or_project="ENG",
            author=None,
            last_modified=observed,
            labels=[],
            version="1",
            content_hash="legacy-limited-doc-hash",
            token_count=8,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=observed,
        )
    )
    await db.insert_memory(old)
    await db.add_memory_source(
        old.id,
        "doc-legacy-limited",
        "confluence",
        old.content,
        support_kind="legacy_limited",
        source_updated_at=observed,
    )
    service = MemoryLifecycleService(
        db=db,
        memory_store=_store(db, RecordingCollection()),
    )

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The corrected legacy source-backed rule.",
        provenance="The user supplied a correction while legacy Support remained gated.",
        reason="Correct the stale legacy-limited claim without claiming Source authority.",
        expected_content_hash=old.content_hash,
        authority=_authority("workspace-admin", "workspace_admin"),
        replacement_kind="revision",
    )

    incumbent = await db.get_memory(old.id)
    challenger = await db.get_memory(result.replacement_memory_id)
    review = await db.get_memory_review(result.review_id)
    assert result.outcome == "review_created"
    assert incumbent is not None and incumbent.status == "active"
    assert challenger is not None and challenger.status == "pending_review"
    assert review is not None and review.status == "pending"
    assert review.expected_support_set_hash is None
    incumbent_sources = await db.get_memory_sources(old.id)
    assert [(source.source_id, source.support_kind) for source in incumbent_sources] == [
        ("src-legacy-limited", "legacy_limited")
    ]

    await ReviewService(
        db=db,
        memory_store=service.memory_store,
    ).approve(
        review.id,
        reviewer="workspace-admin",
        note="Approve the explicitly confirmed correction.",
        expected_fingerprint=memory_review_decision_fingerprint(review),
    )

    approved_incumbent = await db.get_memory(old.id)
    approved_challenger = await db.get_memory(result.replacement_memory_id)
    approved_review = await db.get_memory_review(review.id)
    assert approved_incumbent is not None and approved_incumbent.status == "superseded"
    assert approved_challenger is not None and approved_challenger.status == "active"
    assert approved_review is not None and approved_review.status == "approved"
    assert await db.get_active_memory_support_reference_ids(old.id) == ()
    assert [(source.source_id, source.support_kind) for source in await db.get_memory_sources(old.id)] == [
        ("src-legacy-limited", "legacy_limited")
    ]


@pytest.mark.asyncio
async def test_authorized_correction_fails_atomically_when_support_changes_after_authority_snapshot(
    db: Database,
    monkeypatch,
):
    old = await _source_backed_memory(db, suffix="authority-snapshot", source_owner="alice")
    additional = await _source_backed_memory(db, suffix="late-support", source_owner="bob")
    store = _store(db, RecordingCollection())
    service = MemoryLifecycleService(db=db, memory_store=store)
    original_states = db.get_active_memory_support_states
    injected = False

    async def inject_after_snapshot(result):
        nonlocal injected
        if not injected:
            injected = True
            await _move_source_support(db, from_memory=additional, to_memory=old)
        return result

    async def states_then_change(memory_ids):
        result = await original_states(memory_ids)
        return await inject_after_snapshot(result)

    monkeypatch.setattr(db, "get_active_memory_support_states", states_then_change)

    with pytest.raises(MemoryLifecycleConflict, match="memory_correction_support_set_changed"):
        await service.propose_memory_correction(
            old.id,
            replacement_content="A stale-authority correction must not commit.",
            provenance="Alice proposed this before Bob's Support appeared.",
            reason="Exercise the authority snapshot stale guard.",
            expected_content_hash=old.content_hash,
            authority=_authority("alice", "member"),
        )

    stored_old = await db.get_memory(old.id)
    assert stored_old is not None and stored_old.status == "active"
    assert await db.count_memories() == 1
    assert await db.count_documents("user_correction") == 0
    assert await db.count_memory_reviews() == 0


@pytest.mark.asyncio
async def test_authorized_correction_rolls_back_staging_when_atomic_resolution_fails(
    db: Database,
    monkeypatch,
):
    old = await _source_backed_memory(db, suffix="atomic-rollback", source_owner="alice")
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))

    async def fail_vector_outbox(*_args, **_kwargs):
        raise RuntimeError("forced atomic resolution failure")

    monkeypatch.setattr(db, "_enqueue_review_vector_task_unlocked", fail_vector_outbox)

    with pytest.raises(RuntimeError, match="forced atomic resolution failure"):
        await service.propose_memory_correction(
            old.id,
            replacement_content="This correction must roll back with its Review.",
            provenance="Alice supplied the correction.",
            reason="Exercise atomic staging rollback.",
            expected_content_hash=old.content_hash,
            authority=_authority("alice", "member"),
        )

    stored_old = await db.get_memory(old.id)
    assert stored_old is not None and stored_old.status == "active"
    assert await db.count_memories() == 1
    assert await db.get_active_memory_support_reference_ids(old.id) == ("eref-atomic-rollback",)
    assert await db.count_documents("user_correction") == 0
    assert await db.count_memory_reviews() == 0


@pytest.mark.asyncio
async def test_authorized_correction_rejects_non_correction_document_provenance(
    db: Database,
    monkeypatch,
):
    old = await _source_backed_memory(db, suffix="provenance-guard", source_owner="alice")
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))
    original_builder = service._build_correction_document

    def build_wrong_source_document(**kwargs):
        return replace(original_builder(**kwargs), source="jira")

    monkeypatch.setattr(service, "_build_correction_document", build_wrong_source_document)

    with pytest.raises(MemoryLifecycleConflict, match="memory_correction_target_changed"):
        await service.propose_memory_correction(
            old.id,
            replacement_content="This correction has invalid provenance.",
            provenance="Alice supplied the correction.",
            reason="Exercise the persistence provenance guard.",
            expected_content_hash=old.content_hash,
            authority=_authority("alice", "member"),
        )

    assert await db.count_memories() == 1
    assert await db.count_documents("user_correction") == 0
    assert await db.count_memory_reviews() == 0


@pytest.mark.asyncio
async def test_propose_memory_correction_creates_review_without_complete_source_authority(db: Database):
    old = await _source_backed_memory(db, suffix="review", source_owner="source-owner")
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))
    support_hash = await db.get_memory_support_set_hash(old.id)

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The proposed corrected source-backed rule.",
        provenance="A workspace member proposed the correction.",
        reason="A workspace member reported a stale rule.",
        expected_content_hash=old.content_hash,
        authority=_authority("workspace-member", "member"),
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    review = await db.get_memory_review(result.review_id)
    assert result.outcome == "review_created"
    assert result.status == "pending"
    assert stored_old is not None and stored_old.status == "active"
    assert stored_new is not None and stored_new.status == "pending_review"
    assert await db.get_active_memory_support_reference_ids(old.id) == ("eref-review",)
    assert review is not None and review.status == "pending"
    assert review.expected_support_set_hash == support_hash


@pytest.mark.asyncio
async def test_propose_memory_correction_requires_authority_over_every_supporting_source(db: Database):
    old = await _source_backed_memory(db, suffix="multi-a", source_owner="alice")
    additional = await _source_backed_memory(db, suffix="multi-b", source_owner="bob")
    await _move_source_support(db, from_memory=additional, to_memory=old)
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The proposed multi-source correction.",
        provenance="Alice proposed a correction but cannot manage Bob's Source.",
        reason="One Source owner proposed a multi-source correction.",
        expected_content_hash=old.content_hash,
        authority=_authority("alice", "member"),
    )

    assert result.outcome == "review_created"
    assert set((await db.get_active_memory_support_states((old.id,)))[old.id].reference_ids) == {
        "eref-multi-a",
        "eref-multi-b",
    }


@pytest.mark.asyncio
async def test_propose_memory_correction_self_hosted_owner_has_complete_workspace_authority(db: Database):
    old = await _source_backed_memory(db, suffix="self-host-a", source_owner="alice")
    additional = await _source_backed_memory(db, suffix="self-host-b", source_owner="bob")
    await _move_source_support(db, from_memory=additional, to_memory=old)
    service = MemoryLifecycleService(db=db, memory_store=_store(db, RecordingCollection()))

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="The self-hosted owner corrected the shared claim.",
        provenance="The self-hosted owner supplied the correction.",
        reason="The local owner corrected all local workspace Sources.",
        expected_content_hash=old.content_hash,
        authority=_authority("dev", "owner"),
    )

    assert result.outcome == "applied"
    assert await db.get_active_memory_support_reference_ids(old.id) == ()


@pytest.mark.asyncio
async def test_correction_review_fails_stale_when_support_set_changes(db: Database):
    old = await _source_backed_memory(db, suffix="stale-a", source_owner="source-owner")
    store = _store(db, RecordingCollection())
    service = MemoryLifecycleService(db=db, memory_store=store)
    proposed = await service.propose_memory_correction(
        old.id,
        replacement_content="The proposed correction must be replanned.",
        provenance="A member proposed a correction before another Source added Support.",
        reason="The original proposal used an older Support Set.",
        expected_content_hash=old.content_hash,
        authority=_authority("workspace-member", "member"),
    )
    review = await db.get_memory_review(proposed.review_id)
    assert review is not None
    additional = await _source_backed_memory(db, suffix="stale-b", source_owner="other-owner")
    await _move_source_support(db, from_memory=additional, to_memory=old)

    with pytest.raises(ReviewStaleConflict):
        await ReviewService(db=db, memory_store=store).approve(
            review.id,
            reviewer="workspace-admin",
            note="Approve the old proposal",
            expected_fingerprint=memory_review_decision_fingerprint(review),
        )

    stored_old = await db.get_memory(old.id)
    stored_challenger = await db.get_memory(proposed.replacement_memory_id)
    assert stored_old is not None and stored_old.status == "active"
    assert stored_challenger is not None and stored_challenger.status == "pending_review"


@pytest.mark.asyncio
async def test_replace_agent_claim_memory_updates_claim_lineage(db: Database):
    observed_at = datetime(2026, 6, 28, tzinfo=timezone.utc)
    await db.upsert_source(
        "src-agent-sessions-codex",
        "agent_session",
        "Codex Session",
        "{}",
        "private",
        "andrew.sun01@sap.com",
        created_by_user_id="andrew.sun01@sap.com",
    )
    collection = RecordingCollection()
    store = _store(db, collection)
    created = await AgentKnowledgeBundleService(db=db, memory_store=store).apply_patch_proposal(
        proposal=AgentKnowledgePatchProposal(
            action="create_new_concept",
            concept_id="concept-claude-cli",
            claim_id="claim-claude-cli",
            concept_type="convention",
            title="Claude Code CLI convention",
            claim_text="Use claude-code to invoke Claude Code CLI",
            durable_claim={
                "rule": "Use claude-code to invoke Claude Code CLI",
                "scope": "Claude Code CLI usage.",
            },
            memory_type="fact",
            reason="Initial observed convention.",
            confidence=0.91,
        ),
        owner_user_id="andrew.sun01@sap.com",
        source_id="src-agent-sessions-codex",
        client="codex",
        session_id="session-initial",
        workspace="/workspace",
        repo_identifier="github.com/shno-labs/mem-forge",
        project_key="UNSORTED",
        submitted_at=observed_at,
        source_updated_at=observed_at,
    )
    assert created.memory_id is not None
    old = await db.get_memory(created.memory_id)
    assert old is not None
    await db.enable_lifecycle_gate("src-agent-sessions-codex")
    service = MemoryLifecycleService(db=db, memory_store=store)

    result = await service.propose_memory_correction(
        old.id,
        replacement_content="Invoke Claude Code with `claude`, not `claude-code`.",
        provenance="User corrected the command while reviewing Claude Code CLI usage.",
        reason="User corrected the command name.",
        expected_content_hash=old.content_hash,
        authority=_authority("andrew.sun01@sap.com", "member"),
        replacement_kind="revision",
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    claim = await db.get_agent_claim("claim-claude-cli")
    concept = await db.get_agent_concept("concept-claude-cli")
    new_sources = await db.get_memory_sources(result.replacement_memory_id)
    assert stored_old is not None
    assert stored_old.status == "superseded"
    assert stored_old.superseded_by == result.replacement_memory_id
    assert await db.get_active_memory_support_reference_ids(old.id) == ()
    assert stored_new is not None
    assert stored_new.status == "active"
    assert await db.get_active_memory_support_reference_ids(result.replacement_memory_id)
    assert stored_new.content == "Invoke Claude Code with `claude`, not `claude-code`."
    assert stored_new.extraction_context == (
        "Invoke Claude Code with `claude`, not `claude-code`."
    )
    assert claim is not None
    assert claim["memory_id"] == result.replacement_memory_id
    assert claim["claim_text"] == "Invoke Claude Code with `claude`, not `claude-code`."
    assert concept is not None
    assert "Invoke Claude Code with `claude`, not `claude-code`." in concept["markdown_body"]
    assert "Use claude-code to invoke Claude Code CLI" not in concept["markdown_body"]
    assert [(source.doc_id, source.source_type) for source in new_sources] == [
        ("concept-claude-cli", "agent_session")
    ]
    assert new_sources[0].excerpt == stored_new.extraction_context

    await db.delete_source_cascade("src-agent-sessions-codex")

    assert await db.get_source("src-agent-sessions-codex") is None
    assert await db.get_agent_concept("concept-claude-cli") is None
    assert await db.get_agent_claim("claim-claude-cli") is None
    deleted_source_memory = await db.get_memory(result.replacement_memory_id)
    assert deleted_source_memory is not None
    assert deleted_source_memory.status == "retired"
    assert await db.get_active_memory_support_reference_ids(result.replacement_memory_id) == ()
    for table in (
        "source_projection_runs",
        "source_units",
        "source_observations",
        "source_lifecycle_gates",
        "lifecycle_plans",
        "memory_support_assertions",
    ):
        async with db.db.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
            assert (await cursor.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_create_search_get_memory_round_trip_keeps_provenance_out_of_search(
    db: Database,
    tmp_path,
    monkeypatch,
):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    class RoundTripRuntimeProvider:
        def build_adapters(self, database, memory_collection, *, audit_logger=None):
            return build_sqlite_adapters(
                database,
                memory_collection,
                audit_logger=audit_logger,
            )

        async def build_search_engine(self, database, config, *, audit_logger=None):
            adapters = build_sqlite_adapters(
                database,
                RecordingCollection(),
                audit_logger=audit_logger,
            )
            return SearchEngine(
                relational=adapters.relational,
                keyword=adapters.keyword,
                vector=adapters.vector,
                embed_cfg={},
                config=config.retrieval,
                embedding_provider=lambda _text: None,
            )

    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
        runtime_provider=RoundTripRuntimeProvider(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memories/create",
            json={
                "content": "Use canonical payroll trigger status fields.",
                "provenance": "User confirmed this after validating the payroll smoke flow.",
                "memory_type": "fact",
                "client": "codex",
            },
        )
        payload = response.json()
        search_response = client.post(
            "/api/v1/memories/search",
            json={
                "query": "canonical payroll trigger status fields",
                "include_private": True,
                "top_k": 10,
            },
        )
        detail_response = client.get(
            f"/api/v1/memories/{payload['memory_id']}?include_private=true"
        )

    assert response.status_code == 200, response.text
    assert search_response.status_code == 200, search_response.text
    assert detail_response.status_code == 200, detail_response.text
    search_payload = search_response.json()
    [search_result] = [
        result
        for result in search_payload["results"]
        if result["memory_id"] == payload["memory_id"]
    ]
    assert search_result["summary"] == "Use canonical payroll trigger status fields."
    assert "User confirmed this after validating the payroll smoke flow." not in str(search_result)
    assert "sources" not in search_result
    detail = detail_response.json()
    assert detail["id"] == payload["memory_id"]
    assert detail["content"] == "Use canonical payroll trigger status fields."
    [source] = detail["sources"]
    assert source["doc_id"] == f"user-memory-{payload['memory_id']}"
    assert source["source_type"] == "user_memory"
    assert source["support_kind"] == "extracted"
    assert source["doc_title"] == f"User memory {payload['memory_id']}"
    assert source["source_url"] == f"memforge://user-memory/user-memory-{payload['memory_id']}"
    assert source["content_url"] is None
    assert source["pdf_url"] is None
    assert source["excerpt"] == "User confirmed this after validating the payroll smoke flow."
    stored = await db.get_memory(payload["memory_id"])
    assert payload["status"] == "inserted"
    assert stored is not None
    assert stored.owner_user_id == "andrew.sun01@sap.com"
    assert stored.visibility == "private"
    assert stored.repo_identifier is None
    assert stored.extraction_context == "User confirmed this after validating the payroll smoke flow."
    audit_rows = await db.list_memory_audit_events(
        memory_id=payload["memory_id"],
        event_type="memory_insert_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"
    document = await db.get_document(f"user-memory-{payload['memory_id']}")
    assert document is not None
    assert document.client == "codex"


@pytest.mark.asyncio
async def test_create_memory_route_requires_provenance(db: Database, tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memories/create",
            json={
                "content": "Keep create_memory content durable and reusable.",
                "reason": "Legacy clients used to send this field.",
                "memory_type": "convention",
                "client": "codex",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_memory_route_rejects_legacy_reason_field(db: Database, tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    provenance = "User explicitly asked to keep this convention after reviewing MCP memory behavior."
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memories/create",
            json={
                "content": "Keep create_memory content durable and reusable.",
                "provenance": provenance,
                "reason": "Legacy clients used to send this field.",
                "memory_type": "convention",
                "client": "codex",
            },
        )

    assert response.status_code == 422
    assert "Extra inputs are not permitted" in response.text


@pytest.mark.asyncio
async def test_retire_memory_route_returns_conflict_for_stale_content_hash(db: Database, tmp_path):
    from memforge.server.admin_api import create_admin_app

    memory = _memory("mem-retire-route", "Route guarded fact")
    await db.insert_memory(memory)
    app = create_admin_app(db=db, config=_api_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/memories/{memory.id}/retire",
            json={
                "reason": "User says this is stale",
                "expected_content_hash": "wrong",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "content_hash_mismatch"


@pytest.mark.asyncio
async def test_retire_memory_route_audits_request_principal(db: Database, tmp_path):
    from memforge.server.admin_api import create_admin_app

    memory = _memory("mem-retire-route-audit", "Route audited fact")
    await db.insert_memory(memory)
    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/memories/{memory.id}/retire",
            json={
                "reason": "User says this is stale",
                "expected_content_hash": memory.content_hash,
            },
        )

    assert response.status_code == 200, response.text
    audit_rows = await db.list_memory_audit_events(
        memory_id=memory.id,
        event_type="memory_retire_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"


@pytest.mark.asyncio
async def test_propose_memory_correction_route_audits_request_principal(db: Database, tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    memory = _memory("mem-replace-route-audit", "Route replacement audited fact")
    await db.insert_memory(memory)
    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/memories/{memory.id}/corrections/propose",
            json={
                "replacement_content": "Route replacement corrected fact",
                "provenance": "User supplied the corrected route in chat.",
                "reason": "User corrected this memory.",
                "expected_content_hash": memory.content_hash,
                "replacement_kind": "supersession",
            },
        )
        removed = client.post(
            f"/api/v1/memories/{memory.id}/replace",
            json={
                "replacement_content": "Removed legacy route",
                "provenance": "Legacy route must remain absent.",
                "reason": "The correction contract is unified.",
                "expected_content_hash": memory.content_hash,
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "applied"
    assert removed.status_code == 404
    audit_rows = await db.list_memory_audit_events(
        memory_id=memory.id,
        event_type="memory_supersede_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"


@pytest.mark.asyncio
async def test_propose_memory_correction_route_creates_review_for_workspace_member(
    db: Database,
    tmp_path,
    monkeypatch,
):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)
    old = await _source_backed_memory(db, suffix="route-review", source_owner="source-owner")
    app = create_admin_app(
        db=db,
        config=_api_config(tmp_path),
        principal_resolver=lambda _request: "workspace-member",
        workspace_role_resolver=lambda _request: "member",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/memories/{old.id}/corrections/propose",
            json={
                "replacement_content": "The route-proposed corrected claim.",
                "provenance": "A workspace member supplied this correction.",
                "reason": "The member reported stale source knowledge.",
                "expected_content_hash": old.content_hash,
                "replacement_kind": "supersession",
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["outcome"] == "review_created"
    assert response.json()["status"] == "pending"
    review = await db.get_memory_review(response.json()["review_id"])
    assert review is not None and review.expected_support_set_hash
