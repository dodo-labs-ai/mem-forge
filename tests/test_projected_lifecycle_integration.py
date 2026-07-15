from __future__ import annotations

from datetime import datetime, timezone

import pytest
import pytest_asyncio

from memforge.llm.structured import (
    ContradictionDecision,
    ContradictionResponse,
    ReconciliationDecision,
    ReconciliationResponse,
)
from memforge.memory.engine import MemoryEngine
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    NormalizedContent,
    RawContent,
    RawMemory,
    content_hash,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import (
    project_source_item,
    project_source_unit_tombstone,
)
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "projected-lifecycle.db"))
    await database.connect()
    await database.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc).isoformat()
    await database.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project,
               last_modified, version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("confluence-123", "src-1", "https://example.test/123", "Page", "ENG", now, "2", "h", now),
    )
    try:
        yield database
    finally:
        await database.close()


def _projection(
    *,
    run_id: str,
    body: str,
    item_id: str = "confluence-123",
    prior=None,
    prior_observations=None,
):
    item = ContentItem(
        item_id=item_id,
        title="Page",
        source_url="https://example.test/123",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="2",
        extra={"page_id": "123", "space_key": "ENG"},
    )
    raw = RawContent(item=item, body=body.encode(), content_type="text/html")
    normalized = NormalizedContent(item=item, markdown_body=body)
    return project_source_item(
        source_id="src-1",
        source_type="confluence",
        run_id=run_id,
        item=item,
        raw=raw,
        normalized=normalized,
        prior_unit_revision=prior,
        prior_observation_revisions=prior_observations,
    )


class _ReplacementClient:
    def __init__(self, incumbent_id: str) -> None:
        self.incumbent_id = incumbent_id

    async def reconcile_memories(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ReconciliationResponse(
            decisions=[
                ReconciliationDecision(
                    index=0,
                    action="SUPERSEDE",
                    memory_id=self.incumbent_id,
                    reason="The source now retains A7.",
                )
            ]
        )

    async def detect_contradictions(self, prompt: str, **kwargs):
        del prompt, kwargs
        return ContradictionResponse(
            decisions=[
                ContradictionDecision(
                    pair_index=0,
                    classification="contradiction",
                    reason="Independent source still says A7 is removed.",
                )
            ]
        )


class _OutboxDrainer:
    def __init__(self, database: Database) -> None:
        self.db = database

    async def drain_lifecycle_vector_outbox(self, lifecycle_plan_id: str) -> None:
        for task in await self.db.list_lifecycle_vector_tasks(
            lifecycle_plan_id=lifecycle_plan_id
        ):
            await self.db.complete_lifecycle_vector_task(task.id)


async def _seed_incumbent_support(
    db: Database,
    *,
    projection,
    memory_id: str = "mem-old",
) -> Memory:
    incumbent = Memory(
        id=memory_id,
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
    )
    await db.insert_memory(incumbent)
    await db.add_memory_source(
        incumbent.id,
        "confluence-123",
        "confluence",
        "A7 is removed.",
        source_updated_at=None,
    )
    revision = projection.observation_revisions[0]
    unit = EvidenceUnit(
        id=f"eu-{memory_id}",
        source_id="src-1",
        doc_id="confluence-123",
        doc_revision_id=projection.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=projection.observations[0].id,
        source_lineage_id=projection.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=revision.content,
        excerpt="A7 is removed.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
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
                        observation_id=projection.observations[0].id,
                        observation_revision_id=revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id=f"support-{memory_id}",
            memory_id=incumbent.id,
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash="workspace-eng",
        )
    )
    return incumbent


@pytest.mark.asyncio
async def test_enabled_source_supersedes_incumbent_in_one_atomic_plan(db: Database) -> None:
    first = _projection(
        run_id="projection-1",
        body="A7 is removed.",
        item_id="confluence-old-path",
    )
    await db.record_source_projection(first)
    now = datetime(2026, 7, 14, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="confluence-old-path",
            source="src-1",
            source_url="https://example.test/123",
            title="Page",
            space_or_project="ENG",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="old-hash",
            token_count=10,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    incumbent = Memory(
        id="mem-old",
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
    )
    await db.insert_memory(incumbent)
    await db.add_memory_source(
        incumbent.id,
        "confluence-old-path",
        "confluence",
        "A7 is removed.",
        source_updated_at=None,
    )
    old_revision = first.observation_revisions[0]
    old_unit = EvidenceUnit(
        id="eu-old",
        source_id="src-1",
        doc_id="confluence-old-path",
        doc_revision_id=first.source_unit_revisions[0].id,
        source_type="confluence",
        source_anchor=first.observations[0].id,
        source_lineage_id=first.source_units[0].id,
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content=old_revision.content,
        excerpt="A7 is removed.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace-eng",
    )
    await db.upsert_evidence_unit(old_unit)
    old_reference = (
        await db.record_evidence_references(
            old_unit.id,
            (
                EvidenceReference(
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=first.observations[0].id,
                        observation_revision_id=old_revision.id,
                    ),
                ),
            ),
        )
    )[0]
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-old",
            memory_id=incumbent.id,
            evidence_reference_id=old_reference.id or "",
            source_id="src-1",
            access_context_hash="workspace-eng",
        )
    )
    await db.enable_lifecycle_gate("src-1")

    entity_id = await db.upsert_entity("A7", "A7", ["payroll-result-slot"])
    await db.upsert_source(
        id="src-2",
        type="jira",
        name="Payroll Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id="jira-456",
            source="src-2",
            source_url="https://example.test/browse/PAY-456",
            title="A7 behavior",
            space_or_project="PAY",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash="jira-hash",
            token_count=20,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    cross_source_memory = Memory(
        id="mem-cross-source",
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
        project_key="ENG",
    )
    await db.insert_memory(cross_source_memory)
    await db.add_memory_source(
        cross_source_memory.id,
        "jira-456",
        "jira",
        "A7 is removed.",
        source_updated_at=now,
    )
    await db.link_memory_entity(cross_source_memory.id, entity_id)

    second = _projection(
        run_id="projection-2",
        body="A7 is retained and marked as reduced retro chain.",
        prior=first.source_unit_revisions[0],
        prior_observations={first.observations[0].id: old_revision},
    )
    raw = RawMemory(
        content="A7 is retained and marked as reduced retro chain.",
        memory_type="decision",
        confidence=0.95,
        tags=["payroll", "retro"],
        extraction_context="A7 is retained and marked as reduced retro chain.",
    )
    evidence = build_projected_claim_evidence(
        projection=second,
        raw_memories=(raw,),
        doc_id="confluence-123",
        source_type="confluence",
        project_key="ENG",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace-eng",
        extractor_run_id="sync-2",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
        structured_llm_client=_ReplacementClient(incumbent.id),
    )
    stats = await engine.apply_projected_lifecycle(
        projection=second,
        doc_id="confluence-123",
        raw_memories=[raw],
        doc_type="design-doc",
        project_key="ENG",
        repo_identifier=None,
        entity_ids=[entity_id],
        document_content=raw.content,
        update_mode="full_document",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=datetime(2026, 7, 15, 10, 36, tzinfo=timezone.utc),
    )

    old = await db.get_memory(incumbent.id)
    assert old is not None and old.status == "superseded"
    assert old.superseded_by is not None
    replacement = await db.get_memory(old.superseded_by)
    assert replacement is not None and replacement.status == "active"
    assert stats["superseded"] == 1
    assert stats["contradictions_found"] == 1
    cross_source_review = await db.get_pending_review_for_challenger(replacement.id)
    assert cross_source_review is not None
    assert cross_source_review.kind == "cross_source_conflict"
    assert (await db.get_memory(cross_source_memory.id)).status == "active"
    assert (await db.get_memory(replacement.id)).status == "active"
    persisted_evidence = await db.get_evidence_unit(evidence.units[0].id)
    assert persisted_evidence is not None
    assert persisted_evidence.source_lineage_id == evidence.units[0].source_lineage_id
    assert persisted_evidence.content == evidence.units[0].content
    assert persisted_evidence.extractor_run_id == second.run_id
    assert persisted_evidence.observed_at == "2026-07-15T10:36:00+00:00"
    assert await db.list_lifecycle_vector_tasks() == []


@pytest.mark.asyncio
async def test_enabled_source_tombstone_retires_last_supported_incumbent(db: Database) -> None:
    initial = _projection(run_id="projection-before-delete", body="A7 is removed.")
    await db.record_source_projection(initial)
    incumbent = await _seed_incumbent_support(db, projection=initial)
    await db.enable_lifecycle_gate("src-1")
    tombstone = project_source_unit_tombstone(
        source_type="confluence",
        run_id="projection-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision for revision in initial.observation_revisions
        },
        reason="not_returned_by_authoritative_snapshot",
    )
    engine = MemoryEngine(
        relational=build_sqlite_adapters(db, object()).relational,
        vector=build_sqlite_adapters(db, object()).vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    result = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
    )

    retired = await db.get_memory(incumbent.id)
    assert retired is not None and retired.status == "retired"
    assert result == {"retired": 1, "pending_review": 0, "can_delete_document": True}
    assert await db.list_lifecycle_vector_tasks() == []

    await db.delete_projected_document("confluence-123")

    assert await db.get_document("confluence-123") is None
    assert await db.get_memory_sources(incumbent.id) == []
    assert await db.get_evidence_unit(f"eu-{incumbent.id}") is not None
    assert await db.get_source_projection(initial.run_id) == initial


@pytest.mark.asyncio
async def test_gated_source_tombstone_only_opens_review(db: Database) -> None:
    initial = _projection(run_id="projection-before-gated-delete", body="A7 is removed.")
    await db.record_source_projection(initial)
    incumbent = await _seed_incumbent_support(db, projection=initial)
    tombstone = project_source_unit_tombstone(
        source_type="confluence",
        run_id="projection-gated-delete",
        source_unit=initial.source_units[0],
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={
            revision.observation_id: revision for revision in initial.observation_revisions
        },
        reason="not_returned_by_authoritative_snapshot",
    )
    adapters = build_sqlite_adapters(db, object())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=_OutboxDrainer(db),
    )

    result = await engine.apply_projected_tombstone(
        projection=tombstone,
        doc_id="confluence-123",
        reason="not_returned_by_authoritative_snapshot",
    )

    active = await db.get_memory(incumbent.id)
    assert active is not None and active.status == "active"
    assert result == {"retired": 0, "pending_review": 1, "can_delete_document": False}
    with pytest.raises(ValueError, match="active document support remains"):
        await db.delete_projected_document("confluence-123")
