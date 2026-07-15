from __future__ import annotations

from dataclasses import replace

import pytest
import pytest_asyncio

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
    evidence_reference_id_for,
)
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleCutoverFinding,
    LifecycleGateState,
    LifecycleBackfillJobStatus,
    IncumbentDecision,
    IncumbentDisposition,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    LifecycleReviewStatus,
    ReconciliationScope,
    StaleGuard,
)
from memforge.memory.lifecycle_planner import NewMemoryDefaults, build_lifecycle_plan
from memforge.memory.lifecycle_review import build_lifecycle_review_approval_plan
from memforge.memory.cutover import (
    run_source_lifecycle_backfill,
    run_source_lifecycle_backfill_job,
    run_source_lifecycle_recovery_job,
)
from memforge.models import Memory, ReconcileAction, ReconcileOperation, content_hash
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database, MIGRATIONS
from tests.test_source_projection_store import _projection


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "cutover.db"))
    await database.connect()
    await database.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    memory = Memory(
        id="mem-legacy",
        memory_type="fact",
        content="Legacy claim",
        content_hash=content_hash("Legacy claim"),
    )
    await database.insert_memory(memory)
    await database.record_source_projection(_projection())
    try:
        yield database
    finally:
        await database.close()


def _finding() -> LifecycleCutoverFinding:
    return LifecycleCutoverFinding(
        id="finding-1",
        source_id="src-1",
        memory_id="mem-legacy",
        reason=CutoverFindingReason.OBSERVATION_NOT_FOUND,
        status=CutoverFindingStatus.OPEN,
        available_provenance={"doc_id": "legacy-doc"},
        mapping_attempt={"strategy": "document-id"},
    )


def _unit() -> EvidenceUnit:
    return EvidenceUnit(
        id="eu-backfill-1",
        source_id="src-1",
        doc_id=None,
        doc_revision_id="obsrev-page-1-v2",
        source_type="confluence",
        source_anchor="legacy-compatible-anchor",
        source_lineage_id="unit-page-1",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="Legacy claim",
        excerpt="Legacy claim",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )


def test_cutover_schema_has_a_forward_migration() -> None:
    version, description, statements = next(item for item in MIGRATIONS if item[0] == 48)

    assert version == 48
    assert description == "Add lifecycle cutover gates findings and support assertions"
    assert any("CREATE TABLE IF NOT EXISTS source_lifecycle_gates" in item for item in statements)
    assert any("CREATE TABLE IF NOT EXISTS lifecycle_cutover_findings" in item for item in statements)


def test_backfill_job_schema_has_a_forward_migration() -> None:
    version, description, statements = next(item for item in MIGRATIONS if item[0] == 51)

    assert version == 51
    assert description == "Add durable lifecycle backfill jobs"
    assert any("CREATE TABLE IF NOT EXISTS lifecycle_backfill_jobs" in item for item in statements)


@pytest.mark.asyncio
async def test_new_source_is_destructive_lifecycle_gated_by_default(db: Database) -> None:
    gate = await db.get_lifecycle_gate("src-1")

    assert gate.state is LifecycleGateState.GATED


@pytest.mark.asyncio
async def test_open_finding_blocks_gate_and_resolution_preserves_history(db: Database) -> None:
    finding = _finding()
    await db.upsert_lifecycle_cutover_finding(finding)

    with pytest.raises(ValueError, match="open lifecycle cutover findings"):
        await db.enable_lifecycle_gate("src-1")

    unit = _unit()
    await db.upsert_evidence_unit(unit)
    reference = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-page-1-body",
            observation_revision_id="obsrev-page-1-v2",
        ),
        evidence_unit_id=unit.id,
    )
    reference = replace(reference, id=evidence_reference_id_for(unit.id, reference))
    await db.record_evidence_references(unit.id, (reference,))
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-1",
            memory_id="mem-legacy",
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash="workspace",
        )
    )

    resolved = await db.resolve_lifecycle_cutover_finding(
        finding.id,
        observation_id="obs-page-1-body",
        source_unit_id="unit-page-1",
    )
    gate = await db.enable_lifecycle_gate("src-1")

    assert resolved.status is CutoverFindingStatus.RESOLVED
    assert resolved.created_at == (await db.get_lifecycle_cutover_finding(finding.id)).created_at
    assert gate.state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_finding_cannot_resolve_before_memory_lineage_is_persisted(db: Database) -> None:
    await db.upsert_lifecycle_cutover_finding(_finding())

    with pytest.raises(ValueError, match="validated support lineage"):
        await db.resolve_lifecycle_cutover_finding(
            "finding-1",
            observation_id="obs-page-1-body",
            source_unit_id="unit-page-1",
        )


async def _persist_support_lineage(db: Database) -> EvidenceReference:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    reference = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-page-1-body",
            observation_revision_id="obsrev-page-1-v2",
        ),
        evidence_unit_id=unit.id,
    )
    reference = replace(reference, id=evidence_reference_id_for(unit.id, reference))
    await db.record_evidence_references(unit.id, (reference,))
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-1",
            memory_id="mem-legacy",
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash="workspace",
        )
    )
    return reference


def _retirement_plan(reference: EvidenceReference, support_hash: str) -> LifecyclePlan:
    return LifecyclePlan(
        id="plan-retire-1",
        scope=ReconciliationScope(
            id="scope-retire-1",
            source_id="src-1",
            source_unit_id="unit-page-1",
            base_unit_revision_id="unitrev-page-1-v1",
            target_unit_revision_id="unitrev-page-1-v2",
        ),
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=("mem-legacy",),
            incumbent_decisions=(
                IncumbentDecision(
                    "mem-legacy",
                    IncumbentDisposition.REMOVE_SUPPORT,
                    "authoritative evidence removed",
                ),
            ),
            batch_ids=("batch-1",),
            completed_batch_ids=("batch-1",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=("obsrev-page-1-v2",),
            support_set_hashes={"mem-legacy": support_hash},
        ),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-legacy",
                source_id="src-1",
                evidence_reference_ids=(reference.id or "",),
            ),
            LifecycleMutation(
                LifecycleMutationType.RETIRE_MEMORY,
                memory_id="mem-legacy",
                source_id="src-1",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_lifecycle_plan_applies_support_removal_and_retirement_atomically(db: Database) -> None:
    reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(reference, await db.get_memory_support_set_hash("mem-legacy"))

    await db.apply_lifecycle_plan(plan)
    await db.apply_lifecycle_plan(plan)

    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "retired"
    assert await db.get_lifecycle_plan_status(plan.id) == "applied"


@pytest.mark.asyncio
async def test_gated_review_approval_applies_proposal_and_resolves_review_atomically(
    db: Database,
) -> None:
    reference = await _persist_support_lineage(db)
    incumbent = await db.get_memory("mem-legacy")
    assert incumbent is not None
    support_hash = await db.get_memory_support_set_hash(incumbent.id)
    original = build_lifecycle_plan(
        plan_id="plan-gated-delete",
        scope=ReconciliationScope(
            id="scope-gated-delete",
            source_id="src-1",
            source_unit_id="unit-page-1",
            base_unit_revision_id="unitrev-page-1-v1",
            target_unit_revision_id="unitrev-page-1-v2",
        ),
        gate_state=LifecycleGateState.GATED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=incumbent.id,
                reason="source no longer supports claim",
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_reference_ids={incumbent.id: (reference.id or "",)},
        all_active_support_reference_ids={incumbent.id: (reference.id or "",)},
        support_set_hashes={incumbent.id: support_hash},
        observation_revision_ids=("obsrev-page-1-v2",),
        new_evidence_reference_ids=(),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key=None,
            repo_identifier=None,
            doc_id="legacy-doc",
            source_type="confluence",
            access_context_hash="workspace",
        ),
    )

    await db.apply_lifecycle_plan(original)
    reviews = await db.list_lifecycle_reviews(
        "src-1",
        status=LifecycleReviewStatus.PENDING,
    )
    assert len(reviews) == 1
    assert (await db.get_memory(incumbent.id)).status == "active"  # type: ignore[union-attr]

    await db.enable_lifecycle_gate("src-1")
    payload = await db.get_lifecycle_plan_payload(original.id)
    assert payload is not None
    approval = build_lifecycle_review_approval_plan(reviews[0], payload)
    await db.apply_lifecycle_plan(approval)

    approved = await db.get_lifecycle_review(reviews[0].id)
    retired = await db.get_memory(incumbent.id)
    assert approved is not None and approved.status is LifecycleReviewStatus.APPROVED
    assert retired is not None and retired.status == "retired"
    tasks = await db.list_lifecycle_vector_tasks(source_id="src-1")
    assert {(task.memory_id, task.operation.value) for task in tasks} == {
        (incumbent.id, "delete")
    }


@pytest.mark.asyncio
async def test_stale_lifecycle_plan_rolls_back_without_partial_mutation(db: Database) -> None:
    reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    stale = _retirement_plan(reference, "not-the-current-support-hash")

    with pytest.raises(ValueError, match="support stale guard"):
        await db.apply_lifecycle_plan(stale)

    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "active"
    assert await db.get_lifecycle_plan_status(stale.id) is None


@pytest.mark.asyncio
async def test_mutation_failure_rolls_back_staged_evidence_with_the_plan(db: Database) -> None:
    active_reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    staged_unit = replace(
        _unit(),
        id="eu-staged-rollback",
        source_anchor="obs-page-1-body",
    )
    staged_reference = EvidenceReference(
        id="eref-staged-rollback",
        evidence_unit_id=staged_unit.id,
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-page-1-body",
            observation_revision_id="obsrev-page-1-v2",
        ),
    )
    plan = _retirement_plan(
        active_reference,
        await db.get_memory_support_set_hash("mem-legacy"),
    )
    plan = replace(
        plan,
        id="plan-evidence-rollback",
        evidence_units=(staged_unit,),
        evidence_references=(staged_reference,),
        mutations=(
            replace(
                plan.mutations[0],
                evidence_reference_ids=("eref-not-active-support",),
            ),
            plan.mutations[1],
        ),
    )

    with pytest.raises(ValueError, match="complete active support set"):
        await db.apply_lifecycle_plan(plan)

    assert await db.get_evidence_unit(staged_unit.id) is None
    assert await db.get_lifecycle_plan_status(plan.id) is None
    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "active"


@pytest.mark.asyncio
async def test_mutation_failure_rolls_back_source_projection_with_the_plan(db: Database) -> None:
    active_reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    previous = _projection()
    observation_revision = replace(
        previous.observation_revisions[0],
        id="obsrev-page-1-v3",
        semantic_hash="body-hash-v3",
        content="third body",
    )
    unit_revision = replace(
        previous.source_unit_revisions[0],
        id="unitrev-page-1-v3",
        semantic_hash="unit-hash-v3",
        observation_revision_ids=(observation_revision.id,),
    )
    changed_anchor = replace(
        previous.deltas[0].changed_anchors[0],
        observation_revision_id=observation_revision.id,
    )
    projection = replace(
        previous,
        run_id="projection-run-atomic-rollback",
        observation_revisions=(observation_revision,),
        source_unit_revisions=(unit_revision,),
        deltas=(
            replace(
                previous.deltas[0],
                previous_unit_revision_id=previous.source_unit_revisions[0].id,
                current_unit_revision_id=unit_revision.id,
                changed_anchors=(changed_anchor,),
                fragment_mappings=(),
            ),
        ),
    )
    plan = _retirement_plan(
        active_reference,
        await db.get_memory_support_set_hash("mem-legacy"),
    )
    plan = replace(
        plan,
        id="plan-projection-rollback",
        scope=replace(
            plan.scope,
            base_unit_revision_id=previous.source_unit_revisions[0].id,
            target_unit_revision_id=unit_revision.id,
        ),
        stale_guard=replace(
            plan.stale_guard,
            observation_revision_ids=(observation_revision.id,),
        ),
        mutations=(
            replace(
                plan.mutations[0],
                evidence_reference_ids=("eref-not-active-support",),
            ),
            plan.mutations[1],
        ),
    )

    with pytest.raises(ValueError, match="complete active support set"):
        await db.apply_source_projection_lifecycle(projection, plan)

    current = await db.get_current_source_unit_revision("unit-page-1")
    assert current is not None and current.id == "unitrev-page-1-v2"
    assert await db.get_source_projection(projection.run_id) is None
    assert await db.get_lifecycle_plan_status(plan.id) is None


@pytest.mark.asyncio
async def test_memory_version_stale_guard_rejects_concurrent_incumbent_change(
    db: Database,
) -> None:
    reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(
        reference,
        await db.get_memory_support_set_hash("mem-legacy"),
    )
    plan = replace(
        plan,
        stale_guard=replace(
            plan.stale_guard,
            memory_versions={"mem-legacy": "memory-version-before-concurrent-edit"},
        ),
    )

    with pytest.raises(ValueError, match="Memory stale guard"):
        await db.apply_lifecycle_plan(plan)

    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "active"
    assert await db.get_lifecycle_plan_status(plan.id) is None


@pytest.mark.asyncio
async def test_backfill_maps_exact_document_lineage_and_enables_gate(db: Database) -> None:
    now = "2026-07-15T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified, version,
               content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy-doc", "src-1", "https://example.test/page-1", "Page 1", "ENG", now, "1", "hash", now),
    )
    await db.add_memory_source(
        "mem-legacy",
        "legacy-doc",
        "confluence",
        "new body",
        source_updated_at=None,
    )
    projection = _projection()
    projection = replace(
        projection,
        source_units=(
            replace(
                projection.source_units[0],
                locator={"document_id": "legacy-doc", "url": "https://example.test/page-1"},
            ),
        ),
    )
    # The fixture already persisted the original retry identity. Use a distinct
    # run so the enriched locator is a new immutable projection snapshot.
    await db.record_source_projection(replace(projection, run_id="projection-run-backfill"))

    result = await run_source_lifecycle_backfill(db, "src-1")

    assert result.scanned_memories == 1
    assert result.mapped_memories == 1
    assert result.finding_count == 0
    assert result.gate_enabled is True
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_recovery_reextracts_only_identifiable_documents_then_validates_lineage(
    db: Database,
) -> None:
    now = "2026-07-15T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified, version,
               content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy-doc", "src-1", "https://example.test/page-1", "Page 1", "ENG", now, "1", "hash", now),
    )
    await db.add_memory_source(
        "mem-legacy",
        "legacy-doc",
        "confluence",
        "new body",
        source_updated_at=None,
    )
    requested: list[frozenset[str]] = []

    async def reextract(document_ids: frozenset[str]) -> None:
        requested.append(document_ids)
        projection = _projection()
        await db.record_source_projection(
            replace(
                projection,
                run_id="projection-run-recovered",
                source_units=(
                    replace(
                        projection.source_units[0],
                        locator={
                            "document_id": "legacy-doc",
                            "url": "https://example.test/page-1",
                        },
                    ),
                ),
            )
        )

    completed = await run_source_lifecycle_recovery_job(
        db,
        "src-1",
        job_id="backfill-recovery",
        reextract_documents=reextract,
    )

    assert requested == [frozenset({"legacy-doc"})]
    assert completed.status is LifecycleBackfillJobStatus.COMPLETED
    assert completed.finding_count == 0
    findings = await db.list_lifecycle_cutover_findings("src-1")
    assert len(findings) == 1
    assert findings[0].status is CutoverFindingStatus.RESOLVED
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_backfill_leaves_durable_finding_when_source_unit_cannot_be_located(db: Database) -> None:
    now = "2026-07-15T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified, version,
               content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("missing-doc", "src-1", "https://example.test/missing", "Missing", "ENG", now, "1", "hash", now),
    )
    await db.add_memory_source(
        "mem-legacy",
        "missing-doc",
        "confluence",
        "Legacy claim",
        source_updated_at=None,
    )

    result = await run_source_lifecycle_backfill(db, "src-1")

    assert result.finding_count == 1
    assert result.gate_enabled is False
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.GATED


@pytest.mark.asyncio
async def test_backfill_job_records_completed_counts_and_is_idempotent(db: Database) -> None:
    completed = await run_source_lifecycle_backfill_job(
        db,
        "src-1",
        job_id="backfill-job-1",
    )
    retried = await run_source_lifecycle_backfill_job(
        db,
        "src-1",
        job_id="backfill-job-1",
    )

    assert completed.status is LifecycleBackfillJobStatus.COMPLETED
    assert retried == completed
    assert await db.list_lifecycle_backfill_jobs("src-1") == [completed]


@pytest.mark.asyncio
async def test_backfill_job_failure_is_durable(db: Database, monkeypatch) -> None:
    async def fail_scan(source_id: str):
        del source_id
        raise RuntimeError("projection store unavailable")

    monkeypatch.setattr(db, "list_legacy_memory_provenance", fail_scan)

    with pytest.raises(RuntimeError, match="projection store unavailable"):
        await run_source_lifecycle_backfill_job(
            db,
            "src-1",
            job_id="backfill-job-failed",
        )

    failed = await db.get_lifecycle_backfill_job("backfill-job-failed")
    assert failed is not None
    assert failed.status is LifecycleBackfillJobStatus.FAILED
    assert failed.error == "projection store unavailable"
