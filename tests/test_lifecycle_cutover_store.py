from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import sqlite3
from types import SimpleNamespace

import pytest
import pytest_asyncio

from memforge.memory.evidence import (
    ActiveSupportEvidence,
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
)
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleCutoverFinding,
    LifecycleBackfillJob,
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
    lifecycle_plan_to_payload,
)
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_memory_version,
)
from memforge.memory.lifecycle_review import (
    build_lifecycle_review_approval_plan,
    build_lifecycle_review_refresh_plan,
)
from memforge.memory.cutover import (
    recover_stale_lifecycle_jobs,
    run_source_lifecycle_backfill,
    run_source_lifecycle_backfill_job,
    run_source_lifecycle_recovery_job,
    run_with_lifecycle_activity_heartbeat,
)
from memforge.models import (
    DocumentRecord,
    Memory,
    MemoryReview,
    ReconcileAction,
    ReconcileOperation,
    ReviewKind,
    ReviewStatus,
    content_hash,
)
from memforge.source_activity import SourceActivityConflict, SourceActivityKind
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database, MIGRATIONS
from tests.test_source_projection_store import _projection
from tests.unit_support_fixture import complete_unit_parts, primary_reference, record_unit_support


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
        doc_id="legacy-gate-doc",
        doc_revision_id="unitrev-page-1-v2",
        source_type="confluence",
        source_anchor="obs-page-1-body",
        source_lineage_id="unit-page-1",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="Legacy claim",
        excerpt="Legacy claim",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace",
    )


async def _add_unsupported_legacy_source_edge(
    db: Database,
    *,
    doc_id: str = "legacy-doc",
) -> None:
    now = "2026-07-16T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified,
               version, content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            "src-1",
            f"https://example.test/{doc_id}",
            "Legacy document",
            "ENG",
            now,
            "1",
            "legacy-hash",
            now,
        ),
    )
    await db.add_memory_source(
        "mem-legacy",
        doc_id,
        "confluence",
        "Legacy claim",
        source_updated_at=None,
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
async def test_lifecycle_activity_heartbeat_cancels_operation_with_wrapper() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HeartbeatDatabase:
        async def renew_source_activity(self, **_kwargs) -> None:
            raise AssertionError("long heartbeat interval should not renew")

    async def operation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(
        run_with_lifecycle_activity_heartbeat(
            HeartbeatDatabase(),
            "job-cancelled",
            operation,
            heartbeat_interval_seconds=3600,
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_lifecycle_activity_cancellation_fails_durable_recovery_job(
    db: Database,
    monkeypatch,
) -> None:
    scan_started = asyncio.Event()

    async def block_scan(_source_id: str):
        scan_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(db, "list_legacy_memory_provenance", block_scan)
    job_id = "recovery-cancelled-durably"
    task = asyncio.create_task(
        run_with_lifecycle_activity_heartbeat(
            db,
            job_id,
            lambda: run_source_lifecycle_recovery_job(
                db,
                "src-1",
                job_id=job_id,
            ),
            heartbeat_interval_seconds=3600,
        )
    )
    await scan_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    job = await db.get_lifecycle_backfill_job(job_id)
    assert job is not None
    assert job.status is LifecycleBackfillJobStatus.FAILED
    assert job.error == "lifecycle recovery cancelled"
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


@pytest.mark.asyncio
async def test_repeated_wrapper_cancellation_waits_for_durable_job_cleanup(
    db: Database,
    monkeypatch,
) -> None:
    scan_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    fail_job = db.fail_lifecycle_backfill_job

    async def block_scan(_source_id: str):
        scan_started.set()
        await asyncio.Event().wait()

    async def block_cleanup(job_id: str, *, error: str):
        cleanup_started.set()
        await allow_cleanup.wait()
        return await fail_job(job_id, error=error)

    monkeypatch.setattr(db, "list_legacy_memory_provenance", block_scan)
    monkeypatch.setattr(db, "fail_lifecycle_backfill_job", block_cleanup)
    job_id = "recovery-repeated-cancellation"
    task = asyncio.create_task(
        run_with_lifecycle_activity_heartbeat(
            db,
            job_id,
            lambda: run_source_lifecycle_recovery_job(
                db,
                "src-1",
                job_id=job_id,
            ),
            heartbeat_interval_seconds=3600,
        )
    )
    await scan_started.wait()
    task.cancel()
    await cleanup_started.wait()
    try:
        for _ in range(5):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
    finally:
        allow_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    job = await db.get_lifecycle_backfill_job(job_id)
    assert job is not None
    assert job.status is LifecycleBackfillJobStatus.FAILED
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


@pytest.mark.asyncio
async def test_lifecycle_heartbeat_failure_fails_durable_recovery_job(
    db: Database,
    monkeypatch,
) -> None:
    scan_started = asyncio.Event()

    async def block_scan(_source_id: str):
        scan_started.set()
        await asyncio.Event().wait()

    async def fail_heartbeat(**_kwargs) -> None:
        await scan_started.wait()
        raise RuntimeError("simulated lease loss")

    monkeypatch.setattr(db, "list_legacy_memory_provenance", block_scan)
    monkeypatch.setattr(db, "renew_source_activity", fail_heartbeat)
    job_id = "recovery-heartbeat-failed-durably"

    with pytest.raises(SourceActivityConflict, match="heartbeat stopped"):
        await run_with_lifecycle_activity_heartbeat(
            db,
            job_id,
            lambda: run_source_lifecycle_recovery_job(
                db,
                "src-1",
                job_id=job_id,
            ),
            heartbeat_interval_seconds=0,
        )

    assert scan_started.is_set()
    job = await db.get_lifecycle_backfill_job(job_id)
    assert job is not None
    assert job.status is LifecycleBackfillJobStatus.FAILED
    assert job.error == "lifecycle recovery cancelled"
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


@pytest.mark.asyncio
async def test_lifecycle_activity_heartbeat_prefers_completed_work_when_both_tasks_finish(
    monkeypatch,
) -> None:
    import memforge.memory.cutover as cutover_module

    real_wait = asyncio.wait

    async def wait_for_both(tasks, *, return_when):
        del return_when
        await asyncio.gather(*tasks, return_exceptions=True)
        return set(tasks), set()

    monkeypatch.setattr(cutover_module.asyncio, "wait", wait_for_both)

    class HeartbeatDatabase:
        async def renew_source_activity(self, **_kwargs) -> None:
            raise RuntimeError("lease already released by completed job")

    async def operation() -> str:
        return "completed"

    try:
        result = await run_with_lifecycle_activity_heartbeat(
            HeartbeatDatabase(),
            "job-completed",
            operation,
            heartbeat_interval_seconds=0,
        )
    finally:
        monkeypatch.setattr(cutover_module.asyncio, "wait", real_wait)

    assert result == "completed"


@pytest.mark.asyncio
async def test_finding_upsert_rejects_identity_or_status_change(db: Database) -> None:
    finding = _finding()
    await db.upsert_lifecycle_cutover_finding(finding)

    await db.upsert_lifecycle_cutover_finding(replace(finding, reason=CutoverFindingReason.AMBIGUOUS_OBSERVATION))
    evolved = await db.get_lifecycle_cutover_finding(finding.id)
    assert evolved is not None
    assert evolved.reason is CutoverFindingReason.AMBIGUOUS_OBSERVATION
    assert evolved.status is CutoverFindingStatus.OPEN

    with pytest.raises(ValueError, match="finding identity"):
        await db.upsert_lifecycle_cutover_finding(replace(finding, memory_id="mem-different"))
    with pytest.raises(ValueError, match="open findings"):
        await db.upsert_lifecycle_cutover_finding(
            replace(
                finding,
                id="finding-resolved-insert",
                status=CutoverFindingStatus.RESOLVED,
            )
        )


@pytest.mark.asyncio
async def test_finding_cannot_resolve_before_memory_lineage_is_persisted(db: Database) -> None:
    await db.upsert_lifecycle_cutover_finding(_finding())

    with pytest.raises(ValueError, match="validated support lineage"):
        await db.resolve_lifecycle_cutover_finding(
            "finding-1",
            observation_id="obs-page-1-body",
            source_unit_id="unit-page-1",
        )


async def _attach_legacy_source(db: Database) -> None:
    now = "2026-07-15T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified, version,
               content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "legacy-gate-doc",
            "src-1",
            "https://example.test/legacy-gate-doc",
            "Legacy",
            "ENG",
            now,
            "1",
            "hash",
            now,
        ),
    )
    await db.add_memory_source(
        "mem-legacy",
        "legacy-gate-doc",
        "confluence",
        "Legacy claim",
        source_updated_at=None,
    )


@pytest.mark.asyncio
async def test_gate_requires_validated_support_for_active_source_backed_memory(db: Database) -> None:
    await _attach_legacy_source(db)

    with pytest.raises(ValueError, match="source-backed Memory lacks validated support lineage"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
async def test_gate_rejects_active_support_without_matching_source_provenance(
    db: Database,
) -> None:
    await _attach_legacy_source(db)
    await _persist_exact_support_and_provenance(db)
    await db.db.execute(
        "DELETE FROM memory_sources WHERE memory_id = ? AND source_id = ?",
        ("mem-legacy", "src-1"),
    )
    await db.db.commit()

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 1
    with pytest.raises(ValueError, match="active support lacks source provenance"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
async def test_gate_rejects_source_provenance_without_exact_document_support(
    db: Database,
) -> None:
    await _attach_legacy_source(db)
    await _persist_exact_support_and_provenance(db)
    await _add_unsupported_legacy_source_edge(db, doc_id="wrong-support-doc")

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0
    assert await db.count_active_source_memories_without_support("src-1") == 1
    with pytest.raises(ValueError, match="source-backed Memory lacks validated support lineage"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_statement, corrupt_params",
    (
        ("DELETE FROM evidence_units WHERE id = ?", ("eu-backfill-1",)),
        ("UPDATE evidence_units SET doc_id = NULL WHERE id = ?", ("eu-backfill-1",)),
    ),
    ids=("missing-evidence-unit", "null-document"),
)
async def test_reverse_support_projection_audit_fails_closed_for_corrupt_evidence_chain(
    db: Database,
    corrupt_statement: str,
    corrupt_params: tuple[str, ...],
) -> None:
    await _attach_legacy_source(db)
    await _persist_exact_support_and_provenance(db)
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = OFF")
    await db.db.execute(corrupt_statement, corrupt_params)
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = ON")

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 1


@pytest.mark.asyncio
async def test_gate_ignores_inactive_historical_memory_without_support(db: Database) -> None:
    await _attach_legacy_source(db)
    await db.db.execute("UPDATE memories SET status = 'retired' WHERE id = ?", ("mem-legacy",))
    await db.db.commit()

    gate = await db.enable_lifecycle_gate("src-1")

    assert gate.state is LifecycleGateState.ENABLED


async def _persist_exact_support_and_provenance(db: Database) -> str:
    """Make ``_unit()`` active Support for the fixture Memory and return its id."""

    if not any(source.source_id == "src-1" for source in await db.get_memory_sources("mem-legacy")):
        await _attach_legacy_source(db)
    unit = _unit()
    await record_unit_support(
        db,
        memory_id="mem-legacy",
        unit=unit,
        references=(
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-page-1-body",
                    observation_revision_id="obsrev-page-1-v2",
                )
            ),
        ),
    )
    return unit.id


def _overlapping_source_projection():
    """Project the same provider object through a distinct Configured Source."""

    original = _projection()
    observation = replace(
        original.observations[0],
        id="obs-page-1-body-source-2",
        source_id="src-2",
        source_unit_id="unit-page-1-source-2",
    )
    observation_revision = replace(
        original.observation_revisions[0],
        id="obsrev-page-1-v2-source-2",
        observation_id=observation.id,
    )
    source_unit = replace(
        original.source_units[0],
        id="unit-page-1-source-2",
        source_id="src-2",
    )
    source_unit_revision = replace(
        original.source_unit_revisions[0],
        id="unitrev-page-1-v2-source-2",
        source_unit_id=source_unit.id,
        observation_revision_ids=(observation_revision.id,),
    )
    relation = replace(original.relations[0], from_id=source_unit.id)
    delta = replace(
        original.deltas[0],
        source_unit_id=source_unit.id,
        previous_unit_revision_id="unitrev-page-1-v1-source-2",
        current_unit_revision_id=source_unit_revision.id,
        changed_anchors=(
            replace(
                original.deltas[0].changed_anchors[0],
                observation_id=observation.id,
                observation_revision_id=observation_revision.id,
            ),
        ),
        fragment_mappings=(
            replace(
                original.deltas[0].fragment_mappings[0],
                observation_id=observation.id,
                previous_revision_id="obsrev-page-1-v1-source-2",
                current_revision_id=observation_revision.id,
            ),
        ),
    )
    return replace(
        original,
        run_id="projection-run-source-2",
        source_id="src-2",
        observations=(observation,),
        observation_revisions=(observation_revision,),
        source_units=(source_unit,),
        source_unit_revisions=(source_unit_revision,),
        relations=(relation,),
        deltas=(delta,),
    )


def _overlapping_support_plan(
    *,
    plan_id: str,
    mutation_type: LifecycleMutationType,
    evidence_unit: EvidenceUnit,
    staged_references: tuple[EvidenceReference, ...] = (),
    support_hash: str | None = None,
) -> LifecyclePlan:
    removing = mutation_type is LifecycleMutationType.REMOVE_SUPPORT
    return LifecyclePlan(
        id=plan_id,
        scope=ReconciliationScope(
            id=f"scope-{plan_id}",
            source_id="src-2",
            source_unit_id="unit-page-1-source-2",
            base_unit_revision_id="unitrev-page-1-v1-source-2",
            target_unit_revision_id="unitrev-page-1-v2-source-2",
        ),
        gate_state=(LifecycleGateState.ENABLED if removing else LifecycleGateState.GATED),
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=(("mem-legacy",) if removing else ()),
            incumbent_decisions=(
                (
                    IncumbentDecision(
                        "mem-legacy",
                        IncumbentDisposition.REMOVE_SUPPORT,
                        "overlapping source no longer supports claim",
                    ),
                )
                if removing
                else ()
            ),
            batch_ids=("batch-overlap",),
            completed_batch_ids=("batch-overlap",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=("obsrev-page-1-v2-source-2",),
            support_set_hashes=({"mem-legacy": support_hash} if isinstance(support_hash, str) else {}),
        ),
        mutations=(
            LifecycleMutation(
                mutation_type,
                memory_id="mem-legacy",
                source_id="src-2",
                evidence_unit_ids=(evidence_unit.id,),
                payload={
                    "access_context_hash": "workspace",
                    "document_id": "legacy-gate-doc",
                },
            ),
        ),
        evidence_units=((evidence_unit,) if staged_references else ()),
        evidence_references=staged_references,
    )


async def _attach_overlapping_source_support(
    db: Database,
    *,
    plan_id: str,
) -> EvidenceUnit:
    await _persist_exact_support_and_provenance(db)
    await db.upsert_source(
        id="src-2",
        type="confluence",
        name="Engineering overlap",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await db.record_source_projection(_overlapping_source_projection())
    unit, references = await complete_unit_parts(
        db,
        replace(
            _unit(),
            id=f"eu-{plan_id}",
            source_id="src-2",
            doc_revision_id="unitrev-page-1-v2-source-2",
            source_anchor="obs-page-1-body-source-2",
            source_lineage_id="unit-page-1-source-2",
        ),
        (
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-page-1-body-source-2",
                    observation_revision_id="obsrev-page-1-v2-source-2",
                )
            ),
        ),
    )
    await db.apply_lifecycle_plan(
        _overlapping_support_plan(
            plan_id=plan_id,
            mutation_type=LifecycleMutationType.ATTACH_SUPPORT,
            evidence_unit=unit,
            staged_references=references,
        )
    )
    return unit


async def _attach_same_source_incarnation_support(
    db: Database,
) -> tuple[str, str]:
    """Attach two active Support edges sharing one Memory/source/document.

    Source Unit identity can be reincarnated for a reused document locator, so
    the doc-level ``memory_sources`` row must remain until both Unit-owned
    Support edges are gone.
    """

    first_unit_id = await _persist_exact_support_and_provenance(db)
    original = _projection()
    second_unit = replace(
        original.source_units[0],
        id="unit-page-1-incarnation-2",
        provider_key="page-1-incarnation-2",
    )
    second_observation = replace(
        original.observations[0],
        id="obs-page-1-body-incarnation-2",
        source_unit_id=second_unit.id,
        provider_key="page-1-incarnation-2:body",
    )
    second_observation_revision = replace(
        original.observation_revisions[0],
        id="obsrev-page-1-v2-incarnation-2",
        observation_id=second_observation.id,
    )
    second_unit_revision = replace(
        original.source_unit_revisions[0],
        id="unitrev-page-1-v2-incarnation-2",
        source_unit_id=second_unit.id,
        observation_revision_ids=(second_observation_revision.id,),
    )
    await db.record_source_projection(
        replace(
            original,
            run_id="projection-run-incarnation-2",
            observations=(second_observation,),
            observation_revisions=(second_observation_revision,),
            source_units=(second_unit,),
            source_unit_revisions=(second_unit_revision,),
            relations=(),
            deltas=(),
        )
    )
    second_evidence_unit = replace(
        _unit(),
        id="eu-page-1-incarnation-2",
        doc_revision_id=second_unit_revision.id,
        source_anchor=second_observation.id,
        source_lineage_id=second_unit.id,
    )
    await record_unit_support(
        db,
        memory_id="mem-legacy",
        unit=second_evidence_unit,
        references=(
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=second_observation.id,
                    observation_revision_id=second_observation_revision.id,
                )
            ),
        ),
    )
    await db.enable_lifecycle_gate("src-1")
    return first_unit_id, second_evidence_unit.id


def _remove_support_plan(
    *,
    plan_id: str,
    scope_source_unit_id: str,
    target_unit_revision_id: str,
    observation_revision_id: str,
    evidence_unit_id: str,
    support_hash: str,
) -> LifecyclePlan:
    return LifecyclePlan(
        id=plan_id,
        scope=ReconciliationScope(
            id=f"scope-{plan_id}",
            source_id="src-1",
            source_unit_id=scope_source_unit_id,
            base_unit_revision_id=None,
            target_unit_revision_id=target_unit_revision_id,
        ),
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=("mem-legacy",),
            incumbent_decisions=(
                IncumbentDecision(
                    "mem-legacy",
                    IncumbentDisposition.REMOVE_SUPPORT,
                    "one Source Unit no longer supports the claim",
                ),
            ),
            batch_ids=(f"batch-{plan_id}",),
            completed_batch_ids=(f"batch-{plan_id}",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=(observation_revision_id,),
            support_set_hashes={"mem-legacy": support_hash},
        ),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-legacy",
                source_id="src-1",
                evidence_unit_ids=(evidence_unit_id,),
                payload={"document_id": "legacy-gate-doc"},
            ),
        ),
    )


@pytest.mark.asyncio
async def test_remove_support_preserves_shared_same_source_document_projection(
    db: Database,
) -> None:
    first_unit_id, second_unit_id = await _attach_same_source_incarnation_support(db)
    plan = _remove_support_plan(
        plan_id="plan-remove-first-incarnation",
        scope_source_unit_id="unit-page-1",
        target_unit_revision_id="unitrev-page-1-v2",
        observation_revision_id="obsrev-page-1-v2",
        evidence_unit_id=first_unit_id,
        support_hash=await db.get_memory_support_set_hash("mem-legacy"),
    )

    await db.apply_lifecycle_plan(plan)

    assert await db.get_active_memory_support_unit_ids("mem-legacy") == (second_unit_id,)
    assert [(source.source_id, source.doc_id) for source in await db.get_memory_sources("mem-legacy")] == [
        ("src-1", "legacy-gate-doc")
    ]
    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0


@pytest.mark.asyncio
async def test_overlapping_configured_sources_keep_independent_support_projection(
    db: Database,
) -> None:
    unit = await _attach_overlapping_source_support(
        db,
        plan_id="plan-overlapping-attach",
    )

    rows = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_sources
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-legacy", "legacy-gate-doc"),
    )
    assert [row["source_id"] for row in rows] == ["src-1", "src-2"]
    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0
    assert await db.count_active_supported_memories_without_source_provenance("src-2") == 0

    await db.enable_lifecycle_gate("src-2")
    await db.apply_lifecycle_plan(
        _overlapping_support_plan(
            plan_id="plan-overlapping-remove",
            mutation_type=LifecycleMutationType.REMOVE_SUPPORT,
            evidence_unit=unit,
            support_hash=await db.get_memory_support_set_hash("mem-legacy"),
        )
    )

    rows = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_sources
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-legacy", "legacy-gate-doc"),
    )
    assert [row["source_id"] for row in rows] == ["src-1"]
    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0
    metadata_rows = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_search_metadata_trigram
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-legacy", "legacy-gate-doc"),
    )
    assert [row["source_id"] for row in metadata_rows] == ["src-1"]


@pytest.mark.asyncio
async def test_document_owned_corroboration_preserves_an_overlapping_source_projection(
    db: Database,
) -> None:
    await _attach_overlapping_source_support(
        db,
        plan_id="plan-overlapping-corroborate",
    )

    outcome = await db.corroborate_memory(
        "mem-legacy",
        "legacy-gate-doc",
        "confluence",
        "longer owner source excerpt",
        support_kind="extracted",
        source_updated_at=None,
    )

    assert outcome == "updated"
    rows = await db.db.execute_fetchall(
        """SELECT source_id, excerpt FROM memory_sources
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-legacy", "legacy-gate-doc"),
    )
    assert [(row["source_id"], row["excerpt"]) for row in rows] == [
        ("src-1", "longer owner source excerpt"),
        ("src-2", "Legacy claim"),
    ]


@pytest.mark.asyncio
async def test_remove_memory_source_is_noop_for_missing_memory(db: Database) -> None:
    retired = await db.remove_memory_source(
        "mem-missing",
        "missing-doc",
        source_id="src-1",
    )

    assert retired is False
    async with db.db.execute(
        "SELECT COUNT(*) FROM lifecycle_plans WHERE reconciliation_scope_id = 'direct_support_removal'"
    ) as cursor:
        assert int((await cursor.fetchone())[0]) == 0


@pytest.mark.asyncio
async def test_delete_source_preserves_a_document_projected_by_another_configured_source(
    db: Database,
) -> None:
    overlapping_unit = await _attach_overlapping_source_support(
        db,
        plan_id="plan-overlapping-delete-owner",
    )

    result = await db.delete_source_cascade("src-1")

    assert result.retired_memory_ids == ()
    document = await db.get_document("legacy-gate-doc")
    assert document is not None
    assert document.source == "src-2"
    sources = await db.get_memory_sources("mem-legacy")
    assert [(source.source_id, source.doc_id) for source in sources] == [
        ("src-2", "legacy-gate-doc"),
    ]
    assert await db.get_active_memory_support_unit_ids("mem-legacy") == (overlapping_unit.id,)
    memory = await db.get_memory("mem-legacy")
    assert memory is not None
    assert memory.status == "active"


@pytest.mark.asyncio
async def test_delete_non_owner_source_removes_its_exact_shared_document_projection(
    db: Database,
) -> None:
    await _attach_overlapping_source_support(
        db,
        plan_id="plan-overlapping-delete-non-owner",
    )

    result = await db.delete_source_cascade("src-2")

    assert result.retired_memory_ids == ()
    document = await db.get_document("legacy-gate-doc")
    assert document is not None
    assert document.source == "src-1"
    sources = await db.get_memory_sources("mem-legacy")
    assert [(source.source_id, source.doc_id) for source in sources] == [
        ("src-1", "legacy-gate-doc"),
    ]
    assert await db.get_active_memory_support_unit_ids("mem-legacy") == ("eu-backfill-1",)
    memory = await db.get_memory("mem-legacy")
    assert memory is not None
    assert memory.status == "active"


def _retirement_plan(evidence_unit_id: str, support_hash: str) -> LifecyclePlan:
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
                evidence_unit_ids=(evidence_unit_id,),
            ),
            LifecycleMutation(
                LifecycleMutationType.RETIRE_MEMORY,
                memory_id="mem-legacy",
                source_id="src-1",
            ),
        ),
    )


def _gated_retirement_plan(
    evidence_unit_id: str,
    incumbent: Memory,
    support_hash: str,
    *,
    plan_id: str,
) -> LifecyclePlan:
    return build_lifecycle_plan(
        plan_id=plan_id,
        scope=ReconciliationScope(
            id=f"scope-{plan_id}",
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
        source_support_unit_ids={incumbent.id: (evidence_unit_id,)},
        all_active_support_unit_ids={incumbent.id: (evidence_unit_id,)},
        support_set_hashes={incumbent.id: support_hash},
        observation_revision_ids=("obsrev-page-1-v2",),
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


@pytest.mark.asyncio
async def test_lifecycle_plan_applies_support_removal_and_retirement_atomically(db: Database) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(unit_id, await db.get_memory_support_set_hash("mem-legacy"))
    other = Database(db.db_path)
    await other.connect()

    try:
        await db.apply_lifecycle_plan(plan)
        await db.apply_lifecycle_plan(plan)
        await asyncio.wait_for(
            other.upsert_source(
                id="src-after-plan-retry",
                type="confluence",
                name="Writer lock probe",
                config_json="{}",
                access_policy="workspace",
                owner_user_id="owner-1",
            ),
            timeout=1,
        )
    finally:
        await other.close()

    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "retired"
    assert await db.get_lifecycle_plan_status(plan.id) == "applied"


@pytest.mark.asyncio
async def test_terminal_lifecycle_mutation_stales_all_pending_reviews(
    db: Database,
) -> None:
    challenger = Memory(
        id="mem-review-challenger",
        memory_type="fact",
        content="Independent challenger",
        content_hash=content_hash("Independent challenger"),
    )
    await db.insert_memory(challenger)
    await db.insert_memory_review(
        MemoryReview(
            id="review-terminal-incumbent",
            kind=ReviewKind.SUPERSEDE.value,
            status=ReviewStatus.PENDING.value,
            incumbent_memory_id="mem-legacy",
            challenger_memory_id=challenger.id,
        )
    )
    unit_id = await _persist_exact_support_and_provenance(db)
    incumbent = await db.get_memory("mem-legacy")
    assert incumbent is not None
    support_hash = await db.get_memory_support_set_hash(incumbent.id)
    review_plan = _gated_retirement_plan(
        unit_id,
        incumbent,
        support_hash,
        plan_id="plan-terminal-review",
    )
    await db.apply_lifecycle_plan(review_plan)
    lifecycle_reviews = await db.list_lifecycle_reviews(
        "src-1",
        status=LifecycleReviewStatus.PENDING,
    )
    assert len(lifecycle_reviews) == 1
    await db.enable_lifecycle_gate("src-1")
    plan = replace(
        _retirement_plan(unit_id, support_hash),
        id="plan-terminal-retire",
    )

    await db.apply_lifecycle_plan(plan)

    review = await db.get_memory_review("review-terminal-incumbent")
    assert review is not None
    assert review.status == ReviewStatus.STALE.value
    assert review.resolved_at is not None
    lifecycle_review = await db.get_lifecycle_review(lifecycle_reviews[0].id)
    assert lifecycle_review is not None
    assert lifecycle_review.status is LifecycleReviewStatus.STALE
    assert lifecycle_review.resolved_at is not None


@pytest.mark.asyncio
async def test_gated_review_approval_applies_proposal_and_resolves_review_atomically(
    db: Database,
) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    incumbent = await db.get_memory("mem-legacy")
    assert incumbent is not None
    support_hash = await db.get_memory_support_set_hash(incumbent.id)
    original = _gated_retirement_plan(
        unit_id,
        incumbent,
        support_hash,
        plan_id="plan-gated-delete",
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
    assert {(task.memory_id, task.operation.value) for task in tasks} == {(incumbent.id, "delete")}


@pytest.mark.asyncio
async def test_stale_nonterminal_review_refresh_preserves_other_source_support(
    db: Database,
) -> None:
    source_unit_id = await _persist_exact_support_and_provenance(db)
    other_unit = await _attach_overlapping_source_support(
        db,
        plan_id="plan-review-refresh-other-source",
    )
    await db.enable_lifecycle_gate("src-1")
    incumbent = await db.get_memory("mem-legacy")
    assert incumbent is not None
    support_state = (await db.get_active_memory_support_states((incumbent.id,)))[incumbent.id]
    original = build_lifecycle_plan(
        plan_id="plan-review-refresh-original",
        scope=ReconciliationScope(
            id="scope-review-refresh-original",
            source_id="src-1",
            source_unit_id="unit-page-1",
            base_unit_revision_id="unitrev-page-1-v1",
            target_unit_revision_id="unitrev-page-1-v2",
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=incumbent.id,
                reason="source no longer supports claim",
                flag_for_review=True,
            ),
        ),
        incumbents={incumbent.id: incumbent},
        source_support_unit_ids={incumbent.id: (source_unit_id,)},
        all_active_support_unit_ids={incumbent.id: support_state.unit_ids},
        support_set_hashes={incumbent.id: support_state.support_set_hash},
        observation_revision_ids=("obsrev-page-1-v2",),
        defaults=NewMemoryDefaults(
            visibility="workspace",
            owner_user_id=None,
            project_key=None,
            repo_identifier=None,
            doc_id="legacy-gate-doc",
            source_type="confluence",
            access_context_hash="workspace",
        ),
    )
    assert [mutation.mutation_type for mutation in original.mutations] == [
        LifecycleMutationType.CREATE_REVIEW,
    ]
    await db.apply_lifecycle_plan(original)
    [stale_review] = await db.list_lifecycle_reviews(
        "src-1",
        status=LifecycleReviewStatus.PENDING,
    )

    additional_other_unit = replace(other_unit, id="eu-review-refresh-other-source-additional")
    await record_unit_support(
        db,
        memory_id=incumbent.id,
        unit=additional_other_unit,
        references=(
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-page-1-body-source-2",
                    observation_revision_id="obsrev-page-1-v2-source-2",
                )
            ),
        ),
    )
    await db.resolve_lifecycle_review(stale_review.id, LifecycleReviewStatus.STALE)
    incumbent = await db.get_memory(incumbent.id)
    assert incumbent is not None
    payload = await db.get_lifecycle_plan_payload(original.id)
    assert payload is not None
    refreshed_plan, refreshed_review_id = build_lifecycle_review_refresh_plan(
        await db.get_lifecycle_review(stale_review.id),  # type: ignore[arg-type]
        payload,
        gate_state=LifecycleGateState.ENABLED,
        current_support_set_hash=await db.get_memory_support_set_hash(incumbent.id),
        current_memory_version=lifecycle_memory_version(incumbent),
    )

    await db.apply_lifecycle_plan(refreshed_plan)
    await db.apply_lifecycle_plan(refreshed_plan)
    refreshed_review = await db.get_lifecycle_review(refreshed_review_id)
    assert refreshed_review is not None
    assert refreshed_review.status is LifecycleReviewStatus.PENDING
    assert (await db.get_lifecycle_review(stale_review.id)).status is LifecycleReviewStatus.STALE  # type: ignore[union-attr]

    approval = build_lifecycle_review_approval_plan(refreshed_review, lifecycle_plan_to_payload(refreshed_plan))
    await db.apply_lifecycle_plan(approval)

    assert (await db.get_lifecycle_review(refreshed_review_id)).status is LifecycleReviewStatus.APPROVED  # type: ignore[union-attr]
    remaining = (await db.get_active_memory_support_states((incumbent.id,)))[incumbent.id]
    assert remaining.unit_ids == tuple(sorted((other_unit.id, additional_other_unit.id)))
    assert (await db.get_memory(incumbent.id)).status == "active"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_failed_vector_tasks_rotate_without_starving_new_pending_cleanup(
    db: Database,
) -> None:
    for index in range(3):
        await db.db.execute(
            """INSERT INTO source_deletion_vector_outbox (
                   id, source_id, memory_id, status, attempts, error,
                   created_at, updated_at
               ) VALUES (?, 'src-1', ?, 'failed', 1, 'poison', ?, ?)""",
            (
                f"failed-{index}",
                f"mem-failed-{index}",
                f"2026-07-15T00:00:0{index}Z",
                f"2026-07-15T00:00:0{index}Z",
            ),
        )
    await db.db.execute(
        """INSERT INTO source_deletion_vector_outbox (
               id, source_id, memory_id, status, created_at, updated_at
           ) VALUES ('pending-new', 'src-1', 'mem-pending', 'pending',
                     '2026-07-15T01:00:00Z', '2026-07-15T01:00:00Z')"""
    )
    await db.db.commit()

    first = await db.list_lifecycle_vector_tasks(source_id="src-1", limit=2)
    assert [task.id for task in first] == ["pending-new", "failed-0"]
    await db.complete_lifecycle_vector_task("pending-new")
    await db.fail_lifecycle_vector_task("failed-0", "still poison")

    [next_retry] = await db.list_lifecycle_vector_tasks(source_id="src-1", limit=1)
    assert next_retry.id == "failed-1"


@pytest.mark.asyncio
async def test_stale_lifecycle_plan_rolls_back_without_partial_mutation(db: Database) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    stale = _retirement_plan(unit_id, "not-the-current-support-hash")

    with pytest.raises(ValueError, match="support stale guard"):
        await db.apply_lifecycle_plan(stale)

    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "active"
    assert await db.get_lifecycle_plan_status(stale.id) is None


@pytest.mark.asyncio
async def test_mutation_failure_rolls_back_staged_evidence_with_the_plan(db: Database) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    staged_unit, staged_references = await complete_unit_parts(
        db,
        replace(_unit(), id="eu-staged-rollback"),
        (
            primary_reference(
                SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-page-1-body",
                    observation_revision_id="obsrev-page-1-v2",
                )
            ),
        ),
    )
    plan = _retirement_plan(
        unit_id,
        await db.get_memory_support_set_hash("mem-legacy"),
    )
    plan = replace(
        plan,
        id="plan-evidence-rollback",
        evidence_units=(staged_unit,),
        evidence_references=staged_references,
        mutations=(
            replace(
                plan.mutations[0],
                evidence_unit_ids=("eu-not-active-support",),
            ),
            plan.mutations[1],
        ),
    )

    with pytest.raises(ValueError, match="complete active Evidence Units"):
        await db.apply_lifecycle_plan(plan)

    assert await db.get_evidence_unit(staged_unit.id) is None
    assert await db.get_lifecycle_plan_status(plan.id) is None
    memory = await db.get_memory("mem-legacy")
    assert memory is not None and memory.status == "active"


@pytest.mark.asyncio
async def test_mutation_failure_rolls_back_source_projection_with_the_plan(db: Database) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
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
        unit_id,
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
                evidence_unit_ids=("eu-not-active-support",),
            ),
            plan.mutations[1],
        ),
    )

    with pytest.raises(ValueError, match="complete active Evidence Units"):
        await db.apply_source_projection_lifecycle(projection, plan)

    current = await db.get_current_source_unit_revision("unit-page-1")
    assert current is not None and current.id == "unitrev-page-1-v2"
    assert await db.get_source_projection(projection.run_id) is None
    assert await db.get_lifecycle_plan_status(plan.id) is None


@pytest.mark.asyncio
async def test_stale_source_activity_epoch_rejects_projected_lifecycle_commit(
    db: Database,
) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    lease = await db.acquire_source_activity(
        activity_id="sync-before-fence",
        source_id="src-1",
        kind=SourceActivityKind.SYNC,
    )
    await db.db.execute(
        "UPDATE sources SET activity_epoch = activity_epoch + 1 WHERE id = ?",
        ("src-1",),
    )
    await db.db.commit()
    projection = replace(_projection(), run_id="projection-from-stale-worker")
    plan = replace(
        _retirement_plan(
            unit_id,
            await db.get_memory_support_set_hash("mem-legacy"),
        ),
        id="plan-from-stale-worker",
    )

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.apply_source_projection_lifecycle(
            projection,
            plan,
            expected_source_activity_epoch=lease.epoch,
        )

    assert await db.get_source_projection(projection.run_id) is None
    assert await db.get_lifecycle_plan_status(plan.id) is None


@pytest.mark.asyncio
async def test_memory_version_stale_guard_rejects_concurrent_incumbent_change(
    db: Database,
) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(
        unit_id,
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
@pytest.mark.parametrize(
    ("support_revision_ids", "current_revision_id", "expected_mapped", "expected_findings"),
    (
        (("obsrev-supported",), "obsrev-supported", 1, 0),
        (("obsrev-supported",), "obsrev-newer", 0, 1),
        (("obsrev-old", "obsrev-supported"), "obsrev-supported", 1, 0),
    ),
)
async def test_backfill_trusts_existing_active_support_only_at_current_revision(
    support_revision_ids: tuple[str, ...],
    current_revision_id: str,
    expected_mapped: int,
    expected_findings: int,
) -> None:
    finding = LifecycleCutoverFinding(
        id="finding-supported",
        source_id="src-1",
        memory_id="mem-supported",
        reason=CutoverFindingReason.AMBIGUOUS_OBSERVATION,
        status=CutoverFindingStatus.OPEN,
        available_provenance={"documents": [{"doc_id": "legacy-doc"}]},
        mapping_attempt={"strategy": "legacy"},
    )
    supports = tuple(
        ActiveSupportEvidence(
            memory_id="mem-supported",
            source_id="src-1",
            reference_id=f"ref-{revision_id}",
            evidence_unit_id=f"unit-evidence-{revision_id}",
            role=EvidenceRole.PRIMARY,
            anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION,
                observation_id="obs-supported",
                observation_revision_id=revision_id,
            ),
            excerpt="Exact supported claim",
        )
        for revision_id in support_revision_ids
    )

    class SupportedDb:
        def __init__(self) -> None:
            self.resolved: list[tuple[str, str, str]] = []
            self.enabled: list[str] = []
            self.gated: list[str] = []
            self.upserted: list[LifecycleCutoverFinding] = []
            self.finding_id: str | None = None

        async def list_legacy_memory_provenance(self, source_id: str):
            assert source_id == "src-1"
            return [
                SimpleNamespace(
                    memory_id="mem-supported",
                    doc_id="legacy-doc",
                    source_type="jira",
                    excerpt="ambiguous legacy excerpt",
                )
            ]

        async def get_lifecycle_cutover_finding(self, finding_id: str):
            assert finding_id.startswith("finding-")
            self.finding_id = finding_id
            return replace(finding, id=finding_id)

        async def get_active_memory_support_evidence(self, memory_id: str, *, source_id: str):
            assert (memory_id, source_id) == ("mem-supported", "src-1")
            return supports

        async def get_evidence_unit(self, evidence_unit_id: str):
            assert evidence_unit_id in {support.evidence_unit_id for support in supports}
            return replace(
                _unit(),
                id=evidence_unit_id,
                source_lineage_id="unit-supported",
            )

        async def get_current_source_observation_revisions(self, source_unit_id: str):
            assert source_unit_id == "unit-supported"
            return {"obs-supported": SimpleNamespace(id=current_revision_id)}

        async def resolve_lifecycle_cutover_finding(
            self,
            finding_id: str,
            *,
            observation_id: str,
            source_unit_id: str,
            source_activity=None,
        ):
            assert source_activity is None
            self.resolved.append((finding_id, observation_id, source_unit_id))
            return finding

        async def enable_lifecycle_gate(self, source_id: str, *, source_activity=None) -> None:
            assert source_activity is None
            self.enabled.append(source_id)

        async def find_source_unit_by_document_id(self, *_args):
            if current_revision_id == "obsrev-supported":
                raise AssertionError("current supported Memory must not use legacy provenance")
            return None

        async def upsert_lifecycle_cutover_finding(
            self,
            cutover_finding: LifecycleCutoverFinding,
            *,
            source_activity=None,
        ) -> None:
            assert source_activity is None
            self.upserted.append(cutover_finding)

        async def gate_destructive_lifecycle(
            self,
            source_id: str,
            *,
            reason: str,
            source_activity=None,
        ) -> None:
            assert source_activity is None
            assert reason == "1 open lifecycle cutover finding(s)"
            self.gated.append(source_id)

    database = SupportedDb()

    result = await run_source_lifecycle_backfill(database, "src-1")

    assert result.scanned_memories == 1
    assert result.mapped_memories == expected_mapped
    assert result.finding_count == expected_findings
    assert result.gate_enabled is (expected_findings == 0)
    if expected_findings == 0:
        assert database.resolved == [(database.finding_id, "obs-supported", "unit-supported")]
        assert database.enabled == ["src-1"]
        assert database.upserted == []
        assert database.gated == []
    else:
        assert database.resolved == []
        assert database.enabled == []
        assert len(database.upserted) == 1
        assert database.gated == ["src-1"]


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
    epoch_after_completion = await db.get_source_activity_epoch("src-1")
    retried = await run_source_lifecycle_backfill_job(
        db,
        "src-1",
        job_id="backfill-job-1",
    )

    assert completed.status is LifecycleBackfillJobStatus.COMPLETED
    assert retried == completed
    assert await db.list_lifecycle_backfill_jobs("src-1") == [completed]
    assert await db.get_source_activity_epoch("src-1") == epoch_after_completion
    async with db.db.execute(
        "SELECT COUNT(*) AS count FROM source_activity_leases WHERE source_id = ?",
        ("src-1",),
    ) as cursor:
        assert int((await cursor.fetchone())["count"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [run_source_lifecycle_backfill_job, run_source_lifecycle_recovery_job],
)
async def test_completed_job_retry_gates_a_new_support_invariant_violation(
    db: Database,
    runner,
) -> None:
    job_id = f"completed-before-gap-{runner.__name__}"
    completed = await runner(db, "src-1", job_id=job_id)
    assert completed.status is LifecycleBackfillJobStatus.COMPLETED
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED
    await _add_unsupported_legacy_source_edge(db)

    retried = await runner(db, "src-1", job_id=job_id)

    assert retried == completed
    gate = await db.get_lifecycle_gate("src-1")
    assert gate.state is LifecycleGateState.GATED
    assert gate.reason and "support invariant violation" in gate.reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner",
    [run_source_lifecycle_backfill_job, run_source_lifecycle_recovery_job],
)
@pytest.mark.parametrize(
    "preflight_method",
    [
        "count_active_source_memories_without_support",
        "count_active_supported_memories_without_source_provenance",
    ],
)
async def test_support_preflight_failure_does_not_create_job_or_activity(
    db: Database,
    monkeypatch,
    runner,
    preflight_method: str,
) -> None:
    async def fail_preflight(_source_id: str) -> int:
        raise RuntimeError("support invariant unavailable")

    monkeypatch.setattr(
        db,
        preflight_method,
        fail_preflight,
    )
    job_id = f"preflight-failed-{runner.__name__}-{preflight_method}"

    with pytest.raises(RuntimeError, match="support invariant unavailable"):
        await runner(db, "src-1", job_id=job_id)

    assert await db.get_lifecycle_backfill_job(job_id) is None
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


@pytest.mark.asyncio
async def test_failed_backfill_job_retry_does_not_reacquire_activity(db: Database) -> None:
    failed = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="backfill-job-terminal-failed",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(failed.id)
    failed = await db.fail_lifecycle_backfill_job(failed.id, error="operator blocker")
    epoch_after_failure = await db.get_source_activity_epoch("src-1")

    retried = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id=failed.id,
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )

    assert retried == failed
    assert await db.get_source_activity_epoch("src-1") == epoch_after_failure
    async with db.db.execute(
        "SELECT COUNT(*) AS count FROM source_activity_leases WHERE source_id = ?",
        ("src-1",),
    ) as cursor:
        assert int((await cursor.fetchone())["count"]) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_operation", ["complete", "fail"])
async def test_sqlite_lifecycle_terminal_transition_requires_current_activity_lease(
    db: Database,
    terminal_operation: str,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id=f"sqlite-terminal-with-lost-lease-{terminal_operation}",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "DELETE FROM source_activity_leases WHERE id = ?",
        (job.id,),
    )
    await db.db.commit()

    with pytest.raises(SourceActivityConflict, match="lease is not current"):
        if terminal_operation == "complete":
            await db.complete_lifecycle_backfill_job(
                job.id,
                scanned_memories=3,
                mapped_memories=2,
                finding_count=1,
            )
        else:
            await db.fail_lifecycle_backfill_job(
                job.id,
                error="operator blocker",
            )

    stored = await db.get_lifecycle_backfill_job(job.id)
    assert stored is not None
    assert stored.status is LifecycleBackfillJobStatus.RUNNING


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_fails_expired_job_atomically(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-expired-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    fenced_epoch = await db.get_source_activity_epoch("src-1")

    recovered = await db.recover_stale_lifecycle_backfill_job(
        job.id,
        error="operator recovered expired lifecycle job",
    )

    assert recovered.status is LifecycleBackfillJobStatus.FAILED
    assert recovered.error == "operator recovered expired lifecycle job"
    assert await db.get_source_activity_epoch("src-1") == fenced_epoch + 1
    assert await db.get_active_lifecycle_backfill_job("src-1") is None
    assert not await db.release_source_activity(
        activity_id=job.id,
        capability=job.id,
    )


@pytest.mark.asyncio
async def test_backfill_rejects_recovered_maintenance_authority(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-backfill-stale-authority",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")
    gate_before = await db.get_lifecycle_gate("src-1")

    with pytest.raises(SourceActivityConflict, match="lease is not current"):
        await run_source_lifecycle_backfill(
            db,
            "src-1",
            lifecycle_job_id=job.id,
        )

    assert await db.get_lifecycle_gate("src-1") == gate_before


@pytest.mark.asyncio
async def test_lifecycle_gate_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-gate-stale-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    activity = await db.renew_source_activity(
        activity_id=job.id,
        capability=job.id,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")
    gate_before = await db.get_lifecycle_gate("src-1")

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.enable_lifecycle_gate("src-1", source_activity=activity)

    assert await db.get_lifecycle_gate("src-1") == gate_before


@pytest.mark.asyncio
async def test_evidence_write_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-evidence-stale-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    activity = await db.renew_source_activity(
        activity_id=job.id,
        capability=job.id,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.upsert_evidence_unit(_unit(), source_activity=activity)

    assert await db.get_evidence_unit(_unit().id) is None


@pytest.mark.asyncio
async def test_document_write_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-document-stale-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    activity = await db.renew_source_activity(
        activity_id=job.id,
        capability=job.id,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")
    now = datetime(2026, 7, 19, tzinfo=timezone.utc)
    document = DocumentRecord(
        doc_id="doc-stale-fence",
        source="src-1",
        source_url=None,
        title="Stale fenced document",
        space_or_project=None,
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash="stale-fence-hash",
        token_count=3,
        raw_content_uri=None,
        raw_content_type="text/plain",
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.upsert_document(
            document,
            require_configured_source=True,
            source_activity=activity,
        )

    assert await db.get_document(document.doc_id) is None


@pytest.mark.asyncio
async def test_evidence_reference_write_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-reference-stale-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    activity = await db.renew_source_activity(
        activity_id=job.id,
        capability=job.id,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")
    reference = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-page-1-body",
            observation_revision_id="obsrev-page-1-v2",
        ),
        evidence_unit_id=_unit().id,
    )

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.record_evidence_references(
            _unit().id,
            (reference,),
            source_activity=activity,
        )


@pytest.mark.asyncio
async def test_finding_write_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-finding-stale-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    activity = await db.renew_source_activity(
        activity_id=job.id,
        capability=job.id,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()
    await db.recover_stale_lifecycle_backfill_job(job.id, error="expired")

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.upsert_lifecycle_cutover_finding(
            _finding(),
            source_activity=activity,
        )

    assert await db.get_lifecycle_cutover_finding(_finding().id) is None


@pytest.mark.asyncio
async def test_list_stale_lifecycle_jobs_excludes_current_lease(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-current-maintenance",
        type="confluence",
        name="Current maintenance",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    stale = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-list-stale-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    current = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-list-current-lifecycle-job",
            source_id="src-current-maintenance",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(stale.id)
    await db.start_lifecycle_backfill_job(current.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", stale.id),
    )
    await db.db.commit()

    stale_ids = await db.list_stale_lifecycle_backfill_job_ids(limit=10)

    assert stale_ids == (stale.id,)
    await db.fail_lifecycle_backfill_job(current.id, error="test cleanup")


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_jobs_fails_orphaned_jobs(
    db: Database,
) -> None:
    stale = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-sweep-stale-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(stale.id)
    await db.db.execute(
        "DELETE FROM source_activity_leases WHERE id = ?",
        (stale.id,),
    )
    await db.db.commit()

    recovered = await recover_stale_lifecycle_jobs(db)

    assert tuple(job.id for job in recovered) == (stale.id,)
    assert recovered[0].status is LifecycleBackfillJobStatus.FAILED
    assert recovered[0].error == "source lifecycle maintenance lease expired before completion"


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_jobs_isolates_one_invalid_job(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-second-stale-maintenance",
        type="confluence",
        name="Second stale maintenance",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    invalid = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-sweep-invalid-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    recoverable = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-sweep-recoverable-lifecycle-job",
            source_id="src-second-stale-maintenance",
            status=LifecycleBackfillJobStatus.QUEUED,
            created_at="2026-01-02T00:00:00+00:00",
        )
    )
    await db.start_lifecycle_backfill_job(invalid.id)
    await db.start_lifecycle_backfill_job(recoverable.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET capability = ?, lease_until = ? WHERE id = ?",
        ("invalid-capability", "2000-01-01T00:00:00+00:00", invalid.id),
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", recoverable.id),
    )
    await db.db.commit()

    recovered = await recover_stale_lifecycle_jobs(db)

    assert tuple(job.id for job in recovered) == (recoverable.id,)
    assert (await db.get_lifecycle_backfill_job(invalid.id)).status is (LifecycleBackfillJobStatus.RUNNING)
    assert (await db.get_lifecycle_backfill_job(recoverable.id)).status is (LifecycleBackfillJobStatus.FAILED)


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_refuses_current_lease(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-refuse-current-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)

    with pytest.raises(SourceActivityConflict, match="lease is still current"):
        await db.recover_stale_lifecycle_backfill_job(
            job.id,
            error="must not replace a live lifecycle owner",
        )

    stored = await db.get_lifecycle_backfill_job(job.id)
    assert stored is not None
    assert stored.status is LifecycleBackfillJobStatus.RUNNING
    await db.fail_lifecycle_backfill_job(job.id, error="test cleanup")


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_accepts_missing_lease(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-orphaned-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "DELETE FROM source_activity_leases WHERE id = ?",
        (job.id,),
    )
    await db.db.commit()

    recovered = await db.recover_stale_lifecycle_backfill_job(
        job.id,
        error="operator recovered orphaned lifecycle job",
    )

    assert recovered.status is LifecycleBackfillJobStatus.FAILED
    assert recovered.error == "operator recovered orphaned lifecycle job"
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_serializes_job_reacquisition(
    db: Database,
    monkeypatch,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-serialized-lifecycle-job",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute("DELETE FROM source_activity_leases WHERE id = ?", (job.id,))
    await db.db.commit()
    other = Database(db.db_path)
    await other.connect()
    source_locked = asyncio.Event()
    release_recovery = asyncio.Event()
    original_execute = db.db._execute

    async def pause_after_source_lock(function, *args, **kwargs):
        cursor = await original_execute(function, *args, **kwargs)
        sql = args[0] if args and isinstance(args[0], str) else ""
        if sql.startswith("UPDATE sources SET status = status"):
            source_locked.set()
            await release_recovery.wait()
        return cursor

    monkeypatch.setattr(db.db, "_execute", pause_after_source_lock)
    recovery = asyncio.create_task(
        db.recover_stale_lifecycle_backfill_job(
            job.id,
            error="operator recovered orphaned lifecycle job",
        )
    )
    await asyncio.wait_for(source_locked.wait(), timeout=1)
    reacquisition = asyncio.create_task(other.create_lifecycle_backfill_job(job))
    await asyncio.sleep(0.05)
    assert not reacquisition.done()

    release_recovery.set()
    recovered, retried = await asyncio.gather(recovery, reacquisition)

    assert recovered.status is LifecycleBackfillJobStatus.FAILED
    assert retried.status is LifecycleBackfillJobStatus.FAILED
    assert await other.get_active_lifecycle_backfill_job("src-1") is None
    assert not await other.release_source_activity(activity_id=job.id, capability=job.id)
    await other.close()


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_rejects_lease_identity_mismatch(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-identity-mismatch",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET capability = ?, lease_until = ? WHERE id = ?",
        ("different-capability", "2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.commit()

    with pytest.raises(SourceActivityConflict, match="identity mismatch"):
        await db.recover_stale_lifecycle_backfill_job(job.id, error="must roll back")

    stored = await db.get_lifecycle_backfill_job(job.id)
    assert stored is not None
    assert stored.status is LifecycleBackfillJobStatus.RUNNING


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_is_idempotent_after_failure(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-idempotent-failure",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute("DELETE FROM source_activity_leases WHERE id = ?", (job.id,))
    await db.db.commit()
    first = await db.recover_stale_lifecycle_backfill_job(job.id, error="first recovery")

    second = await db.recover_stale_lifecycle_backfill_job(job.id, error="second recovery")

    assert second == first
    assert second.error == "first recovery"


@pytest.mark.asyncio
async def test_recover_stale_lifecycle_backfill_job_rolls_back_partial_failure(
    db: Database,
) -> None:
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-recover-rollback",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await db.start_lifecycle_backfill_job(job.id)
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", job.id),
    )
    await db.db.execute(
        """CREATE TRIGGER abort_stale_recovery
           BEFORE UPDATE OF status ON lifecycle_backfill_jobs
           WHEN NEW.id = 'sqlite-recover-rollback' AND NEW.status = 'failed'
           BEGIN SELECT RAISE(ABORT, 'forced recovery failure'); END"""
    )
    await db.db.commit()
    fenced_epoch = await db.get_source_activity_epoch("src-1")

    with pytest.raises(sqlite3.IntegrityError, match="forced recovery failure"):
        await db.recover_stale_lifecycle_backfill_job(job.id, error="must roll back")

    stored = await db.get_lifecycle_backfill_job(job.id)
    assert stored is not None
    assert stored.status is LifecycleBackfillJobStatus.RUNNING
    assert await db.get_source_activity_epoch("src-1") == fenced_epoch
    async with db.db.execute(
        "SELECT COUNT(*) AS count FROM source_activity_leases WHERE id = ?",
        (job.id,),
    ) as cursor:
        assert int((await cursor.fetchone())["count"]) == 1


@pytest.mark.asyncio
async def test_recovery_gates_enabled_source_before_scanning_missing_support(
    db: Database,
    monkeypatch,
) -> None:
    await _add_unsupported_legacy_source_edge(db)
    now = "2026-07-16T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO source_lifecycle_gates (
               source_id, state, reason, audited_at, enabled_at, updated_at
           ) VALUES (?, 'enabled', NULL, ?, ?, ?)""",
        ("src-1", now, now, now),
    )
    await db.db.commit()

    async def fail_scan(_source_id: str):
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(db, "list_legacy_memory_provenance", fail_scan)

    with pytest.raises(RuntimeError, match="simulated audit failure"):
        await run_source_lifecycle_recovery_job(
            db,
            "src-1",
            job_id="recovery-gates-before-scan",
        )

    gate = await db.get_lifecycle_gate("src-1")
    assert gate.state is LifecycleGateState.GATED
    assert gate.reason and "support invariant violation" in gate.reason


@pytest.mark.asyncio
async def test_source_allows_only_one_active_lifecycle_job(db: Database) -> None:
    first = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="source-rebaseline-first",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    assert await db.get_active_lifecycle_backfill_job("src-1") == first

    with pytest.raises(ValueError, match="source lifecycle job already active"):
        await db.create_lifecycle_backfill_job(
            LifecycleBackfillJob(
                id="source-rebaseline-second",
                source_id="src-1",
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )

    assert await db.create_lifecycle_backfill_job(first) == first

    await db.start_lifecycle_backfill_job(first.id)
    assert (await db.get_active_lifecycle_backfill_job("src-1")).status is LifecycleBackfillJobStatus.RUNNING
    await db.fail_lifecycle_backfill_job(first.id, error="test terminal state")
    assert await db.get_active_lifecycle_backfill_job("src-1") is None


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
