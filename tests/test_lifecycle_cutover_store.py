from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
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
    MemorySupportAssertion,
    evidence_reference_id_for,
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
)
from memforge.memory.lifecycle_planner import NewMemoryDefaults, build_lifecycle_plan
from memforge.memory.lifecycle_review import build_lifecycle_review_approval_plan
from memforge.genes.local_markdown_gene import LocalMarkdownGene
from memforge.memory.cutover import (
    list_agent_session_lifecycle_migration_candidates,
    recover_stale_lifecycle_jobs,
    reconstruct_historical_source_projection,
    repair_lifecycle_cutover_finding,
    run_source_lifecycle_backfill,
    run_source_lifecycle_backfill_job,
    run_source_lifecycle_recovery_job,
    run_with_lifecycle_activity_heartbeat,
)
from memforge.models import (
    ContentItem,
    DocumentRecord,
    Memory,
    MemoryReview,
    NormalizedContent,
    RawContent,
    ReconcileAction,
    ReconcileOperation,
    ReviewKind,
    ReviewStatus,
    content_hash,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_activity import SourceActivityConflict, SourceActivityKind
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database, MIGRATIONS
from memforge.storage.document_store import LocalDocumentStore
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
async def test_agent_session_migration_inventory_includes_hidden_gate_and_enabled_lineage_gap(
    db: Database,
    monkeypatch,
) -> None:
    for source_id in ("src-agent-hidden", "src-agent-enabled-gap", "src-agent-healthy"):
        await db.upsert_source(
            id=source_id,
            type="agent_session",
            name="Agent Session",
            config_json="{}",
            access_policy="private",
            owner_user_id=f"owner-{source_id}",
        )
        memory_id = f"mem-{source_id}"
        doc_id = f"doc-{source_id}"
        await db.insert_memory(
            Memory(
                id=memory_id,
                memory_type="fact",
                content=f"claim for {source_id}",
                content_hash=content_hash(f"claim for {source_id}"),
                visibility="private",
                owner_user_id=f"owner-{source_id}",
            )
        )
        await db.db.execute(
            """INSERT INTO documents (
                   doc_id, source, source_url, title, space_or_project, last_modified,
                   version, content_hash, last_synced
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                source_id,
                f"agent-knowledge://{source_id}/{doc_id}",
                "Agent concept",
                "agent-session",
                "2026-07-16T00:00:00+00:00",
                "1",
                "hash",
                "2026-07-16T00:00:00+00:00",
            ),
        )
        await db.add_memory_source(
            memory_id,
            doc_id,
            "agent_session",
            f"claim for {source_id}",
            source_updated_at=None,
        )

    # Simulate a legacy writer that created unsupported provenance after a gate
    # had already been enabled. The public gate API correctly rejects this state.
    await db.db.execute(
        """INSERT INTO source_lifecycle_gates (
               source_id, state, reason, audited_at, enabled_at, updated_at
           ) VALUES (?, 'enabled', NULL, ?, ?, ?)""",
        (
            "src-agent-enabled-gap",
            "2026-07-16T00:00:00+00:00",
            "2026-07-16T00:00:00+00:00",
            "2026-07-16T00:00:00+00:00",
        ),
    )
    await db.db.commit()

    healthy_projection = _projection()
    healthy_observation = replace(
        healthy_projection.observations[0],
        id="obs-agent-healthy",
        source_id="src-agent-healthy",
        source_unit_id="unit-agent-healthy",
        observation_type="agent_concept",
        provider_key="doc-src-agent-healthy:concept",
    )
    healthy_observation_revision = replace(
        healthy_projection.observation_revisions[0],
        id="obsrev-agent-healthy",
        observation_id=healthy_observation.id,
    )
    healthy_source_unit = replace(
        healthy_projection.source_units[0],
        id="unit-agent-healthy",
        source_id="src-agent-healthy",
        unit_type="agent_concept",
        provider_key="doc-src-agent-healthy",
        locator={"document_id": "doc-src-agent-healthy"},
    )
    healthy_source_unit_revision = replace(
        healthy_projection.source_unit_revisions[0],
        id="unitrev-agent-healthy",
        source_unit_id=healthy_source_unit.id,
        observation_revision_ids=(healthy_observation_revision.id,),
    )
    await db.record_source_projection(
        replace(
            healthy_projection,
            run_id="projection-agent-healthy",
            source_id="src-agent-healthy",
            source_type="agent_session",
            observations=(healthy_observation,),
            observation_revisions=(healthy_observation_revision,),
            source_units=(healthy_source_unit,),
            source_unit_revisions=(healthy_source_unit_revision,),
            relations=(),
            deltas=(),
        )
    )
    healthy_unit = replace(
        _unit(),
        id="eu-agent-healthy",
        source_id="src-agent-healthy",
        doc_id="doc-src-agent-healthy",
        doc_revision_id=healthy_observation_revision.id,
        source_type="agent_session",
        source_lineage_id=healthy_source_unit.id,
    )
    await db.upsert_evidence_unit(healthy_unit)
    healthy_reference = EvidenceReference(
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-agent-healthy",
            observation_revision_id=healthy_observation_revision.id,
        ),
        evidence_unit_id=healthy_unit.id,
    )
    healthy_reference = replace(
        healthy_reference,
        id=evidence_reference_id_for(healthy_unit.id, healthy_reference),
    )
    await db.record_evidence_references(healthy_unit.id, (healthy_reference,))
    await db.upsert_memory_support_assertion(
        MemorySupportAssertion(
            id="support-agent-healthy",
            memory_id="mem-src-agent-healthy",
            evidence_reference_id=healthy_reference.id or "",
            source_id="src-agent-healthy",
            access_context_hash="private:owner-src-agent-healthy",
        )
    )
    await db.enable_lifecycle_gate("src-agent-healthy")

    async def forbid_content_scan(_source_id: str):
        raise AssertionError("candidate inventory must not load Memory content or excerpts")

    monkeypatch.setattr(db, "list_legacy_memory_provenance", forbid_content_scan)

    candidates = await list_agent_session_lifecycle_migration_candidates(db)

    assert [candidate.source_id for candidate in candidates] == [
        "src-agent-enabled-gap",
        "src-agent-hidden",
    ]
    assert [candidate.active_memory_count for candidate in candidates] == [1, 1]
    assert [candidate.missing_support_count for candidate in candidates] == [1, 1]
    assert candidates[0].gate_state is LifecycleGateState.ENABLED
    assert candidates[1].gate_state is LifecycleGateState.GATED


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
async def test_resolved_finding_retry_preserves_history_and_enabled_gate(
    db: Database,
) -> None:
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
    await db.upsert_lifecycle_cutover_finding(finding)

    assert resolved.status is CutoverFindingStatus.RESOLVED
    assert resolved.created_at == (await db.get_lifecycle_cutover_finding(finding.id)).created_at
    assert gate.state is LifecycleGateState.ENABLED
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED
    assert (await db.get_lifecycle_cutover_finding(finding.id)).status is CutoverFindingStatus.RESOLVED


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
async def test_gate_ignores_inactive_historical_memory_without_support(db: Database) -> None:
    await _attach_legacy_source(db)
    await db.db.execute("UPDATE memories SET status = 'retired' WHERE id = ?", ("mem-legacy",))
    await db.db.commit()

    gate = await db.enable_lifecycle_gate("src-1")

    assert gate.state is LifecycleGateState.ENABLED


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
async def test_terminal_lifecycle_mutation_stales_pending_memory_reviews(
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
            kind=ReviewKind.CROSS_SOURCE_CONFLICT.value,
            status=ReviewStatus.PENDING.value,
            incumbent_memory_id="mem-legacy",
            challenger_memory_id=challenger.id,
        )
    )
    reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(reference, await db.get_memory_support_set_hash("mem-legacy"))

    await db.apply_lifecycle_plan(plan)

    review = await db.get_memory_review("review-terminal-incumbent")
    assert review is not None
    assert review.status == ReviewStatus.STALE.value
    assert review.resolved_at is not None


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
    assert {(task.memory_id, task.operation.value) for task in tasks} == {(incumbent.id, "delete")}


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
async def test_stale_source_activity_epoch_rejects_projected_lifecycle_commit(
    db: Database,
) -> None:
    active_reference = await _persist_support_lineage(db)
    await db.enable_lifecycle_gate("src-1")
    lease = await db.acquire_source_activity(
        activity_id="sync-before-rebaseline",
        source_id="src-1",
        kind=SourceActivityKind.SYNC,
    )
    await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="rebaseline-fence",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    projection = replace(_projection(), run_id="projection-from-stale-worker")
    plan = replace(
        _retirement_plan(
            active_reference,
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
@pytest.mark.parametrize(
    ("current_revision_id", "expected_mapped", "expected_findings"),
    (("obsrev-supported", 1, 0), ("obsrev-newer", 0, 1)),
)
async def test_backfill_trusts_existing_active_support_only_at_current_revision(
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
    support = ActiveSupportEvidence(
        memory_id="mem-supported",
        source_id="src-1",
        reference_id="ref-supported",
        evidence_unit_id="unit-evidence-supported",
        role=EvidenceRole.PRIMARY,
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id="obs-supported",
            observation_revision_id="obsrev-supported",
        ),
        excerpt="Exact supported claim",
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
            return (support,)

        async def get_evidence_unit(self, evidence_unit_id: str):
            assert evidence_unit_id == support.evidence_unit_id
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
async def test_local_markdown_synthetic_canary_projects_and_closes_cutover(
    db: Database,
    tmp_path,
) -> None:
    source_id = "src-local-markdown-canary"
    document_id = "local-markdown-vault-a-design-md"
    memory_id = "mem-local-markdown-canary"
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Local Markdown Canary",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await db.insert_memory(
        Memory(
            id=memory_id,
            memory_type="decision",
            content="Keep A7.",
            content_hash=content_hash("Keep A7."),
        )
    )

    packages_dir = tmp_path / "local-markdown-canary"
    packages_dir.mkdir()
    package_path = packages_dir / "design.json"
    package_path.write_text(
        json.dumps(
            {
                "package_kind": "local_markdown_document",
                "doc_id": document_id,
                "title": "Design",
                "source_url": "file:///vault-a/design.md",
                "last_modified": now.isoformat(),
                "content_type": "text/markdown",
                "space_or_project": "vault-a",
                "version": "1",
                "author": "Ada",
                "vault_id": "vault-a",
                "relative_path": "design.md",
                "file_lineage_id": "file-77",
                "markdown": "# Design\n\nKeep A7.",
            }
        ),
        encoding="utf-8",
    )
    gene = LocalMarkdownGene({"documents_dir": str(packages_dir)}, source_id)
    await gene.authenticate()
    items = [item async for item in gene.discover()]
    assert len(items) == 1
    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)
    projection = project_source_item(
        source_id=source_id,
        source_type="local_markdown",
        run_id="local-markdown-canary-run",
        item=items[0],
        raw=raw,
        normalized=normalized,
    )
    await db.record_source_projection(projection)
    await db.upsert_document(
        DocumentRecord(
            doc_id=document_id,
            source=source_id,
            source_url=items[0].source_url,
            title=items[0].title,
            space_or_project=items[0].space_or_project,
            author=items[0].author,
            last_modified=now,
            labels=items[0].labels,
            version=items[0].version,
            content_hash=content_hash(normalized.markdown_body),
            token_count=4,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        memory_id,
        document_id,
        "local_markdown",
        "Keep A7.",
        source_updated_at=now,
    )

    result = await run_source_lifecycle_backfill(db, source_id)

    assert projection.source_units[0].unit_type == "local_file"
    assert projection.source_units[0].provider_key == "vault-a:file-77"
    assert projection.observations[0].observation_type == "file_content"
    assert result.scanned_memories == 1
    assert result.mapped_memories == 1
    assert result.finding_count == 0
    assert result.gate_enabled is True
    assert (await db.get_lifecycle_gate(source_id)).state is LifecycleGateState.ENABLED


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
        repair_projections=reextract,
    )

    assert requested == [frozenset({"legacy-doc"})]
    assert completed.status is LifecycleBackfillJobStatus.COMPLETED
    assert completed.finding_count == 0
    findings = await db.list_lifecycle_cutover_findings("src-1")
    assert len(findings) == 1
    assert findings[0].status is CutoverFindingStatus.RESOLVED
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_cutover_reconstructs_historical_projection_from_exact_stored_artifacts(
    db: Database,
    tmp_path,
) -> None:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    raw_payload = {
        "conversation_type": "group_chat",
        "messages": [
            {
                "id": "message-1",
                "from": "Ada",
                "content": "Legacy claim",
                "time": now.isoformat(),
                "attachments": [],
            }
        ],
    }
    store = LocalDocumentStore(str(tmp_path / "artifacts"))
    raw_uri = store.store_raw(
        "src-1",
        "Historical Teams block",
        json.dumps(raw_payload).encode(),
        "application/json",
    )
    normalized_uri = store.store_normalized(
        "src-1",
        "Historical Teams block",
        "# Historical Teams block\n\nLegacy claim",
    )
    await db.upsert_document(
        DocumentRecord(
            doc_id="teams-historical-window",
            source="src-1",
            source_url="https://teams.microsoft.com/l/message/conversation-1/message-1",
            title="Historical Teams block",
            space_or_project="PCC",
            author="Ada",
            last_modified=now,
            labels=["group_chat"],
            version="v1",
            content_hash="historical-hash",
            token_count=3,
            raw_content_uri=raw_uri,
            raw_content_type="application/json",
            normalized_content_uri=normalized_uri,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        "mem-legacy",
        "teams-historical-window",
        "teams",
        "Legacy claim",
        source_updated_at=None,
    )
    assert (await run_source_lifecycle_backfill(db, "src-1")).finding_count == 1

    projections = []

    async def reconstruct(document_ids: frozenset[str]) -> None:
        for document_id in document_ids:
            projections.append(
                await reconstruct_historical_source_projection(
                    db,
                    store,
                    source_id="src-1",
                    source_type="teams",
                    document_id=document_id,
                )
            )

    async def unexpected_reextract(document_ids: frozenset[str]) -> None:
        raise AssertionError(f"re-extraction should not run: {document_ids}")

    completed = await run_source_lifecycle_recovery_job(
        db,
        "src-1",
        job_id="historical-reconstruction",
        reconstruct_documents=reconstruct,
        repair_projections=unexpected_reextract,
    )

    assert projections[0].checkpoint["cutover_repair"] is True
    assert await db.find_source_unit_by_document_id("src-1", "teams-historical-window") is not None
    assert completed.mapped_memories == 1
    assert completed.finding_count == 0
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_cutover_reconstructs_agent_session_projection_from_canonical_concept(
    db: Database,
    tmp_path,
) -> None:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-agent",
        type="agent_session",
        name="Codex Session",
        config_json="{}",
        access_policy="private",
        owner_user_id="owner-1",
    )
    markdown = "# Durable convention\n\nLegacy claim\n"
    await db.upsert_agent_concept(
        concept_id="akb-concept-1",
        source_id="src-agent",
        owner_user_id="owner-1",
        workspace="workspace-1",
        repo_identifier="repo-1",
        concept_type="convention",
        concept_path="conventions/durable-convention.md",
        title="Durable convention",
        markdown_body=markdown,
        frontmatter={"source_type": "agent_session"},
        observed_at=now,
    )
    await db.upsert_document(
        DocumentRecord(
            doc_id="akb-concept-1",
            source="src-agent",
            source_url="agent-knowledge://owner-1/akb-concept-1",
            title="Durable convention",
            space_or_project="workspace-1",
            author="codex",
            last_modified=now,
            labels=["convention"],
            version=content_hash(markdown),
            content_hash=content_hash(markdown),
            token_count=None,
            raw_content_uri=None,
            raw_content_type="text/markdown",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
            client="codex",
        )
    )
    await db.add_memory_source(
        "mem-legacy",
        "akb-concept-1",
        "agent_session",
        "Legacy claim",
        source_updated_at=None,
    )
    assert (await run_source_lifecycle_backfill(db, "src-agent")).finding_count == 1

    async def reconstruct(document_ids: frozenset[str]) -> None:
        for document_id in document_ids:
            await reconstruct_historical_source_projection(
                db,
                LocalDocumentStore(str(tmp_path / "missing-artifacts")),
                source_id="src-agent",
                source_type="agent_session",
                document_id=document_id,
            )

    completed = await run_source_lifecycle_recovery_job(
        db,
        "src-agent",
        job_id="agent-concept-reconstruction",
        reconstruct_documents=reconstruct,
    )

    source_unit = await db.find_source_unit_by_document_id("src-agent", "akb-concept-1")
    assert source_unit is not None
    assert source_unit.unit_type == "agent_session_window"
    assert completed.mapped_memories == 1
    assert completed.finding_count == 0
    assert (await db.get_lifecycle_gate("src-agent")).state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_ambiguous_cutover_finding_requires_exact_observation_repair(db: Database) -> None:
    now = datetime(2026, 7, 15, tzinfo=timezone.utc)
    item = ContentItem(
        item_id="teams-window-ambiguous",
        title="Ambiguous Teams window",
        source_url="https://teams.microsoft.com/l/message/conversation-1/message-1",
        last_modified=now,
        content_type="application/json",
        version="v1",
        extra={"window_id": "teams-window-ambiguous", "conversation_id": "conversation-1"},
    )
    raw = RawContent(
        item=item,
        body=json.dumps(
            {
                "messages": [
                    {
                        "id": "message-1",
                        "content": "Repeated quote",
                        "attachments": [],
                        "time": "2026-07-15T10:00:00Z",
                    },
                    {
                        "id": "message-2",
                        "content": "Repeated quote",
                        "attachments": [],
                        "time": "2026-07-15T10:01:00Z",
                    },
                ]
            }
        ).encode(),
        content_type="application/json",
    )
    projection = project_source_item(
        source_id="src-1",
        source_type="teams",
        run_id="projection-ambiguous",
        item=item,
        raw=raw,
        normalized=NormalizedContent(item=item, markdown_body="Repeated quote"),
    )
    await db.record_source_projection(projection)
    await db.upsert_document(
        DocumentRecord(
            doc_id=item.item_id,
            source="src-1",
            source_url=item.source_url,
            title=item.title,
            space_or_project="PCC",
            author=None,
            last_modified=now,
            labels=[],
            version="v1",
            content_hash="ambiguous-hash",
            token_count=2,
            raw_content_uri=None,
            raw_content_type="application/json",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )
    await db.add_memory_source(
        "mem-legacy",
        item.item_id,
        "teams",
        "Repeated quote…",
        source_updated_at=None,
    )
    result = await run_source_lifecycle_backfill(db, "src-1")
    assert result.finding_count == 1
    finding = (await db.list_lifecycle_cutover_findings("src-1"))[0]
    assert finding.reason is CutoverFindingReason.AMBIGUOUS_OBSERVATION

    selected_observation_id = projection.observations[0].id
    with pytest.raises(ValueError, match="requires an exact evidence_quote"):
        await repair_lifecycle_cutover_finding(
            db,
            source_id="src-1",
            finding_id=finding.id,
            observation_id=selected_observation_id,
        )
    repaired = await repair_lifecycle_cutover_finding(
        db,
        source_id="src-1",
        finding_id=finding.id,
        observation_id=selected_observation_id,
        evidence_quote="Repeated quote",
        operator_id="operator-1",
    )
    final = await run_source_lifecycle_backfill(db, "src-1")

    assert repaired.status is CutoverFindingStatus.RESOLVED
    assert repaired.observation_id == selected_observation_id
    assert final.finding_count == 0
    assert final.gate_enabled is True
    async with db.db.execute(
        "SELECT source_metadata_json FROM evidence_units WHERE source_id = ?",
        ("src-1",),
    ) as cursor:
        metadata = json.loads((await cursor.fetchone())["source_metadata_json"])
    assert metadata["operator_selected_observation"] is True
    assert metadata["operator_id"] == "operator-1"
    assert metadata["legacy_excerpt_replaced"] is True


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
async def test_support_preflight_failure_does_not_create_job_or_activity(
    db: Database,
    monkeypatch,
    runner,
) -> None:
    async def fail_preflight(_source_id: str) -> int:
        raise RuntimeError("support invariant unavailable")

    monkeypatch.setattr(
        db,
        "count_active_source_memories_without_support",
        fail_preflight,
    )
    job_id = f"preflight-failed-{runner.__name__}"

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
async def test_rebaseline_rejects_recovered_maintenance_epoch(
    db: Database,
) -> None:
    job = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="sqlite-rebaseline-stale-epoch",
            source_id="src-1",
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
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
        await db.rebaseline_source_lifecycle(
            "src-1",
            source_activity=activity,
        )

    assert await db.get_source_projection("projection-run-1") is not None


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
async def test_support_write_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    references = await db.record_evidence_references(
        _unit().id,
        (
            EvidenceReference(
                role=EvidenceRole.PRIMARY,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id="obs-page-1-body",
                    observation_revision_id="obsrev-page-1-v2",
                ),
                evidence_unit_id=_unit().id,
            ),
        ),
    )
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-support-stale-fence",
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
    reference_id = references[0].id or ""

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.upsert_memory_support_assertion(
            MemorySupportAssertion(
                id="support-stale-fence",
                memory_id="mem-legacy",
                evidence_reference_id=reference_id,
                source_id="src-1",
                access_context_hash="access-a",
            ),
            source_activity=activity,
        )

    assert reference_id not in await db.get_active_memory_support_reference_ids("mem-legacy")


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
async def test_finding_resolution_rejects_recovered_maintenance_fence(
    db: Database,
) -> None:
    finding = _finding()
    unit = _unit()
    await db.upsert_lifecycle_cutover_finding(finding)
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
            id="support-finding-stale-fence",
            memory_id="mem-legacy",
            evidence_reference_id=reference.id or "",
            source_id="src-1",
            access_context_hash="workspace",
        )
    )
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="sqlite-finding-resolution-stale-fence",
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
        await db.resolve_lifecycle_cutover_finding(
            finding.id,
            observation_id="obs-page-1-body",
            source_unit_id="unit-page-1",
            source_activity=activity,
        )

    assert (await db.get_lifecycle_cutover_finding(finding.id)).status is (CutoverFindingStatus.OPEN)


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
