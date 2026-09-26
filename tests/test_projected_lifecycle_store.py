from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
)
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    LifecycleGateState,
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
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database
from tests.test_source_projection_store import _projection
from tests.unit_support_fixture import complete_unit_parts, primary_reference, record_unit_support


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "lifecycle.db"))
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
        id="mem-1",
        memory_type="fact",
        content="Source claim",
        content_hash=content_hash("Source claim"),
    )
    await database.insert_memory(memory)
    await database.record_source_projection(_projection())
    try:
        yield database
    finally:
        await database.close()


async def _expired_sync_activity(db: Database, activity_id: str):
    activity = await db.acquire_source_activity(
        activity_id=activity_id,
        source_id="src-1",
        kind=SourceActivityKind.SYNC,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", activity_id),
    )
    await db.db.commit()
    return activity


def _unit() -> EvidenceUnit:
    return EvidenceUnit(
        id="eu-1",
        source_id="src-1",
        doc_id="gate-doc",
        doc_revision_id="unitrev-page-1-v2",
        source_type="confluence",
        source_anchor="obs-page-1-body",
        source_lineage_id="unit-page-1",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="Source claim",
        excerpt="Source claim",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        access_context_hash="workspace",
    )


@pytest.mark.asyncio
async def test_new_source_is_destructive_lifecycle_gated_by_default(db: Database) -> None:
    gate = await db.get_lifecycle_gate("src-1")

    assert gate.state is LifecycleGateState.GATED


async def _attach_source_document(db: Database, doc_id: str = "gate-doc") -> None:
    """Record ``doc_id`` as Source provenance of the fixture Memory."""

    now = "2026-07-15T00:00:00+00:00"
    await db.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified, version,
               content_hash, last_synced
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            "src-1",
            f"https://example.test/{doc_id}",
            "Engineering page",
            "ENG",
            now,
            "1",
            "hash",
            now,
        ),
    )
    await db.add_memory_source(
        "mem-1",
        doc_id,
        "confluence",
        "Source claim",
        source_updated_at=None,
    )


@pytest.mark.asyncio
async def test_gate_requires_validated_support_for_active_source_backed_memory(db: Database) -> None:
    await _attach_source_document(db)

    with pytest.raises(ValueError, match="source-backed Memory lacks validated support lineage"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
async def test_gate_rejects_active_support_without_matching_source_provenance(
    db: Database,
) -> None:
    await _attach_source_document(db)
    await _persist_exact_support_and_provenance(db)
    await db.db.execute(
        "DELETE FROM memory_sources WHERE memory_id = ? AND source_id = ?",
        ("mem-1", "src-1"),
    )
    await db.db.commit()

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 1
    with pytest.raises(ValueError, match="active support lacks source provenance"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
async def test_gate_rejects_source_provenance_without_exact_document_support(
    db: Database,
) -> None:
    await _attach_source_document(db)
    await _persist_exact_support_and_provenance(db)
    await _attach_source_document(db, doc_id="wrong-support-doc")

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0
    assert await db.count_active_source_memories_without_support("src-1") == 1
    with pytest.raises(ValueError, match="source-backed Memory lacks validated support lineage"):
        await db.enable_lifecycle_gate("src-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corrupt_statement, corrupt_params",
    (
        ("DELETE FROM evidence_units WHERE id = ?", ("eu-1",)),
        ("UPDATE evidence_units SET doc_id = NULL WHERE id = ?", ("eu-1",)),
    ),
    ids=("missing-evidence-unit", "null-document"),
)
async def test_reverse_support_projection_audit_fails_closed_for_corrupt_evidence_chain(
    db: Database,
    corrupt_statement: str,
    corrupt_params: tuple[str, ...],
) -> None:
    await _attach_source_document(db)
    await _persist_exact_support_and_provenance(db)
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = OFF")
    await db.db.execute(corrupt_statement, corrupt_params)
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = ON")

    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 1


@pytest.mark.asyncio
async def test_gate_ignores_inactive_historical_memory_without_support(db: Database) -> None:
    await _attach_source_document(db)
    await db.db.execute("UPDATE memories SET status = 'retired' WHERE id = ?", ("mem-1",))
    await db.db.commit()

    gate = await db.enable_lifecycle_gate("src-1")

    assert gate.state is LifecycleGateState.ENABLED


@pytest.mark.asyncio
async def test_gate_route_enables_a_gated_source_only_when_support_is_complete(
    db: Database,
    tmp_path,
) -> None:
    await _attach_source_document(db)
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    app = create_admin_app(db=db, config=config, principal_resolver=lambda request: "owner-1")

    with TestClient(app) as client:
        refused = client.post("/api/v1/sources/src-1/memory-lifecycle/gate")
        await _persist_exact_support_and_provenance(db)
        enabled = client.post("/api/v1/sources/src-1/memory-lifecycle/gate")

    assert refused.status_code == 409
    assert refused.json()["detail"] == "source-backed Memory lacks validated support lineage"
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["gate"]["state"] == LifecycleGateState.ENABLED.value
    assert (await db.get_lifecycle_gate("src-1")).state is LifecycleGateState.ENABLED


async def _persist_exact_support_and_provenance(db: Database) -> str:
    """Make ``_unit()`` active Support for the fixture Memory and return its id."""

    if not any(source.source_id == "src-1" for source in await db.get_memory_sources("mem-1")):
        await _attach_source_document(db)
    unit = _unit()
    await record_unit_support(
        db,
        memory_id="mem-1",
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
            mandatory_incumbent_ids=("mem-1",),
            incumbent_decisions=(
                (
                    IncumbentDecision(
                        "mem-1",
                        IncumbentDisposition.REMOVE_SUPPORT,
                        "overlapping source no longer supports claim",
                    )
                    if removing
                    else IncumbentDecision("mem-1", IncumbentDisposition.KEEP, "overlapping source states the claim")
                ),
            ),
            batch_ids=("batch-overlap",),
            completed_batch_ids=("batch-overlap",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=("obsrev-page-1-v2-source-2",),
            support_set_hashes=({"mem-1": support_hash} if isinstance(support_hash, str) else {}),
        ),
        mutations=(
            LifecycleMutation(
                mutation_type,
                memory_id="mem-1",
                source_id="src-2",
                evidence_unit_ids=(evidence_unit.id,),
                payload={
                    "access_context_hash": "workspace",
                    "document_id": "gate-doc",
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
        memory_id="mem-1",
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
            mandatory_incumbent_ids=("mem-1",),
            incumbent_decisions=(
                IncumbentDecision(
                    "mem-1",
                    IncumbentDisposition.REMOVE_SUPPORT,
                    "one Source Unit no longer supports the claim",
                ),
            ),
            batch_ids=(f"batch-{plan_id}",),
            completed_batch_ids=(f"batch-{plan_id}",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=(observation_revision_id,),
            support_set_hashes={"mem-1": support_hash},
        ),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-1",
                source_id="src-1",
                evidence_unit_ids=(evidence_unit_id,),
                payload={"document_id": "gate-doc"},
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
        support_hash=await db.get_memory_support_set_hash("mem-1"),
    )

    await db.apply_lifecycle_plan(plan)

    assert await db.get_active_memory_support_unit_ids("mem-1") == (second_unit_id,)
    assert [(source.source_id, source.doc_id) for source in await db.get_memory_sources("mem-1")] == [
        ("src-1", "gate-doc")
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
        ("mem-1", "gate-doc"),
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
            support_hash=await db.get_memory_support_set_hash("mem-1"),
        )
    )

    rows = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_sources
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-1", "gate-doc"),
    )
    assert [row["source_id"] for row in rows] == ["src-1"]
    assert await db.count_active_supported_memories_without_source_provenance("src-1") == 0
    metadata_rows = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_search_metadata_trigram
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-1", "gate-doc"),
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
        "mem-1",
        "gate-doc",
        "confluence",
        "longer owner source excerpt",
        support_kind="extracted",
        source_updated_at=None,
    )

    assert outcome == "updated"
    rows = await db.db.execute_fetchall(
        """SELECT source_id, excerpt FROM memory_sources
             WHERE memory_id = ? AND doc_id = ? ORDER BY source_id""",
        ("mem-1", "gate-doc"),
    )
    assert [(row["source_id"], row["excerpt"]) for row in rows] == [
        ("src-1", "longer owner source excerpt"),
        ("src-2", "Source claim"),
    ]


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
    document = await db.get_document("gate-doc")
    assert document is not None
    assert document.source == "src-2"
    sources = await db.get_memory_sources("mem-1")
    assert [(source.source_id, source.doc_id) for source in sources] == [
        ("src-2", "gate-doc"),
    ]
    assert await db.get_active_memory_support_unit_ids("mem-1") == (overlapping_unit.id,)
    memory = await db.get_memory("mem-1")
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
    document = await db.get_document("gate-doc")
    assert document is not None
    assert document.source == "src-1"
    sources = await db.get_memory_sources("mem-1")
    assert [(source.source_id, source.doc_id) for source in sources] == [
        ("src-1", "gate-doc"),
    ]
    assert await db.get_active_memory_support_unit_ids("mem-1") == ("eu-1",)
    memory = await db.get_memory("mem-1")
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
            mandatory_incumbent_ids=("mem-1",),
            incumbent_decisions=(
                IncumbentDecision(
                    "mem-1",
                    IncumbentDisposition.REMOVE_SUPPORT,
                    "authoritative evidence removed",
                ),
            ),
            batch_ids=("batch-1",),
            completed_batch_ids=("batch-1",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=("obsrev-page-1-v2",),
            support_set_hashes={"mem-1": support_hash},
        ),
        mutations=(
            LifecycleMutation(
                LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-1",
                source_id="src-1",
                evidence_unit_ids=(evidence_unit_id,),
            ),
            LifecycleMutation(
                LifecycleMutationType.RETIRE_MEMORY,
                memory_id="mem-1",
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
            doc_id="gate-doc",
            source_type="confluence",
            access_context_hash="workspace",
        ),
    )


@pytest.mark.asyncio
async def test_lifecycle_plan_applies_support_removal_and_retirement_atomically(db: Database) -> None:
    unit_id = await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    plan = _retirement_plan(unit_id, await db.get_memory_support_set_hash("mem-1"))
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

    memory = await db.get_memory("mem-1")
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
            incumbent_memory_id="mem-1",
            challenger_memory_id=challenger.id,
        )
    )
    unit_id = await _persist_exact_support_and_provenance(db)
    incumbent = await db.get_memory("mem-1")
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
    incumbent = await db.get_memory("mem-1")
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
    incumbent = await db.get_memory("mem-1")
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
            doc_id="gate-doc",
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
    assert set(remaining.unit_ids) == {other_unit.id, additional_other_unit.id}
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

    memory = await db.get_memory("mem-1")
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
        await db.get_memory_support_set_hash("mem-1"),
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
    memory = await db.get_memory("mem-1")
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
        await db.get_memory_support_set_hash("mem-1"),
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
            await db.get_memory_support_set_hash("mem-1"),
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
        await db.get_memory_support_set_hash("mem-1"),
    )
    plan = replace(
        plan,
        stale_guard=replace(
            plan.stale_guard,
            memory_versions={"mem-1": "memory-version-before-concurrent-edit"},
        ),
    )

    with pytest.raises(ValueError, match="Memory stale guard"):
        await db.apply_lifecycle_plan(plan)

    memory = await db.get_memory("mem-1")
    assert memory is not None and memory.status == "active"
    assert await db.get_lifecycle_plan_status(plan.id) is None


@pytest.mark.asyncio
async def test_lifecycle_gate_rejects_an_expired_activity_lease(
    db: Database,
) -> None:
    activity = await _expired_sync_activity(db, "sqlite-gate-stale-fence")
    gate_before = await db.get_lifecycle_gate("src-1")

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.enable_lifecycle_gate("src-1", source_activity=activity)

    assert await db.get_lifecycle_gate("src-1") == gate_before


@pytest.mark.asyncio
async def test_evidence_write_rejects_an_expired_activity_lease(
    db: Database,
) -> None:
    activity = await _expired_sync_activity(db, "sqlite-evidence-stale-fence")

    with pytest.raises(SourceActivityConflict, match="source activity"):
        await db.upsert_evidence_unit(_unit(), source_activity=activity)

    assert await db.get_evidence_unit(_unit().id) is None


@pytest.mark.asyncio
async def test_document_write_rejects_an_expired_activity_lease(
    db: Database,
) -> None:
    activity = await _expired_sync_activity(db, "sqlite-document-stale-fence")
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
async def test_evidence_reference_write_rejects_an_expired_activity_lease(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    activity = await _expired_sync_activity(db, "sqlite-reference-stale-fence")
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




def _staged_claim(content: str):
    """A Candidate whose resolved Evidence is one part of the fixture page body."""
    import hashlib

    from memforge.memory.evidence import EvidencePartKind, ResolvedEvidencePart, ResolvedEvidenceSelection
    from memforge.models import RawMemory

    digest = hashlib.sha256(content.encode()).hexdigest()
    return RawMemory(
        content=content,
        memory_type="fact",
        resolved_evidence_selection=ResolvedEvidenceSelection(
            source_id="src-1", source_unit_id="unit-page-1", target_unit_revision_id="unitrev-page-1-v2",
            access_context_hash="workspace", catalog_digest="catalog", compiler_contract_version=1,
            parts=(
                ResolvedEvidencePart(
                    role=EvidenceRole.PRIMARY,
                    kind=EvidencePartKind.TEXT,
                    anchor=SourceAnchor(
                        kind=AnchorKind.REVISION_RANGE, observation_id="obs-page-1-body",
                        observation_revision_id="obsrev-page-1-v2", range_start=0, range_end=len(content),
                    ),
                    raw_content_sha256=digest,
                    presentation_sha256=digest,
                    excerpt=content,
                ),
            ),
        ),
    )


async def _coordinator_review_plan(
    db: Database, *, plan_id: str, outcome: str = "raised", coordinator_reviews=(),
) -> LifecyclePlan:
    """One revision's Plan over mem-1 under partial coverage.

    ``raised`` raises the contradiction again; ``decided`` reads mem-1 as
    supported by its unchanged Evidence without the conflict; ``undecided``
    keeps mem-1 unchanged without a decision.
    """
    from memforge.models import CoordinatorProposal, CoordinatorReview

    incumbent = await db.get_memory("mem-1")
    assert incumbent is not None
    candidate = _staged_claim("One reviewer approves payroll.")
    operation = {
        "raised": ReconcileOperation(
            action=ReconcileAction.NOOP,
            memory_id=incumbent.id,
            reason="partial coverage",
            support_revalidation_skipped=True,
            reviews=(CoordinatorReview(candidate, CoordinatorProposal.SUPERSEDE, "one versus two reviewers"),),
        ),
        "decided": ReconcileOperation(
            action=ReconcileAction.NOOP, memory_id=incumbent.id, memory=_staged_claim(incumbent.content),
            reason="supported",
        ),
        "undecided": ReconcileOperation(
            action=ReconcileAction.NOOP, memory_id=incumbent.id, reason="capacity",
            support_revalidation_skipped=True,
        ),
    }[outcome]
    return build_lifecycle_plan(
        plan_id=plan_id,
        scope=ReconciliationScope(
            id=f"scope-{plan_id}", source_id="src-1", source_unit_id="unit-page-1",
            base_unit_revision_id="unitrev-page-1-v1", target_unit_revision_id="unitrev-page-1-v2",
        ),
        gate_state=LifecycleGateState.ENABLED,
        operations=(operation,),
        incumbents={incumbent.id: incumbent},
        source_support_unit_ids={incumbent.id: ("eu-1",)},
        all_active_support_unit_ids={incumbent.id: ("eu-1",)},
        support_set_hashes={incumbent.id: await db.get_memory_support_set_hash(incumbent.id)},
        observation_revision_ids=("obsrev-page-1-v2",),
        evidence_unit_ids_by_claim_hash={
            content_hash(candidate.content): ("eu-candidate",),
            content_hash(incumbent.content): ("eu-1",),
        },
        defaults=NewMemoryDefaults(
            visibility="workspace", owner_user_id=None, project_key=None, repo_identifier=None,
            doc_id="gate-doc", source_type="confluence", access_context_hash="workspace",
        ),
        coordinator_reviews=coordinator_reviews,
    )


async def _coordinator_reviews(db: Database):
    return await db.list_lifecycle_reviews(incumbent_memory_ids=("mem-1",))


@pytest.mark.asyncio
async def test_a_raised_again_coordinator_review_is_refreshed_in_place(db: Database) -> None:
    from memforge.memory.coordinator_review import coordinator_review_id
    from memforge.models import CoordinatorProposal

    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    await db.apply_lifecycle_plan(await _coordinator_review_plan(db, plan_id="plan-conflict-1"))
    [created] = await _coordinator_reviews(db)
    assert created.id == coordinator_review_id(
        "unit-page-1", "mem-1", CoordinatorProposal.SUPERSEDE, "One reviewer approves payroll.",
    )
    assert created.staged_evidence["stale_guard"] == {
        "support_set_hash": await db.get_memory_support_set_hash("mem-1"),
        "memory_version": lifecycle_memory_version(await db.get_memory("mem-1")),
    }

    await db.apply_lifecycle_plan(await _coordinator_review_plan(
        db, plan_id="plan-conflict-2", coordinator_reviews=(created,),
    ))

    [refreshed] = await _coordinator_reviews(db)
    assert refreshed.id == created.id and refreshed.status is LifecycleReviewStatus.PENDING
    assert refreshed.lifecycle_plan_id == "plan-conflict-2"
    assert refreshed.created_at == created.created_at


@pytest.mark.asyncio
async def test_a_conflict_the_revision_no_longer_raises_closes_stale_and_reopens_when_raised(db: Database) -> None:
    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    await db.apply_lifecycle_plan(await _coordinator_review_plan(db, plan_id="plan-conflict-1"))
    [pending] = await _coordinator_reviews(db)

    # A revision that keeps mem-1 without deciding it leaves the conflict pending as it is.
    held = await _coordinator_review_plan(
        db, plan_id="plan-conflict-held", outcome="undecided", coordinator_reviews=(pending,),
    )
    assert not any(mutation.mutation_type is LifecycleMutationType.RESOLVE_REVIEW for mutation in held.mutations)

    closing = await _coordinator_review_plan(
        db, plan_id="plan-conflict-gone", outcome="decided", coordinator_reviews=(pending,),
    )
    assert closing.mutations[0].mutation_type is LifecycleMutationType.RESOLVE_REVIEW
    await db.apply_lifecycle_plan(closing)
    [stale] = await _coordinator_reviews(db)
    assert stale.status is LifecycleReviewStatus.STALE and stale.resolved_at is not None

    await db.apply_lifecycle_plan(await _coordinator_review_plan(
        db, plan_id="plan-conflict-again", coordinator_reviews=(stale,),
    ))
    [reopened] = await _coordinator_reviews(db)
    assert (reopened.id, reopened.status, reopened.resolved_at) == (pending.id, LifecycleReviewStatus.PENDING, None)
    assert reopened.lifecycle_plan_id == "plan-conflict-again"


@pytest.mark.asyncio
@pytest.mark.parametrize("decided", [LifecycleReviewStatus.REJECTED, LifecycleReviewStatus.APPROVED])
async def test_a_decided_conflict_is_never_raised_again(db: Database, decided) -> None:
    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    await db.apply_lifecycle_plan(await _coordinator_review_plan(db, plan_id="plan-conflict-1"))
    [pending] = await _coordinator_reviews(db)
    await db.db.execute("UPDATE lifecycle_reviews SET status = ? WHERE id = ?", (decided.value, pending.id))
    await db.db.commit()
    [decided_review] = await _coordinator_reviews(db)

    again = await _coordinator_review_plan(db, plan_id="plan-conflict-2", coordinator_reviews=(decided_review,))

    assert not any(mutation.mutation_type is LifecycleMutationType.CREATE_REVIEW for mutation in again.mutations)
    assert [decision.disposition for decision in again.coverage_proof.incumbent_decisions] == [
        IncumbentDisposition.KEEP,
    ]
    # Plan apply refuses to raise a decided Review again even when a plan asks it to.
    raising = await _coordinator_review_plan(db, plan_id="plan-conflict-3")
    with pytest.raises(ValueError, match="cannot be raised again"):
        await db.apply_lifecycle_plan(raising)
    assert await db.get_lifecycle_plan_status(raising.id) is None
    [unchanged] = await _coordinator_reviews(db)
    assert unchanged.status is decided


@pytest.mark.asyncio
async def test_review_mutations_refuse_a_foreign_incumbent_and_an_unknown_resolution(db: Database) -> None:
    from dataclasses import replace

    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    raising = await _coordinator_review_plan(db, plan_id="plan-conflict-1")
    [create] = [item for item in raising.mutations if item.mutation_type is LifecycleMutationType.CREATE_REVIEW]
    # A row of the same ID on another incumbent; only its identity matters here.
    await db.db.execute("PRAGMA foreign_keys = OFF")
    await db.db.execute(
        """INSERT INTO lifecycle_reviews (id, lifecycle_plan_id, incumbent_memory_id, status,
               staged_evidence_json, reason, created_at)
           VALUES (?, 'plan-other', 'mem-other', 'pending', '{}', 'other', '2026-01-01T00:00:00+00:00')""",
        (create.payload["review_id"],),
    )
    await db.db.commit()
    await db.db.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(ValueError, match="belongs to another incumbent"):
        await db.apply_lifecycle_plan(raising)
    await db.db.execute("DELETE FROM lifecycle_reviews WHERE id = ?", (create.payload["review_id"],))
    await db.db.commit()

    await db.apply_lifecycle_plan(raising)
    [pending] = await _coordinator_reviews(db)
    closing = await _coordinator_review_plan(
        db, plan_id="plan-conflict-gone", outcome="decided", coordinator_reviews=(pending,),
    )
    resolve, *rest = closing.mutations
    unknown = replace(closing, mutations=(
        replace(resolve, payload={**resolve.payload, "status": LifecycleReviewStatus.PENDING.value}), *rest,
    ))
    with pytest.raises(ValueError, match="cannot be resolved as pending"):
        await db.apply_lifecycle_plan(unknown)
    [unchanged] = await _coordinator_reviews(db)
    assert unchanged.status is LifecycleReviewStatus.PENDING


@pytest.mark.asyncio
async def test_a_coordinator_review_is_approved_against_its_own_guard(db: Database) -> None:
    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    creating = await _coordinator_review_plan(db, plan_id="plan-conflict-1")
    await db.apply_lifecycle_plan(creating)
    [review] = await _coordinator_reviews(db)
    payload = await db.get_lifecycle_plan_payload(creating.id)

    approval = build_lifecycle_review_approval_plan(review, payload)

    guard = review.staged_evidence["stale_guard"]
    assert approval.stale_guard.support_set_hashes == {"mem-1": guard["support_set_hash"]}
    assert approval.stale_guard.memory_versions == {"mem-1": guard["memory_version"]}
    await db.db.execute("UPDATE memories SET updated_at = ? WHERE id = 'mem-1'", ("2030-01-01T00:00:00+00:00",))
    await db.db.commit()
    with pytest.raises(ValueError, match="stale guard"):
        await db.apply_lifecycle_plan(approval)


@pytest.mark.asyncio
async def test_a_coordinator_review_is_not_refreshed_by_hand(db: Database) -> None:
    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    creating = await _coordinator_review_plan(db, plan_id="plan-conflict-1")
    await db.apply_lifecycle_plan(creating)
    [review] = await _coordinator_reviews(db)
    stale = replace(review, status=LifecycleReviewStatus.STALE)

    with pytest.raises(ValueError, match="next revision"):
        build_lifecycle_review_refresh_plan(
            stale, await db.get_lifecycle_plan_payload(creating.id), gate_state=LifecycleGateState.ENABLED,
            current_support_set_hash="hash", current_memory_version="version",
        )


@pytest.mark.asyncio
async def test_lifecycle_reviews_are_listed_by_incumbent(db: Database) -> None:
    await _persist_exact_support_and_provenance(db)
    await db.enable_lifecycle_gate("src-1")
    await db.apply_lifecycle_plan(await _coordinator_review_plan(db, plan_id="plan-conflict-1"))

    assert [review.incumbent_memory_id for review in await _coordinator_reviews(db)] == ["mem-1"]
    assert await db.list_lifecycle_reviews(incumbent_memory_ids=("mem-other",)) == []
    assert await db.list_lifecycle_reviews(incumbent_memory_ids=()) == []
