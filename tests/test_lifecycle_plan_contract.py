from __future__ import annotations

import pytest

from memforge.memory.lifecycle_plan import (
    CoverageProof,
    IncumbentDecision,
    IncumbentDisposition,
    LifecycleGateState,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    ReconciliationScope,
    StaleGuard,
)


def _scope() -> ReconciliationScope:
    return ReconciliationScope(
        id="scope-1",
        source_id="src-1",
        source_unit_id="unit-1",
        base_unit_revision_id="unitrev-1",
        target_unit_revision_id="unitrev-2",
    )


def _proof(*decisions: IncumbentDecision) -> CoverageProof:
    return CoverageProof(
        mandatory_incumbent_ids=("mem-a", "mem-b"),
        incumbent_decisions=decisions,
        batch_ids=("batch-1", "batch-2"),
        completed_batch_ids=("batch-1", "batch-2"),
    )


def _guard() -> StaleGuard:
    return StaleGuard(
        observation_revision_ids=("obsrev-1", "obsrev-2"),
        support_set_hashes={"mem-a": "support-a", "mem-b": "support-b"},
    )


def test_coverage_proof_requires_one_terminal_decision_per_incumbent() -> None:
    with pytest.raises(ValueError, match="missing incumbent decisions"):
        _proof(
            IncumbentDecision(
                memory_id="mem-a",
                disposition=IncumbentDisposition.KEEP,
                reason="disjoint evidence",
            )
        ).validate()

    with pytest.raises(ValueError, match="duplicate incumbent decision"):
        _proof(
            IncumbentDecision("mem-a", IncumbentDisposition.KEEP, "first"),
            IncumbentDecision("mem-a", IncumbentDisposition.KEEP, "duplicate"),
            IncumbentDecision("mem-b", IncumbentDisposition.KEEP, "keep"),
        ).validate()


def test_coverage_proof_requires_all_batches() -> None:
    proof = CoverageProof(
        mandatory_incumbent_ids=("mem-a",),
        incumbent_decisions=(
            IncumbentDecision("mem-a", IncumbentDisposition.KEEP, "checked"),
        ),
        batch_ids=("batch-1", "batch-2"),
        completed_batch_ids=("batch-1",),
    )

    with pytest.raises(ValueError, match="incomplete reconciliation batches"):
        proof.validate()


def test_gated_source_rejects_destructive_plan() -> None:
    plan = LifecyclePlan(
        id="plan-1",
        scope=_scope(),
        gate_state=LifecycleGateState.GATED,
        coverage_proof=_proof(
            IncumbentDecision("mem-a", IncumbentDisposition.REMOVE_SUPPORT, "removed"),
            IncumbentDecision("mem-b", IncumbentDisposition.KEEP, "disjoint"),
        ),
        stale_guard=_guard(),
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-a",
                source_id="src-1",
                evidence_reference_ids=("eref-a",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="lifecycle gate"):
        plan.validate()


def test_gated_source_still_allows_create_and_attach_support() -> None:
    plan = LifecyclePlan(
        id="plan-1",
        scope=_scope(),
        gate_state=LifecycleGateState.GATED,
        coverage_proof=_proof(
            IncumbentDecision("mem-a", IncumbentDisposition.KEEP, "checked"),
            IncumbentDecision("mem-b", IncumbentDisposition.KEEP, "checked"),
        ),
        stale_guard=_guard(),
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.CREATE_MEMORY,
                memory_id="mem-new",
                source_id="src-1",
            ),
            LifecycleMutation(
                mutation_type=LifecycleMutationType.ATTACH_SUPPORT,
                memory_id="mem-new",
                source_id="src-1",
                evidence_reference_ids=("eref-new",),
            ),
        ),
    )

    plan.validate()


def test_supersession_is_distinct_from_support_removal_and_retirement() -> None:
    plan = LifecyclePlan(
        id="plan-1",
        scope=_scope(),
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=_proof(
            IncumbentDecision(
                "mem-a",
                IncumbentDisposition.SUPERSEDE,
                "replaced",
                replacement_memory_id="mem-new",
            ),
            IncumbentDecision("mem-b", IncumbentDisposition.REMOVE_SUPPORT, "removed"),
        ),
        stale_guard=_guard(),
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.SUPERSEDE_MEMORY,
                memory_id="mem-a",
                replacement_memory_id="mem-new",
                source_id="src-1",
            ),
            LifecycleMutation(
                mutation_type=LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-b",
                source_id="src-1",
                evidence_reference_ids=("eref-b",),
            ),
            LifecycleMutation(
                mutation_type=LifecycleMutationType.RETIRE_MEMORY,
                memory_id="mem-b",
                source_id="src-1",
            ),
        ),
    )

    plan.validate()
    assert [item.mutation_type for item in plan.mutations] == [
        LifecycleMutationType.SUPERSEDE_MEMORY,
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.RETIRE_MEMORY,
    ]


def test_plan_rejects_mutation_for_memory_outside_incumbent_ledger() -> None:
    plan = LifecyclePlan(
        id="plan-1",
        scope=_scope(),
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=_proof(
            IncumbentDecision("mem-a", IncumbentDisposition.KEEP, "keep"),
            IncumbentDecision("mem-b", IncumbentDisposition.KEEP, "keep"),
        ),
        stale_guard=_guard(),
        mutations=(
            LifecycleMutation(
                mutation_type=LifecycleMutationType.REMOVE_SUPPORT,
                memory_id="mem-not-covered",
                source_id="src-1",
                evidence_reference_ids=("eref-x",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="outside mandatory incumbent ledger"):
        plan.validate()
