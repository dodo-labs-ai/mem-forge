from __future__ import annotations

import pytest

from memforge.memory.lifecycle_plan import (
    LifecycleGateState,
    LifecycleMutationType,
    LifecycleReview,
    LifecycleReviewStatus,
    ReconciliationScope,
    lifecycle_plan_to_payload,
)
from memforge.memory.lifecycle_planner import NewMemoryDefaults, build_lifecycle_plan
from memforge.memory.lifecycle_review import build_lifecycle_review_approval_plan
from memforge.models import Memory, RawMemory, ReconcileAction, ReconcileOperation, content_hash


def _memory(memory_id: str = "mem-old") -> Memory:
    return Memory(
        id=memory_id,
        memory_type="decision",
        content="A7 is removed.",
        content_hash=content_hash("A7 is removed."),
    )


def _replacement() -> RawMemory:
    return RawMemory(
        content="A7 is retained and marked as reduced retro chain.",
        memory_type="decision",
        confidence=0.9,
        tags=["payroll", "retro"],
        extraction_context="A7 is retained",
    )


def _scope() -> ReconciliationScope:
    return ReconciliationScope(
        id="scope-1",
        source_id="src-1",
        source_unit_id="unit-1",
        base_unit_revision_id="unitrev-1",
        target_unit_revision_id="unitrev-2",
    )


def _defaults() -> NewMemoryDefaults:
    return NewMemoryDefaults(
        visibility="workspace",
        owner_user_id=None,
        project_key="PAY",
        repo_identifier=None,
        doc_id="PAY-1",
        source_type="jira",
        access_context_hash="workspace-pay",
    )


def _build(*, gate: LifecycleGateState, all_support=("eref-old",), flagged=False):
    old = _memory()
    return build_lifecycle_plan(
        plan_id="plan-1",
        scope=_scope(),
        gate_state=gate,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory_id=old.id,
                memory=_replacement(),
                reason="current source changed",
                flag_for_review=flagged,
            ),
        ),
        incumbents={old.id: old},
        source_support_reference_ids={old.id: ("eref-old",)},
        all_active_support_reference_ids={old.id: all_support},
        support_set_hashes={old.id: "support-hash"},
        observation_revision_ids=("obsrev-2",),
        new_evidence_reference_ids=("eref-new",),
        defaults=_defaults(),
    )


def test_gated_replacement_stages_review_without_mutating_incumbent() -> None:
    plan = _build(gate=LifecycleGateState.GATED)

    assert [item.mutation_type for item in plan.mutations] == [
        LifecycleMutationType.CREATE_REVIEW,
    ]
    assert plan.coverage_proof.incumbent_decisions[0].disposition.value == "review"
    staged = plan.mutations[0].payload["staged_evidence"]
    assert [item["mutation_type"] for item in staged["proposed_mutations"]] == [
        "create_memory",
        "attach_support",
        "remove_support",
        "supersede_memory",
    ]


def test_pending_review_builds_fresh_atomic_approval_plan() -> None:
    original = _build(gate=LifecycleGateState.GATED)
    mutation = original.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=original.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.PENDING,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
    )

    approval = build_lifecycle_review_approval_plan(review, lifecycle_plan_to_payload(original))

    assert approval.gate_state is LifecycleGateState.ENABLED
    assert approval.coverage_proof.mandatory_incumbent_ids == ("mem-old",)
    assert [item.mutation_type for item in approval.mutations] == [
        LifecycleMutationType.CREATE_MEMORY,
        LifecycleMutationType.ATTACH_SUPPORT,
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.SUPERSEDE_MEMORY,
        LifecycleMutationType.RESOLVE_REVIEW,
    ]
    assert approval.stale_guard.support_set_hashes == {"mem-old": "support-hash"}


def test_enabled_local_replacement_is_create_attach_remove_supersede() -> None:
    plan = _build(gate=LifecycleGateState.ENABLED)

    assert [item.mutation_type for item in plan.mutations] == [
        LifecycleMutationType.CREATE_MEMORY,
        LifecycleMutationType.ATTACH_SUPPORT,
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.SUPERSEDE_MEMORY,
    ]
    decision = plan.coverage_proof.incumbent_decisions[0]
    assert decision.replacement_memory_id is not None
    assert set(plan.stale_guard.memory_versions) == {"mem-old"}
    assert plan.stale_guard.memory_versions["mem-old"].startswith("memory-version-")


def test_support_outside_current_scope_routes_replacement_to_review() -> None:
    plan = _build(
        gate=LifecycleGateState.ENABLED,
        all_support=("eref-old", "eref-other-source"),
    )

    assert [item.mutation_type for item in plan.mutations] == [
        LifecycleMutationType.CREATE_REVIEW,
    ]
    proposed = plan.mutations[0].payload["staged_evidence"]["proposed_mutations"]
    assert "supersede_memory" not in {item["mutation_type"] for item in proposed}
    assert "refresh_memory_index" in {item["mutation_type"] for item in proposed}


def test_planner_rejects_incomplete_incumbent_ledger() -> None:
    old = _memory()

    with pytest.raises(ValueError, match="missing lifecycle operation"):
        build_lifecycle_plan(
            plan_id="plan-1",
            scope=_scope(),
            gate_state=LifecycleGateState.ENABLED,
            operations=(),
            incumbents={old.id: old},
            source_support_reference_ids={old.id: ("eref-old",)},
            all_active_support_reference_ids={old.id: ("eref-old",)},
            support_set_hashes={old.id: "support-hash"},
            observation_revision_ids=("obsrev-2",),
            new_evidence_reference_ids=("eref-new",),
            defaults=_defaults(),
        )
