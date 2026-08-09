from __future__ import annotations

from dataclasses import replace

import pytest

from memforge.memory.lifecycle_plan import (
    LifecycleGateState,
    LifecycleMutationType,
    LifecycleReview,
    LifecycleReviewStatus,
    ReconciliationScope,
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
from memforge.memory.relation_discovery_contract import relation_discovery_request_id
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
        evidence_quote="A7 is retained",
        extraction_context="the entire untrusted source document",
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
        actor_user_id="reviewer-1",
        entity_ids_by_claim_hash={content_hash(_replacement().content): (7, 11)},
    )


def _build(*, gate: LifecycleGateState, all_support=("eref-old",), flagged=False, defaults=None):
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
        defaults=defaults or _defaults(),
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
    create_payload = staged["proposed_mutations"][0]["payload"]["memory"]
    assert create_payload["extraction_context"] == "A7 is retained"
    assert create_payload["document_source"]["excerpt"] == "A7 is retained"


def test_gated_noop_evidence_rebind_stages_review_without_mutating_incumbent() -> None:
    old = _memory()
    candidate = RawMemory(
        content=old.content,
        memory_type=old.memory_type,
        confidence=0.9,
        evidence_quote="A7 is removed.",
        support_validation={"validated_quote": "A7 is removed."},
    )

    plan = build_lifecycle_plan(
        plan_id="plan-noop-rebind",
        scope=_scope(),
        gate_state=LifecycleGateState.GATED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=old.id,
                memory=candidate,
                reason="claim unchanged; evidence moved to current revision",
            ),
        ),
        incumbents={old.id: old},
        source_support_reference_ids={old.id: ("eref-old",)},
        all_active_support_reference_ids={old.id: ("eref-old",)},
        support_set_hashes={old.id: "support-hash"},
        observation_revision_ids=("obsrev-2",),
        new_evidence_reference_ids=("eref-new",),
        defaults=_defaults(),
    )

    assert [item.mutation_type for item in plan.mutations] == [
        LifecycleMutationType.CREATE_REVIEW,
    ]
    assert plan.coverage_proof.incumbent_decisions[0].disposition.value == "review"
    proposed = plan.mutations[0].payload["staged_evidence"]["proposed_mutations"]
    assert [item["mutation_type"] for item in proposed] == [
        "remove_support",
        "attach_support",
    ]
    assert proposed[0]["evidence_reference_ids"] == ["eref-old"]
    assert proposed[1]["evidence_reference_ids"] == ["eref-new"]

    mutation = plan.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=plan.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.PENDING,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
    )
    approval = build_lifecycle_review_approval_plan(review, lifecycle_plan_to_payload(plan))

    assert [item.mutation_type for item in approval.mutations] == [
        LifecycleMutationType.RESOLVE_REVIEW,
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.ATTACH_SUPPORT,
    ]
    assert approval.coverage_proof.incumbent_decisions[0].disposition.value == "keep"


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
        LifecycleMutationType.RESOLVE_REVIEW,
        LifecycleMutationType.CREATE_MEMORY,
        LifecycleMutationType.ATTACH_SUPPORT,
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.SUPERSEDE_MEMORY,
    ]
    assert approval.stale_guard.support_set_hashes == {"mem-old": "support-hash"}
    [request] = approval.relation_discovery_requests
    replacement_id = approval.coverage_proof.incumbent_decisions[0].replacement_memory_id
    assert replacement_id is not None
    assert request.id == relation_discovery_request_id(
        lifecycle_plan_id=approval.id,
        memory_id=replacement_id,
        expected_content_hash=content_hash(_replacement().content.strip()),
    )
    assert request.memory_id == replacement_id
    assert request.expected_content_hash == content_hash(_replacement().content.strip())
    assert request.source_id == "src-1"
    assert request.source_unit_id == "unit-1"
    assert request.source_unit_revision_id == "unitrev-2"
    assert request.doc_id == "PAY-1"
    assert request.actor_user_id == "reviewer-1"
    assert request.entity_ids == (7, 11)


def test_pending_review_without_activation_does_not_enqueue_relation_discovery() -> None:
    old = _memory()
    original = build_lifecycle_plan(
        plan_id="plan-delete-review",
        scope=_scope(),
        gate_state=LifecycleGateState.GATED,
        operations=(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=old.id,
                reason="current source removed the claim",
            ),
        ),
        incumbents={old.id: old},
        source_support_reference_ids={old.id: ("eref-old",)},
        all_active_support_reference_ids={old.id: ("eref-old",)},
        support_set_hashes={old.id: "support-hash"},
        observation_revision_ids=("obsrev-2",),
        new_evidence_reference_ids=(),
        defaults=_defaults(),
    )
    mutation = original.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=original.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.PENDING,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
    )

    approval = build_lifecycle_review_approval_plan(
        review,
        lifecycle_plan_to_payload(original),
    )

    assert approval.relation_discovery_requests == ()


def test_stale_review_refresh_creates_only_a_new_pending_decision() -> None:
    original = _build(
        gate=LifecycleGateState.ENABLED,
        all_support=("eref-old", "eref-other-source"),
    )
    mutation = original.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=original.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.STALE,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
        source_id="src-1",
    )

    refreshed, refreshed_review_id = build_lifecycle_review_refresh_plan(
        review,
        lifecycle_plan_to_payload(original),
        gate_state=LifecycleGateState.ENABLED,
        current_support_set_hash="current-support-hash",
        current_memory_version=lifecycle_memory_version(_memory()),
    )
    retried, retried_review_id = build_lifecycle_review_refresh_plan(
        review,
        lifecycle_plan_to_payload(original),
        gate_state=LifecycleGateState.ENABLED,
        current_support_set_hash="current-support-hash",
        current_memory_version=lifecycle_memory_version(_memory()),
    )

    assert retried == refreshed
    assert retried_review_id == refreshed_review_id
    assert refreshed.id.startswith("lifecycle-review-refresh-")
    assert refreshed.coverage_proof.incumbent_decisions[0].disposition.value == "review"
    assert refreshed.stale_guard.support_set_hashes == {"mem-old": "current-support-hash"}
    assert [item.mutation_type for item in refreshed.mutations] == [
        LifecycleMutationType.CREATE_REVIEW,
    ]
    [create_review] = refreshed.mutations
    assert create_review.payload["review_id"] == refreshed_review_id
    assert create_review.payload["staged_evidence"]["refreshed_from_review_id"] == review.id


def test_refresh_rejects_a_non_stale_lifecycle_review() -> None:
    original = _build(
        gate=LifecycleGateState.ENABLED,
        all_support=("eref-old", "eref-other-source"),
    )
    mutation = original.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=original.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.PENDING,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
        source_id="src-1",
    )

    with pytest.raises(ValueError, match="only a stale lifecycle review"):
        build_lifecycle_review_refresh_plan(
            review,
            lifecycle_plan_to_payload(original),
            gate_state=LifecycleGateState.ENABLED,
            current_support_set_hash="current-support-hash",
            current_memory_version=lifecycle_memory_version(_memory()),
        )


def test_refresh_rejects_a_terminal_proposal_that_needs_replanning() -> None:
    original = _build(gate=LifecycleGateState.GATED)
    mutation = original.mutations[0]
    review = LifecycleReview(
        id=str(mutation.payload["review_id"]),
        lifecycle_plan_id=original.id,
        incumbent_memory_id=mutation.memory_id,
        status=LifecycleReviewStatus.STALE,
        staged_evidence=mutation.payload["staged_evidence"],
        reason=str(mutation.payload["reason"]),
        source_id="src-1",
    )

    with pytest.raises(ValueError, match="requires source replanning"):
        build_lifecycle_review_refresh_plan(
            review,
            lifecycle_plan_to_payload(original),
            gate_state=LifecycleGateState.ENABLED,
            current_support_set_hash="changed-support-hash",
            current_memory_version=lifecycle_memory_version(_memory()),
        )


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


def test_private_lifecycle_plan_persists_owner_as_relation_discovery_actor() -> None:
    plan = _build(
        gate=LifecycleGateState.ENABLED,
        defaults=replace(
            _defaults(),
            visibility="private",
            owner_user_id="owner-1",
            actor_user_id=None,
        ),
    )

    [request] = plan.relation_discovery_requests
    assert request.actor_user_id == "owner-1"


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
