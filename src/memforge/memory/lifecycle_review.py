"""Approval planning for durable, source-local lifecycle reviews."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from memforge.memory.lifecycle_plan import (
    CoverageProof,
    IncumbentDecision,
    IncumbentDisposition,
    LifecycleGateState,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    ReconciliationScope,
    StaleGuard,
)


def build_lifecycle_review_approval_plan(
    review: LifecycleReview,
    original_plan_payload: Mapping[str, object],
) -> LifecyclePlan:
    """Turn a pending proposal into a fresh atomic plan with original stale guards.

    Approval never mutates from the review row alone. The original complete plan
    supplies the source-unit revision and incumbent snapshots, while the review
    carries only the mutations proposed for its one incumbent.
    """

    if review.status is not LifecycleReviewStatus.PENDING:
        raise ValueError(f"lifecycle review is already {review.status.value}")
    scope_payload = _mapping(original_plan_payload.get("scope"), "scope")
    source_id = _text(scope_payload.get("source_id"), "scope.source_id")
    stale_payload = _mapping(original_plan_payload.get("stale_guard"), "stale_guard")
    support_hashes = _string_mapping(stale_payload.get("support_set_hashes"), "support_set_hashes")
    memory_versions = _string_mapping(stale_payload.get("memory_versions"), "memory_versions")
    incumbent_id = review.incumbent_memory_id
    if incumbent_id not in support_hashes or incumbent_id not in memory_versions:
        raise ValueError("review incumbent is absent from original stale guard")

    disposition = IncumbentDisposition(
        _text(review.staged_evidence.get("proposed_disposition"), "proposed_disposition")
    )
    replacement_id = review.staged_evidence.get("replacement_memory_id")
    if replacement_id is not None and not isinstance(replacement_id, str):
        raise ValueError("replacement_memory_id must be a string")
    raw_mutations = review.staged_evidence.get("proposed_mutations")
    if not isinstance(raw_mutations, Sequence) or isinstance(raw_mutations, (str, bytes)):
        raise ValueError("lifecycle review lacks proposed mutations")
    proposed = tuple(_deserialize_mutation(value, source_id, incumbent_id) for value in raw_mutations)
    if not proposed:
        raise ValueError("lifecycle review has no proposed mutations")

    resolution = LifecycleMutation(
        mutation_type=LifecycleMutationType.RESOLVE_REVIEW,
        memory_id=incumbent_id,
        source_id=source_id,
        payload={"review_id": review.id, "status": LifecycleReviewStatus.APPROVED.value},
    )
    scope = ReconciliationScope(
        id=f"{_text(scope_payload.get('id'), 'scope.id')}:review:{review.id}",
        source_id=source_id,
        source_unit_id=_text(scope_payload.get("source_unit_id"), "scope.source_unit_id"),
        base_unit_revision_id=_optional_text(scope_payload.get("base_unit_revision_id")),
        target_unit_revision_id=_optional_text(scope_payload.get("target_unit_revision_id")),
        dependency_unit_ids=tuple(
            str(value) for value in _sequence(scope_payload.get("dependency_unit_ids", ()))
        ),
    )
    plan = LifecyclePlan(
        id=f"lifecycle-review-approval-{review.id}",
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=(incumbent_id,),
            incumbent_decisions=(
                IncumbentDecision(
                    memory_id=incumbent_id,
                    disposition=disposition,
                    reason=review.reason or "approved lifecycle review",
                    replacement_memory_id=(
                        replacement_id if disposition is IncumbentDisposition.SUPERSEDE else None
                    ),
                ),
            ),
            batch_ids=(f"{scope.id}:batch:0",),
            completed_batch_ids=(f"{scope.id}:batch:0",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=tuple(
                str(value) for value in _sequence(stale_payload.get("observation_revision_ids", ()))
            ),
            support_set_hashes={incumbent_id: support_hashes[incumbent_id]},
            memory_versions={incumbent_id: memory_versions[incumbent_id]},
        ),
        mutations=(*proposed, resolution),
    )
    plan.validate()
    return plan


def _deserialize_mutation(
    value: object,
    source_id: str,
    incumbent_id: str,
) -> LifecycleMutation:
    raw = _mapping(value, "proposed_mutation")
    mutation_source_id = _text(raw.get("source_id"), "mutation.source_id")
    if mutation_source_id != source_id:
        raise ValueError("review mutation belongs to another source")
    mutation_type = LifecycleMutationType(_text(raw.get("mutation_type"), "mutation_type"))
    if mutation_type in {LifecycleMutationType.CREATE_REVIEW, LifecycleMutationType.RESOLVE_REVIEW}:
        raise ValueError("review proposal contains a nested review mutation")
    memory_id = _text(raw.get("memory_id"), "mutation.memory_id")
    replacement_memory_id = _optional_text(raw.get("replacement_memory_id"))
    evidence_ids = tuple(str(item) for item in _sequence(raw.get("evidence_reference_ids", ())))
    payload = _mapping(raw.get("payload", {}), "mutation.payload")
    mutation = LifecycleMutation(
        mutation_type=mutation_type,
        memory_id=memory_id,
        source_id=mutation_source_id,
        evidence_reference_ids=evidence_ids,
        replacement_memory_id=replacement_memory_id,
        payload=dict(payload),
    )
    if mutation_type in {
        LifecycleMutationType.REMOVE_SUPPORT,
        LifecycleMutationType.SUPERSEDE_MEMORY,
        LifecycleMutationType.RETIRE_MEMORY,
        LifecycleMutationType.REFRESH_MEMORY_INDEX,
    } and memory_id != incumbent_id:
        raise ValueError("review proposal destructively targets another incumbent")
    return mutation


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("expected a sequence")
    return value


def _string_mapping(value: object, name: str) -> dict[str, str]:
    raw = _mapping(value, name)
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in raw.items()):
        raise ValueError(f"{name} must contain string values")
    return dict(raw)  # type: ignore[arg-type]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected a string or null")
    return value
