"""Approval planning for durable, source-local lifecycle reviews."""

from __future__ import annotations

import hashlib
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
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryRequest,
    relation_discovery_request_id,
)


def build_lifecycle_review_approval_plan(
    review: LifecycleReview,
    original_plan_payload: Mapping[str, object],
    *,
    reviewer: str | None = None,
    review_note: str | None = None,
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

    plan_id = f"lifecycle-review-approval-{review.id}"
    relation_discovery_requests = _relation_discovery_requests(
        review,
        proposed=proposed,
        plan_id=plan_id,
        source_id=source_id,
        source_unit_id=_text(scope_payload.get("source_unit_id"), "scope.source_unit_id"),
        source_unit_revision_id=_optional_text(scope_payload.get("target_unit_revision_id")),
    )

    resolution = LifecycleMutation(
        mutation_type=LifecycleMutationType.RESOLVE_REVIEW,
        memory_id=incumbent_id,
        source_id=source_id,
        payload={
            "review_id": review.id,
            "status": LifecycleReviewStatus.APPROVED.value,
            "reviewer": reviewer,
            "review_note": review_note,
        },
    )
    scope = ReconciliationScope(
        id=f"{_text(scope_payload.get('id'), 'scope.id')}:review:{review.id}",
        source_id=source_id,
        source_unit_id=_text(scope_payload.get("source_unit_id"), "scope.source_unit_id"),
        base_unit_revision_id=_optional_text(scope_payload.get("base_unit_revision_id")),
        target_unit_revision_id=_optional_text(scope_payload.get("target_unit_revision_id")),
        dependency_unit_ids=tuple(str(value) for value in _sequence(scope_payload.get("dependency_unit_ids", ()))),
    )
    plan = LifecyclePlan(
        id=plan_id,
        scope=scope,
        gate_state=LifecycleGateState.ENABLED,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=(incumbent_id,),
            incumbent_decisions=(
                IncumbentDecision(
                    memory_id=incumbent_id,
                    disposition=disposition,
                    reason=review.reason or "approved lifecycle review",
                    replacement_memory_id=(replacement_id if disposition is IncumbentDisposition.SUPERSEDE else None),
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
        # Resolve first inside the same transaction so terminal mutations stale
        # only other pending review work. Any later failure rolls approval back.
        mutations=(resolution, *proposed),
        relation_discovery_requests=relation_discovery_requests,
    )
    plan.validate()
    return plan


def build_lifecycle_review_refresh_plan(
    review: LifecycleReview,
    original_plan_payload: Mapping[str, object],
    *,
    gate_state: LifecycleGateState,
    current_support_set_hash: str,
    current_memory_version: str,
) -> tuple[LifecyclePlan, str]:
    """Reissue one stale proposal against the incumbent's current snapshot.

    The stale Review remains immutable audit history. The returned plan creates
    a new pending Review only; approving that Review still uses the ordinary
    lifecycle decision transaction and its complete stale guards.
    """

    if review.status is not LifecycleReviewStatus.STALE:
        raise ValueError("only a stale lifecycle review can be refreshed")
    if not current_support_set_hash or not current_memory_version:
        raise ValueError("lifecycle review refresh requires a current incumbent snapshot")

    scope_payload = _mapping(original_plan_payload.get("scope"), "scope")
    stale_payload = _mapping(original_plan_payload.get("stale_guard"), "stale_guard")
    source_id = _text(scope_payload.get("source_id"), "scope.source_id")
    if review.source_id is not None and review.source_id != source_id:
        raise ValueError("lifecycle review belongs to another source")
    incumbent_id = review.incumbent_memory_id
    original_support_hashes = _string_mapping(
        stale_payload.get("support_set_hashes"),
        "support_set_hashes",
    )
    original_memory_versions = _string_mapping(
        stale_payload.get("memory_versions"),
        "memory_versions",
    )
    if incumbent_id not in original_support_hashes or incumbent_id not in original_memory_versions:
        raise ValueError("review incumbent is absent from original stale guard")
    raw_mutations = review.staged_evidence.get("proposed_mutations")
    if not isinstance(raw_mutations, Sequence) or isinstance(raw_mutations, (str, bytes)):
        raise ValueError("lifecycle review lacks proposed mutations")
    proposed = tuple(_deserialize_mutation(value, source_id, incumbent_id) for value in raw_mutations)
    if not proposed:
        raise ValueError("lifecycle review has no proposed mutations")
    if any(
        mutation.mutation_type
        in {
            LifecycleMutationType.SUPERSEDE_MEMORY,
            LifecycleMutationType.RETIRE_MEMORY,
        }
        for mutation in proposed
    ):
        raise ValueError(
            "stale terminal lifecycle proposal requires source replanning, not Review refresh"
        )

    source_unit_id = _text(scope_payload.get("source_unit_id"), "scope.source_unit_id")
    target_unit_revision_id = _optional_text(scope_payload.get("target_unit_revision_id"))
    observation_revision_ids = tuple(
        str(value) for value in _sequence(stale_payload.get("observation_revision_ids", ()))
    )
    refresh_digest = hashlib.sha256(
        "\x1f".join(
            (
                review.id,
                source_id,
                source_unit_id,
                target_unit_revision_id or "",
                current_support_set_hash,
                current_memory_version,
                *observation_revision_ids,
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    plan_id = f"lifecycle-review-refresh-{refresh_digest}"
    refreshed_review_id = f"review-{refresh_digest}"
    scope = ReconciliationScope(
        id=f"{_text(scope_payload.get('id'), 'scope.id')}:refresh:{refresh_digest}",
        source_id=source_id,
        source_unit_id=source_unit_id,
        base_unit_revision_id=_optional_text(scope_payload.get("base_unit_revision_id")),
        target_unit_revision_id=target_unit_revision_id,
        dependency_unit_ids=tuple(
            str(value) for value in _sequence(scope_payload.get("dependency_unit_ids", ()))
        ),
    )
    staged_evidence = dict(review.staged_evidence)
    staged_evidence["refreshed_from_review_id"] = review.id
    create_review = LifecycleMutation(
        mutation_type=LifecycleMutationType.CREATE_REVIEW,
        memory_id=incumbent_id,
        source_id=source_id,
        payload={
            "review_id": refreshed_review_id,
            "reason": review.reason or "refreshed lifecycle review required",
            "staged_evidence": staged_evidence,
        },
    )
    plan = LifecyclePlan(
        id=plan_id,
        scope=scope,
        gate_state=gate_state,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=(incumbent_id,),
            incumbent_decisions=(
                IncumbentDecision(
                    memory_id=incumbent_id,
                    disposition=IncumbentDisposition.REVIEW,
                    reason=review.reason or "refreshed lifecycle review required",
                ),
            ),
            batch_ids=(f"{scope.id}:batch:0",),
            completed_batch_ids=(f"{scope.id}:batch:0",),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=observation_revision_ids,
            support_set_hashes={incumbent_id: current_support_set_hash},
            memory_versions={incumbent_id: current_memory_version},
        ),
        mutations=(create_review,),
    )
    plan.validate()
    return plan, refreshed_review_id


def _relation_discovery_requests(
    review: LifecycleReview,
    *,
    proposed: tuple[LifecycleMutation, ...],
    plan_id: str,
    source_id: str,
    source_unit_id: str,
    source_unit_revision_id: str | None,
) -> tuple[RelationDiscoveryRequest, ...]:
    activations = {
        mutation.memory_id: mutation
        for mutation in proposed
        if mutation.mutation_type
        in {
            LifecycleMutationType.CREATE_MEMORY,
            LifecycleMutationType.REACTIVATE_MEMORY,
        }
    }
    if not activations:
        return ()
    if len(activations) != 1:
        raise ValueError("lifecycle review relation discovery requires one activated Memory")

    seed = _mapping(
        review.staged_evidence.get("relation_discovery_seed"),
        "relation_discovery_seed",
    )
    memory_id = _text(seed.get("memory_id"), "relation_discovery_seed.memory_id")
    expected_content_hash = _text(
        seed.get("expected_content_hash"),
        "relation_discovery_seed.expected_content_hash",
    )
    activation = activations.get(memory_id)
    if activation is None:
        raise ValueError("relation discovery seed does not identify the activated Memory")
    if activation.mutation_type is LifecycleMutationType.CREATE_MEMORY:
        memory_payload = _mapping(activation.payload.get("memory"), "create_memory.payload.memory")
        activation_content_hash = _text(memory_payload.get("content_hash"), "memory.content_hash")
    else:
        activation_content_hash = _text(
            activation.payload.get("expected_content_hash"),
            "reactivate_memory.expected_content_hash",
        )
    if activation_content_hash != expected_content_hash:
        raise ValueError("relation discovery seed content hash does not match activation")
    if (
        _text(seed.get("source_id"), "relation_discovery_seed.source_id") != source_id
        or _text(seed.get("source_unit_id"), "relation_discovery_seed.source_unit_id") != source_unit_id
        or _optional_text(seed.get("source_unit_revision_id")) != source_unit_revision_id
    ):
        raise ValueError("relation discovery seed belongs to another reconciliation scope")
    entity_ids = tuple(_integer_sequence(seed.get("entity_ids", ()), "relation_discovery_seed.entity_ids"))
    actor_user_id = _optional_text(seed.get("actor_user_id"))
    doc_id = _text(seed.get("doc_id"), "relation_discovery_seed.doc_id")
    return (
        RelationDiscoveryRequest(
            id=relation_discovery_request_id(
                lifecycle_plan_id=plan_id,
                memory_id=memory_id,
                expected_content_hash=expected_content_hash,
            ),
            memory_id=memory_id,
            expected_content_hash=expected_content_hash,
            source_id=source_id,
            source_unit_id=source_unit_id,
            source_unit_revision_id=source_unit_revision_id,
            doc_id=doc_id,
            actor_user_id=actor_user_id,
            entity_ids=entity_ids,
        ),
    )


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
    if (
        mutation_type
        in {
            LifecycleMutationType.REMOVE_SUPPORT,
            LifecycleMutationType.SUPERSEDE_MEMORY,
            LifecycleMutationType.RETIRE_MEMORY,
            LifecycleMutationType.REFRESH_MEMORY_INDEX,
        }
        and memory_id != incumbent_id
    ):
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


def _integer_sequence(value: object, name: str) -> tuple[int, ...]:
    raw = _sequence(value)
    if any(not isinstance(item, int) or isinstance(item, bool) for item in raw):
        raise ValueError(f"{name} must contain integers")
    return tuple(raw)


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
