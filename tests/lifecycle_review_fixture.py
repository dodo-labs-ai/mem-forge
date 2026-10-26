"""A Lifecycle Plan that holds one old Memory's decision for Review.

A LifecycleReview stays readable and decidable: it is listed, approved,
rejected and refreshed, and it contests the exact Support it stages for
removal. Tests of those paths build the creating Plan here from the direct Plan
the planner would commit, so they do not depend on which path raises a Review.
"""

from __future__ import annotations

from dataclasses import replace

from memforge.memory.lifecycle_plan import (
    IncumbentDisposition,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    lifecycle_stable_id,
)
from memforge.models import RawMemory, ReconcileAction


def hold_for_review(
    plan: LifecyclePlan,
    memory_id: str,
    *,
    action: ReconcileAction,
    candidate: RawMemory | None = None,
    replacement_memory_id: str | None = None,
) -> LifecyclePlan:
    """Move one incumbent's decided mutations, and its replacement's creation, into a pending Review."""

    [decision] = [item for item in plan.coverage_proof.incumbent_decisions if item.memory_id == memory_id]
    held_ids = {memory_id} | ({replacement_memory_id} if replacement_memory_id else set())
    held = tuple(mutation for mutation in plan.mutations if mutation.memory_id in held_ids)
    staged: dict[str, object] = {"action": action.value}
    if candidate is not None:
        staged["candidate"] = {"content": candidate.content, "memory_type": candidate.memory_type}
    staged["proposed_disposition"] = decision.disposition.value
    staged["replacement_memory_id"] = replacement_memory_id
    staged["proposed_mutations"] = [_serialize(mutation) for mutation in held]
    requests = {request.memory_id: request for request in plan.relation_discovery_requests}
    if replacement_memory_id is not None:
        request = requests[replacement_memory_id]
        staged["relation_discovery_seed"] = {
            "memory_id": request.memory_id,
            "expected_content_hash": request.expected_content_hash,
            "source_id": request.source_id,
            "source_unit_id": request.source_unit_id,
            "source_unit_revision_id": request.source_unit_revision_id,
            "doc_id": request.doc_id,
            "actor_user_id": request.actor_user_id,
            "entity_ids": list(request.entity_ids),
        }
    review = LifecycleMutation(
        LifecycleMutationType.CREATE_REVIEW,
        memory_id=memory_id,
        source_id=plan.scope.source_id,
        payload={
            "review_id": lifecycle_stable_id("review", plan.id, memory_id, action.value),
            "reason": decision.reason,
            "staged_evidence": staged,
        },
    )
    held_plan = replace(
        plan,
        coverage_proof=replace(
            plan.coverage_proof,
            incumbent_decisions=tuple(
                replace(item, disposition=IncumbentDisposition.REVIEW, replacement_memory_id=None)
                if item.memory_id == memory_id
                else item
                for item in plan.coverage_proof.incumbent_decisions
            ),
        ),
        mutations=(*(mutation for mutation in plan.mutations if mutation.memory_id not in held_ids), review),
        relation_discovery_requests=tuple(
            request for request in plan.relation_discovery_requests if request.memory_id not in held_ids
        ),
    )
    held_plan.validate()
    return held_plan


def _serialize(mutation: LifecycleMutation) -> dict[str, object]:
    return {
        "mutation_type": mutation.mutation_type.value,
        "memory_id": mutation.memory_id,
        "source_id": mutation.source_id,
        "evidence_unit_ids": list(mutation.evidence_unit_ids),
        "replacement_memory_id": mutation.replacement_memory_id,
        "payload": dict(mutation.payload),
    }
