"""A Lifecycle Review as an earlier version stored it.

No revision raises a Lifecycle Review. One that an earlier version raised held
one old Memory's decided mutations, and the creation of its replacement, as
staged evidence beside the Plan that decided everything else. Such a Review
stays readable and, while pending, decidable, and it contests the exact Support
it staged for removal. Tests of those paths build the stored rows here from the
direct Plan the planner commits today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from memforge.memory.lifecycle_plan import (
    LifecycleMutation,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    lifecycle_plan_to_payload,
    lifecycle_stable_id,
)
from memforge.models import RawMemory, ReconcileAction
from memforge.storage.database import Database


@dataclass(frozen=True, slots=True)
class StoredLifecycleReview:
    """The creating Plan's stored payload and the Review row it held."""

    plan_payload: Mapping[str, object]
    review: LifecycleReview


def held_for_review(
    plan: LifecyclePlan,
    memory_id: str,
    *,
    action: ReconcileAction,
    candidate: RawMemory | None = None,
    replacement_memory_id: str | None = None,
) -> StoredLifecycleReview:
    """Hold one incumbent's decided mutations, and its replacement's creation, in a pending Review."""

    [decision] = [item for item in plan.coverage_proof.incumbent_decisions if item.memory_id == memory_id]
    held_ids = {memory_id} | ({replacement_memory_id} if replacement_memory_id else set())
    held = tuple(mutation for mutation in plan.mutations if mutation.memory_id in held_ids)
    staged: dict[str, object] = {"action": action.value}
    if candidate is not None:
        staged["candidate"] = {"content": candidate.content, "memory_type": candidate.memory_type}
    staged["proposed_disposition"] = decision.disposition.value
    staged["replacement_memory_id"] = replacement_memory_id
    staged["proposed_mutations"] = [_serialize(mutation) for mutation in held]
    if replacement_memory_id is not None:
        [request] = [item for item in plan.relation_discovery_requests if item.memory_id == replacement_memory_id]
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
    payload = lifecycle_plan_to_payload(plan)
    payload["mutations"] = [
        item for item in payload["mutations"] if item["memory_id"] not in held_ids  # type: ignore[index]
    ]
    payload["relation_discovery_requests"] = [
        item
        for item in payload["relation_discovery_requests"]  # type: ignore[union-attr]
        if item["memory_id"] not in held_ids
    ]
    for item in payload["coverage_proof"]["incumbent_decisions"]:  # type: ignore[index]
        if item["memory_id"] == memory_id:
            item["disposition"] = "review"
            item["replacement_memory_id"] = None
    review = LifecycleReview(
        id=lifecycle_stable_id("review", plan.id, memory_id, action.value),
        lifecycle_plan_id=plan.id,
        incumbent_memory_id=memory_id,
        status=LifecycleReviewStatus.PENDING,
        staged_evidence=staged,
        reason=decision.reason,
        source_id=plan.scope.source_id,
    )
    return StoredLifecycleReview(plan_payload=payload, review=review)


async def store_lifecycle_review(db: Database, stored: StoredLifecycleReview) -> LifecycleReview:
    """Write the applied creating Plan and its pending Review, and read the Review back."""

    payload_json = json.dumps(stored.plan_payload, sort_keys=True)
    scope = stored.plan_payload["scope"]
    assert isinstance(scope, Mapping)
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO lifecycle_plans (
               id, reconciliation_scope_id, source_id, source_unit_id,
               target_unit_revision_id, status, payload_json, payload_hash,
               created_at, applied_at
           ) VALUES (?, ?, ?, ?, ?, 'applied', ?, ?, ?, ?)""",
        (
            stored.review.lifecycle_plan_id,
            scope["id"],
            scope["source_id"],
            scope["source_unit_id"],
            scope["target_unit_revision_id"],
            payload_json,
            hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            now,
            now,
        ),
    )
    await db.db.execute(
        """INSERT INTO lifecycle_reviews (
               id, lifecycle_plan_id, incumbent_memory_id, status,
               staged_evidence_json, reason, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            stored.review.id,
            stored.review.lifecycle_plan_id,
            stored.review.incumbent_memory_id,
            stored.review.status.value,
            json.dumps(stored.review.staged_evidence, sort_keys=True),
            stored.review.reason,
            now,
        ),
    )
    await db.db.commit()
    review = await db.get_lifecycle_review(stored.review.id)
    assert review is not None
    return review


def _serialize(mutation: LifecycleMutation) -> dict[str, object]:
    return {
        "mutation_type": mutation.mutation_type.value,
        "memory_id": mutation.memory_id,
        "source_id": mutation.source_id,
        "evidence_unit_ids": list(mutation.evidence_unit_ids),
        "replacement_memory_id": mutation.replacement_memory_id,
        "payload": dict(mutation.payload),
    }
