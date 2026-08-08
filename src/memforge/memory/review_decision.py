"""Stable guards for Review decisions and caller-supplied manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Mapping, Sequence

from memforge.memory.lifecycle_plan import LifecycleReview
from memforge.models import Memory, MemoryReview


DECISION_FINGERPRINT_VERSION = "review-decision-v1"


@dataclass(frozen=True, slots=True)
class ReviewVectorTask:
    """Durable derived-index work owned by a workbench Review decision."""

    id: str
    review_id: str
    memory_id: str
    operation: Literal["upsert", "delete"]
    status: Literal["pending", "completed", "failed"]
    attempts: int = 0
    error: str | None = None


def memory_review_decision_fingerprint(
    review: MemoryReview,
    related_challengers: Sequence[Memory] = (),
) -> str:
    """Bind a workbench decision to the exact Review and participant versions."""

    return _fingerprint(
        {
            "origin": "memory",
            "id": review.id,
            "kind": review.kind,
            "incumbent_memory_id": review.incumbent_memory_id,
            "challenger_memory_id": review.challenger_memory_id,
            "expected_incumbent_updated_at": review.expected_incumbent_updated_at,
            "expected_challenger_updated_at": review.expected_challenger_updated_at,
            "replacement_kind": review.replacement_kind,
            "created_at": review.created_at.isoformat() if review.created_at else None,
            "related_challengers": [
                {
                    "id": memory.id,
                    "content_hash": memory.content_hash,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                }
                for memory in sorted(related_challengers, key=lambda item: item.id)
            ],
        }
    )


def lifecycle_review_decision_fingerprint(review: LifecycleReview) -> str:
    """Bind a projected decision to its exact staged Lifecycle Plan evidence."""

    return _fingerprint(
        {
            "origin": "lifecycle",
            "id": review.id,
            "lifecycle_plan_id": review.lifecycle_plan_id,
            "incumbent_memory_id": review.incumbent_memory_id,
            "staged_evidence": review.staged_evidence,
            "created_at": review.created_at,
            "source_id": review.source_id,
        }
    )


def _fingerprint(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    return f"{DECISION_FINGERPRINT_VERSION}:{digest}"
