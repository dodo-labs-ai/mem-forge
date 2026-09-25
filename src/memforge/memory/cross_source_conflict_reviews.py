"""Read a workspace's Cross-Source Conflict Reviews.

Cross-document discovery records relations and creates no Review (ADR 0037).
The Cross-Source Conflict Reviews a workspace still holds are people's decisions
on Memory pairs: they seed the relation evaluation set and are converted once
into relations and Relation Dismissals.

Both read a decided Review as a relation label: a confirmed Review says the two
statements contradict each other and a dismissed one says both statements hold.
A person may relabel a decided Review by its id, and the evaluation set and the
conversion then take that label.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Protocol

from memforge.memory.cross_document_relation import CrossDocumentRelationLabel
from memforge.models import Memory, MemoryReview, ReviewStatus

CROSS_SOURCE_CONFLICT_REVIEW_KIND = "cross_source_conflict"
_PAGE_SIZE = 100

# A person's decision on a Cross-Source Conflict Review, as a relation label.
REVIEW_DECISION_LABELS = {
    ReviewStatus.APPROVED.value: CrossDocumentRelationLabel.CONTRADICTS,
    ReviewStatus.REJECTED.value: CrossDocumentRelationLabel.NONE,
}


class ReviewLabelOverrideError(ValueError):
    """A relabel names a Review that is not a decided Cross-Source Conflict Review."""


class CrossSourceConflictReviewStore(Protocol):
    async def list_memory_reviews(
        self,
        status: str | None = None,
        kind: str | None = None,
        limit: int = ...,
        offset: int = 0,
    ) -> list[MemoryReview]: ...


async def list_cross_source_conflict_reviews(
    store: CrossSourceConflictReviewStore,
    *,
    status: str | None = None,
) -> list[MemoryReview]:
    """Every Cross-Source Conflict Review, optionally of one status."""

    reviews: list[MemoryReview] = []
    while True:
        page = await store.list_memory_reviews(
            status=status,
            kind=CROSS_SOURCE_CONFLICT_REVIEW_KIND,
            limit=_PAGE_SIZE,
            offset=len(reviews),
        )
        reviews.extend(page)
        if len(page) < _PAGE_SIZE:
            return reviews


def decided_review_labels(
    reviews: Iterable[MemoryReview],
    label_overrides: Mapping[str, CrossDocumentRelationLabel],
) -> dict[str, CrossDocumentRelationLabel]:
    """The relation label of every decided Review, by Review id.

    A decided Review takes its decision's label unless ``label_overrides``
    relabels it. Relabeling a Review that is pending, stale or absent is an error.
    """

    labels = {
        review.id: label_overrides.get(review.id, REVIEW_DECISION_LABELS[review.status])
        for review in reviews
        if review.status in REVIEW_DECISION_LABELS
    }
    unknown = sorted(set(label_overrides) - set(labels))
    if unknown:
        raise ReviewLabelOverrideError("label overrides name no decided cross-source Review: " + ", ".join(unknown))
    return labels


def review_memories_unchanged(review: MemoryReview, *, challenger: Memory, incumbent: Memory) -> bool:
    """Whether both Memories still have the versions the Review was created for.

    A Review records each Memory's ``updated_at``, not its content, so any later
    write to either Memory counts as a change.
    """

    return _same_version(challenger.updated_at, review.expected_challenger_updated_at) and _same_version(
        incumbent.updated_at, review.expected_incumbent_updated_at
    )


def _same_version(updated_at: datetime | None, expected: str | None) -> bool:
    if updated_at is None or not expected:
        return False
    return _utc(updated_at) == _utc(datetime.fromisoformat(expected))


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment
