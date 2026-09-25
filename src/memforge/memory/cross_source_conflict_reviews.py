"""Read a workspace's Cross-Source Conflict Reviews.

Cross-document discovery records relations and creates no Review (ADR 0037).
The Cross-Source Conflict Reviews a workspace still holds are people's decisions
on Memory pairs: they seed the relation evaluation set and are converted once
into relations and Relation Dismissals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from memforge.models import Memory, MemoryReview

CROSS_SOURCE_CONFLICT_REVIEW_KIND = "cross_source_conflict"
_PAGE_SIZE = 100


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
