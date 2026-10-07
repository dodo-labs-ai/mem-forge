"""Caller-scoped Review reads shared by the queue and Memory navigation.

Adapters return candidates and facts; this module decides visibility and open
membership before ordering, pagination or choosing one link. Reads never change
Review status or weaken the decision path's transactional stale guards.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from memforge.memory.lifecycle_plan import LifecycleReviewQueueEntry, LifecycleReviewStatus
from memforge.models import Memory, MemoryReview, MemoryStatus, ReviewKind, ReviewStatus
from memforge.source_access import source_is_discoverable
from memforge.storage.adapters.context import AccessScope
from memforge.storage.admin_review import ReviewAdminReader

REVIEW_QUEUE_READ_SIZE = 500
"""Storage page size for complete queue enumeration, not a business coverage cap."""


def _dt_iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def review_access_scope(user_id: str) -> AccessScope:
    """Review participants span lifecycle statuses independently of page filters."""
    return AccessScope(
        user_id=user_id, include_private=True,
        allowed_statuses=("active", "pending_review", "superseded", "retired"),
        active_project=None, scope_mode="project-first",
    )


def review_is_visible(
    participant_ids: Sequence[str], visible_ids: set[str],
    sources: Sequence[Mapping[str, Any] | None], *, scope: AccessScope,
) -> bool:
    """A Review exposes its whole participant/Source set or none of it."""
    return visible_ids.issuperset(participant_ids) and all(
        source is not None and source_is_discoverable(source, viewer_id=scope.user_id)
        for source in sources
    )


def is_memory_review_stale(review: MemoryReview, incumbent: Memory | None, challenger: Memory | None) -> bool:
    """Detect drift between the review's pinned timestamps and current memories."""
    if review.status != "pending":
        return False
    if incumbent is None or challenger is None:
        return True
    actual_incumbent = incumbent.updated_at.isoformat() if incumbent.updated_at else None
    actual_challenger = challenger.updated_at.isoformat() if challenger.updated_at else None
    if review.expected_incumbent_updated_at is not None and review.expected_incumbent_updated_at != actual_incumbent:
        return True
    if review.expected_challenger_updated_at is not None and review.expected_challenger_updated_at != actual_challenger:
        return True
    return False

@dataclass(frozen=True)
class QueuedMemoryReview:
    review: MemoryReview
    related_challenger_ids: tuple[str, ...]
    sources: tuple[Mapping[str, Any], ...]
    """The Sources behind every participant, each one discoverable by the caller."""

    @property
    def sort_key(self) -> tuple[str, str]:
        return _dt_iso(self.review.created_at) or "", self.review.id


@dataclass(frozen=True)
class QueuedLifecycleReview:
    entry: LifecycleReviewQueueEntry
    source: Mapping[str, Any]

    @property
    def sort_key(self) -> tuple[str, str]:
        return self.entry.created_at or "", self.entry.id


@dataclass(frozen=True)
class ReviewQueue:
    """Every Review the caller may see in one queue view, newest first."""

    entries: list[QueuedMemoryReview | QueuedLifecycleReview]
    memories: Mapping[str, Memory]
    """The Memories already read to build the queue, reused when a page is shown."""


def _lifecycle_queue_status(status: str | None) -> LifecycleReviewStatus | None:
    if status is None:
        return None
    return LifecycleReviewStatus.PENDING if status == "open" else LifecycleReviewStatus(status)


async def _memory_reviews_in_queue(reader: ReviewAdminReader, status: str | None) -> list[MemoryReview]:
    """The Memory Reviews a queue status can show; ``stale`` includes pending Reviews that have drifted."""
    reviews: list[MemoryReview] = []
    for review_status in (ReviewStatus.PENDING.value, ReviewStatus.STALE.value) if status == "stale" else (status,):
        offset = 0
        while True:
            chunk = await reader.list_memory_reviews(
                status=review_status,
                kind=ReviewKind.SUPERSEDE.value,
                limit=REVIEW_QUEUE_READ_SIZE,
                offset=offset,
            )
            reviews.extend(chunk)
            if len(chunk) < REVIEW_QUEUE_READ_SIZE:
                break
            offset += len(chunk)
    return reviews


async def read_review_queue(
    reader: ReviewAdminReader,
    *,
    scope: AccessScope,
    status: str | None,
    origin: Literal["memory", "lifecycle"] | None,
    source_id: str | None,
) -> ReviewQueue:
    """Order and filter one queue view from narrow reads.

    A Review is shown when the caller can see each of its Memories and
    discover each Source behind them. Staged Lifecycle evidence and Memory
    provenance are left for the page.
    """
    memory_reviews = await _memory_reviews_in_queue(reader, status) if origin in {None, "memory"} else []
    lifecycle_entries = (
        await reader.list_lifecycle_review_queue_entries(source_id, status=_lifecycle_queue_status(status))
        if origin in {None, "lifecycle"}
        else []
    )
    return await filter_review_queue(
        reader, scope=scope, memory_reviews=memory_reviews,
        lifecycle_entries=lifecycle_entries, status=status, source_id=source_id,
    )


async def filter_review_queue(
    reader: ReviewAdminReader,
    *,
    scope: AccessScope,
    memory_reviews: Sequence[MemoryReview],
    lifecycle_entries: Sequence[LifecycleReviewQueueEntry],
    status: str | None = None,
    source_id: str | None = None,
) -> ReviewQueue:
    """Apply the same participant, Source and version rules to every Review read."""
    related_by_review: dict[str, list[str]] = {}
    if memory_reviews:
        for related in await reader.list_memory_review_related_challengers_many([review.id for review in memory_reviews]):
            related_by_review.setdefault(related.review_id, []).append(related.challenger_memory_id)
    memory_review_participants = {
        review.id: (
            review.incumbent_memory_id,
            review.challenger_memory_id,
            *related_by_review.get(review.id, ()),
        )
        for review in memory_reviews
    }

    # Staleness compares a pending Memory Review with its Memories' versions, and
    # a Lifecycle candidate takes part only once it exists as a Memory.
    compared_ids = tuple(
        dict.fromkeys(
            (
                *(
                    memory_id
                    for review in memory_reviews
                    if review.status == ReviewStatus.PENDING.value
                    for memory_id in (review.incumbent_memory_id, review.challenger_memory_id)
                ),
                *(entry.candidate_memory_id for entry in lifecycle_entries if entry.candidate_memory_id),
            )
        )
    )
    memories = {memory.id: memory for memory in await reader.list_memories_by_ids(compared_ids)} if compared_ids else {}
    lifecycle_participants = {
        entry.id: (
            entry.incumbent_memory_id,
            *((entry.candidate_memory_id,) if entry.candidate_memory_id in memories else ()),
        )
        for entry in lifecycle_entries
    }
    participant_ids = tuple(
        dict.fromkeys(
            memory_id
            for participants in (*memory_review_participants.values(), *lifecycle_participants.values())
            for memory_id in participants
        )
    )
    visible_ids = (
        await reader.filter_visible_ids(participant_ids, scope) if participant_ids else set()
    )
    memory_source_ids = (
        await reader.get_memory_source_ids_many(participant_ids)
        if participant_ids
        else {}
    )
    sources_by_id = {str(source["id"]): source for source in await reader.list_sources()}

    entries: list[QueuedMemoryReview | QueuedLifecycleReview] = []
    for review in memory_reviews:
        participants = memory_review_participants[review.id]
        review_source_ids = tuple(
            dict.fromkeys(
                review_source_id
                for memory_id in participants
                for review_source_id in memory_source_ids.get(memory_id, ())
            )
        )
        if source_id is not None and source_id not in review_source_ids:
            continue
        sources = tuple(sources_by_id.get(review_source_id) for review_source_id in review_source_ids)
        if not review_is_visible(participants, visible_ids, sources, scope=scope):
            continue
        stale = is_memory_review_stale(
            review,
            memories.get(review.incumbent_memory_id),
            memories.get(review.challenger_memory_id),
        )
        if status == "open" and stale:
            continue
        if status == "stale" and review.status != ReviewStatus.STALE.value and not stale:
            continue
        entries.append(QueuedMemoryReview(review, tuple(related_by_review.get(review.id, ())), sources))
    for entry in lifecycle_entries:
        source = sources_by_id.get(entry.source_id or "")
        participants = lifecycle_participants[entry.id]
        source_ids = tuple(dict.fromkeys((
            entry.source_id or "",
            *(source_id for memory_id in participants for source_id in memory_source_ids.get(memory_id, ())),
        )))
        sources = tuple(sources_by_id.get(source_id) for source_id in source_ids)
        if not review_is_visible(participants, visible_ids, sources, scope=scope):
            continue
        entries.append(QueuedLifecycleReview(entry, source))
    entries.sort(key=lambda queued: queued.sort_key, reverse=True)
    return ReviewQueue(entries=entries, memories=memories)


async def read_open_review_ids(
    reader: ReviewAdminReader,
    memories: Sequence[Memory],
    *,
    scope: AccessScope,
) -> Mapping[str, str]:
    """Choose the newest visible open Review for each waiting Memory on a page.

    Candidate reads cover every pending Review of the requested Memories, so a
    hidden or stale newer candidate cannot displace an older eligible one.
    Review participants may have different lifecycle statuses from the page.
    """
    waiting = {memory.id for memory in memories if memory.status == MemoryStatus.PENDING_REVIEW.value}
    if not waiting:
        return {}
    memory_reviews = await reader.list_pending_memory_reviews_for_memories(tuple(waiting))
    lifecycle_entries = await reader.list_lifecycle_review_queue_entries(
        status=LifecycleReviewStatus.PENDING, memory_ids=tuple(waiting),
    )
    review_scope = review_access_scope(scope.user_id)
    queue = await filter_review_queue(
        reader, scope=review_scope, memory_reviews=memory_reviews,
        lifecycle_entries=lifecycle_entries, status="open",
    )
    result: dict[str, str] = {}
    for queued in queue.entries:
        if isinstance(queued, QueuedMemoryReview):
            ids = (queued.review.incumbent_memory_id, queued.review.challenger_memory_id, *queued.related_challenger_ids)
            review_id = queued.review.id
        else:
            ids = (queued.entry.incumbent_memory_id,)
            review_id = queued.entry.id
        for memory_id in waiting.intersection(ids):
            result.setdefault(memory_id, review_id)
    return result
