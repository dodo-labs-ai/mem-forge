"""Facts consumed by canonical caller-scoped Review reads.

Candidate methods are internal reads, not authorized user results. The shared
Review reader applies AccessScope, complete participant/Source visibility and
pinned versions before exposing a queue or navigation link.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from memforge.memory.lifecycle_plan import LifecycleReviewQueueEntry, LifecycleReviewStatus
from memforge.models import Memory, MemoryReview, MemoryReviewRelatedChallenger
from memforge.storage.adapters.context import AccessScope


class ReviewAdminReader(Protocol):
    async def list_memory_reviews(
        self, status: str | None = None, kind: str | None = None,
        limit: int = 100, offset: int = 0,
    ) -> list[MemoryReview]: ...

    async def list_pending_memory_reviews_for_memories(
        self, memory_ids: Sequence[str],
    ) -> list[MemoryReview]:
        """All pending supersede candidates with any requested participant.

        Include related challengers; do not select a newest candidate before
        shared eligibility has been evaluated. Deduplicate Reviews across ids.
        """
        ...

    async def list_lifecycle_review_queue_entries(
        self, source_id: str | None = None, *,
        status: LifecycleReviewStatus | None = None,
        memory_ids: Sequence[str] | None = None,
    ) -> list[LifecycleReviewQueueEntry]:
        """Narrow candidates, optionally restricted to requested incumbents.

        None means no incumbent filter; an empty sequence returns no candidates.
        Staged Evidence remains in storage. Include an existing candidate Memory
        id so the shared reader checks its visibility as well as the incumbent.
        """
        ...

    async def list_memory_review_related_challengers_many(
        self, review_ids: Sequence[str],
    ) -> list[MemoryReviewRelatedChallenger]: ...

    async def list_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]: ...

    async def filter_visible_ids(self, ids: Sequence[str], scope: AccessScope) -> set[str]: ...

    async def get_memory_source_ids_many(
        self, memory_ids: Sequence[str],
    ) -> dict[str, tuple[str, ...]]: ...

    async def list_sources(self) -> list[dict[str, Any]]: ...
