"""Review service - lifecycle-safe approval and rejection of memory reviews.

A ``MemoryReview`` records a human decision point for a quarantined challenger
memory. Approving promotes the challenger and marks the incumbent superseded;
rejecting retires the challenger. Both paths keep SQLite, FTS5, and ChromaDB
in lockstep through the existing ``MemoryStore`` operations.

Optimistic concurrency: each review snapshots the incumbent and challenger
``updated_at`` at creation time. If either has drifted when the review
resolves, the service refuses to mutate and surfaces the drift so the UI can
re-pin expectations or reload.
"""

from __future__ import annotations

from dataclasses import dataclass

from memforge.memory.store import MemoryStore
from memforge.models import (
    Memory,
    MemoryReview,
    ReviewKind,
    ReviewStatus,
)
from memforge.storage.database import Database

__all__ = [
    "ReviewService",
    "ReviewError",
    "ReviewNotFound",
    "ReviewAlreadyResolved",
    "ReviewStaleConflict",
    "ReviewKindUnsupported",
]


class ReviewError(Exception):
    """Base class for review service errors."""


class ReviewNotFound(ReviewError):
    """The requested review id does not exist."""


class ReviewAlreadyResolved(ReviewError):
    """The review is no longer pending (already approved, rejected, or stale)."""

    def __init__(self, review: MemoryReview) -> None:
        super().__init__(f"Review {review.id} is already {review.status}")
        self.review = review


class ReviewStaleConflict(ReviewError):
    """Underlying memories changed since the review was created."""

    def __init__(self, review: MemoryReview, *, incumbent: Memory | None, challenger: Memory | None) -> None:
        super().__init__(f"Review {review.id} is stale")
        self.review = review
        self.incumbent = incumbent
        self.challenger = challenger


class ReviewKindUnsupported(ReviewError):
    """The review kind is not handled by this service version."""


@dataclass
class ResolvedReview:
    """The result of an approve/reject call."""
    review: MemoryReview
    incumbent: Memory | None
    challenger: Memory | None


class ReviewService:
    """Resolve memory reviews with the same index discipline as the sync pipeline."""

    def __init__(self, db: Database, memory_store: MemoryStore) -> None:
        self.db = db
        self.memory_store = memory_store

    async def approve(
        self,
        review_id: str,
        *,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> ResolvedReview:
        review = await self._load_pending(review_id)
        incumbent, challenger = await self._load_pair(review)
        related_challengers = await self._load_related_challengers(review)
        self._guard_supersede(review, incumbent, challenger)
        await self._guard_fresh(review, incumbent, challenger)

        context = self.memory_store.operation_context()
        retired_related: list[Memory] = []
        await self.memory_store.promote_quarantined_challenger(
            incumbent=incumbent,
            challenger=challenger,
            replacement_reason=review.reason,
            review_id=review.id,
            context=context,
        )
        try:
            for related in related_challengers:
                if related.status != "pending_review":
                    continue
                await self.memory_store.retire_memory(
                    related.id,
                    reason="review_redundant",
                    context=context,
                    review_id=review.id,
                )
                retired_related.append(related)
        except Exception:
            await self.memory_store.restore_review_transition(
                incumbent=incumbent,
                challenger=challenger,
                context=context,
                review_id=review.id,
                reason="review_approve_resolution_rollback",
            )
            for related in retired_related:
                await self.memory_store.restore_review_transition(
                    incumbent=None,
                    challenger=related,
                    context=context,
                    review_id=review.id,
                    reason="review_related_rollback",
                )
            raise

        try:
            await self.db.resolve_memory_review(
                review_id,
                status=ReviewStatus.APPROVED.value,
                reviewer=reviewer,
                review_note=note,
            )
        except Exception as exc:
            await self.memory_store.restore_review_transition(
                incumbent=incumbent,
                challenger=challenger,
                context=context,
                review_id=review.id,
                reason="review_approve_resolution_rollback",
            )
            for related in retired_related:
                await self.memory_store.restore_review_transition(
                    incumbent=None,
                    challenger=related,
                    context=context,
                    review_id=review.id,
                    reason="review_related_rollback",
                )
            await self.memory_store.record_review_decision(
                "review_resolution_failed",
                memory_id=challenger.id,
                review_id=review.id,
                reviewer=reviewer,
                reason=review.reason,
                context=context,
                payload={"target_status": ReviewStatus.APPROVED.value},
                error=str(exc),
            )
            raise
        await self.memory_store.record_review_decision(
            "review_approved",
            memory_id=challenger.id,
            review_id=review.id,
            reviewer=reviewer,
            reason=review.reason,
            context=context,
            payload={"incumbent_memory_id": incumbent.id},
        )
        return ResolvedReview(
            review=await self.db.get_memory_review(review_id),  # type: ignore[arg-type]
            incumbent=await self.db.get_memory(incumbent.id),
            challenger=await self.db.get_memory(challenger.id),
        )

    async def reject(
        self,
        review_id: str,
        *,
        reviewer: str | None = None,
        note: str | None = None,
    ) -> ResolvedReview:
        review = await self._load_pending(review_id)
        if not note or not note.strip():
            raise ReviewError("A note is required when rejecting a review")
        incumbent, challenger = await self._load_pair(review)
        related_challengers = await self._load_related_challengers(review)
        self._guard_supersede(review, incumbent, challenger)
        await self._guard_fresh(review, incumbent, challenger)

        context = self.memory_store.operation_context()
        retired_related: list[Memory] = []
        await self.memory_store.retire_memory(
            challenger.id,
            reason="rejected",
            context=context,
            review_id=review.id,
        )
        try:
            for related in related_challengers:
                if related.status != "pending_review":
                    continue
                await self.memory_store.retire_memory(
                    related.id,
                    reason="rejected",
                    context=context,
                    review_id=review.id,
                )
                retired_related.append(related)
        except Exception:
            await self.memory_store.restore_review_transition(
                incumbent=None,
                challenger=challenger,
                context=context,
                review_id=review.id,
                reason="review_reject_resolution_rollback",
            )
            for related in retired_related:
                await self.memory_store.restore_review_transition(
                    incumbent=None,
                    challenger=related,
                    context=context,
                    review_id=review.id,
                    reason="review_related_rollback",
                )
            raise

        try:
            await self.db.resolve_memory_review(
                review_id,
                status=ReviewStatus.REJECTED.value,
                reviewer=reviewer,
                review_note=note,
            )
        except Exception as exc:
            await self.memory_store.restore_review_transition(
                incumbent=None,
                challenger=challenger,
                context=context,
                review_id=review.id,
                reason="review_reject_resolution_rollback",
            )
            for related in retired_related:
                await self.memory_store.restore_review_transition(
                    incumbent=None,
                    challenger=related,
                    context=context,
                    review_id=review.id,
                    reason="review_related_rollback",
                )
            await self.memory_store.record_review_decision(
                "review_resolution_failed",
                memory_id=challenger.id,
                review_id=review.id,
                reviewer=reviewer,
                reason="rejected",
                context=context,
                payload={"target_status": ReviewStatus.REJECTED.value},
                error=str(exc),
            )
            raise
        await self.memory_store.record_review_decision(
            "review_rejected",
            memory_id=challenger.id,
            review_id=review.id,
            reviewer=reviewer,
            reason="rejected",
            context=context,
        )
        return ResolvedReview(
            review=await self.db.get_memory_review(review_id),  # type: ignore[arg-type]
            incumbent=await self.db.get_memory(incumbent.id),
            challenger=await self.db.get_memory(challenger.id),
        )

    async def refresh(self, review_id: str) -> ResolvedReview:
        """Re-pin a review's expected timestamps to the current memories.

        Useful when the incumbent or challenger drifted after a sync and the
        reviewer wants to retake the decision against the updated state.
        """
        review = await self.db.get_memory_review(review_id)
        if review is None:
            raise ReviewNotFound(review_id)
        incumbent, challenger = await self._load_pair(review)
        self._guard_supersede(review, incumbent, challenger)

        if challenger.status not in ("pending_review",) or incumbent.status != "active":
            # The lifecycle has already moved past where the review can apply.
            raise ReviewError(
                f"Cannot refresh review {review.id}: incumbent status is "
                f"{incumbent.status!r}, challenger status is {challenger.status!r}"
            )

        await self.db.refresh_memory_review_expectations(
            review_id,
            expected_incumbent_updated_at=(
                incumbent.updated_at.isoformat() if incumbent.updated_at else None
            ),
            expected_challenger_updated_at=(
                challenger.updated_at.isoformat() if challenger.updated_at else None
            ),
        )
        return ResolvedReview(
            review=await self.db.get_memory_review(review_id),  # type: ignore[arg-type]
            incumbent=incumbent,
            challenger=challenger,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _load_pending(self, review_id: str) -> MemoryReview:
        review = await self.db.get_memory_review(review_id)
        if review is None:
            raise ReviewNotFound(review_id)
        if review.status != ReviewStatus.PENDING.value:
            raise ReviewAlreadyResolved(review)
        return review

    async def _load_pair(self, review: MemoryReview) -> tuple[Memory, Memory]:
        incumbent = await self.db.get_memory(review.incumbent_memory_id)
        challenger = await self.db.get_memory(review.challenger_memory_id)
        if incumbent is None or challenger is None:
            raise ReviewError(
                f"Review {review.id} references missing memories "
                f"(incumbent={'present' if incumbent else 'missing'}, "
                f"challenger={'present' if challenger else 'missing'})"
            )
        # ``Database.get_memory`` does not populate ``entity_refs`` (entities
        # are linked through ``memory_entities``). Hydrate from the join so
        # downstream embedding text and any caller depending on entity_refs
        # see the same coverage that was set at extraction time.
        incumbent.entity_refs = await self.db.get_memory_entity_names(incumbent.id)
        challenger.entity_refs = await self.db.get_memory_entity_names(challenger.id)
        return incumbent, challenger

    async def _load_related_challengers(self, review: MemoryReview) -> list[Memory]:
        related: list[Memory] = []
        for row in await self.db.list_memory_review_related_challengers(review.id):
            challenger = await self.db.get_memory(row.challenger_memory_id)
            if challenger is None:
                continue
            challenger.entity_refs = await self.db.get_memory_entity_names(challenger.id)
            related.append(challenger)
        return related

    def _guard_supersede(
        self,
        review: MemoryReview,
        incumbent: Memory,
        challenger: Memory,
    ) -> None:
        if review.kind != ReviewKind.SUPERSEDE.value:
            raise ReviewKindUnsupported(
                f"Review kind {review.kind!r} is not supported in this version"
            )
        if challenger.status != "pending_review":
            raise ReviewError(
                f"Challenger {challenger.id} has status {challenger.status!r}; "
                f"expected pending_review"
            )
        if incumbent.status != "active":
            raise ReviewError(
                f"Incumbent {incumbent.id} has status {incumbent.status!r}; "
                f"expected active"
            )

    async def _guard_fresh(
        self,
        review: MemoryReview,
        incumbent: Memory,
        challenger: Memory,
    ) -> None:
        actual_incumbent = incumbent.updated_at.isoformat() if incumbent.updated_at else None
        actual_challenger = challenger.updated_at.isoformat() if challenger.updated_at else None

        incumbent_drift = (
            review.expected_incumbent_updated_at is not None
            and review.expected_incumbent_updated_at != actual_incumbent
        )
        challenger_drift = (
            review.expected_challenger_updated_at is not None
            and review.expected_challenger_updated_at != actual_challenger
        )

        if incumbent_drift or challenger_drift:
            await self.db.resolve_memory_review(
                review.id,
                status=ReviewStatus.STALE.value,
                reviewer=review.reviewer,
                review_note=review.review_note,
            )
            raise ReviewStaleConflict(review, incumbent=incumbent, challenger=challenger)
