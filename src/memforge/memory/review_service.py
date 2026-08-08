"""Review service - lifecycle-safe approval and rejection of memory reviews.

A ``MemoryReview`` records a human decision point for a quarantined challenger
memory. Approving promotes the challenger and marks the incumbent superseded;
rejecting retires the challenger. The Review CAS, Memory rows, and relational
search projection commit in one database transaction. The vector projection is
published as Review-owned durable work and reconciled from that committed truth;
a projection failure is retried and never compensated by reversing the lifecycle
decision.

Optimistic concurrency: each review snapshots the incumbent and challenger
``updated_at`` at creation time. If either has drifted when the review
resolves, the service refuses to mutate and surfaces the drift so the UI can
re-pin expectations or reload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from memforge.memory.store import MemoryStore
from memforge.memory.review_decision import memory_review_decision_fingerprint
from memforge.models import (
    Memory,
    MemoryReview,
    MemoryReviewRelatedChallenger,
    ReviewKind,
    ReviewStatus,
)

__all__ = [
    "ReviewService",
    "ReviewError",
    "ReviewNotFound",
    "ReviewAlreadyResolved",
    "ReviewStaleConflict",
    "ReviewKindUnsupported",
    "ReviewResolutionStore",
]


class ReviewResolutionStore(Protocol):
    """Narrow shared persistence contract for one workbench Review decision."""

    async def get_memory_review(self, review_id: str) -> MemoryReview | None: ...
    async def get_memory(self, memory_id: str) -> Memory | None: ...
    async def get_memory_entity_names(self, memory_id: str) -> list[str]: ...
    async def list_memory_review_related_challengers(
        self,
        review_id: str,
    ) -> list[MemoryReviewRelatedChallenger]: ...
    async def resolve_memory_review(
        self,
        review_id: str,
        *,
        status: str,
        reviewer: str | None,
        review_note: str | None,
    ) -> None: ...
    async def apply_memory_review_resolution(
        self,
        review: MemoryReview,
        *,
        status: str,
        reviewer: str | None,
        review_note: str | None,
        incumbent: Memory,
        challenger: Memory,
        related_challengers: Sequence[Memory],
    ) -> None: ...


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

    def __init__(self, db: ReviewResolutionStore, memory_store: MemoryStore) -> None:
        self.db = db
        self.memory_store = memory_store

    async def approve(
        self,
        review_id: str,
        *,
        reviewer: str | None = None,
        note: str | None = None,
        expected_fingerprint: str,
    ) -> ResolvedReview:
        review = await self._load_pending(review_id)
        incumbent, challenger = await self._load_pair(review)
        related_challengers = await self._load_related_challengers(review)
        self._guard_decision_fingerprint(
            review,
            related_challengers,
            expected_fingerprint,
            incumbent=incumbent,
            challenger=challenger,
        )
        if review.kind == ReviewKind.CROSS_SOURCE_CONFLICT.value:
            return await self._resolve_cross_source_finding(
                review,
                incumbent=incumbent,
                challenger=challenger,
                status=ReviewStatus.APPROVED,
                reviewer=reviewer,
                note=note,
            )
        self._guard_supersede(review, incumbent, challenger)
        await self._guard_fresh(review, incumbent, challenger)
        self._guard_related_pending(review, related_challengers, incumbent, challenger)
        context = self.memory_store.operation_context()
        await self._apply_atomic_resolution(
            review,
            status=ReviewStatus.APPROVED,
            reviewer=reviewer,
            note=note,
            incumbent=incumbent,
            challenger=challenger,
            related_challengers=related_challengers,
            context=context,
        )
        vector_sync = await self._reconcile_committed_vectors(
            review,
            reviewer=reviewer,
            context=context,
            memory_ids=(incumbent.id, challenger.id, *(item.id for item in related_challengers)),
        )
        await self.memory_store.record_audit_event(
            "memory_supersede_committed",
            "committed",
            context=context,
            memory_id=incumbent.id,
            candidate_id=challenger.id,
            review_id=review.id,
            actor_id=reviewer,
            reason=review.reason,
            payload={"old_memory_id": incumbent.id, "new_memory_id": challenger.id},
        )
        await self.memory_store.record_review_decision(
            "review_approved",
            memory_id=challenger.id,
            review_id=review.id,
            reviewer=reviewer,
            reason=review.reason,
            context=context,
            payload={"incumbent_memory_id": incumbent.id, "vector_sync": vector_sync},
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
        expected_fingerprint: str,
    ) -> ResolvedReview:
        review = await self._load_pending(review_id)
        if not note or not note.strip():
            raise ReviewError("A note is required when rejecting a review")
        incumbent, challenger = await self._load_pair(review)
        related_challengers = await self._load_related_challengers(review)
        self._guard_decision_fingerprint(
            review,
            related_challengers,
            expected_fingerprint,
            incumbent=incumbent,
            challenger=challenger,
        )
        if review.kind == ReviewKind.CROSS_SOURCE_CONFLICT.value:
            return await self._resolve_cross_source_finding(
                review,
                incumbent=incumbent,
                challenger=challenger,
                status=ReviewStatus.REJECTED,
                reviewer=reviewer,
                note=note,
            )
        self._guard_supersede(review, incumbent, challenger)
        await self._guard_fresh(review, incumbent, challenger)
        self._guard_related_pending(review, related_challengers, incumbent, challenger)
        context = self.memory_store.operation_context()
        await self._apply_atomic_resolution(
            review,
            status=ReviewStatus.REJECTED,
            reviewer=reviewer,
            note=note,
            incumbent=incumbent,
            challenger=challenger,
            related_challengers=related_challengers,
            context=context,
        )
        vector_sync = await self._reconcile_committed_vectors(
            review,
            reviewer=reviewer,
            context=context,
            memory_ids=(challenger.id, *(item.id for item in related_challengers)),
        )
        await self.memory_store.record_review_decision(
            "review_rejected",
            memory_id=challenger.id,
            review_id=review.id,
            reviewer=reviewer,
            reason="rejected",
            context=context,
            payload={"vector_sync": vector_sync},
        )
        return ResolvedReview(
            review=await self.db.get_memory_review(review_id),  # type: ignore[arg-type]
            incumbent=await self.db.get_memory(incumbent.id),
            challenger=await self.db.get_memory(challenger.id),
        )

    @staticmethod
    def _guard_decision_fingerprint(
        review: MemoryReview,
        related_challengers: tuple[Memory, ...],
        expected_fingerprint: str,
        *,
        incumbent: Memory,
        challenger: Memory,
    ) -> None:
        if memory_review_decision_fingerprint(review, related_challengers) != expected_fingerprint:
            raise ReviewStaleConflict(
                review,
                incumbent=incumbent,
                challenger=challenger,
            )

    @staticmethod
    def _guard_related_pending(
        review: MemoryReview,
        related_challengers: tuple[Memory, ...],
        incumbent: Memory,
        challenger: Memory,
    ) -> None:
        if any(memory.status != "pending_review" for memory in related_challengers):
            raise ReviewStaleConflict(
                review,
                incumbent=incumbent,
                challenger=challenger,
            )

    async def _apply_atomic_resolution(
        self,
        review: MemoryReview,
        *,
        status: ReviewStatus,
        reviewer: str | None,
        note: str | None,
        incumbent: Memory,
        challenger: Memory,
        related_challengers: tuple[Memory, ...],
        context,
    ) -> None:
        try:
            await self.db.apply_memory_review_resolution(
                review,
                status=status.value,
                reviewer=reviewer,
                review_note=note,
                incumbent=incumbent,
                challenger=challenger,
                related_challengers=related_challengers,
            )
        except Exception as exc:
            await self.memory_store.record_review_decision(
                "review_resolution_failed",
                memory_id=challenger.id,
                review_id=review.id,
                reviewer=reviewer,
                reason=review.reason,
                context=context,
                payload={"target_status": status.value},
                error=str(exc),
            )
            if not isinstance(exc, ValueError):
                raise
            if "active source support" in str(exc):
                raise ReviewError(str(exc)) from exc
            current = await self.db.get_memory_review(review.id)
            if current is None:
                raise ReviewNotFound(review.id) from exc
            if current.status != ReviewStatus.PENDING.value:
                raise ReviewAlreadyResolved(current) from exc
            raise ReviewStaleConflict(
                current,
                incumbent=await self.db.get_memory(incumbent.id),
                challenger=await self.db.get_memory(challenger.id),
            ) from exc

    async def _reconcile_committed_vectors(
        self,
        review: MemoryReview,
        *,
        reviewer: str | None,
        context,
        memory_ids: tuple[str, ...],
    ) -> str:
        try:
            delivery = await self.memory_store.attempt_review_vector_delivery(review.id)
        except Exception as exc:
            await self.memory_store.record_review_decision(
                "review_vector_sync_failed",
                memory_id=review.challenger_memory_id,
                review_id=review.id,
                reviewer=reviewer,
                reason=review.reason,
                context=context,
                payload={"memory_ids": list(memory_ids)},
                error=str(exc),
            )
            return "durable_retry_pending"
        if delivery.pending:
            await self.memory_store.record_review_decision(
                "review_vector_sync_failed",
                memory_id=review.challenger_memory_id,
                review_id=review.id,
                reviewer=reviewer,
                reason=review.reason,
                context=context,
                payload={
                    "memory_ids": list(memory_ids),
                    "error_types": list(delivery.error_types),
                },
                error="durable Review vector delivery remains pending",
            )
            return "durable_retry_pending"
        return "synchronized"

    async def _resolve_cross_source_finding(
        self,
        review: MemoryReview,
        *,
        incumbent: Memory,
        challenger: Memory,
        status: ReviewStatus,
        reviewer: str | None,
        note: str | None,
    ) -> ResolvedReview:
        """Acknowledge or dismiss a cross-source finding without lifecycle mutation."""
        await self._guard_fresh(review, incumbent, challenger)
        await self._resolve_pending_review(
            review.id,
            status=status.value,
            reviewer=reviewer,
            review_note=note,
        )
        await self.memory_store.record_review_decision(
            "cross_source_review_resolved",
            memory_id=challenger.id,
            review_id=review.id,
            reviewer=reviewer,
            reason=review.reason,
            context=self.memory_store.operation_context(),
            payload={
                "incumbent_memory_id": incumbent.id,
                "resolution": status.value,
                "destructive_action": False,
            },
        )
        return ResolvedReview(
            review=await self.db.get_memory_review(review.id),  # type: ignore[arg-type]
            incumbent=await self.db.get_memory(incumbent.id),
            challenger=await self.db.get_memory(challenger.id),
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
            raise ReviewKindUnsupported(f"Review kind {review.kind!r} is not supported in this version")
        if challenger.status != "pending_review":
            raise ReviewError(f"Challenger {challenger.id} has status {challenger.status!r}; expected pending_review")
        if incumbent.status != "active":
            raise ReviewError(f"Incumbent {incumbent.id} has status {incumbent.status!r}; expected active")

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
            await self._resolve_pending_review(
                review.id,
                status=ReviewStatus.STALE.value,
                reviewer=review.reviewer,
                review_note=review.review_note,
            )
            raise ReviewStaleConflict(review, incumbent=incumbent, challenger=challenger)

    async def _resolve_pending_review(
        self,
        review_id: str,
        *,
        status: str,
        reviewer: str | None,
        review_note: str | None,
    ) -> None:
        try:
            await self.db.resolve_memory_review(
                review_id,
                status=status,
                reviewer=reviewer,
                review_note=review_note,
            )
        except ValueError as exc:
            current = await self.db.get_memory_review(review_id)
            if current is not None and current.status != ReviewStatus.PENDING.value:
                raise ReviewAlreadyResolved(current) from exc
            raise
