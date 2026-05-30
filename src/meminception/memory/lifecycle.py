"""Lean lifecycle helpers for memory state transitions."""

from __future__ import annotations

from meminception.models import MemoryStatus, ReconcileAction, ReconcileOperation

RETIRED_REASONS = {
    "source_deleted",
    "expired",
    "admin_hidden",
    "rejected",
    "privacy_removed",
    "no_support",
}


def normalize_memory_status(status: str | MemoryStatus) -> str:
    """Return the canonical stored status.

    ``decayed`` remains accepted as a compatibility alias, but new writes store
    ``retired`` because the status means hidden/retired, not ranking decay.
    """
    raw = status.value if isinstance(status, MemoryStatus) else str(status)
    return "retired" if raw == "decayed" else raw


def allowed_search_statuses(include_superseded: bool = False) -> tuple[str, ...]:
    """Statuses visible to normal agent retrieval."""
    return ("active", "superseded") if include_superseded else ("active",)


def requires_human_review(op: ReconcileOperation, corroboration_count: int = 0) -> bool:
    """Whether a reconciliation operation is too risky to apply automatically."""
    if op.action not in (ReconcileAction.SUPERSEDE, ReconcileAction.DELETE):
        return False
    return bool(op.flag_for_review) or corroboration_count >= 3
