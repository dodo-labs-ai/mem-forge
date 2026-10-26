"""Stored Review rows for tests of the reads and decisions they still get.

A pending supersede Review comes from a Memory correction, and related
challengers and Lifecycle Reviews come only from earlier versions. Tests that
need such a row in a given state write it here, as storage holds it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from memforge.models import MemoryReview
from memforge.storage.database import Database


async def insert_stored_memory_review(db: Database, review: MemoryReview) -> None:
    """Write one ``memory_reviews`` row exactly as given."""

    created_at = (review.created_at or datetime.now(timezone.utc)).isoformat()
    await db.db.execute(
        """INSERT INTO memory_reviews (
               id, kind, status, incumbent_memory_id, challenger_memory_id,
               reason, review_note, reviewer,
               expected_incumbent_updated_at, expected_challenger_updated_at,
               expected_support_set_hash, replacement_kind, created_at, resolved_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            review.id,
            review.kind,
            review.status,
            review.incumbent_memory_id,
            review.challenger_memory_id,
            review.reason,
            review.review_note,
            review.reviewer,
            review.expected_incumbent_updated_at,
            review.expected_challenger_updated_at,
            review.expected_support_set_hash,
            review.replacement_kind,
            created_at,
            review.resolved_at.isoformat() if review.resolved_at else None,
        ),
    )
    await db.db.commit()


async def attach_stored_related_challenger(
    db: Database,
    review_id: str,
    challenger_memory_id: str,
    *,
    reason: str | None = None,
) -> None:
    """Write one ``memory_review_related_challengers`` row for a stored Review."""

    await db.db.execute(
        """INSERT INTO memory_review_related_challengers (
               review_id, challenger_memory_id, reason, created_at
           ) VALUES (?, ?, ?, ?)""",
        (review_id, challenger_memory_id, reason, datetime.now(timezone.utc).isoformat()),
    )
    await db.db.commit()
