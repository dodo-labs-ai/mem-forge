from dataclasses import replace
from datetime import datetime, timezone

from memforge.memory.lifecycle_plan import LifecycleReview, LifecycleReviewStatus
from memforge.memory.review_decision import (
    lifecycle_review_decision_fingerprint,
    memory_review_decision_fingerprint,
)
from memforge.models import Memory, MemoryReview, content_hash


def test_memory_review_fingerprint_changes_with_pinned_participant_version() -> None:
    created = datetime(2026, 8, 8, tzinfo=timezone.utc)
    review = MemoryReview(
        id="review-1",
        kind="supersede",
        status="pending",
        incumbent_memory_id="memory-old",
        challenger_memory_id="memory-new",
        expected_incumbent_updated_at="2026-08-08T00:00:00+00:00",
        expected_challenger_updated_at="2026-08-08T00:00:01+00:00",
        created_at=created,
    )

    original = memory_review_decision_fingerprint(review)
    changed = memory_review_decision_fingerprint(
        MemoryReview(
            **{
                **review.__dict__,
                "expected_challenger_updated_at": "2026-08-08T00:00:02+00:00",
            }
        )
    )

    assert original.startswith("review-decision-v1:")
    assert original != changed


def test_memory_review_fingerprint_binds_related_membership_but_not_terminal_status() -> None:
    created = datetime(2026, 8, 8, tzinfo=timezone.utc)
    review = MemoryReview(
        id="review-related",
        kind="supersede",
        status="pending",
        incumbent_memory_id="memory-old",
        challenger_memory_id="memory-new",
        expected_incumbent_updated_at="2026-08-08T00:00:00+00:00",
        expected_challenger_updated_at="2026-08-08T00:00:01+00:00",
        created_at=created,
    )
    related = Memory(
        id="memory-related",
        memory_type="fact",
        content="Related proposal",
        content_hash=content_hash("Related proposal"),
        created_at=created,
        status="pending_review",
    )

    without_related = memory_review_decision_fingerprint(review)
    with_related = memory_review_decision_fingerprint(review, (related,))
    terminal = memory_review_decision_fingerprint(
        replace(review, status="approved"),
        (replace(related, status="retired"),),
    )

    assert without_related != with_related
    assert terminal == with_related


def test_lifecycle_review_fingerprint_is_mapping_order_independent() -> None:
    first = LifecycleReview(
        id="review-2",
        lifecycle_plan_id="plan-1",
        incumbent_memory_id="memory-1",
        status=LifecycleReviewStatus.PENDING,
        staged_evidence={"candidate": {"content": "new"}, "proposed_disposition": "supersede"},
        source_id="source-1",
    )
    second = LifecycleReview(
        id="review-2",
        lifecycle_plan_id="plan-1",
        incumbent_memory_id="memory-1",
        status=LifecycleReviewStatus.PENDING,
        staged_evidence={"proposed_disposition": "supersede", "candidate": {"content": "new"}},
        source_id="source-1",
    )

    assert lifecycle_review_decision_fingerprint(first) == lifecycle_review_decision_fingerprint(second)
    assert lifecycle_review_decision_fingerprint(first) == lifecycle_review_decision_fingerprint(
        replace(first, status=LifecycleReviewStatus.APPROVED)
    )
