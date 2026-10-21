from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memforge.conversations.coverage import (
    VISIBILITY_MARGIN,
    ProvenInterval,
    conversation_coverage,
    proven_interval,
)
from memforge.conversations.manifest import PollEvidence
from memforge.conversations.port import ConversationLabel, ConversationRef, Interval, PollOutcome, ThreadingModel

NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
START = NOW - timedelta(days=30)


def _conversation(*, certified: datetime | None = None, replies: bool = True) -> ConversationRef:
    return ConversationRef(
        label=ConversationLabel(conversation_id="c1", kind="group_chat", name="Chat"),
        threading=ThreadingModel.INLINE,
        poll_includes_replies=replies,
        visible_from_certified=certified,
    )


def _poll(outcome: PollOutcome, **evidence) -> PollEvidence:
    return PollEvidence(requested=Interval(start=START), outcome=outcome, started_at=NOW, **evidence)


def test_reaching_the_requested_start_proves_the_whole_interval_up_to_the_margin():
    interval = proven_interval(_poll(PollOutcome.REACHED_START), _conversation(), received_at=NOW)

    assert interval == ProvenInterval(start=START, end=NOW - VISIBILITY_MARGIN)


def test_running_out_of_history_proves_only_from_the_earliest_message_returned():
    earliest = NOW - timedelta(days=3)
    interval = proven_interval(
        _poll(PollOutcome.HISTORY_EXHAUSTED, earliest_returned_at=earliest), _conversation(), received_at=NOW
    )

    assert interval.start == earliest
    assert not interval.contains(earliest - timedelta(seconds=1))


def test_certified_visibility_extends_an_exhausted_poll_to_the_requested_start():
    interval = proven_interval(
        _poll(PollOutcome.HISTORY_EXHAUSTED, earliest_returned_at=NOW - timedelta(days=3)),
        _conversation(certified=START - timedelta(days=1)),
        received_at=NOW,
    )

    assert interval.start == START


def test_an_exhausted_poll_that_returned_nothing_proves_nothing():
    interval = proven_interval(_poll(PollOutcome.HISTORY_EXHAUSTED), _conversation(), received_at=NOW)

    assert interval.empty


def test_a_poll_stopped_early_proves_just_after_the_last_complete_page():
    page_time = NOW - timedelta(days=10)
    interval = proven_interval(
        _poll(PollOutcome.STOPPED_EARLY, earliest_complete_page_time=page_time), _conversation(), received_at=NOW
    )

    assert not interval.contains(page_time)
    assert interval.contains(page_time + timedelta(milliseconds=1))


def test_the_earlier_of_the_device_clock_and_the_receipt_bounds_the_end():
    device_ahead = proven_interval(
        PollEvidence(requested=Interval(start=START), outcome=PollOutcome.REACHED_START, started_at=NOW + timedelta(hours=2)),
        _conversation(),
        received_at=NOW,
    )

    assert device_ahead.end == NOW - VISIBILITY_MARGIN


def test_a_backfill_proves_no_more_than_its_requested_end():
    backfill_end = NOW - timedelta(days=5)
    interval = proven_interval(
        PollEvidence(
            requested=Interval(start=START, end=backfill_end), outcome=PollOutcome.REACHED_START, started_at=NOW
        ),
        _conversation(),
        received_at=NOW,
    )

    assert interval == ProvenInterval(start=START, end=backfill_end)


def test_interval_bounds_are_half_open():
    interval = ProvenInterval(start=START, end=NOW)

    assert interval.contains(START)
    assert not interval.contains(NOW)


def test_replies_are_proven_absent_only_when_polls_return_them():
    inside = NOW - timedelta(days=1)
    with_replies = conversation_coverage(_conversation(), [_poll(PollOutcome.REACHED_START)], received_at=NOW)
    without_replies = conversation_coverage(
        _conversation(replies=False), [_poll(PollOutcome.REACHED_START)], received_at=NOW
    )

    assert with_replies.proves_absent(inside, is_reply=True)
    assert not without_replies.proves_absent(inside, is_reply=True)
    assert without_replies.proves_absent(inside, is_reply=False)
