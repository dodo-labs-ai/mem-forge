"""Proven intervals: what one poll proves about which messages exist.

A poll of a readable conversation proves the half-open creation-time interval
``[start, end)`` it read completely: every message the provider shows in it
was returned. The conversation layer computes the interval from the poll's
raw evidence and never takes an executor's conclusion.

- ``end`` is the earlier of the poll's start on the executor's clock and the
  server's receipt of the manifest, minus a visibility margin, and no later
  than the requested end. The margin leaves messages the provider has not yet
  returned to the next poll instead of placing them inside this one unread.
- ``start`` depends on how paging ended:

  - it reached the requested start: the requested start;
  - history ran out first: the requested start when the provider certifies
    visible history that far back, the certified start when it certifies a
    later one, and otherwise the earliest message returned, since the provider
    may hide older history;
  - paging stopped early: just after the earliest message time on the last
    fully read page, which covers what was read contiguously from the newest
    message. A message at that same time may sit on the next, unread page.

"The earliest message returned" assumes visible history is a suffix: the
provider hides only messages before some time, never a stretch in the middle.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from memforge.conversations.manifest import PollEvidence
from memforge.conversations.port import ConversationRef, PollOutcome

VISIBILITY_MARGIN = timedelta(minutes=5)
JUST_AFTER = timedelta(microseconds=1)


@dataclass(frozen=True, slots=True)
class ProvenInterval:
    """A half-open interval ``[start, end)`` a poll read completely; empty when ``start >= end``."""

    start: datetime
    end: datetime

    @property
    def empty(self) -> bool:
        return self.start >= self.end

    def contains(self, at: datetime) -> bool:
        return self.start <= at < self.end


def proven_interval(
    poll: PollEvidence,
    conversation: ConversationRef,
    *,
    received_at: datetime,
    margin: timedelta = VISIBILITY_MARGIN,
) -> ProvenInterval:
    """The interval one poll proves complete."""

    end = min(poll.started_at, received_at) - margin
    if poll.requested.end is not None:
        end = min(end, poll.requested.end)
    requested_start = poll.requested.start
    if poll.outcome is PollOutcome.REACHED_START:
        start = requested_start
    elif poll.outcome is PollOutcome.HISTORY_EXHAUSTED:
        certified = conversation.visible_from_certified
        if certified is not None:
            start = max(requested_start, certified)
        elif poll.earliest_returned_at is not None:
            start = poll.earliest_returned_at
        else:
            start = end
    else:
        start = poll.earliest_complete_page_time + JUST_AFTER if poll.earliest_complete_page_time else end
    return ProvenInterval(start=max(start, requested_start), end=end)


@dataclass(frozen=True, slots=True)
class ConversationCoverage:
    """Everything this attempt proved about one conversation."""

    intervals: tuple[ProvenInterval, ...]
    # Whether a poll returns replies as well as top-level messages.
    covers_replies: bool

    def proves_absent(self, created_at: datetime, *, is_reply: bool) -> bool:
        """Whether a message the attempt did not return is proven gone.

        A top-level message, or a reply when polls return replies, is gone
        when a proven interval holds its creation time.
        """

        return (not is_reply or self.covers_replies) and any(
            interval.contains(created_at) for interval in self.intervals
        )


def conversation_coverage(
    conversation: ConversationRef,
    polls: Iterable[PollEvidence],
    *,
    received_at: datetime,
    margin: timedelta = VISIBILITY_MARGIN,
) -> ConversationCoverage:
    return ConversationCoverage(
        intervals=tuple(
            proven_interval(poll, conversation, received_at=received_at, margin=margin) for poll in polls
        ),
        covers_replies=conversation.poll_includes_replies,
    )
