"""A model of collection attempts over one conversation, for the tests of the backfill rule.

A :class:`Conversation` holds the ground truth, the creation time of every
message, and answers a poll as a provider that pages newest first does: a page
holds ``page_size`` messages, a poll that reads ``pages`` pages and has more to
read stops early, complete from the oldest message it read, and a poll that
reads every page ends ``reached_start`` or ``history_exhausted``. Messages
created within ``hidden`` of the poll's start are not shown yet.

A :class:`Simulation` runs attempts: each builds its request from the previous
manifest, as the server does, answers it from the conversation, and keeps the
request, the manifest and the floor the request was built with. It states what
each poll read independently of the layer: a poll that reached its start or
ran out of history read ``[requested start, proven end)``, and one that
stopped early read its proven interval.

``old_rule`` plans backfills as the server did before the bounded rule: one
per stopped poll, from the requested start of the poll it continues, clipped
only to the retention cutoff.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from memforge.conversations.coverage import JUST_AFTER, proven_interval
from memforge.conversations.layer import ConversationLayer, ConversationSelection, retention_cutoff
from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    ConversationEvidence,
    PollEvidence,
    RequestedConversation,
)
from memforge.conversations.port import (
    ConversationAccess,
    ConversationLabel,
    ConversationRef,
    ConversationStatus,
    Interval,
    PollOutcome,
    ThreadingModel,
)

CONVERSATION = "c1"
REF = ConversationRef(
    label=ConversationLabel(conversation_id=CONVERSATION, kind="group_chat", name=CONVERSATION),
    threading=ThreadingModel.INLINE,
    poll_includes_replies=True,
)

Range = tuple[datetime, datetime]
# How many pages a poll reads before it stops, given its interval and its
# index in the request (0 is the regular poll); ``None`` reads every page.
Pages = Callable[[Interval, int], "int | None"]


def union(ranges: Iterable[Range]) -> list[Range]:
    merged: list[Range] = []
    for start, end in sorted(item for item in ranges if item[0] < item[1]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def without(ranges: Iterable[Range], removed: Iterable[Range]) -> list[Range]:
    cuts = union(removed)
    result: list[Range] = []
    for start, end in union(ranges):
        for cut_start, cut_end in cuts:
            if cut_end <= start or cut_start >= end:
                continue
            if cut_start > start:
                result.append((start, cut_start))
            start = max(start, cut_end)
        if start < end:
            result.append((start, end))
    return result


def history_floor(selection: ConversationSelection, now: datetime) -> datetime:
    floor = now - timedelta(days=selection.initial_history_days)
    cutoff = retention_cutoff(now, selection.rolling_retention_days)
    return floor if cutoff is None else max(floor, cutoff)


def poll_read(poll: PollEvidence, *, received_at: datetime) -> Range:
    """What one poll read, as the backfill rule defines it."""

    proven = proven_interval(poll, REF, received_at=received_at)
    if poll.outcome is PollOutcome.STOPPED_EARLY:
        return proven.start, proven.end
    return poll.requested.start, proven.end


def manifest(at: datetime, *polls: PollEvidence, received_at: datetime | None = None) -> CollectionManifest:
    """A manifest in which the conversation was read by ``polls``."""

    return CollectionManifest(
        source_id="src-chat",
        collection_attempt_id=f"attempt-{at.isoformat()}",
        config_revision="cfg",
        record_schema="conversation-message:1",
        received_at=received_at or at,
        conversations=(
            ConversationEvidence(
                conversation_id=CONVERSATION,
                status=ConversationStatus(ConversationAccess.READABLE, "ok"),
                ref=REF,
                polls=polls,
            ),
        ),
        messages=(),
    )


def unreadable_manifest(at: datetime) -> CollectionManifest:
    return CollectionManifest(
        source_id="src-chat",
        collection_attempt_id=f"attempt-{at.isoformat()}",
        config_revision="cfg",
        record_schema="conversation-message:1",
        received_at=at,
        conversations=(
            ConversationEvidence(
                conversation_id=CONVERSATION, status=ConversationStatus(ConversationAccess.NO_ACCESS, "HTTP 403")
            ),
        ),
        messages=(),
    )


def backfills(request: CollectionRequest | RequestedConversation) -> list[Range]:
    conversation = request.conversations[0] if isinstance(request, CollectionRequest) else request
    return [(interval.start, interval.end) for interval in conversation.intervals if interval.end is not None]


def old_rule_request(
    *, selection: ConversationSelection, now: datetime, previous: CollectionManifest | None
) -> CollectionRequest:
    """The request the server built before the bounded rule: one backfill per stopped poll."""

    reach = timedelta(days=selection.initial_history_days)
    cutoff = retention_cutoff(now, selection.rolling_retention_days)
    regular_start = now - reach if cutoff is None else max(now - reach, cutoff)
    evidence = previous.conversation(CONVERSATION) if previous is not None else None
    intervals = [Interval(start=regular_start)]
    for poll in evidence.polls if evidence is not None else ():
        if poll.outcome is PollOutcome.STOPPED_EARLY and poll.earliest_complete_page_time is not None:
            start = poll.requested.start if cutoff is None else max(poll.requested.start, cutoff)
            end = poll.earliest_complete_page_time + JUST_AFTER
            if start < end:
                intervals.append(Interval(start=start, end=end))
    return CollectionRequest(conversations=(RequestedConversation(CONVERSATION, tuple(intervals)),))


def probe_5(previous: CollectionManifest) -> list[Range]:
    """The pre-deploy probe for an old-rule manifest the bounded rule cannot read soundly.

    The ranges of ``[regular requested start, highest backfill requested end)``
    that lie in no backfill request and outside every range a poll of the
    manifest read, when the regular poll stopped early and carries at least
    one backfill; empty otherwise.
    """

    evidence = previous.conversation(CONVERSATION)
    if evidence is None or evidence.ref is None:
        return []
    regular = next(poll for poll in evidence.polls if poll.requested.end is None)
    requested = [(poll.requested.start, poll.requested.end) for poll in evidence.polls if poll.requested.end]
    if regular.outcome is not PollOutcome.STOPPED_EARLY or not requested:
        return []
    top = max(end for _start, end in requested)
    reads = [poll_read(poll, received_at=previous.received_at) for poll in evidence.polls]
    return without([(regular.requested.start, top)], [*requested, *reads])


@dataclass
class Conversation:
    """The ground truth of one conversation: when each of its messages was created."""

    times: list[datetime] = field(default_factory=list)

    def post(self, *times: datetime) -> None:
        self.times = sorted([*self.times, *times])

    def between(self, start: datetime, end: datetime) -> list[datetime]:
        return [at for at in self.times if start <= at < end]

    def poll(
        self, interval: Interval, *, at: datetime, page_size: int, pages: int | None, hidden: timedelta
    ) -> tuple[PollEvidence, list[datetime]]:
        """One poll of ``interval`` started at ``at``, and the creation times of the messages it returned."""

        visible = [time for time in self.times if time <= at - hidden]
        inside = sorted(
            (time for time in visible if time >= interval.start and (interval.end is None or time < interval.end)),
            reverse=True,
        )
        if pages is not None and len(inside) > pages * page_size:
            read = inside[: pages * page_size]
            return (
                PollEvidence(
                    requested=interval,
                    outcome=PollOutcome.STOPPED_EARLY,
                    started_at=at,
                    earliest_returned_at=read[-1],
                    earliest_complete_page_time=read[-1],
                    stop_reason="budget",
                ),
                read,
            )
        before = [time for time in visible if time < interval.start]
        earliest = inside[-1] if inside else None
        return (
            PollEvidence(
                requested=interval,
                outcome=PollOutcome.REACHED_START if before else PollOutcome.HISTORY_EXHAUSTED,
                started_at=at,
                earliest_returned_at=earliest,
                earliest_complete_page_time=max(before) if before else earliest,
            ),
            inside,
        )


@dataclass(frozen=True)
class Attempt:
    """One collection attempt: its request, the floor it was built with, its manifest and what each poll returned."""

    at: datetime
    floor: datetime
    selection: ConversationSelection
    request: CollectionRequest
    manifest: CollectionManifest
    returned: tuple[tuple[datetime, ...], ...]

    @property
    def readable(self) -> bool:
        return self.manifest.conversations[0].ref is not None

    @property
    def polls(self) -> tuple[PollEvidence, ...]:
        return self.manifest.conversations[0].polls

    @property
    def backfills(self) -> list[Range]:
        return backfills(self.request)

    @property
    def regular(self) -> PollEvidence:
        return self.polls[0]

    @property
    def regular_end(self) -> datetime:
        return proven_interval(self.regular, REF, received_at=self.manifest.received_at).end

    def reads(self) -> list[Range]:
        return [poll_read(poll, received_at=self.manifest.received_at) for poll in self.polls]


class Simulation:
    def __init__(self, conversation: Conversation, *, old_rule: bool = False) -> None:
        self.conversation = conversation
        self.old_rule = old_rule
        self.attempts: list[Attempt] = []

    @property
    def previous(self) -> CollectionManifest | None:
        return self.attempts[-1].manifest if self.attempts else None

    def request(self, selection: ConversationSelection, now: datetime) -> CollectionRequest:
        plan = old_rule_request if self.old_rule else ConversationLayer.collection_request
        return plan(selection=selection, now=now, previous=self.previous)

    def run(
        self,
        selection: ConversationSelection,
        *,
        at: datetime,
        pages: Pages | Sequence[int | None] = (),
        page_size: int = 1,
        hidden: timedelta = timedelta(0),
        received_after: timedelta = timedelta(0),
        readable: bool = True,
    ) -> Attempt:
        """One attempt at ``at``; ``pages`` says how many pages each poll reads, every page by default."""

        request = self.request(selection, at)
        intervals = request.conversations[0].intervals
        if not callable(pages):
            listed = list(pages)
            pages_of = lambda _interval, index: listed[index] if index < len(listed) else None  # noqa: E731
        else:
            pages_of = pages
        if readable:
            answers = [
                self.conversation.poll(
                    interval, at=at, page_size=page_size, pages=pages_of(interval, index), hidden=hidden
                )
                for index, interval in enumerate(intervals)
            ]
            answered = manifest(at, *(poll for poll, _read in answers), received_at=at + received_after)
            returned = tuple(tuple(read) for _poll, read in answers)
        else:
            answered = unreadable_manifest(at)
            returned = ()
        attempt = Attempt(
            at=at,
            floor=history_floor(selection, at),
            selection=selection,
            request=request,
            manifest=answered,
            returned=returned,
        )
        self.attempts.append(attempt)
        return attempt

    def adopt(self, answered: CollectionManifest, *, selection: ConversationSelection) -> Attempt:
        """Take ``answered`` as the last attempt, as if this simulation had run it."""

        intervals = tuple(poll.requested for poll in answered.conversations[0].polls)
        attempt = Attempt(
            at=answered.received_at,
            floor=history_floor(selection, answered.received_at),
            selection=selection,
            request=CollectionRequest(conversations=(RequestedConversation(CONVERSATION, intervals),)),
            manifest=answered,
            returned=(),
        )
        self.attempts.append(attempt)
        return attempt

    def read_so_far(self) -> list[Range]:
        return union(read for attempt in self.attempts if attempt.readable for read in attempt.reads())

    def unrequested(self, selection: ConversationSelection, now: datetime) -> list[Range]:
        """What lies after the next floor and below the last regular proven end that no attempt read and the next request's backfills leave out."""

        last = self.attempts[-1]
        floor = history_floor(selection, now)
        return without([(floor, last.regular_end)], [*self.read_so_far(), *backfills(self.request(selection, now))])
