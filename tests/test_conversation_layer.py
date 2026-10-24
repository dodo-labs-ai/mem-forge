from __future__ import annotations

import random
from collections.abc import Collection
from datetime import datetime, timedelta, timezone

import pytest

from memforge.conversations.coverage import JUST_AFTER, VISIBILITY_MARGIN
from memforge.conversations.layer import ConversationLayer, ConversationSelection, retention_cutoff
from memforge.conversations.manifest import (
    CollectionManifest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    manifest_from_payload,
    manifest_to_payload,
    request_from_payload,
    request_to_payload,
)
from memforge.conversations.membership import units_to_read
from memforge.conversations.port import (
    ConversationAccess,
    ConversationLabel,
    ConversationRef,
    ConversationStatus,
    Interval,
    PollOutcome,
    ThreadingModel,
)
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.time_scope import MAX_INITIAL_HISTORY_DAYS
from memforge.conversations.units import Member, Span, StoredUnit, UnitKind, UnitRef
from memforge.source_projection import UnitRemovalReason
from tests import conversation_attempts_fixture as attempts

SOURCE = "src-chat"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
DAY = timedelta(days=1)
HOUR = timedelta(hours=1)
MINUTE = timedelta(minutes=1)


class InMemoryProjection:
    """The current Units of one Source, answering the reads the conversation layer makes.

    A Unit's recorded span is its members' span, and every span bound is
    inclusive, as the Source Projection records them. Reading a Unit it does
    not hold fails, as the reader contract requires of a listed Unit without
    a current revision.
    """

    def __init__(self, *units: StoredUnit) -> None:
        self.units = list(units)

    def _of(self, conversation_id: str) -> list[StoredUnit]:
        return [unit for unit in self.units if unit.ref.conversation_id == conversation_id]

    async def holders(self, source_id: str, conversation_id: str, message_ids: Collection[str]):
        holders: dict[str, tuple[UnitRef, ...]] = {}
        for unit in self._of(conversation_id):
            for member in unit.members:
                if member.identity.message_id in message_ids:
                    holders[member.identity.message_id] = (*holders.get(member.identity.message_id, ()), unit.ref)
        return holders

    async def thread_units(self, source_id: str, conversation_id: str):
        return tuple(unit.ref for unit in self._of(conversation_id) if unit.kind is UnitKind.THREAD)

    async def read(self, units: Collection[UnitRef]):
        wanted = {unit.unit_id for unit in units}
        if missing := sorted(wanted - {unit.ref.unit_id for unit in self.units}):
            raise ValueError(f"conversation Units {missing} were listed as current but have no current revision")
        return tuple(unit for unit in self.units if unit.ref.unit_id in wanted)

    async def units_spanning(self, source_id: str, conversation_id: str, start: datetime, end: datetime):
        return tuple(
            unit
            for unit in self._of(conversation_id)
            if (span := Span.of(unit.members)) is not None and span.earliest <= end and span.latest >= start
        )

    async def units_ending_before(self, source_id: str, conversation_id: str, cutoff: datetime):
        return tuple(
            unit
            for unit in self._of(conversation_id)
            if (span := Span.of(unit.members)) is not None and span.latest < cutoff
        )

    async def units_of(self, source_id: str, conversation_id: str):
        return tuple(unit.ref for unit in self._of(conversation_id))

    async def conversations_held(self, source_id: str):
        return frozenset(unit.ref.conversation_id for unit in self.units)


def _record(conversation_id: str, message_id: str, at: datetime, text: str = "hello") -> dict[str, object]:
    return {"id": message_id, "content": text, "time": at.isoformat(), "conversation_id": conversation_id}


def _member(conversation_id: str, message_id: str, at: datetime) -> Member:
    record = _record(conversation_id, message_id, at)
    return Member(MessageIdentity(conversation_id, message_id), at, record, record_revision(record))


def _window(key: str, conversation_id: str, *members: Member) -> StoredUnit:
    return StoredUnit(
        ref=UnitRef(unit_id=key, document_id=f"doc-{key}", provider_key=key, kind=UnitKind.WINDOW,
                    conversation_id=conversation_id),
        members=members,
    )


def _ref(conversation_id: str) -> ConversationRef:
    return ConversationRef(
        label=ConversationLabel(conversation_id=conversation_id, kind="group_chat", name=conversation_id),
        threading=ThreadingModel.INLINE,
        poll_includes_replies=True,
    )


def _readable(conversation_id: str, *, outcome: PollOutcome = PollOutcome.REACHED_START, days: int = 30):
    return ConversationEvidence(
        conversation_id=conversation_id,
        status=ConversationStatus(ConversationAccess.READABLE, "ok"),
        ref=_ref(conversation_id),
        polls=(PollEvidence(requested=Interval(start=NOW - timedelta(days=days)), outcome=outcome, started_at=NOW),),
    )


def _status(conversation_id: str, access: ConversationAccess) -> ConversationEvidence:
    return ConversationEvidence(conversation_id=conversation_id, status=ConversationStatus(access, "provider said so"))


def _manifest(*conversations: ConversationEvidence, messages=(), attempt: str = "run-1:attempt:1"):
    return CollectionManifest(
        source_id=SOURCE,
        collection_attempt_id=attempt,
        config_revision="cfg-1",
        record_schema="conversation-message:1",
        received_at=NOW,
        conversations=conversations,
        messages=tuple(messages),
    )


def _manifest_message(conversation_id: str, message_id: str, at: datetime, text: str = "hello"):
    record = _record(conversation_id, message_id, at, text)
    message = ManifestMessage(
        identity=MessageIdentity(conversation_id, message_id), revision=record_revision(record), created_at=at
    )
    return message, record


def _selection(*conversation_ids: str, retention: int | None = None, history: int = 30) -> ConversationSelection:
    return ConversationSelection(
        conversation_ids=conversation_ids, initial_history_days=history, rolling_retention_days=retention
    )


def _removals(plan) -> list[tuple[str, UnitRemovalReason]]:
    return sorted((removal.unit.provider_key, removal.reason) for removal in plan.removals)


async def test_a_new_message_opens_a_window():
    message, record = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    layer = ConversationLayer(InMemoryProjection())

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    (unit,) = plan.units
    assert unit.kind is UnitKind.WINDOW
    assert [member.identity.message_id for member in unit.members] == ["m1"]
    assert plan.removals == () and plan.unreadable == ()


async def test_a_body_must_match_its_manifest_revision():
    message, record = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    layer = ConversationLayer(InMemoryProjection())

    with pytest.raises(ValueError, match="does not match"):
        await layer.plan(
            manifest=_manifest(_readable("c1"), messages=[message]),
            bodies={message.identity: {**record, "content": "tampered"}},
            selection=_selection("c1"),
            attempt_started_at=NOW,
        )


async def test_a_window_not_returned_inside_the_proven_interval_is_absent():
    old = _window("w-old", "c1", _member("c1", "m0", NOW - timedelta(days=2)))
    layer = ConversationLayer(InMemoryProjection(old))

    plan = await layer.plan(manifest=_manifest(_readable("c1")), bodies={}, selection=_selection("c1"),
                            attempt_started_at=NOW)

    assert _removals(plan) == [("w-old", UnitRemovalReason.ABSENT_FROM_SOURCE)]


async def test_running_out_of_history_removes_nothing_older_than_what_was_returned():
    old = _window("w-old", "c1", _member("c1", "m0", NOW - timedelta(days=20)))
    message, record = _manifest_message("c1", "m1", NOW - timedelta(days=1))
    evidence = ConversationEvidence(
        conversation_id="c1",
        status=ConversationStatus(ConversationAccess.READABLE, "ok"),
        ref=_ref("c1"),
        polls=(
            PollEvidence(
                requested=Interval(start=NOW - timedelta(days=30)),
                outcome=PollOutcome.HISTORY_EXHAUSTED,
                started_at=NOW,
                earliest_returned_at=message.created_at,
            ),
        ),
    )
    layer = ConversationLayer(InMemoryProjection(old))

    plan = await layer.plan(manifest=_manifest(evidence, messages=[message]), bodies={message.identity: record},
                            selection=_selection("c1"), attempt_started_at=NOW)

    assert plan.removals == ()


async def test_a_gone_conversation_loses_every_unit():
    projection = InMemoryProjection(
        _window("w1", "c1", _member("c1", "m1", NOW - timedelta(days=200))),
        _window("w2", "c1", _member("c1", "m2", NOW - timedelta(days=1))),
    )
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(_status("c1", ConversationAccess.GONE)), bodies={},
                            selection=_selection("c1"), attempt_started_at=NOW)

    assert _removals(plan) == [("w1", UnitRemovalReason.ABSENT_FROM_SOURCE), ("w2", UnitRemovalReason.ABSENT_FROM_SOURCE)]


async def test_lost_access_keeps_support_but_still_applies_retention_and_is_reported():
    projection = InMemoryProjection(
        _window("w-old", "c1", _member("c1", "m1", NOW - timedelta(days=400))),
        _window("w-new", "c1", _member("c1", "m2", NOW - timedelta(days=1))),
        _window("w-other", "c2", _member("c2", "m3", NOW - timedelta(days=1))),
    )
    layer = ConversationLayer(projection)

    plan = await layer.plan(
        manifest=_manifest(_status("c1", ConversationAccess.NO_ACCESS), _readable("c2", days=365)),
        bodies={},
        selection=_selection("c1", "c2", retention=365, history=365),
        attempt_started_at=NOW,
    )

    assert _removals(plan) == [("w-old", UnitRemovalReason.OUTSIDE_ROLLING_RETENTION), ("w-other", UnitRemovalReason.ABSENT_FROM_SOURCE)]
    assert [evidence.conversation_id for evidence in plan.unreadable] == ["c1"]


async def test_a_conversation_without_a_status_in_this_attempt_is_left_alone():
    projection = InMemoryProjection(_window("w-old", "c1", _member("c1", "m1", NOW - timedelta(days=400))))
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(), bodies={}, selection=_selection("c1", retention=365),
                            attempt_started_at=NOW)

    assert plan.removals == ()


async def test_deselected_conversations_are_removed_from_scope():
    projection = InMemoryProjection(_window("w", "c-gone", _member("c-gone", "m", NOW - timedelta(days=1))))
    layer = ConversationLayer(projection)

    plan = await layer.plan(manifest=_manifest(), bodies={}, selection=_selection("c1"), attempt_started_at=NOW,
                            deselected_conversation_ids=["c-gone"])

    assert _removals(plan) == [("w", UnitRemovalReason.REMOVED_FROM_SCOPE)]


async def test_a_manifest_conversation_outside_the_selection_is_refused():
    layer = ConversationLayer(InMemoryProjection())

    with pytest.raises(ValueError, match="outside the selection"):
        await layer.plan(manifest=_manifest(_readable("c9")), bodies={}, selection=_selection("c1"),
                         attempt_started_at=NOW)


def test_initial_history_cannot_exceed_rolling_retention():
    with pytest.raises(ValueError, match="cannot exceed"):
        _selection("c1", retention=30, history=60)


@pytest.mark.parametrize("retention", [None, 365, 730, 1095])
def test_initial_history_is_bounded_to_one_year_whatever_the_retention(retention):
    assert _selection("c1", retention=retention, history=MAX_INITIAL_HISTORY_DAYS).initial_history_days == 365
    for history in (0, MAX_INITIAL_HISTORY_DAYS + 1):
        with pytest.raises(ValueError, match="between 1 and 365 days|cannot exceed"):
            _selection("c1", retention=retention, history=history)


def test_the_request_polls_initial_history_within_the_ceiling_and_retention():
    selection = ConversationSelection(
        conversation_ids=("c1",), initial_history_days=60, rolling_retention_days=365,
        regular_ceiling=timedelta(days=30),
    )

    request = ConversationLayer.collection_request(selection=selection, now=NOW)

    (conversation,) = request.conversations
    assert conversation.intervals == (Interval(start=NOW - timedelta(days=30)),)
    assert request_from_payload(request_to_payload(request)) == request


def test_a_poll_stopped_early_is_backfilled_by_the_next_attempt_from_the_history_floor():
    stopped_at = NOW - timedelta(days=10)
    # The previous attempt ran a day earlier, so its regular poll started a day before this floor.
    previous = attempts.manifest(
        NOW - DAY,
        PollEvidence(
            requested=Interval(start=NOW - timedelta(days=31)),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=NOW - DAY,
            earliest_returned_at=stopped_at,
            earliest_complete_page_time=stopped_at,
        ),
    )
    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW, previous=previous)

    (conversation,) = request.conversations
    assert conversation.intervals == (
        Interval(start=NOW - timedelta(days=30)),
        Interval(start=NOW - timedelta(days=30), end=stopped_at + JUST_AFTER),
    )
    assert conversation.last_known == attempts.REF.label


# ----------------------------------------------------------------------
# Bounded backfills: what the previous attempt left unread after the floor
# ----------------------------------------------------------------------


def _daily_chat(days: int = 30) -> attempts.Conversation:
    """One message a day, the newest half a day before ``NOW``."""

    return attempts.Conversation([NOW - timedelta(days=day, hours=12) for day in range(days)])


def _complete_from(poll: PollEvidence) -> datetime:
    assert poll.earliest_complete_page_time is not None
    return poll.earliest_complete_page_time


def _just_after(at: datetime) -> datetime:
    return at + JUST_AFTER


def test_a_backfill_that_stops_early_is_continued_by_one_backfill_below_it():
    selection = _selection("c1")
    simulation = attempts.Simulation(_daily_chat())
    first = simulation.run(selection, at=NOW, pages=[3])
    stopped = _complete_from(first.regular)
    second = simulation.run(selection, at=NOW + HOUR, pages=[3, 5])
    assert second.backfills == [(second.floor, _just_after(stopped))]
    assert second.polls[1].outcome is PollOutcome.STOPPED_EARLY

    request = simulation.request(selection, NOW + 2 * HOUR)

    # Below the backfill's stop, one backfill continues it; above, the regular
    # poll stopped at the same message again, so only that instant is requested.
    assert attempts.backfills(request) == [
        (attempts.history_floor(selection, NOW + 2 * HOUR), _just_after(_complete_from(second.polls[1]))),
        (stopped, _just_after(stopped)),
    ]


@pytest.mark.parametrize("backfill_pages", [None, 3], ids=["backfill_completed", "backfill_stopped"])
def test_a_regular_poll_that_stops_above_the_previous_attempts_read_leaves_no_range_unrequested(backfill_pages):
    selection = _selection("c1")
    chat = _daily_chat()
    simulation = attempts.Simulation(chat)
    first = simulation.run(selection, at=NOW, pages=[3])
    c1, e1 = _complete_from(first.regular), first.regular_end
    # More new messages than the next regular poll pages through.
    chat.post(*(NOW + timedelta(minutes=minutes) for minutes in (10, 20, 30, 40, 50)))
    second = simulation.run(selection, at=NOW + HOUR, pages=[2, backfill_pages])
    c2 = _complete_from(second.regular)
    assert c2 > e1

    request = simulation.request(selection, NOW + 2 * HOUR)

    upper = (_just_after(c1), _just_after(c2))
    if backfill_pages is None:
        assert attempts.backfills(request) == [upper]
    else:
        lower = (attempts.history_floor(selection, NOW + 2 * HOUR), _just_after(_complete_from(second.polls[1])))
        assert attempts.backfills(request) == [lower, upper]
    assert attempts.without([(e1, c2)], attempts.backfills(request)) == []
    assert simulation.unrequested(selection, NOW + 2 * HOUR) == []


def test_a_regular_poll_and_a_backfill_that_both_stop_early_leave_two_backfills():
    selection = _selection("c1")
    simulation = attempts.Simulation(_daily_chat())
    first = simulation.run(selection, at=NOW, pages=[3])
    c1 = _complete_from(first.regular)
    second = simulation.run(selection, at=NOW + HOUR, pages=[2, 2])
    c2, b2 = _complete_from(second.regular), _complete_from(second.polls[1])
    # The regular poll stopped between the previous backfill's end and the previous regular poll's proven end.
    assert _just_after(c1) < c2 < first.regular_end

    request = simulation.request(selection, NOW + 2 * HOUR)

    assert attempts.backfills(request) == [
        (attempts.history_floor(selection, NOW + 2 * HOUR), _just_after(b2)),
        (_just_after(c1), _just_after(c2)),
    ]


def test_over_three_or_more_attempts_the_upper_backfill_never_re_reads_what_the_previous_lower_backfill_read():
    selection = _selection("c1")
    chat = _daily_chat()
    simulation = attempts.Simulation(chat)
    c1 = _complete_from(simulation.run(selection, at=NOW, pages=[3]).regular)
    chat.post(NOW + 10 * MINUTE, NOW + 20 * MINUTE, NOW + 30 * MINUTE)
    second = simulation.run(selection, at=NOW + HOUR, pages=[3, 2])
    c2, b2 = _complete_from(second.regular), _complete_from(second.polls[1])
    chat.post(NOW + 70 * MINUTE, NOW + 80 * MINUTE, NOW + 90 * MINUTE)
    third = simulation.run(selection, at=NOW + 2 * HOUR, pages=[3, 2, None])
    assert third.backfills == [(third.floor, _just_after(b2)), (_just_after(c1), _just_after(c2))]
    assert third.polls[2].outcome is not PollOutcome.STOPPED_EARLY
    c3, b3 = _complete_from(third.regular), _complete_from(third.polls[1])

    request = simulation.request(selection, NOW + 3 * HOUR)

    assert attempts.backfills(request) == [
        (attempts.history_floor(selection, NOW + 3 * HOUR), _just_after(b3)),
        (_just_after(c2), _just_after(c3)),
    ]
    assert attempts.without(attempts.backfills(request), [third.reads()[1]]) == attempts.backfills(request)


def test_unread_ranges_above_the_lowest_are_spanned_by_one_backfill():
    selection = _selection("c1")
    chat = _daily_chat()
    simulation = attempts.Simulation(chat)
    c1 = _complete_from(simulation.run(selection, at=NOW, pages=[3]).regular)
    chat.post(*(NOW + timedelta(minutes=minutes) for minutes in (10, 20, 30, 40, 50)))
    c2 = _complete_from(simulation.run(selection, at=NOW + HOUR, pages=[2, 2]).regular)
    chat.post(*(NOW + timedelta(minutes=minutes) for minutes in (70, 80, 90, 100, 110)))
    third = simulation.run(selection, at=NOW + 2 * HOUR, pages=[2, 1, 2])
    lower, regular = third.polls[1], third.regular
    assert third.backfills[1] == (_just_after(c1), _just_after(c2))
    assert all(poll.outcome is PollOutcome.STOPPED_EARLY for poll in third.polls)

    request = simulation.request(selection, NOW + 3 * HOUR)

    # Three ranges are unread: below the lower stop, below the upper stop, and
    # below the regular stop. The second backfill spans the upper two.
    assert attempts.backfills(request) == [
        (attempts.history_floor(selection, NOW + 3 * HOUR), _just_after(_complete_from(lower))),
        (_just_after(c1), _just_after(_complete_from(regular))),
    ]
    assert attempts.without([third.reads()[2]], attempts.backfills(request)) == []


def test_a_chat_whose_regular_poll_keeps_stopping_still_reads_further_back():
    selection = _selection("c1")
    chat = _daily_chat()
    simulation = attempts.Simulation(chat)
    lower_ends = []
    for index in range(6):
        at = NOW + index * HOUR
        chat.post(at - 30 * MINUTE, at - 20 * MINUTE)
        attempt = simulation.run(selection, at=at, pages=[1, 2, None])
        if attempt.backfills:
            lower_ends.append(attempt.backfills[0][1])

    assert len(lower_ends) == 5
    assert all(later < earlier for earlier, later in zip(lower_ends, lower_ends[1:]))


def test_a_chat_whose_regular_poll_keeps_stopping_settles_once_the_lower_backfill_reaches_the_floor():
    selection = _selection("c1", history=10)
    chat = attempts.Conversation([NOW - timedelta(hours=12 * index + 6) for index in range(20)])
    simulation = attempts.Simulation(chat)
    # New messages below one poll's budget, then quiet attempts, then new messages again.
    inflow = [2] * 12 + [0] * 3 + [2] * 4
    for index, arriving in enumerate(inflow):
        at = NOW + index * HOUR
        chat.post(*(at - timedelta(minutes=10 * (count + 1)) for count in range(arriving)))
        simulation.run(selection, at=at, pages=lambda _interval, _index: 3)
    history = simulation.attempts
    assert all(attempt.regular.outcome is PollOutcome.STOPPED_EARLY for attempt in history)
    completed = next(
        index
        for index, attempt in enumerate(history)
        if attempt.backfills
        and attempt.backfills[0][0] == attempt.floor
        and attempt.polls[1].outcome is not PollOutcome.STOPPED_EARLY
    )
    assert completed < 12

    for previous, attempt in zip(history[completed:], history[completed + 1 :]):
        ((start, end),) = attempt.backfills
        assert start > attempt.floor
        assert start >= min(max(backfill_end for _start, backfill_end in previous.backfills), _complete_from(previous.regular))
    for index in range(completed + 1, len(history)):
        if inflow[index - 1] == 0:
            instant = _complete_from(history[index - 1].regular)
            assert history[index].backfills == [(instant, _just_after(instant))]


def _stopping_chat() -> attempts.Conversation:
    """Twenty messages, twelve hours apart, all after a ten-day floor at ``NOW``."""

    return attempts.Conversation([NOW - timedelta(hours=12 * index + 6) for index in range(20)])


@pytest.mark.parametrize("case", ["complete_regular_poll", "floor_rose_past_the_stop", "unreadable_attempt"])
def test_a_stop_after_an_attempt_that_left_nothing_unread_restarts_at_the_floor(case):
    selection = _selection("c1", history=10)
    chat = _stopping_chat()
    simulation = attempts.Simulation(chat)
    at = NOW
    if case == "complete_regular_poll":
        simulation.run(selection, at=at)
    elif case == "floor_rose_past_the_stop":
        held = len(chat.between(attempts.history_floor(selection, at), at))
        first = simulation.run(selection, at=at, pages=[held - 1])
        at += 20 * HOUR
        assert _complete_from(first.regular) < attempts.history_floor(selection, at)
    else:
        simulation.run(selection, at=at, pages=[3])
        at += HOUR
        simulation.run(selection, at=at, readable=False)
    at += HOUR
    assert attempts.backfills(simulation.request(selection, at)) == []
    stopped = simulation.run(selection, at=at, pages=[3])

    request = simulation.request(selection, at + HOUR)

    assert attempts.backfills(request) == [
        (attempts.history_floor(selection, at + HOUR), _just_after(_complete_from(stopped.regular)))
    ]


@pytest.mark.parametrize("case", ["no_backfill", "lower_backfill_on_its_way_down", "lower_backfill_completed"])
def test_an_initial_history_expansion_is_backfilled_while_the_regular_poll_keeps_stopping(case):
    narrow, wide = _selection("c1", history=10), _selection("c1", history=20)
    chat = attempts.Conversation([NOW - timedelta(hours=6 * index + 3) for index in range(160)])
    simulation = attempts.Simulation(chat)
    simulation.run(narrow, at=NOW, pages=[3])
    if case != "no_backfill":
        chat.post(NOW + 10 * MINUTE, NOW + 20 * MINUTE, NOW + 30 * MINUTE)
        simulation.run(narrow, at=NOW + HOUR, pages=[2, None if case == "lower_backfill_completed" else 3])
    previous = simulation.attempts[-1]
    old_start = previous.regular.requested.start
    at = previous.at + HOUR

    floor = attempts.history_floor(wide, at)
    lowest = attempts.backfills(simulation.request(wide, at))[0]

    if case == "lower_backfill_completed":
        assert lowest == (floor, old_start)
    else:
        # Joined with the lower backfill's remainder, which starts at the old floor.
        assert lowest[0] == floor
        assert lowest[1] > old_start
    assert simulation.unrequested(wide, at) == []
    for _index in range(2):
        chat.post(at - 20 * MINUTE, at - 10 * MINUTE)
        simulation.run(wide, at=at, pages=[2, 3, 3])
        at += HOUR
        assert simulation.unrequested(wide, at) == []


def test_a_regular_poll_that_stops_inside_the_visibility_margin_leaves_no_range_unrequested():
    selection = _selection("c1")
    chat = _daily_chat()
    simulation = attempts.Simulation(chat)
    simulation.run(selection, at=NOW, pages=[3])
    at = NOW + HOUR
    chat.post(at - 4 * MINUTE, at - 3 * MINUTE, at - 2 * MINUTE)
    second = simulation.run(selection, at=at, pages=[2, None])
    instant, proven_end = _complete_from(second.regular), second.regular_end
    assert instant >= proven_end
    top = max(end for _start, end in second.backfills)

    requested = attempts.backfills(simulation.request(selection, at + HOUR))

    assert attempts.without([(instant, _just_after(instant))], requested) == [(instant, _just_after(instant))]
    assert attempts.without([(top, proven_end)], requested) == []
    chat.post(at + 30 * MINUTE, at + 40 * MINUTE)
    third = simulation.run(selection, at=at + HOUR, pages=[1, None])
    assert _complete_from(third.regular) > instant
    assert simulation.unrequested(selection, at + 2 * HOUR) == []


def _old_rule_chat() -> attempts.Conversation:
    """Sixty days of messages, four a day."""

    return attempts.Conversation([NOW - timedelta(hours=6 * index + 3) for index in range(240)])


def _post_new(chat: attempts.Conversation, at: datetime) -> None:
    chat.post(at - 30 * MINUTE, at - 20 * MINUTE, at - 10 * MINUTE)


def _old_attempts_widened_last(simulation: attempts.Simulation) -> tuple[ConversationSelection, datetime]:
    """Old-rule attempts whose last one widened Initial History and stopped, every backfill starting at a regular start before the widening."""

    narrow, wide = _selection("c1", history=10), _selection("c1", history=20)
    simulation.old_rule = True
    simulation.run(narrow, at=NOW, pages=[3])
    _post_new(simulation.conversation, NOW + HOUR)
    simulation.run(narrow, at=NOW + HOUR, pages=[3, 3])
    _post_new(simulation.conversation, NOW + 2 * HOUR)
    simulation.run(wide, at=NOW + 2 * HOUR, pages=[3, 3, 3])
    return wide, NOW + 3 * HOUR


# The old-manifest example in abstract days: Initial History 200 at t=1000,
# narrowed to 20 at t=1010 and widened to 60 at t=1110.
def _t(day: float) -> datetime:
    return NOW + timedelta(days=day - 1120)


def _abstract_widened_manifest() -> CollectionManifest:
    return attempts.manifest(
        _t(1110),
        PollEvidence(
            requested=Interval(start=_t(1050)),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=_t(1110),
            earliest_returned_at=_t(1100),
            earliest_complete_page_time=_t(1100),
        ),
        PollEvidence(
            requested=Interval(start=_t(800), end=_t(901)),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=_t(1110),
            earliest_returned_at=_t(850),
            earliest_complete_page_time=_t(850),
        ),
        PollEvidence(
            requested=Interval(start=_t(1080), end=_t(1091)),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=_t(1110),
            earliest_returned_at=_t(1085),
            earliest_complete_page_time=_t(1085),
        ),
    )


def _never_requested(previous: CollectionManifest, floor: datetime) -> list[attempts.Range]:
    """The part of ``[floor, T)`` in no backfill request of ``previous`` and outside every range its polls read."""

    polls = previous.conversations[0].polls
    requested = [(poll.requested.start, poll.requested.end) for poll in polls if poll.requested.end is not None]
    reads = [attempts.poll_read(poll, received_at=previous.received_at) for poll in polls]
    return attempts.without([(floor, max(end for _start, end in requested))], [*requested, *reads])


def _old_two_backfills_from_the_floor(simulation: attempts.Simulation):
    selection = _selection("c1", history=30)
    simulation.old_rule = True
    simulation.run(selection, at=NOW, pages=[3])
    _post_new(simulation.conversation, NOW + HOUR)
    simulation.run(selection, at=NOW + HOUR, pages=[3, 3])
    _post_new(simulation.conversation, NOW + 2 * HOUR)
    third = simulation.run(selection, at=NOW + 2 * HOUR, pages=[3, 3, 3])
    assert len(third.backfills) == 2
    return selection, NOW + 3 * HOUR, None


def _old_backfills_from_earlier_regular_starts(narrowed: bool):
    def run(simulation: attempts.Simulation):
        old = ConversationSelection(conversation_ids=("c1",), initial_history_days=20, rolling_retention_days=30)
        simulation.old_rule = True
        at = NOW
        for pages in ([3], [3, 3], [3, 3, 3], [3, 3, 3, 3]):
            _post_new(simulation.conversation, at)
            attempt = simulation.run(old, at=at, pages=pages)
            at += 4 * DAY
        # Each continuation starts at an earlier regular start; the oldest is clipped to the retention cutoff.
        starts = sorted({start for start, _end in attempt.backfills})
        assert len(starts) > 1
        assert starts[0] == retention_cutoff(attempt.at, 30)
        new = (
            ConversationSelection(conversation_ids=("c1",), initial_history_days=15, rolling_retention_days=30)
            if narrowed
            else old
        )
        return new, at, None

    return run


def _old_widened_then_extra_sync(simulation: attempts.Simulation):
    wide, at = _old_attempts_widened_last(simulation)
    _post_new(simulation.conversation, at)
    extra = simulation.run(wide, at=at, pages=[3, None, 3, 3])
    return wide, at + HOUR, extra.manifest


def _abstract_widening_then_extra_sync(simulation: attempts.Simulation):
    widened = _selection("c1", history=60)
    simulation.conversation.post(*(_t(day + 0.5) for day in range(1000, 1115)))
    simulation.adopt(_abstract_widened_manifest(), selection=widened)
    simulation.old_rule = True
    extra = simulation.run(widened, at=_t(1115), pages=[3])
    # The extra sync continues the stopped regular poll from its widened start.
    assert extra.backfills[0] == (_t(1050), _just_after(_t(1100)))
    return widened, _t(1120), extra.manifest


@pytest.mark.parametrize(
    "old_attempts",
    [
        _old_two_backfills_from_the_floor,
        _old_backfills_from_earlier_regular_starts(narrowed=False),
        _old_backfills_from_earlier_regular_starts(narrowed=True),
        _old_widened_then_extra_sync,
        _abstract_widening_then_extra_sync,
    ],
    ids=[
        "two_backfills_from_the_floor",
        "backfills_from_earlier_regular_starts",
        "backfills_from_earlier_regular_starts_narrowed",
        "extra_sync_after_a_widening_from_the_regular_start",
        "extra_sync_after_a_widening_with_an_older_chain_below",
    ],
)
def test_the_first_attempt_after_the_upgrade_loses_no_range_from_an_old_manifest(old_attempts):
    simulation = attempts.Simulation(_old_rule_chat())
    selection, at, extra_sync = old_attempts(simulation)
    if extra_sync is not None:
        assert attempts.probe_5(extra_sync) == []
    simulation.old_rule = False

    assert simulation.unrequested(selection, at) == []
    for _index in range(3):
        _post_new(simulation.conversation, at)
        simulation.run(selection, at=at, pages=[3, 3, 3])
        at += HOUR
        assert simulation.unrequested(selection, at) == []


@pytest.mark.parametrize("shape", ["backfills_from_the_regular_start_before_the_widening", "older_chain_below"])
def test_an_old_manifest_widened_in_its_last_attempt_leaves_the_added_range_to_the_pre_deploy_sync(shape):
    if shape == "older_chain_below":
        selection, at, previous = _selection("c1", history=60), _t(1120), _abstract_widened_manifest()
        expected_gap = [(_t(1060), _t(1080))]
    else:
        simulation = attempts.Simulation(_old_rule_chat())
        selection, at = _old_attempts_widened_last(simulation)
        previous = simulation.previous
        expected_gap = None

    request = ConversationLayer.collection_request(selection=selection, now=at, previous=previous)

    gap = _never_requested(previous, attempts.history_floor(selection, at))
    assert gap
    if expected_gap is not None:
        assert gap == expected_gap
    # The documented behaviour: the request asks for none of it, and the probe flags it.
    assert attempts.without(gap, attempts.backfills(request)) == gap
    assert attempts.without(gap, attempts.probe_5(previous)) == []


def test_a_backfill_ending_within_the_visibility_margin_of_its_start_is_requested_again_above_its_proven_end():
    end = NOW - 2 * MINUTE
    previous = attempts.manifest(
        NOW,
        PollEvidence(
            requested=Interval(start=NOW - 30 * DAY),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=NOW,
            earliest_returned_at=NOW - 2 * DAY,
            earliest_complete_page_time=NOW - 2 * DAY,
        ),
        PollEvidence(
            requested=Interval(start=NOW - 30 * DAY, end=end),
            outcome=PollOutcome.REACHED_START,
            started_at=NOW,
            earliest_complete_page_time=NOW - 31 * DAY,
        ),
    )

    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW + HOUR, previous=previous)

    assert attempts.without([(NOW - VISIBILITY_MARGIN, end)], attempts.backfills(request)) == []


@pytest.mark.parametrize(
    ("outcome", "earliest", "complete"),
    [
        (PollOutcome.REACHED_START, NOW - 3 * DAY, NOW - 31 * DAY),
        (PollOutcome.HISTORY_EXHAUSTED, NOW - 2 * DAY, NOW - 2 * DAY),
        (PollOutcome.HISTORY_EXHAUSTED, None, None),
    ],
    ids=["reached_start", "history_exhausted_well_after_the_floor", "history_exhausted_with_nothing_returned"],
)
def test_a_range_a_poll_read_to_its_start_or_to_the_end_of_history_is_not_backfilled(outcome, earliest, complete):
    previous = attempts.manifest(
        NOW,
        PollEvidence(
            requested=Interval(start=NOW - 30 * DAY),
            outcome=outcome,
            started_at=NOW,
            earliest_returned_at=earliest,
            earliest_complete_page_time=complete,
        ),
    )

    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW + HOUR, previous=previous)

    assert attempts.backfills(request) == []


def test_a_completed_backfill_and_a_complete_regular_poll_end_the_chain():
    previous = attempts.manifest(
        NOW,
        PollEvidence(
            requested=Interval(start=NOW - 30 * DAY),
            outcome=PollOutcome.REACHED_START,
            started_at=NOW,
            earliest_returned_at=NOW - 3 * DAY,
            earliest_complete_page_time=NOW - 31 * DAY,
        ),
        PollEvidence(
            requested=Interval(start=NOW - 30 * DAY, end=NOW - 10 * DAY),
            outcome=PollOutcome.REACHED_START,
            started_at=NOW,
            earliest_returned_at=NOW - 12 * DAY,
            earliest_complete_page_time=NOW - 31 * DAY,
        ),
    )

    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW + HOUR, previous=previous)

    assert attempts.backfills(request) == []


def test_a_backfill_never_starts_before_initial_history_or_the_retention_cutoff():
    previous = attempts.manifest(
        NOW - DAY,
        PollEvidence(
            requested=Interval(start=NOW - 61 * DAY),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=NOW - DAY,
            earliest_returned_at=NOW - 10 * DAY,
            earliest_complete_page_time=NOW - 10 * DAY,
        ),
        PollEvidence(
            requested=Interval(start=NOW - 120 * DAY, end=NOW - 50 * DAY),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=NOW - DAY,
            earliest_returned_at=NOW - 70 * DAY,
            earliest_complete_page_time=NOW - 70 * DAY,
        ),
    )
    selection = ConversationSelection(conversation_ids=("c1",), initial_history_days=20, rolling_retention_days=25)

    request = ConversationLayer.collection_request(selection=selection, now=NOW, previous=previous)

    assert attempts.backfills(request) == [(NOW - 20 * DAY, _just_after(NOW - 10 * DAY))]
    cutoff = retention_cutoff(NOW, 25)
    assert cutoff is not None and all(start >= cutoff for start, _end in attempts.backfills(request))


def test_a_regular_poll_bounded_by_a_ceiling_is_not_backfilled_below_its_start():
    selection = ConversationSelection(
        conversation_ids=("c1",), initial_history_days=60, regular_ceiling=timedelta(days=30)
    )
    previous = attempts.manifest(
        NOW - DAY,
        PollEvidence(
            requested=Interval(start=NOW - 31 * DAY),
            outcome=PollOutcome.STOPPED_EARLY,
            started_at=NOW - DAY,
            earliest_returned_at=NOW - 10 * DAY,
            earliest_complete_page_time=NOW - 10 * DAY,
        ),
    )

    request = ConversationLayer.collection_request(selection=selection, now=NOW, previous=previous)

    assert request.conversations[0].intervals[0] == Interval(start=NOW - 30 * DAY)
    assert attempts.backfills(request) == [(NOW - 31 * DAY, _just_after(NOW - 10 * DAY))]


@pytest.mark.parametrize("access", [ConversationAccess.NO_ACCESS, ConversationAccess.GONE])
def test_an_unreadable_conversation_carries_no_backfill(access):
    previous = _manifest(_status("c1", access))

    request = ConversationLayer.collection_request(selection=_selection("c1"), now=NOW + HOUR, previous=previous)

    assert request.conversations[0].intervals == (Interval(start=NOW + HOUR - 30 * DAY),)


# ----------------------------------------------------------------------
# Seeded sequences of attempts
# ----------------------------------------------------------------------


def _random_chat(rng: random.Random, at: datetime) -> attempts.Conversation:
    """Sixty days of history on an hourly grid, so messages may share a time."""

    return attempts.Conversation([at - timedelta(hours=rng.randrange(1, 60 * 24)) for _ in range(rng.randrange(20, 150))])


def _arrivals(rng: random.Random, chat: attempts.Conversation, since: datetime, until: datetime, count: int) -> None:
    """``count`` new messages in ``(since, until]`` on a minute grid, some within the visibility margin of ``until``."""

    span = max(1, int((until - since) / MINUTE))
    for _ in range(count):
        minutes = rng.randrange(0, min(span, 5)) if rng.random() < 0.25 else rng.randrange(0, span)
        chat.post(until - timedelta(minutes=minutes))


def _next_attempt_time(rng: random.Random, at: datetime) -> datetime:
    roll = rng.random()
    if roll < 0.3:
        return at + timedelta(minutes=rng.randint(6, 60))
    if roll < 0.9:
        return at + timedelta(hours=rng.randint(1, 48))
    return at + timedelta(days=rng.randint(3, 12))


def _changed_history(rng: random.Random, days: int, *, widen: float, narrow: float) -> int:
    roll = rng.random()
    if roll < widen:
        return min(60, days + rng.randint(1, 20))
    if roll < widen + narrow:
        return max(1, days - rng.randint(1, 10))
    return days


def _restart_is_explained(previous: attempts.Attempt, attempt: attempts.Attempt) -> bool:
    """Whether a request whose lowest backfill starts at the floor follows a case §10.3 names.

    It continues a backfill of the previous request that the previous attempt
    did not finish at that point; it follows an Initial History expansion, or
    an attempt that left nothing unread after the floor: a previous request
    without backfills (after a complete regular poll or an unreadable
    attempt), or a floor that rose past every previous backfill; or the floor
    is the instant the previous regular poll is complete from.
    """

    floor = attempt.floor
    point = [(floor, _just_after(floor))]
    if any(start <= floor < end for start, end in previous.backfills) and attempts.without(point, previous.reads()):
        return True
    if attempt.request.conversations[0].intervals[0].start < previous.regular.requested.start:
        return True
    if not previous.backfills or floor >= max(end for _start, end in previous.backfills):
        return True
    return previous.regular.earliest_complete_page_time == floor


def _arbitrary_stops(rng: random.Random) -> None:
    at = NOW
    chat = _random_chat(rng, at)
    simulation = attempts.Simulation(chat)
    days = rng.randint(2, 30)
    for _attempt in range(rng.randint(5, 18)):
        days = _changed_history(rng, days, widen=0.15, narrow=0.15)
        selection = _selection("c1", history=days)
        if simulation.attempts and simulation.attempts[-1].readable:
            assert simulation.unrequested(selection, at) == []
        page_size = rng.randint(1, 4)
        attempt = simulation.run(
            selection,
            at=at,
            pages=lambda _interval, _index: None if rng.random() < 0.35 else rng.randint(1, 3),
            page_size=page_size,
            hidden=timedelta(minutes=rng.choice([0, 0, 1, 3])),
            received_after=timedelta(seconds=rng.randint(0, 120)),
            readable=rng.random() > 0.08,
        )
        assert len(attempt.backfills) <= 2
        assert all(start >= attempt.floor for start, _end in attempt.backfills)
        following = _next_attempt_time(rng, at)
        _arrivals(rng, chat, at, following, rng.randint(0, 12))
        at = following


def _budget_stops(rng: random.Random) -> None:
    at = NOW
    chat = _random_chat(rng, at)
    simulation = attempts.Simulation(chat)
    days = rng.randint(2, 30)
    for _attempt in range(rng.randint(5, 18)):
        days = _changed_history(rng, days, widen=0.1, narrow=0.1)
        selection = _selection("c1", history=days)
        if simulation.attempts and simulation.attempts[-1].readable:
            assert simulation.unrequested(selection, at) == []
        attempt = simulation.run(
            selection,
            at=at,
            # The time budget lets a poll page through two pages, give or take one.
            pages=lambda _interval, _index: 2 + rng.randint(-1, 1),
            page_size=5,
            hidden=timedelta(minutes=rng.choice([0, 0, 1, 3])),
            received_after=timedelta(seconds=rng.randint(0, 120)),
            readable=rng.random() > 0.05,
        )
        assert len(attempt.backfills) <= 2
        previous = simulation.attempts[-2] if len(simulation.attempts) > 1 else None
        if previous is not None and previous.readable and attempt.backfills and attempt.backfills[0][0] == attempt.floor:
            assert _restart_is_explained(previous, attempt)
        roll = rng.random()
        count = 0 if roll < 0.3 else rng.randint(20, 40) if roll < 0.35 else rng.randint(1, 8)
        following = _next_attempt_time(rng, at)
        _arrivals(rng, chat, at, following, count)
        at = following


def _settled_inflow(rng: random.Random) -> None:
    at = NOW
    # Dense history, so the regular poll keeps stopping, and no message within
    # the visibility margin or hidden, so the margin plays no part.
    chat = attempts.Conversation(
        [at - timedelta(minutes=rng.randrange(10, 15 * 24 * 60)) for _ in range(rng.randrange(250, 350))]
    )
    simulation = attempts.Simulation(chat)
    selection = _selection("c1", history=10)
    completed: int | None = None
    for index in range(rng.randint(10, 20)):
        attempt = simulation.run(selection, at=at, pages=lambda _interval, _index: 2, page_size=5)
        if completed is not None and index > completed:
            before, previous = simulation.attempts[index - 2], simulation.attempts[index - 1]
            arrived = len(chat.between(before.at, previous.at))
            instant = _complete_from(previous.regular)
            at_instant = sum(1 for time in chat.times if time == instant)
            assert sum(len(returned) for returned in attempt.returned[1:]) <= arrived + at_instant
        if completed is None and any(
            start == attempt.floor and poll.outcome is not PollOutcome.STOPPED_EARLY
            for (start, _end), poll in zip(attempt.backfills, attempt.polls[1:])
        ):
            completed = index
        following = at + timedelta(hours=rng.randint(1, 12))
        # New messages below one poll's budget of ten, none within the margin.
        for _ in range(rng.randint(0, 9)):
            chat.post(following - timedelta(minutes=rng.randrange(10, max(11, int((following - at) / MINUTE)))))
        at = following


@pytest.mark.parametrize(
    ("run", "sequences"),
    [(_arbitrary_stops, 1000), (_budget_stops, 1000), (_settled_inflow, 300)],
    ids=["arbitrary_stops", "budget_stops", "settled_inflow"],
)
def test_over_any_sequence_of_attempts_no_range_after_the_floor_is_left_unrequested(run, sequences):
    """Seeded sequences of attempts over a modelled conversation (``tests/conversation_attempts_fixture``).

    - Arbitrary stops: the floor rises with time, drops when Initial History
      is widened and rises when it is narrowed; the visibility margin applies,
      recent messages may be hidden, messages may share a time, and some
      attempts cannot read the conversation. After each readable attempt,
      every read so far and the next request's backfills cover
      ``[next floor, last regular proven end)``, and a request carries at most
      two backfills.
    - Budget stops: polls stop only on a jittered page budget, with quiet
      attempts and bursts. Besides the above, a request starts at the floor
      only in the cases CD §10.3 names, so re-reads from the floor stay bounded.
    - Settled inflow: Initial History is fixed, the margin plays no part, and
      new messages stay below one poll's budget. Once the lower backfill has
      completed, each attempt's backfills read at most the messages that
      arrived between the two attempts before it plus the messages at one
      instant.
    """

    for seed in range(sequences):
        run(random.Random(seed))


def test_the_manifest_round_trips_through_its_payload():
    message, _record_value = _manifest_message("c1", "m1", NOW - timedelta(hours=1))
    manifest = _manifest(_readable("c1"), _status("c2", ConversationAccess.NO_ACCESS), messages=[message])

    assert manifest_from_payload(manifest_to_payload(manifest)) == manifest


def test_a_manifest_message_must_belong_to_a_readable_conversation():
    message, _record_value = _manifest_message("c2", "m1", NOW)

    with pytest.raises(ValueError, match="readable conversation"):
        _manifest(_status("c2", ConversationAccess.NO_ACCESS), messages=[message])


async def test_a_new_unit_ending_before_the_retention_cutoff_is_not_projected():
    message, record = _manifest_message("c1", "m-old", NOW - timedelta(days=40))
    layer = ConversationLayer(InMemoryProjection())

    plan = await layer.plan(
        manifest=_manifest(_readable("c1", days=60), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1", retention=30, history=30),
        attempt_started_at=NOW,
    )

    assert plan.units == () and plan.removals == ()


async def test_a_changed_unit_ending_before_the_retention_cutoff_is_removed():
    old = _member("c1", "m-old", NOW - timedelta(days=40))
    layer = ConversationLayer(InMemoryProjection(_window("w-old", "c1", old)))
    message, _ = _manifest_message("c1", "m-old", old.created_at, "edited")
    record = _record("c1", "m-old", old.created_at, "edited")

    plan = await layer.plan(
        manifest=_manifest(_readable("c1", days=60), messages=[message]),
        bodies={message.identity: record},
        selection=_selection("c1", retention=30, history=30),
        attempt_started_at=NOW,
    )

    assert plan.units == ()
    assert _removals(plan) == [("w-old", UnitRemovalReason.OUTSIDE_ROLLING_RETENTION)]


async def test_a_held_message_needs_no_body_at_its_stored_revision():
    held = _member("c1", "m1", NOW - timedelta(hours=2))
    layer = ConversationLayer(InMemoryProjection(_window("w", "c1", held)))
    message, _ = _manifest_message("c1", "m1", held.created_at)
    new, new_record = _manifest_message("c1", "m2", NOW - timedelta(hours=1))

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[message, new]),
        bodies={new.identity: new_record},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    (unit,) = plan.units
    assert unit.provider_key == "w"
    assert [member.identity.message_id for member in unit.members] == ["m1", "m2"]


async def test_a_message_whose_waived_body_is_no_longer_held_is_kept_until_the_next_attempt():
    held = _member("c1", "m1", NOW - timedelta(hours=2))
    layer = ConversationLayer(InMemoryProjection(_window("w", "c1", held)))
    edited, _ = _manifest_message("c1", "m1", held.created_at, "edited since the body was waived")

    plan = await layer.plan(
        manifest=_manifest(_readable("c1"), messages=[edited]),
        bodies={},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    assert plan.units == () and plan.removals == ()


def _thread(key: str, conversation_id: str, root_id: str, *members: Member) -> StoredUnit:
    return StoredUnit(
        ref=UnitRef(unit_id=key, document_id=f"doc-{key}", provider_key=key, kind=UnitKind.THREAD,
                    conversation_id=conversation_id, thread_root_id=root_id),
        members=members,
    )


class SpyProjection(InMemoryProjection):
    """Records the Units each ``read`` asks for."""

    def __init__(self, *units: StoredUnit) -> None:
        super().__init__(*units)
        self.reads: list[set[str]] = []

    async def read(self, units: Collection[UnitRef]):
        self.reads.append({unit.provider_key for unit in units})
        return await super().read(units)


async def test_the_layer_reads_exactly_the_units_the_read_set_names():
    hour = timedelta(hours=1)
    root_r, root_q = _member("c1", "r", NOW - 9 * hour), _member("c1", "q", NOW - 8 * hour)
    projection = SpyProjection(
        _window("w-rooted", "c1", root_r),
        _thread("t-rooted", "c1", "r", root_r, _member("c1", "x", NOW - 7 * hour)),
        _window("w-lending", "c1", root_q),
        _thread("t-rootless", "c1", "q", _member("c1", "z", NOW - 6 * hour)),
        _window("w-edited", "c1", _member("c1", "a", NOW - 5 * hour)),
        _window("w-elsewhere", "c1", _member("c1", "b", NOW - 4 * hour)),
    )
    edited, edited_record = _manifest_message("c1", "a", NOW - 5 * hour, "edited")
    reply, reply_record = _manifest_message("c1", "y", NOW - 3 * hour)
    reply = ManifestMessage(
        identity=reply.identity, revision=reply.revision, created_at=reply.created_at, thread_root_id="r"
    )

    await ConversationLayer(projection).plan(
        manifest=_manifest(_readable("c1"), messages=[edited, reply]),
        bodies={edited.identity: edited_record, reply.identity: reply_record},
        selection=_selection("c1"),
        attempt_started_at=NOW,
    )

    threads = {unit.thread_root_id: unit for unit in await projection.thread_units(SOURCE, "c1")}
    named = units_to_read(
        threads=threads,
        holders=await projection.holders(SOURCE, "c1", {"a", "y", "r", *threads}),
        message_ids={"a", "y"},
        replied_roots={"r"},
    )
    assert projection.reads[0] == {unit.provider_key for unit in named}
    assert projection.reads[0] == {"w-rooted", "t-rooted", "w-lending", "t-rootless", "w-edited"}
