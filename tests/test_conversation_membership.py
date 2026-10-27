from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.conversations.coverage import ConversationCoverage, ProvenInterval
from memforge.conversations.membership import (
    ConversationState,
    ReturnedMessage,
    plan_conversation,
)
from memforge.conversations.port import ConversationLabel, ConversationRef, ThreadingModel
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.units import Member, StoredUnit, UnitKind, UnitRef, thread_unit_key, window_unit_key
from memforge.source_projection import UnitRemovalReason

SOURCE = "src-chat"
RECORD_SCHEMA = "conversation-message:1"
CONVERSATION = "c1"
DAY = datetime(2026, 10, 20, tzinfo=timezone.utc)


def _at(hhmm: str) -> datetime:
    hours, minutes = (int(part) for part in hhmm.split(":"))
    return DAY.replace(hour=hours, minute=minutes)


def _member(message_id: str, hhmm: str, text: str = "hello") -> Member:
    record = {"content": text, "time": _at(hhmm).isoformat(), "conversation_id": CONVERSATION}
    return Member(
        identity=MessageIdentity(CONVERSATION, message_id),
        created_at=_at(hhmm),
        record=record,
        revision=record_revision(record),
        record_schema=RECORD_SCHEMA,
    )


def _returned(message_id: str, hhmm: str, text: str = "hello", *, root: str | None = None) -> ReturnedMessage:
    return ReturnedMessage(member=_member(message_id, hhmm, text), thread_root_id=root)


def _window(key: str, *members: Member) -> StoredUnit:
    return StoredUnit(
        ref=UnitRef(unit_id=f"unit-{key}", document_id=f"doc-{key}", provider_key=key, kind=UnitKind.WINDOW,
                    conversation_id=CONVERSATION),
        members=members,
    )


def _thread(key: str, root: str, *members: Member) -> StoredUnit:
    return StoredUnit(
        ref=UnitRef(unit_id=f"unit-{key}", document_id=f"doc-{key}", provider_key=key, kind=UnitKind.THREAD,
                    conversation_id=CONVERSATION, thread_root_id=root),
        members=members,
    )


def _conversation(threading: ThreadingModel = ThreadingModel.INLINE) -> ConversationRef:
    return ConversationRef(
        label=ConversationLabel(conversation_id=CONVERSATION, kind="channel", name="General"),
        threading=threading,
        poll_includes_replies=True,
    )


def _coverage(start: str | None = None, end: str = "23:59") -> ConversationCoverage:
    intervals = () if start is None else (ProvenInterval(start=_at(start), end=_at(end)),)
    return ConversationCoverage(intervals=intervals, covers_replies=True)


def holders_of(units) -> dict[str, tuple[UnitRef, ...]]:
    """The Units holding each message, by message id, as the store answers for a projection of ``units``."""

    holders: dict[str, dict[str, UnitRef]] = {}
    for unit in units:
        for member in unit.members:
            holders.setdefault(member.identity.message_id, {})[unit.ref.unit_id] = unit.ref
    return {message_id: tuple(refs.values()) for message_id, refs in holders.items()}


def _plan(returned=(), units=(), *, coverage=None, threading=ThreadingModel.INLINE, threads=(), seen=()):
    """Plan over ``units`` read with their members, and ``threads`` known only by reference.

    As the layer reads it, a thread is rooted when it holds its root, and the
    store's holders are those of every Unit given.
    """

    read_threads = [unit for unit in units if unit.kind is UnitKind.THREAD]
    return plan_conversation(
        source_id=SOURCE,
        conversation=_conversation(threading),
        coverage=coverage or _coverage(),
        returned=returned,
        seen_without_record=frozenset(MessageIdentity(CONVERSATION, message_id) for message_id in seen),
        state=ConversationState(
            conversation_id=CONVERSATION,
            units=tuple(units),
            holders=holders_of((*units, *threads)),
            threads={
                **{unit.thread_root_id: unit.ref for unit in threads},
                **{unit.thread_root_id: unit.ref for unit in read_threads},
            },
            rooted_threads=frozenset(
                unit.thread_root_id
                for unit in (*threads, *read_threads)
                if any(member.identity.message_id == unit.thread_root_id for member in unit.members)
            ),
        ),
    )


def _members(planned) -> dict[str, list[str]]:
    return {unit.provider_key: [member.identity.message_id for member in unit.members] for unit in planned.units}


def test_new_top_level_messages_open_windows_by_the_policy():
    plan = _plan([_returned("a", "09:00"), _returned("b", "09:30"), _returned("c", "12:00")])

    assert _members(plan) == {
        window_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, anchor_message_id="a"): ["a", "b"],
        window_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, anchor_message_id="c"): ["c"],
    }
    assert all(unit.stored is None for unit in plan.units)


def test_an_unchanged_unit_is_not_projected_again():
    window = _window("w", _member("a", "09:00"))

    plan = _plan([_returned("a", "09:00")], [window], coverage=_coverage("08:00"))

    assert plan.units == () and plan.removals == ()


def test_an_edited_message_replaces_its_record_in_place():
    window = _window("w", _member("a", "09:00"), _member("b", "09:10"))

    plan = _plan([_returned("b", "09:10", "edited")], [window])

    (unit,) = plan.units
    assert unit.provider_key == "w" and unit.stored is window
    assert [member.record["content"] for member in unit.members] == ["hello", "edited"]


def test_a_member_outside_every_proven_interval_is_carried():
    window = _window("w", _member("a", "09:00"), _member("b", "10:40"))

    plan = _plan([_returned("b", "10:40"), _returned("c", "10:50")], [window], coverage=_coverage("10:30"))

    assert _members(plan) == {"w": ["a", "b", "c"]}


def test_only_the_members_inside_the_interval_are_removed_from_a_straddling_window():
    window = _window("w", _member("a", "10:00"), _member("b", "10:40"))

    plan = _plan([], [window], coverage=_coverage("10:30"))

    assert _members(plan) == {"w": ["a"]}
    assert plan.removals == ()


def test_a_window_whose_every_member_is_proven_absent_is_removed_as_absent():
    window = _window("w", _member("a", "10:00"))

    plan = _plan([], [window], coverage=_coverage("09:00"))

    assert plan.units == ()
    assert [(removal.unit.provider_key, removal.reason) for removal in plan.removals] == [
        ("w", UnitRemovalReason.ABSENT_FROM_SOURCE)
    ]


def test_a_root_gaining_replies_moves_to_its_thread_over_two_runs():
    root = _member("r", "09:00")
    window = _window("w", root, _member("b", "09:20"))
    thread_key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")

    first = _plan([_returned("x", "09:40", root="r")], [window])

    assert _members(first) == {thread_key: ["r", "x"]}
    (thread,) = first.units
    assert thread.kind is UnitKind.THREAD and thread.thread_root_id == "r"

    committed_thread = _thread(thread_key, "r", root, _member("x", "09:40"))
    second = _plan([], [window, committed_thread])

    assert _members(second) == {"w": ["b"]}


def test_a_moved_root_that_was_the_only_member_empties_its_window():
    root = _member("r", "09:00")
    window = _window("w", root)

    plan = _plan([], [window, _thread("t", "r", root, _member("x", "09:40"))])

    assert [(removal.unit.provider_key, removal.reason) for removal in plan.removals] == [("w", UnitRemovalReason.EMPTIED)]


def test_a_root_of_an_unloaded_committed_thread_leaves_its_window():
    window = _window("w", _member("r", "09:00"), _member("b", "09:20"))
    thread = _thread("t", "r", _member("r", "09:00"), _member("x", "09:40"))

    plan = _plan([], [window], threads=[thread])

    assert _members(plan) == {"w": ["b"]}


def test_a_thread_without_its_root_takes_it_from_the_window_before_the_window_gives_it_up():
    root = _member("r", "09:00")
    window = _window("w", root, _member("b", "09:20"))
    rootless = _thread("t", "r", _member("x", "09:40"))

    first = _plan([], [window, rootless])

    assert _members(first) == {"t": ["r", "x"]}

    second = _plan([], [window, _thread("t", "r", root, _member("x", "09:40"))])

    assert _members(second) == {"w": ["b"]}


def test_a_thread_without_its_root_takes_the_root_when_it_is_returned():
    rootless = _thread("t", "r", _member("x", "09:40"))

    plan = _plan([_returned("r", "09:00")], [rootless])

    assert _members(plan) == {"t": ["r", "x"]}


def test_a_thread_known_but_not_read_cannot_take_a_reply():
    thread = _thread("t", "r", _member("r", "09:00"))

    with pytest.raises(ValueError, match="was not read"):
        _plan([_returned("y", "15:00", root="r")], [], threads=[thread])


def test_new_replies_join_their_existing_thread():
    thread = _thread("t", "r", _member("r", "09:00"), _member("x", "09:40"))

    plan = _plan([_returned("y", "15:00", root="r")], [thread])

    assert _members(plan) == {"t": ["r", "x", "y"]}


def test_a_new_root_returned_with_its_replies_never_enters_a_window():
    plan = _plan([_returned("r", "09:00"), _returned("x", "09:10", root="r")])

    key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")
    assert _members(plan) == {key: ["r", "x"]}


def test_post_reply_posts_are_threads_from_the_start():
    plan = _plan([_returned("p", "09:00")], threading=ThreadingModel.POST_REPLY)

    key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="p")
    assert _members(plan) == {key: ["p"]}


def test_flat_conversations_have_windows_only():
    plan = _plan([_returned("r", "09:00"), _returned("x", "09:10", root="r")], threading=ThreadingModel.FLAT)

    key = window_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, anchor_message_id="r")
    assert _members(plan) == {key: ["r", "x"]}


def test_a_message_in_a_thread_and_a_window_is_updated_in_both_until_the_root_moves():
    root = _member("r", "09:00")
    window = _window("w", root)
    thread_key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")

    first = _plan([_returned("r", "09:00", "edited"), _returned("x", "09:10", root="r")], [window])

    assert _members(first) == {thread_key: ["r", "x"], "w": ["r"]}
    assert {unit.provider_key: unit.members[0].record["content"] for unit in first.units} == {
        thread_key: "edited",
        "w": "edited",
    }


def test_equal_message_ids_in_two_conversations_never_match():
    other = StoredUnit(
        ref=UnitRef(unit_id="u2", document_id="d2", provider_key="w2", kind=UnitKind.WINDOW, conversation_id="c2"),
        members=(
            Member(identity=MessageIdentity("c2", "a"), created_at=_at("09:00"), record={}, revision="x", record_schema=RECORD_SCHEMA),
        ),
    )

    with pytest.raises(ValueError, match="another conversation"):
        _plan([_returned("a", "09:00")], [other])
    assert MessageIdentity("c1", "a") != MessageIdentity("c2", "a")


def test_spans_follow_the_decided_membership():
    window = _window("w", _member("a", "09:50"), _member("b", "10:40"))

    plan = _plan([_returned("b", "10:40"), _returned("c", "10:50")], [window], coverage=_coverage("10:30"))

    (unit,) = plan.units
    assert (unit.span.earliest, unit.span.latest) == (_at("09:50"), _at("10:50"))


def test_a_root_proven_absent_does_not_join_the_thread_its_reply_opens():
    window = _window("w", _member("r", "10:00"), _member("m", "10:10"))
    thread_key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")

    plan = _plan([_returned("m", "10:10"), _returned("x", "10:20", root="r")], [window], coverage=_coverage("09:00"))

    assert _members(plan) == {thread_key: ["x"], "w": ["m"]}


def test_a_message_seen_without_its_record_is_carried_and_never_proven_absent():
    window = _window("w", _member("a", "10:00"), _member("b", "10:10"))

    plan = plan_conversation(
        source_id=SOURCE,
        conversation=_conversation(),
        coverage=_coverage("09:00"),
        returned=[],
        state=ConversationState(conversation_id=CONVERSATION, units=(window,), holders=holders_of([window])),
        seen_without_record={MessageIdentity(CONVERSATION, "a")},
    )

    assert _members(plan) == {"w": ["a"]}


def _undated(message_id: str, hhmm: str, text: str = "hello") -> Member:
    """A member stored with a record of an earlier schema, placed by its Observation's source time."""

    record = {"attachments": [], "content": text, "deleted": False}
    return Member(
        identity=MessageIdentity(CONVERSATION, message_id),
        created_at=_at(hhmm),
        record=record,
        revision=record_revision(record),
        record_schema=RECORD_SCHEMA,
    )


def test_an_undated_member_is_never_proven_absent():
    window = _window("w", _undated("a", "10:00"), _member("b", "10:10"))

    plan = _plan([], [window], coverage=_coverage("09:00"))

    assert _members(plan) == {"w": ["a"]}
    assert plan.removals == ()


def test_a_returned_message_replaces_its_undated_record():
    window = _window("w", _undated("a", "10:00", "old"))

    plan = _plan([_returned("a", "10:00", "old")], [window], coverage=_coverage("09:00"))

    (unit,) = plan.units
    assert unit.members[0].record["time"] == _at("10:00").isoformat()


def test_an_undated_root_joins_the_thread_its_reply_opens():
    window = _window("w", _undated("r", "10:00"), _member("m", "10:10"))
    thread_key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")

    plan = _plan([_returned("m", "10:10"), _returned("x", "10:20", root="r")], [window], coverage=_coverage("09:00"))

    assert _members(plan)[thread_key] == ["r", "x"]


def test_an_undated_reply_is_never_proven_absent_from_its_thread():
    thread = _thread("t", "r", _member("r", "10:00"), _undated("x", "10:20"))

    plan = _plan([_returned("r", "10:00")], [thread], coverage=_coverage("09:00"))

    assert plan.units == ()
    assert plan.removals == ()


def test_a_plan_refuses_a_state_that_lacks_a_unit_its_read_set_names():
    window = _window("w", _member("a", "09:00"))
    thread = _thread("t", "r", _member("r", "10:00"))

    # Without the holder read, the edited message would open a second window.
    with pytest.raises(ValueError, match="^Unit w was not read for this plan$"):
        plan_conversation(
            source_id=SOURCE,
            conversation=_conversation(),
            coverage=_coverage(),
            returned=[_returned("a", "09:00", "edited")],
            state=ConversationState(conversation_id=CONVERSATION, units=(), holders=holders_of([window])),
        )
    with pytest.raises(ValueError, match="Unit t was not read for this plan; Unit w was not read for this plan"):
        plan_conversation(
            source_id=SOURCE,
            conversation=_conversation(),
            coverage=_coverage(),
            returned=[_returned("a", "09:00", "edited"), _returned("y", "10:30", root="r")],
            state=ConversationState(
                conversation_id=CONVERSATION,
                units=(),
                holders=holders_of([window, thread]),
                threads={"r": thread.ref},
                rooted_threads=frozenset({"r"}),
            ),
        )


def _decided(plan, unit: StoredUnit) -> list[str]:
    """The message ids a stored Unit holds after the plan: projected, removed, or unchanged."""

    if any(removal.unit.provider_key == unit.provider_key for removal in plan.removals):
        return []
    planned = {planned.provider_key: planned for planned in plan.units}.get(unit.provider_key, unit)
    return [member.identity.message_id for member in planned.members]


@pytest.mark.parametrize(
    ("case", "coverage_start", "lent"),
    [
        ("returned", "09:00", True),
        ("seen without record", "09:00", True),
        ("undated", "09:00", True),
        ("inside an interval", "09:00", False),
        ("outside every interval", "10:05", True),
    ],
)
def test_a_held_root_is_lent_to_its_thread_exactly_when_it_is_not_proven_absent(case, coverage_start, lent):
    root = _undated("r", "10:00") if case == "undated" else _member("r", "10:00")
    window = _window("w", root, _member("m", "10:10"))
    thread_key = thread_unit_key(source_id=SOURCE, conversation_id=CONVERSATION, thread_root_id="r")
    returned = [_returned("m", "10:10"), _returned("x", "10:20", root="r")]
    if case == "returned":
        returned.append(_returned("r", "10:00"))

    plan = _plan(
        returned,
        [window],
        coverage=_coverage(coverage_start),
        seen=("r",) if case == "seen without record" else (),
    )

    thread_members = _members(plan)[thread_key]
    assert ("r" in thread_members) is lent
    assert ("r" in _decided(plan, window)) is lent
