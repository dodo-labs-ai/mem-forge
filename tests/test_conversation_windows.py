from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memforge.conversations.records import MessageIdentity
from memforge.conversations.units import Span, window_unit_key
from memforge.conversations.windows import assign_to_windows

CONVERSATION = "19:chat@thread.v2"


def _at(hhmm: str, day: int = 1) -> datetime:
    hours, minutes = (int(part) for part in hhmm.split(":"))
    return datetime(2026, 10, day, hours, minutes, tzinfo=timezone.utc)


def _message(message_id: str, hhmm: str, day: int = 1) -> tuple[MessageIdentity, datetime]:
    return MessageIdentity(CONVERSATION, message_id), _at(hhmm, day)


def _open(identity: MessageIdentity) -> str:
    return f"w-{identity.message_id}"


def _windows(assignment) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for identity, key in assignment.window_of.items():
        grouped.setdefault(key, []).append(identity.message_id)
    return {key: sorted(ids) for key, ids in sorted(grouped.items())}


def test_messages_within_the_gap_share_a_window_and_an_hour_gap_is_inclusive():
    assignment = assign_to_windows(
        {}, [_message("a", "09:00"), _message("b", "10:00"), _message("c", "11:01")], open_window=_open
    )

    assert _windows(assignment) == {"w-a": ["a", "b"], "w-c": ["c"]}
    assert assignment.opened == {"w-a", "w-c"}


def test_arrival_history_decides_the_windows():
    first = assign_to_windows({}, [_message("a", "10:00"), _message("c", "11:40")], open_window=_open)
    assert _windows(first) == {"w-a": ["a"], "w-c": ["c"]}

    later = assign_to_windows(
        {"w-a": Span(_at("10:00"), _at("10:00")), "w-c": Span(_at("11:40"), _at("11:40"))},
        [_message("b", "10:50")],
        open_window=_open,
    )
    assert later.window_of == {MessageIdentity(CONVERSATION, "b"): "w-a"}
    assert later.opened == frozenset()

    together = assign_to_windows(
        {}, [_message("a", "10:00"), _message("c", "11:40"), _message("b", "10:50")], open_window=_open
    )
    assert _windows(together) == {"w-a": ["a", "b", "c"]}


def test_a_contained_message_joins_the_window_that_started_latest():
    assignment = assign_to_windows(
        {"w-early": Span(_at("09:00"), _at("12:00")), "w-late": Span(_at("10:00"), _at("11:00"))},
        [_message("m", "10:30")],
        open_window=_open,
    )

    assert assignment.window_of == {MessageIdentity(CONVERSATION, "m"): "w-late"}


def test_extending_forward_takes_precedence_over_extending_backward():
    assignment = assign_to_windows(
        {"w-before": Span(_at("09:00"), _at("10:00")), "w-after": Span(_at("10:40"), _at("11:00"))},
        [_message("m", "10:30")],
        open_window=_open,
    )

    assert assignment.window_of == {MessageIdentity(CONVERSATION, "m"): "w-before"}


def test_ties_go_to_the_smaller_window_key():
    assignment = assign_to_windows(
        {"w-b": Span(_at("09:00"), _at("10:00")), "w-a": Span(_at("08:00"), _at("10:00"))},
        [_message("m", "10:20")],
        open_window=_open,
    )

    assert assignment.window_of == {MessageIdentity(CONVERSATION, "m"): "w-a"}


def test_a_window_never_spans_more_than_a_day():
    start = _at("00:00")
    window = Span(start, start + timedelta(hours=23, minutes=30))
    assignment = assign_to_windows(
        {"w-day": window},
        [(MessageIdentity(CONVERSATION, "late"), start + timedelta(hours=24, minutes=10))],
        open_window=_open,
    )

    assert assignment.window_of == {MessageIdentity(CONVERSATION, "late"): "w-late"}


def test_spans_grow_within_one_assignment():
    assignment = assign_to_windows(
        {"w-a": Span(_at("09:00"), _at("09:00"))},
        [_message("b", "09:50"), _message("c", "10:40"), _message("d", "11:30")],
        open_window=_open,
    )

    assert _windows(assignment) == {"w-a": ["b", "c", "d"]}


def test_a_message_whose_window_already_exists_joins_it():
    assignment = assign_to_windows(
        {"w-a": Span(_at("09:00"), _at("09:00"))}, [_message("a", "12:00")], open_window=_open
    )

    assert _windows(assignment) == {"w-a": ["a"]}
    assert assignment.opened == frozenset()


def test_window_keys_are_canonical_and_scoped_to_their_conversation():
    first = window_unit_key(source_id="src", conversation_id="c1", anchor_message_id="m1")
    other = window_unit_key(source_id="src", conversation_id="c2", anchor_message_id="m1")

    assert first.startswith("conv-window:v1:")
    assert first != other
    assert first == window_unit_key(anchor_message_id="m1", conversation_id="c1", source_id="src")
    assert "=" not in first
