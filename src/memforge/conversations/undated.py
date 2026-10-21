"""Undated records: stored message records without their own creation time.

A Source that joined the conversation layer may hold message records of an
earlier record schema that carry no ``/time``, such as Teams'
``teams-message`` version 1. Every record the layer collects is dated; an
undated record exists only as stored, and this module is the one place that
reads it.

An undated member takes its Observation's source time as its place in time:
the edit time of a message edited before it was stored, otherwise its
creation time. That time can be later than the message's creation, so an
undated member is never proven absent: it leaves its Unit only by retention,
by a scope change, or as a root moving to its thread. A poll that returns
the message replaces the record with a dated one. Ordering by that time can
place an edited undated root after its first replies.

A Unit payload names the time of each undated member beside its members, so
projection places them exactly as the conversation layer did.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from memforge.conversations.records import record_source_time
from memforge.source_time import parse_source_time

UNDATED_MEMBER_TIMES_KEY = "undated_member_times"


def is_undated(record: Mapping[str, object]) -> bool:
    """Whether a stored record carries no creation time of its own."""

    return parse_source_time(record.get("time")) is None


def stored_member_time(record: Mapping[str, object], *, observed_at: str | None) -> datetime:
    """A stored member's place in time: its record's creation time, or for an undated record its source time."""

    created_at = parse_source_time(record.get("time"))
    if created_at is None:
        created_at = parse_source_time(observed_at)
    if created_at is None:
        raise ValueError("a stored message record needs its creation time or its Observation's source time")
    return created_at


def member_source_time(record: Mapping[str, object], created_at: datetime) -> object:
    """A member's source time: its record's edit or creation time, or for an undated member its place in time."""

    return created_at.isoformat() if is_undated(record) else record_source_time(record)


def undated_member_times(members: Iterable[tuple[str, Mapping[str, object], datetime]]) -> dict[str, str]:
    """The time of each undated member of a Unit, by message id, for its payload."""

    return {message_id: created_at.isoformat() for message_id, record, created_at in members if is_undated(record)}


def payload_member_times(payload: Mapping[str, object], member: Mapping[str, object]) -> tuple[datetime, object, str]:
    """A payload member's creation time, source time and the time text it is shown with.

    A dated record gives all three; an undated member's are the time its
    payload names, which is the source time it was stored with.
    """

    record = member["record"]
    if not is_undated(record):
        return stored_member_time(record, observed_at=None), record_source_time(record), str(record["time"])
    times = payload.get(UNDATED_MEMBER_TIMES_KEY) or {}
    stored = times.get(str(member["message_id"])) if isinstance(times, Mapping) else None
    created_at = stored_member_time(record, observed_at=stored)
    return created_at, created_at.isoformat(), created_at.isoformat()
