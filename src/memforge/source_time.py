"""The time a source itself gives a piece of content.

A source time is the provider's own record of when content took its current
form: a page version, a commit, the last edit of a comment or message, an agent
session event. It is never the time MemForge discovered, fetched, received or
synced the content. A source that records no such time has no source time, and
callers keep the value absent instead of substituting another clock.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

SOURCE_UPDATED_AT_KEY = "source_updated_at"
"""``NormalizedContent.source_semantics`` key for the time of a Unit's body."""

WESTERNMOST_ZONE = timezone(timedelta(hours=-12), "UTC-12")
"""UTC-12, the westernmost offset in use, where a local time names its latest instant.

A wall-clock time with no offset, such as ``2026-03-01 15:00``, names a
different instant in every timezone, and the latest of them is the one at
UTC-12: 2026-03-02 03:00 UTC. Comparing that instant with ``since`` never
hides a change. Written at UTC-12, an instant reads back in any timezone as
itself or earlier.
"""


def utc_minute_floor(moment: datetime) -> datetime:
    """The start of the UTC minute holding ``moment``.

    A provider that compares times only to the minute, such as a filter that
    takes ``on_or_after`` a minute, is asked from this floor, so nothing
    within that minute is skipped. An aware time is converted to UTC, a naive
    time is read as UTC, and the seconds and microseconds are dropped. The
    floor is for times the provider already gives in UTC, such as Notion's
    ``created_time``, so a naive value is UTC wall time; it is not the
    ambiguous offset-less local time that :data:`WESTERNMOST_ZONE` reads::

        plus_two = timezone(timedelta(hours=2))
        utc_minute_floor(datetime(2026, 3, 1, 9, 30, 40, tzinfo=timezone.utc))  # 2026-03-01 09:30 UTC
        utc_minute_floor(datetime(2026, 3, 1, 11, 30, 40, tzinfo=plus_two))     # 2026-03-01 09:30 UTC
        utc_minute_floor(datetime(2026, 3, 1, 9, 30, 40))                       # 2026-03-01 09:30 UTC
    """

    if moment.tzinfo is None or moment.utcoffset() is None:
        utc = moment.replace(tzinfo=timezone.utc)
    else:
        utc = moment.astimezone(timezone.utc)
    return utc.replace(second=0, microsecond=0)


def parse_source_time(value: object) -> datetime | None:
    """Parse a source time into UTC; ``None`` when the source gave none.

    Accepts an offset-aware ``datetime`` or ISO 8601 string. A value without a
    timezone offset cannot be placed on the timeline and is rejected.
    """

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip())
    else:
        raise ValueError("source time must be an ISO 8601 string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source time must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def source_time_iso(value: object) -> str | None:
    """A source time as a UTC ISO 8601 string; ``None`` when the source gave none."""

    parsed = parse_source_time(value)
    return parsed.isoformat() if parsed is not None else None


def reported_source_time(value: object) -> str | None:
    """A time a provider reported for content, as UTC ISO 8601.

    ``None`` when the provider gave none, or gave one without a timezone offset
    or in another format: such a value cannot be placed on the timeline, so the
    content has no usable source time.
    """

    try:
        return source_time_iso(value)
    except ValueError:
        logger.warning("Source time %r is not an offset-aware ISO 8601 time; treated as unknown", value)
        return None


def latest_source_time(values: Iterable[object]) -> str | None:
    """The latest of several reported source times, ignoring absent and unusable ones."""

    times = [parse_source_time(time) for value in values if (time := reported_source_time(value)) is not None]
    return max(times).isoformat() if times else None
