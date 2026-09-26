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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SOURCE_UPDATED_AT_KEY = "source_updated_at"
"""``NormalizedContent.source_semantics`` key for the time of a Unit's body."""


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
