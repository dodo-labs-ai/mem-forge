"""The shared source-time helpers: the UTC minute floor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memforge.source_time import utc_minute_floor


def test_utc_minute_floor_drops_seconds_and_microseconds() -> None:
    floor = utc_minute_floor(datetime(2026, 3, 1, 9, 30, 40, 123456, tzinfo=timezone.utc))

    assert floor == datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    assert floor.tzinfo is timezone.utc


def test_utc_minute_floor_converts_an_offset_to_utc() -> None:
    # 00:15:59 at UTC+05:45 is the previous day's 18:30:59 in UTC.
    moment = datetime(2026, 3, 2, 0, 15, 59, tzinfo=timezone(timedelta(hours=5, minutes=45)))

    floor = utc_minute_floor(moment)

    assert floor == datetime(2026, 3, 1, 18, 30, tzinfo=timezone.utc)
    assert floor.utcoffset() == timedelta(0)


def test_utc_minute_floor_reads_a_naive_time_as_utc() -> None:
    floor = utc_minute_floor(datetime(2026, 3, 1, 9, 30, 40))

    assert floor == datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    assert floor.tzinfo is timezone.utc
