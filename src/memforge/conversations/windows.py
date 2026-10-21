"""Window Policy ``gap-window-v1``: assigning top-level messages to windows.

Windows group nearby messages by a 60-minute gap and end at a 24-hour span.
New messages are assigned one by one in ``(creation time, message id)`` order,
and each assignment extends its window's span at once, so the first run and
every later run follow the same rule. A message already in a Unit stays
there; the policy only places new ones, and windows never merge. A version of
the policy never reinterprets a window another version formed.

For each new message, in order of precedence:

1. **Contained**: among windows whose span contains the message time, the
   one that started latest; on a tie, the smaller key.
2. **Extends forward**: among windows that end before the message, at most
   the gap before it, and whose span stays within the cap with it, the
   smallest gap; on a tie, the smaller key.
3. **Extends backward**: among windows that start after the message, at most
   the gap after it, and whose span stays within the cap with it, the
   smallest gap; on a tie, the smaller key.
4. Otherwise a new window, keyed by the message that opened it.

The result depends on arrival history: messages at 10:00 and 11:40 form two
windows, and one at 10:50 that arrives later joins the first; all three
arriving together form one window. For the same arrival history the result is
deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from memforge.conversations.records import MessageIdentity
from memforge.conversations.units import Span


@dataclass(frozen=True, slots=True)
class WindowPolicy:
    version: str
    gap: timedelta
    max_span: timedelta


GAP_WINDOW_V1 = WindowPolicy(version="gap-window-v1", gap=timedelta(minutes=60), max_span=timedelta(hours=24))


@dataclass(frozen=True, slots=True)
class WindowAssignment:
    """Where each new message goes, and the windows this assignment opened."""

    window_of: Mapping[MessageIdentity, str]
    opened: frozenset[str]


def assign_to_windows(
    windows: Mapping[str, Span],
    messages: Iterable[tuple[MessageIdentity, datetime]],
    *,
    open_window: Callable[[MessageIdentity], str],
    policy: WindowPolicy = GAP_WINDOW_V1,
) -> WindowAssignment:
    """Assign new top-level messages to the existing ``windows`` of one conversation, or open new ones.

    ``windows`` maps each window's key to its current span. ``open_window``
    names the window a message opens; when that window already exists, the
    message joins it.
    """

    spans = dict(windows)
    window_of: dict[MessageIdentity, str] = {}
    opened: set[str] = set()
    for identity, created_at in sorted(messages, key=lambda pair: (pair[1], pair[0].message_id)):
        key = _choose_window(spans, created_at, policy)
        if key is None:
            key = open_window(identity)
        span = spans.get(key)
        if span is None:
            opened.add(key)
            spans[key] = Span(earliest=created_at, latest=created_at)
        else:
            # A message that once opened a window and returns opens it again.
            spans[key] = Span(earliest=min(span.earliest, created_at), latest=max(span.latest, created_at))
        window_of[identity] = key
    return WindowAssignment(window_of=window_of, opened=frozenset(opened))


def _choose_window(spans: Mapping[str, Span], at: datetime, policy: WindowPolicy) -> str | None:
    containing = [key for key, span in spans.items() if span.earliest <= at <= span.latest]
    if containing:
        latest_start = max(spans[key].earliest for key in containing)
        return min(key for key in containing if spans[key].earliest == latest_start)
    forward = [
        (at - span.latest, key)
        for key, span in spans.items()
        if span.latest < at and at - span.latest <= policy.gap and at - span.earliest <= policy.max_span
    ]
    if forward:
        return min(forward)[1]
    backward = [
        (span.earliest - at, key)
        for key, span in spans.items()
        if span.earliest > at and span.earliest - at <= policy.gap and span.latest - at <= policy.max_span
    ]
    if backward:
        return min(backward)[1]
    return None
