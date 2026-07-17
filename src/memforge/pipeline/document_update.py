"""Document update planning for memory extraction.

This module decides how an updated source item should be processed. It keeps the
decision source-agnostic: genes still produce stable normalized markdown, while
the sync pipeline decides whether an update is small enough for diff-guided
extraction or should fall back to full-document extraction.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "DEFAULT_MAX_CHANGED_RATIO",
    "DEFAULT_MAX_DIFF_CHARS",
    "DEFAULT_MAX_DIFF_LINES",
    "DocumentUpdatePlan",
    "plan_document_update",
    "quote_overlaps_current_changes",
]

DEFAULT_MAX_DIFF_LINES = 400
DEFAULT_MAX_DIFF_CHARS = 40_000
DEFAULT_MAX_CHANGED_RATIO = 0.40


@dataclass(frozen=True)
class DocumentUpdatePlan:
    """Decision for processing a changed normalized document."""

    mode: Literal["diff_guided", "full_document"]
    reason: str
    data_shape: str
    changed_hunks: str | None = None
    diff_line_count: int = 0
    added_lines: int = 0
    removed_lines: int = 0
    changed_ratio: float = 0.0
    current_changed_ranges: tuple[tuple[int, int], ...] = ()
    fallback_from: str | None = None
    thresholds: dict[str, float | int] = field(default_factory=dict)


def plan_document_update(
    *,
    previous_content: str | None,
    updated_content: str,
    data_shape: str,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
) -> DocumentUpdatePlan:
    """Choose diff-guided or full-document extraction for an updated source item."""
    thresholds = {
        "max_diff_lines": max_diff_lines,
        "max_diff_chars": max_diff_chars,
        "max_changed_ratio": max_changed_ratio,
    }

    if previous_content is None:
        return DocumentUpdatePlan(
            mode="full_document",
            reason="previous_content_missing",
            data_shape=data_shape,
            fallback_from="diff_guided",
            thresholds=thresholds,
        )

    previous_lines = previous_content.splitlines()
    updated_lines = updated_content.splitlines()
    current_changed_ranges = _current_changed_ranges(
        previous_lines=previous_lines,
        updated_lines=updated_lines,
        updated_content=updated_content,
    )
    diff_lines = list(
        difflib.unified_diff(
            previous_lines,
            updated_lines,
            fromfile="previous",
            tofile="updated",
            lineterm="",
        )
    )
    diff_text = "\n".join(diff_lines)

    added_lines = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed_lines = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    changed_ratio = (added_lines + removed_lines) / max(len(previous_lines), 20)

    if not diff_text:
        return DocumentUpdatePlan(
            mode="full_document",
            reason="empty_diff",
            data_shape=data_shape,
            fallback_from="diff_guided",
            thresholds=thresholds,
        )

    if len(diff_text) > max_diff_chars:
        return DocumentUpdatePlan(
            mode="full_document",
            reason="diff_payload_too_large",
            data_shape=data_shape,
            changed_hunks=diff_text,
            diff_line_count=len(diff_lines),
            added_lines=added_lines,
            removed_lines=removed_lines,
            changed_ratio=changed_ratio,
            current_changed_ranges=current_changed_ranges,
            fallback_from="diff_guided",
            thresholds=thresholds,
        )

    if len(diff_lines) > max_diff_lines:
        return DocumentUpdatePlan(
            mode="full_document",
            reason="diff_too_large",
            data_shape=data_shape,
            changed_hunks=diff_text,
            diff_line_count=len(diff_lines),
            added_lines=added_lines,
            removed_lines=removed_lines,
            changed_ratio=changed_ratio,
            current_changed_ranges=current_changed_ranges,
            fallback_from="diff_guided",
            thresholds=thresholds,
        )

    if changed_ratio > max_changed_ratio:
        return DocumentUpdatePlan(
            mode="full_document",
            reason="changed_ratio_too_high",
            data_shape=data_shape,
            changed_hunks=diff_text,
            diff_line_count=len(diff_lines),
            added_lines=added_lines,
            removed_lines=removed_lines,
            changed_ratio=changed_ratio,
            current_changed_ranges=current_changed_ranges,
            fallback_from="diff_guided",
            thresholds=thresholds,
        )

    return DocumentUpdatePlan(
        mode="diff_guided",
        reason="small_diff",
        data_shape=data_shape,
        changed_hunks=diff_text,
        diff_line_count=len(diff_lines),
        added_lines=added_lines,
        removed_lines=removed_lines,
        changed_ratio=changed_ratio,
        current_changed_ranges=current_changed_ranges,
        thresholds=thresholds,
    )


def quote_overlaps_current_changes(
    updated_content: str,
    evidence_quote: str,
    current_changed_ranges: tuple[tuple[int, int], ...],
) -> bool:
    """Return whether an exact current quote intersects an inserted/replaced range."""

    quote = evidence_quote.strip()
    if not quote or not current_changed_ranges:
        return False
    offset = updated_content.find(quote)
    while offset >= 0:
        quote_end = offset + len(quote)
        if any(offset < range_end and quote_end > range_start for range_start, range_end in current_changed_ranges):
            return True
        offset = updated_content.find(quote, offset + 1)
    return False


def _current_changed_ranges(
    *,
    previous_lines: list[str],
    updated_lines: list[str],
    updated_content: str,
) -> tuple[tuple[int, int], ...]:
    """Map inserted/replaced updated lines to merged character ranges."""

    line_offsets = [0]
    for line in updated_content.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    if len(line_offsets) <= len(updated_lines):
        line_offsets.append(len(updated_content))

    ranges = []
    # Match unified_diff's SequenceMatcher defaults so the executable ranges
    # grant exactly the same authority the model saw in changed_hunks.
    matcher = difflib.SequenceMatcher(a=previous_lines, b=updated_lines)
    for tag, _previous_start, _previous_end, current_start, current_end in matcher.get_opcodes():
        if tag not in {"insert", "replace"} or current_start == current_end:
            continue
        ranges.append((line_offsets[current_start], line_offsets[current_end]))
    if not ranges:
        return ()
    merged = [ranges[0]]
    for range_start, range_end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if range_start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, range_end))
        else:
            merged.append((range_start, range_end))
    return tuple(merged)
