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
    "DEFAULT_MAX_DIFF_LINES",
    "DocumentUpdatePlan",
    "plan_document_update",
]

DEFAULT_MAX_DIFF_LINES = 400
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
    fallback_from: str | None = None
    thresholds: dict[str, float | int] = field(default_factory=dict)


def plan_document_update(
    *,
    previous_content: str | None,
    updated_content: str,
    data_shape: str,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_changed_ratio: float = DEFAULT_MAX_CHANGED_RATIO,
) -> DocumentUpdatePlan:
    """Choose diff-guided or full-document extraction for an updated source item."""
    thresholds = {
        "max_diff_lines": max_diff_lines,
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
        thresholds=thresholds,
    )
