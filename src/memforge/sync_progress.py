"""Typed, storage-safe progress snapshots for source synchronization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SYNC_PROGRESS_PHASES = frozenset(
    {
        "waiting_for_device",
        "connecting",
        "discovering",
        "uploading",
        "processing",
        "reconciling",
    }
)
SYNC_PROGRESS_UNITS = frozenset(
    {"item", "page", "file", "issue", "message", "conversation"}
)
_SOURCE_PROGRESS_UNITS = {
    "confluence": "page",
    "github_pages": "page",
    "github_repo": "file",
    "jira": "issue",
    "local_markdown": "file",
    "teams": "message",
}


def source_progress_unit(source_type: str) -> str:
    return _SOURCE_PROGRESS_UNITS.get(source_type, "item")


def source_sync_progress_from_pipeline(
    value: Mapping[str, Any],
    *,
    source_type: str,
) -> dict[str, Any] | None:
    """Translate generic pipeline counters into the public progress contract."""
    phase = str(value.get("phase") or "")
    if phase == "detecting_deletions":
        phase = "reconciling"
    if phase == "complete":
        return None
    if phase not in {"discovering", "processing", "reconciling"}:
        return None
    snapshot: dict[str, Any] = {"schema_version": 1, "phase": phase}
    completed = _non_negative_int(value.get("current", 0), "progress.completed")
    total = _non_negative_int(value.get("total", 0), "progress.total")
    progress: dict[str, Any] = {
        "completed": completed,
        "unit": source_progress_unit(source_type),
    }
    if total > 0:
        progress["total"] = total
    snapshot["progress"] = progress
    counts = {
        "changed": value.get("docs_updated"),
        "failed": value.get("docs_failed"),
        "memories_created": value.get("memories_extracted"),
    }
    normalized_counts = {key: count for key, count in counts.items() if count is not None}
    if normalized_counts:
        snapshot["counts"] = normalized_counts
    return normalize_sync_progress_snapshot(snapshot)


def normalize_sync_progress_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the public source-progress contract."""
    allowed = {"schema_version", "phase", "progress", "source_time_range", "counts"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown sync progress fields: {', '.join(sorted(unknown))}")
    if value.get("schema_version") != 1:
        raise ValueError("sync progress schema_version must be 1")
    phase = str(value.get("phase") or "")
    if phase not in SYNC_PROGRESS_PHASES:
        raise ValueError(f"unsupported sync progress phase: {phase or '<empty>'}")

    normalized: dict[str, Any] = {"schema_version": 1, "phase": phase}
    progress = value.get("progress")
    if progress is not None:
        if not isinstance(progress, Mapping) or set(progress) - {"completed", "total", "unit"}:
            raise ValueError("sync progress.progress has unsupported fields")
        completed = _non_negative_int(progress.get("completed"), "progress.completed")
        total_value = progress.get("total")
        total = None if total_value is None else _non_negative_int(total_value, "progress.total")
        unit = str(progress.get("unit") or "")
        if unit not in SYNC_PROGRESS_UNITS:
            raise ValueError(f"unsupported sync progress unit: {unit or '<empty>'}")
        normalized_progress: dict[str, Any] = {"completed": completed, "unit": unit}
        if total is not None:
            normalized_progress["total"] = total
            normalized_progress["completed"] = min(completed, total)
        normalized["progress"] = normalized_progress

    time_range = value.get("source_time_range")
    if time_range is not None:
        if not isinstance(time_range, Mapping) or set(time_range) - {"start", "end"}:
            raise ValueError("sync progress.source_time_range has unsupported fields")
        normalized_range = {
            key: str(time_range[key])[:64]
            for key in ("start", "end")
            if time_range.get(key)
        }
        if normalized_range:
            normalized["source_time_range"] = normalized_range

    counts = value.get("counts")
    if counts is not None:
        count_fields = {"changed", "failed", "memories_created"}
        if not isinstance(counts, Mapping) or set(counts) - count_fields:
            raise ValueError("sync progress.counts has unsupported fields")
        normalized_counts = {
            key: _non_negative_int(counts[key], f"counts.{key}")
            for key in count_fields
            if counts.get(key) is not None
        }
        if normalized_counts:
            normalized["counts"] = normalized_counts
    return normalized


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed
