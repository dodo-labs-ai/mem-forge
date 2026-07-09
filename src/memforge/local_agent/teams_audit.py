"""Compact audit helpers for Teams local-agent sync runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any

_HASH_PREFIX = "sha256:"

_DROP_KEYS = frozenset(
    {
        "authorization",
        "bearer_token",
        "token",
        "raw_message_body",
        "message_body",
        "content",
        "participant_display_names",
        "imdisplayname",
    }
)

_HASH_KEYS = frozenset(
    {
        "raw_conversation_id",
        "conversation_key",
        "raw_message_id",
        "raw_root_message_id",
        "metadata_sync_state",
        "metadata_backward_link",
    }
)

_CLAIM_COUNT_KEYS = (
    "claim_add",
    "claim_update",
    "claim_supersede",
    "claim_noop",
    "claim_rejected_ambiguous",
)


def redact_teams_audit_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe Teams audit event without raw sensitive values."""
    redacted: dict[str, Any] = {}
    for key, value in event.items():
        normalized_key = str(key)
        lowered = normalized_key.lower()
        if lowered in _DROP_KEYS:
            continue
        if lowered in _HASH_KEYS:
            redacted[f"{normalized_key}_hash"] = _stable_hash(value)
            continue
        redacted[normalized_key] = _json_safe(value)
    return redacted


def write_teams_audit_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append one redacted Teams audit event as JSONL and return it."""
    redacted = redact_teams_audit_event(event)
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(redacted, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.chmod(path, 0o600)
    return redacted


def validate_teams_audit_run(events: Iterable[Mapping[str, Any]]) -> list[str]:
    """Validate the run-level Teams audit invariants used for next-day checks."""
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        run_id = str(event.get("run_id") or "")
        if run_id:
            by_run[run_id].append(event)

    errors: list[str] = []
    for run_id, run_events in sorted(by_run.items()):
        polls = [event for event in run_events if event.get("event") == "teams_conversation_poll"]
        projections = [event for event in run_events if event.get("event") == "teams_window_projection"]
        patches = [event for event in run_events if event.get("event") == "teams_memory_patch"]
        summaries = [event for event in run_events if event.get("event") == "teams_sync_run"]

        for poll in polls:
            _validate_poll(run_id, poll, errors)
        _validate_projection_linkage(run_id, projections, patches, errors)
        if summaries:
            _validate_window_totals(run_id, summaries[-1], projections, patches, errors)
            _validate_claim_totals(run_id, summaries[-1], patches, errors)

    return errors


def _validate_poll(run_id: str, poll: Mapping[str, Any], errors: list[str]) -> None:
    if poll.get("pagination_complete") is not True:
        errors.append(f"poll pagination incomplete for run {run_id}")
    if poll.get("access_probe_status") != "ok":
        errors.append(f"poll access probe not ok for run {run_id}")
    if not poll.get("covered_created_from") or not poll.get("covered_created_to"):
        errors.append(f"poll coverage range missing for run {run_id}")

    raw_messages = _int_value(poll.get("raw_messages_seen"))
    unique_messages = _int_value(poll.get("unique_message_keys_seen"))
    duplicate_messages = _int_value(poll.get("duplicate_raw_messages"))
    if raw_messages != unique_messages + duplicate_messages:
        errors.append(f"poll raw/unique counts do not reconcile for run {run_id}")

    selected_messages = _int_value(poll.get("selected_message_keys_seen"))
    if not selected_messages and "selected_message_keys_seen" not in poll:
        selected_messages = unique_messages
    terminal_actions = sum(
        _int_value(poll.get(key))
        for key in (
            "upsert_new",
            "upsert_updated",
            "upsert_unchanged",
            "explicit_delete_markers",
            "missing_once_candidates",
        )
    )
    if selected_messages != terminal_actions:
        errors.append(f"poll ledger action counts do not reconcile for run {run_id}")


def _validate_projection_linkage(
    run_id: str,
    projections: list[Mapping[str, Any]],
    patches: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    patches_by_revision: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for patch in patches:
        patches_by_revision[(str(patch.get("window_id_hash") or ""), str(patch.get("revision_hash") or ""))].append(
            patch
        )

    projections_by_revision: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for projection in projections:
        key = (str(projection.get("window_id_hash") or ""), str(projection.get("revision_hash") or ""))
        projections_by_revision[key].append(projection)

        receipt_status = projection.get("receipt_status")
        matching_patches = patches_by_revision.get(key, [])
        if receipt_status == "new" and len(matching_patches) != 1:
            errors.append(
                f"new projection missing one matching memory patch for run {run_id} "
                f"window {key[0]} revision {key[1]}"
            )
        if receipt_status == "existing":
            if matching_patches:
                errors.append(
                    f"existing projection has unexpected memory patch for run {run_id} "
                    f"window {key[0]} revision {key[1]}"
                )
            if not projection.get("receipt_skip_reason"):
                errors.append(
                    f"existing projection missing skip reason for run {run_id} "
                    f"window {key[0]} revision {key[1]}"
                )

    for key, repeated in projections_by_revision.items():
        if len(repeated) > 1 and any(event.get("receipt_status") != "existing" for event in repeated):
            errors.append(
                f"duplicate new projection for run {run_id} "
                f"window {key[0]} revision {key[1]}"
            )


def _validate_claim_totals(
    run_id: str,
    summary: Mapping[str, Any],
    patches: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    for key in _CLAIM_COUNT_KEYS:
        expected = _int_value(summary.get(key))
        actual = sum(_int_value(patch.get(key)) for patch in patches)
        if expected != actual:
            errors.append(f"sync summary {key} does not match memory patches for run {run_id}")


def _validate_window_totals(
    run_id: str,
    summary: Mapping[str, Any],
    projections: list[Mapping[str, Any]],
    patches: list[Mapping[str, Any]],
    errors: list[str],
) -> None:
    if "selected_windows" in summary and _int_value(summary.get("selected_windows")) != len(projections):
        errors.append(f"sync summary selected_windows does not match projections for run {run_id}")

    pushed_patches = [
        patch for patch in patches
        if str(patch.get("patch_status") or "pushed") == "pushed"
    ]
    failed_patches = [
        patch for patch in patches
        if str(patch.get("patch_status") or "") == "failed"
    ]
    if "pushed_windows" in summary and _int_value(summary.get("pushed_windows")) != len(pushed_patches):
        errors.append(f"sync summary pushed_windows does not match pushed patches for run {run_id}")
    if "failed_windows" in summary and _int_value(summary.get("failed_windows")) != len(failed_patches):
        errors.append(f"sync summary failed_windows does not match failed patches for run {run_id}")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return f"{_HASH_PREFIX}{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in sorted(value.items())}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(nested) for nested in value]
    return str(value)


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
