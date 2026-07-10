"""Shared execution contract for local-agent-backed sources."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


LOCAL_AGENT_SYNC_OPERATIONS = frozenset(
    {
        "github_repo_sync",
        "jira_sync",
        "local_markdown_sync",
        "teams_sync",
    }
)

_IMMUTABLE_EXECUTION_MODE_FIELDS = {
    "github_repo": ("connection_mode",),
    "jira": ("sync_mode",),
}


def _source_config(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        if isinstance(parsed, Mapping):
            return parsed
    return {}


def local_agent_sync_operation(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> str | None:
    """Return the daemon sync operation for a canonical source configuration."""
    normalized_type = str(source_type or "").strip().lower()
    normalized_config = _source_config(config)
    if normalized_type == "teams":
        return "teams_sync"
    if normalized_type == "jira" and str(
        normalized_config.get("sync_mode") or ""
    ).strip().lower() == "local_agent":
        return "jira_sync"
    if normalized_type == "local_markdown":
        return "local_markdown_sync"
    if normalized_type == "github_repo" and str(
        normalized_config.get("connection_mode") or ""
    ).strip().lower() == "local_push":
        return "github_repo_sync"
    return None


def is_local_agent_backed_source(source: Mapping[str, Any]) -> bool:
    return (
        local_agent_sync_operation(
            str(source.get("type") or ""),
            source.get("config"),
        )
        is not None
    )


def source_execution_descriptor(
    source_type: str,
    config: Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    """Return the canonical execution contract exposed to source clients."""
    normalized_type = str(source_type or "").strip().lower()
    operation = local_agent_sync_operation(normalized_type, config)
    return {
        "kind": "local_agent" if operation is not None else "server",
        "operation": operation,
        "immutable_config_fields": list(
            _IMMUTABLE_EXECUTION_MODE_FIELDS.get(normalized_type, ())
        ),
    }


def execution_owner_user_id(source: Mapping[str, Any]) -> str | None:
    value = str(source.get("execution_owner_user_id") or "").strip()
    return value or None


def source_with_sync_inputs(
    source: Mapping[str, Any],
    inputs: list[Any],
) -> dict[str, Any]:
    """Project immutable raw inputs into the connector's runtime manifest."""
    latest_entries: dict[str, dict[str, Any]] = {}
    for source_input in sorted(
        inputs,
        key=lambda item: int(getattr(item, "input_generation", 0)),
    ):
        metadata = getattr(source_input, "metadata", {})
        entry = metadata.get("manifest_entry") if isinstance(metadata, Mapping) else None
        if not isinstance(entry, Mapping):
            continue
        doc_id = str(entry.get("doc_id") or "").strip()
        raw_uri = str(getattr(source_input, "raw_uri", "") or "").strip()
        if not doc_id or not raw_uri:
            continue
        latest_entries[doc_id] = {**entry, "package_uri": raw_uri}
    projected = dict(source)
    if latest_entries:
        config = dict(_source_config(source.get("config")))
        config["local_agent_package_manifest"] = list(latest_entries.values())
        projected["config"] = config
    return projected
