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
