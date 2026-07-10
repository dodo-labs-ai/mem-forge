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


def execution_owner_user_id(source: Mapping[str, Any]) -> str | None:
    value = str(source.get("execution_owner_user_id") or "").strip()
    return value or None
