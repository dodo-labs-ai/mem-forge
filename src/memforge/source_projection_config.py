"""Canonical provider namespace and projection-scope config boundaries."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any


_NAMESPACE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "confluence": ("base_url",),
    "jira": ("base_url",),
    "github_repo": ("repo_url",),
    # vault_id is the portable source namespace; root is a device-local locator
    # and may change without changing file lineage.
    "local_markdown": ("vault_id",),
    "agent_session": ("client",),
}

_SCOPE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "confluence": (
        "sync_mode",
        "spaces",
        "page_tree_root",
        "exclude_labels",
        "include_children",
    ),
    "jira": (
        "query_mode",
        "projects",
        "jql",
        "jql_filter",
        "issue_types",
        "include_comments",
    ),
    "github_repo": ("ref", "include_paths", "exclude_paths", "include_extensions"),
    "github_pages": (
        "sync_mode",
        "page_url",
        "root_url",
        "pages",
        "max_depth",
        "exclude_url_patterns",
    ),
    "local_markdown": ("include", "exclude"),
    # These three Teams settings change frozen-window membership and therefore
    # belong to projection scope, not operational retry/pacing configuration.
    "teams": (
        "conversation_ids",
        "channels",
        "group_chats",
        "individual_chats",
        "max_age_days",
        "conversation_gap_minutes",
        "max_block_messages",
    ),
}

_SET_LIKE_FIELDS = frozenset(
    {
        "spaces",
        "exclude_labels",
        "projects",
        "issue_types",
        "include_paths",
        "exclude_paths",
        "include_extensions",
        "pages",
        "exclude_url_patterns",
        "include",
        "exclude",
        "conversation_ids",
        "channels",
        "group_chats",
        "individual_chats",
    }
)
_URL_FIELDS = frozenset({"base_url", "repo_url", "page_url", "root_url"})


def canonical_provider_namespace(source_type: str, config: Mapping[str, Any]) -> dict[str, object]:
    """Return immutable identity-bearing provider configuration only."""

    return _canonical_fields(config, _NAMESPACE_FIELDS.get(source_type, ()))


def canonical_projection_scope(source_type: str, config: Mapping[str, Any]) -> dict[str, object]:
    """Return only fields that can change projected unit/observation membership."""

    scope = _canonical_fields(config, _SCOPE_FIELDS.get(source_type, ()))
    if source_type == "confluence":
        mode = str(scope.get("sync_mode") or "").lower()
        mode = mode if mode in {"page_tree", "space"} else ("page_tree" if scope.get("page_tree_root") else "space")
        scope["sync_mode"] = mode
        if mode == "page_tree":
            scope.pop("spaces", None)
        else:
            scope.pop("page_tree_root", None)
            scope.pop("include_children", None)
    if source_type == "jira":
        query_mode = str(scope.get("query_mode") or "simple").lower()
        query_mode = "advanced" if query_mode == "advanced" else "simple"
        scope["query_mode"] = query_mode
        if query_mode == "advanced":
            scope.pop("projects", None)
            scope.pop("jql_filter", None)
            scope.pop("issue_types", None)
        else:
            scope.pop("jql", None)
    return scope


def projection_scope_fingerprint(scope: Mapping[str, object]) -> str:
    """Return the stable identity of one canonical Projection Scope."""

    payload = json.dumps(
        dict(scope),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def projection_access_fingerprint(access_context: Mapping[str, object]) -> str:
    """Return the exact access identity embedded in Source Unit revisions."""

    payload = json.dumps(
        dict(access_context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def projection_scope_transition_id(
    source_id: str,
    previous_scope: Mapping[str, object],
    target_scope: Mapping[str, object],
    *,
    predecessor_transition_id: str | None = None,
) -> str:
    """Return a retry-stable identity for one scope-transition cycle."""

    payload = json.dumps(
        {
            "source_id": source_id,
            "previous_scope": dict(previous_scope),
            "target_scope": dict(target_scope),
            "predecessor_transition_id": predecessor_transition_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"scope-transition-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _canonical_fields(config: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in fields:
        if field not in config:
            continue
        value = config[field]
        if field in _SET_LIKE_FIELDS:
            values = _list_value(value)
            if values:
                result[field] = values
            continue
        if isinstance(value, str):
            value = value.strip()
            if field in _URL_FIELDS:
                value = value.rstrip("/")
            if value:
                result[field] = value
            continue
        if value is not None:
            result[field] = value
    return result


def _list_value(value: object) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()})
