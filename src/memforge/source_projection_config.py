"""Canonical provider namespace and projection-scope config boundaries."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

from memforge.source_types import source_type_descriptor


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

    descriptor = source_type_descriptor(source_type)
    return _canonical_fields(config, descriptor.namespace_fields if descriptor else ())


def canonical_projection_scope(source_type: str, config: Mapping[str, Any]) -> dict[str, object]:
    """Return only fields that can change projected unit/observation membership."""

    descriptor = source_type_descriptor(source_type)
    if descriptor is None:
        return {}
    scope = _canonical_fields(config, descriptor.scope_fields)
    return descriptor.canonical_scope(scope) if descriptor.canonical_scope else scope


def excluded_scope_config(
    source_type: str,
    current_config: Mapping[str, Any],
    excluded_scope: Mapping[str, object],
) -> dict[str, Any]:
    """The configuration that lists a scope the user's change excluded, built from the current one.

    Every field that is not a scope field, credentials and connection
    settings included, is kept as it is. The scope fields are those of the
    excluded scope, canonicalized again so a stored scope from before a field
    was added still reads its default, and a current scope field the excluded
    scope leaves out is removed. For example, a Jira Source changed from
    ``{"query_mode": "simple", "projects": ["PAY", "OPS"]}`` to
    ``{"query_mode": "advanced", "jql": "project = PAY"}`` lists its excluded
    scope with ``projects`` and without ``jql``, and with the current token.
    """

    descriptor = source_type_descriptor(source_type)
    if descriptor is None:
        raise ValueError(f"Unknown source type {source_type!r}")
    config = {key: value for key, value in current_config.items() if key not in descriptor.scope_fields}
    config.update(canonical_projection_scope(source_type, excluded_scope))
    return config


def projection_scope_fingerprint(scope: Mapping[str, object]) -> str:
    """Return the stable identity of one canonical Projection Scope."""

    payload = json.dumps(
        dict(scope),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_access_context(source: Mapping[str, Any]) -> dict[str, object]:
    """Return the access context one configured Source projects its Units under."""

    return {
        "access_policy": str(source.get("access_policy") or "workspace"),
        "owner_user_id": source.get("owner_user_id"),
    }


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
