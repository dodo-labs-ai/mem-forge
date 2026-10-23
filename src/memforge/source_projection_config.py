"""Canonical provider namespace and projection-scope config boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
from typing import Any, Protocol

from memforge.source_projection import ProjectionScopeTransition
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
    """Return only fields that can change projected unit/observation membership.

    An absent or empty scope field reads its descriptor default, so a scope
    stored before the field was added equals one that sets the default. The
    same function canonicalizes a configuration and a stored scope.
    """

    descriptor = source_type_descriptor(source_type)
    if descriptor is None:
        return {}
    scope = {**descriptor.scope_defaults, **_canonical_fields(config, descriptor.scope_fields)}
    return descriptor.canonical_scope(scope) if descriptor.canonical_scope else scope


def canonical_stored_scope(
    source_type: str,
    scope: Mapping[str, object],
    *,
    stored_target: Mapping[str, object],
    configured_scope: Mapping[str, object],
) -> dict[str, object]:
    """A scope stored with a transition, canonicalized again against the current configuration.

    An absent field reads its descriptor default. A scope field that
    ``stored_target``, the target of the Source's open transition as stored,
    lacks and ``configured_scope`` sets reads the configured value: every
    scope change writes the configuration and its transition together, and a
    later change supersedes the transition, so only a target stored before
    the field existed can lack it, and the field has not changed since. The
    transition's previous scope and every target it superseded are older
    still, so they read it too. For example, a GitHub Pages subtree pinned to
    ``branch: gh-pages`` whose transition was stored before ``branch`` was a
    scope field reads ``gh-pages`` in both scopes.
    """

    stored = canonical_projection_scope(source_type, stored_target)
    predating = {field: value for field, value in configured_scope.items() if field not in stored}
    canonical = canonical_projection_scope(source_type, scope)
    return canonical_projection_scope(
        source_type,
        {**{field: value for field, value in predating.items() if field not in canonical}, **canonical},
    )


def canonical_scope_transition(
    source_type: str,
    transition: ProjectionScopeTransition,
    configured_scope: Mapping[str, object],
) -> ProjectionScopeTransition:
    """The stored transition with both scopes canonicalized again (:func:`canonical_stored_scope`).

    Every reader that compares a stored scope with a configured one, or hands
    it on, reads it through this, so a transition opened before a scope field
    was added matches the configuration it targets. For example, a Jira
    target stored as ``{"projects": ["PAY"]}`` reads
    ``{"projects": ["PAY"], "query_mode": "simple"}``.
    """

    def canonical(scope: Mapping[str, object]) -> dict[str, object]:
        return canonical_stored_scope(
            source_type, scope, stored_target=transition.target_scope, configured_scope=configured_scope
        )

    return replace(
        transition,
        previous_scope=canonical(transition.previous_scope),
        target_scope=canonical(transition.target_scope),
    )


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


class ExcludedScopesUnavailable(ValueError):
    """The superseded chain of an open transition cannot be read, so its excluded scopes are unknown."""


class _TransitionReader(Protocol):
    async def get_projection_scope_transition(self, transition_id: str) -> ProjectionScopeTransition | None: ...


async def excluded_projection_scopes(
    db: _TransitionReader,
    transition: ProjectionScopeTransition,
    *,
    source_type: str,
    config: Mapping[str, Any],
) -> list[dict[str, object]]:
    """The distinct canonical scopes an open transition excludes: the last applied scope and every superseded target.

    The superseded chain is walked by ``predecessor_transition_id`` back to the
    last applied scope. A scope equal to the new target is not excluded. Each
    scope is canonicalized again against the configured one
    (:func:`canonical_stored_scope`), so one stored before a scope field was
    added reads that field as the open transition does. ``config`` is the
    Source's current configuration. Raises ``ExcludedScopesUnavailable`` when
    a transition of the chain is missing, belongs to another Source or loops.
    """

    target = canonical_projection_scope(source_type, transition.target_scope)
    stored = await db.get_projection_scope_transition(transition.id)
    if stored is None:
        raise ExcludedScopesUnavailable(f"the transition {transition.id} is missing")
    configured = canonical_projection_scope(source_type, config)

    def canonical(scope: Mapping[str, object]) -> dict[str, object]:
        return canonical_stored_scope(source_type, scope, stored_target=stored.target_scope, configured_scope=configured)

    chain = [transition.previous_scope]
    predecessor_id = transition.predecessor_transition_id
    walked: set[str] = set()
    while predecessor_id is not None:
        if predecessor_id in walked:
            raise ExcludedScopesUnavailable(f"the superseded transitions loop back to {predecessor_id}")
        walked.add(predecessor_id)
        predecessor = await db.get_projection_scope_transition(predecessor_id)
        if predecessor is None or predecessor.source_id != transition.source_id:
            raise ExcludedScopesUnavailable(f"the superseded transition {predecessor_id} is missing")
        chain.append(predecessor.target_scope)
        predecessor_id = predecessor.predecessor_transition_id
    scopes: list[dict[str, object]] = []
    for scope in chain:
        excluded = canonical(scope)
        if excluded != target and excluded not in scopes:
            scopes.append(excluded)
    return scopes


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
    previous_cycle_transition_id: str | None = None,
) -> str:
    """Return a retry-stable identity for one scope-transition cycle.

    ``previous_cycle_transition_id`` is the Source's latest transition of any
    status, or ``None`` for its first. Each cycle therefore has its own
    identity, also when it repeats the scopes of an earlier one (ADR 0005),
    while two identical concurrent changes compute the same one. It is not
    the superseded transition a transition records as its predecessor, though
    a superseding change reads the same id for both. Its hash key keeps the
    spelling every stored identity was computed with.
    """

    payload = json.dumps(
        {
            "source_id": source_id,
            "previous_scope": dict(previous_scope),
            "target_scope": dict(target_scope),
            "predecessor_transition_id": previous_cycle_transition_id,
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
