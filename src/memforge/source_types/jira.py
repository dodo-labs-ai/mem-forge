"""Jira: the issues a JQL selects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import jira as adapter
from memforge.source_types.config import SourceConfigPolicy, secret_replaced
from memforge.source_types.descriptor import SourceTypeDescriptor

_SOURCE_OWNED_COOKIE_FIELDS = (
    "jira_cookie",
    "jira_cookie_encrypted",
    "jira_cookie_configured",
    "jira_cookie_hint",
    "jira_cookie_decrypt_failed",
)
_PAT_FIELDS = ("pat", "pat_encrypted", "pat_configured", "pat_hint", "pat_decrypt_failed")


def _canonical_scope(scope: dict[str, object]) -> dict[str, object]:
    query_mode = "advanced" if str(scope["query_mode"]).lower() == "advanced" else "simple"
    scope["query_mode"] = query_mode
    if query_mode == "advanced":
        scope.pop("projects", None)
        scope.pop("jql_filter", None)
        scope.pop("issue_types", None)
    else:
        scope.pop("jql", None)
    return scope


def _complete_config(config: dict[str, Any], source_id: str, existing_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Carry the authentication method over, and name the method.

    A blank ``auth_mode`` keeps the stored one. A new Source without one gets
    the default method (:func:`~memforge.genes.jira_gene.jira_auth_mode`), so
    every stored configuration names its method.
    """

    from memforge.genes.jira_gene import jira_auth_mode

    existing_config = existing_config or {}
    if not str(config.get("auth_mode") or "").strip():
        stored = str(existing_config.get("auth_mode") or "").strip()
        if stored:
            config["auth_mode"] = stored
    config["auth_mode"] = jira_auth_mode(config)
    return config


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    if any(key in config for key in ("jira_cookie", "jira_cookie_encrypted")):
        raise ValueError("Jira browser sessions are managed by shared auth sessions, not source config")


def _facts(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    from memforge.genes.jira_gene import _build_jql
    from memforge.genes.toolkit.config import list_config

    advanced = str(config.get("query_mode") or "simple").strip().lower() == "advanced"
    projects = list_config(config.get("projects"))
    if not (str(config.get("jql") or "").strip() if advanced else projects):
        return []
    scoped = {**config, "projects": projects}
    if "issue_types" in config:
        scoped["issue_types"] = list_config(config["issue_types"])
    return [("Effective query at sync", _build_jql(scoped, "updated >= '<last sync>'"))]


def _store_config(config: dict[str, Any]) -> dict[str, Any]:
    from memforge.genes.jira_gene import JIRA_AUTH_MODE_COOKIE, jira_auth_mode

    # Browser sessions belong to the shared auth sessions, never to one Source.
    for key in _SOURCE_OWNED_COOKIE_FIELDS:
        config.pop(key, None)
    if jira_auth_mode(config) == JIRA_AUTH_MODE_COOKIE:
        for key in _PAT_FIELDS:
            config.pop(key, None)
    return config


def _credentials_changed(config: dict[str, Any], existing_config: Mapping[str, Any]) -> bool:
    from memforge.genes.jira_gene import JIRA_AUTH_MODE_COOKIE, jira_auth_mode

    auth_mode = jira_auth_mode(_complete_config(dict(config), "", existing_config))
    if auth_mode != jira_auth_mode(existing_config):
        return True
    if auth_mode == JIRA_AUTH_MODE_COOKIE:
        return False
    return secret_replaced("pat", config, existing_config)


DESCRIPTOR = SourceTypeDescriptor(
    source_type="jira",
    gene="memforge.genes.jira_gene:JiraGene",
    unit_noun="issue",
    namespace_fields=("base_url",),
    scope_fields=("query_mode", "projects", "jql", "jql_filter", "issue_types", "include_comments"),
    scope_defaults={"query_mode": "simple"},
    canonical_scope=_canonical_scope,
    project_native=adapter.project_native,
    evidence_profiles=adapter.CURRENT_OBSERVATION_PROFILES,
    legacy_evidence_profiles=adapter.LEGACY_OBSERVATION_PROFILES,
    record_schemas=adapter.CANONICAL_RECORD_SCHEMAS,
    acquisition_contract=2,
    config=SourceConfigPolicy(
        extra_secret_fields=("jira_cookie",),
        complete=_complete_config,
        validate=_validate_config,
        store=_store_config,
        credentials_changed=_credentials_changed,
        facts=_facts,
    ),
)
