"""Jira: the issues a JQL selects, read by the server or by the local agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import jira as adapter
from memforge.source_types.config import LocalInbox, SourceConfigPolicy, secret_has_value
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor

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


def _read_by_local_agent(config: Mapping[str, Any]) -> bool:
    return str(config.get("sync_mode") or "").strip().lower() == "local_agent"


def jira_auth_mode(config: Mapping[str, Any], existing_config: Mapping[str, Any] | None = None) -> str:
    """The configured authentication method; a stored token without one means a token."""

    existing_config = existing_config or {}
    configured_auth_mode = config.get("auth_mode") or existing_config.get("auth_mode")
    if configured_auth_mode is None and secret_has_value("pat", config, existing_config):
        return "pat"
    return str(configured_auth_mode or "browser_cookie").strip().lower()


def _complete_config(config: dict[str, Any], source_id: str, existing_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not str(config.get("sync_mode") or "").strip():
        sync_mode = str((existing_config or {}).get("sync_mode") or "").strip()
        if sync_mode:
            config["sync_mode"] = sync_mode
    return config


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    _validate_auth(config, existing_config)
    validate_scope(config)


def _validate_auth(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    if any(key in config for key in ("jira_cookie", "jira_cookie_encrypted")):
        raise ValueError("Jira browser sessions are managed by shared auth sessions, not source config")
    auth_mode = jira_auth_mode(config, existing_config)
    if auth_mode not in {"browser_cookie", "pat"}:
        raise ValueError("Jira Authentication Method must be Browser session or Personal access token")

    sync_mode = str(config.get("sync_mode") or existing_config.get("sync_mode") or "cloud").strip().lower()
    if sync_mode not in {"cloud", "local_agent"}:
        raise ValueError("Jira Sync Location must be Cloud or Local daemon")
    if sync_mode == "local_agent" and auth_mode != "browser_cookie":
        raise ValueError("Jira Local daemon sync currently requires Browser session authentication")

    if auth_mode == "browser_cookie":
        return

    if secret_has_value("pat", config, existing_config):
        return
    raise ValueError("Jira Personal Access Token is required for pat authentication")


def validate_scope(config: Mapping[str, Any]) -> None:
    """Require the field that actually drives discovery for the chosen query mode.

    ``projects`` is optional at the schema level so advanced JQL queries (which
    embed their own project clause) do not have to duplicate it.
    """
    mode = str(config.get("query_mode") or "simple").strip().lower()
    if mode == "advanced":
        if not str(config.get("jql") or "").strip():
            raise ValueError("Jira JQL is required in advanced query mode")
        return
    projects = config.get("projects")
    has_projects = (isinstance(projects, list) and any(str(p).strip() for p in projects)) or (
        isinstance(projects, str) and any(p.strip() for p in projects.split(","))
    )
    if not has_projects:
        raise ValueError("Jira Projects to Sync is required in simple query mode")


def _store_config(config: dict[str, Any], local_inbox: LocalInbox) -> dict[str, Any]:
    # Browser sessions belong to the shared auth sessions, never to one Source.
    for key in _SOURCE_OWNED_COOKIE_FIELDS:
        config.pop(key, None)
    if jira_auth_mode(config) == "browser_cookie":
        for key in _PAT_FIELDS:
            config.pop(key, None)
    if _read_by_local_agent(config):
        config["local_agent_documents_dir"] = local_inbox()
    else:
        config.pop("local_agent_documents_dir", None)
    return config


def _credentials_changed(config: dict[str, Any], existing_config: Mapping[str, Any]) -> bool:
    auth_mode = jira_auth_mode(config, existing_config)
    if auth_mode != jira_auth_mode(existing_config):
        return True
    if auth_mode == "browser_cookie":
        return False
    incoming = config.get("pat")
    if not isinstance(incoming, str) or not incoming.strip():
        return False
    return _stored_secret_value("pat", existing_config) != incoming.strip()


def _stored_secret_value(field: str, config: Mapping[str, Any]) -> str | None:
    from memforge.source_secrets import SecretConfigurationError, decrypt_secret

    encrypted = config.get(f"{field}_encrypted")
    if encrypted:
        try:
            return decrypt_secret(str(encrypted))
        except SecretConfigurationError:
            return None
    plaintext = config.get(field)
    if isinstance(plaintext, str) and plaintext.strip():
        return plaintext.strip()
    return None


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
    local_agent=LocalAgentCollection(
        operation="jira_sync",
        job_config_fields=frozenset(
            {
                "base_url",
                "auth_mode",
                "sync_mode",
                "projects",
                "issue_types",
                "include_comments",
                "query_mode",
                "jql",
                "jql_filter",
            }
        ),
        collects=_read_by_local_agent,
        immutable_execution_fields=("sync_mode",),
        package_submitter="memforge.local_adapter:submit_jira_request",
        replay_adapter="memforge.local_agent.replay_adapter:JiraReplayAdapter",
        accepts_artifact_inputs=True,
    ),
    config=SourceConfigPolicy(
        extra_secret_fields=("jira_cookie",),
        complete=_complete_config,
        validate=_validate_config,
        store=_store_config,
        credentials_changed=_credentials_changed,
    ),
)
