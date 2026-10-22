"""Jira: the issues a JQL selects, read by the server or by the local agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import jira as adapter
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor


def _canonical_scope(scope: dict[str, object]) -> dict[str, object]:
    query_mode = "advanced" if str(scope.get("query_mode") or "simple").lower() == "advanced" else "simple"
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


DESCRIPTOR = SourceTypeDescriptor(
    source_type="jira",
    gene="memforge.genes.jira_gene:JiraGene",
    unit_noun="issue",
    namespace_fields=("base_url",),
    scope_fields=("query_mode", "projects", "jql", "jql_filter", "issue_types", "include_comments"),
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
        replay_adapter="memforge.local_agent.replay_adapter:JiraReplayAdapter",
    ),
)
