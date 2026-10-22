"""GitHub repository: files at a ref, pulled by the server or pushed by the local agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import github_repo as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor


def _pushed_by_local_agent(config: Mapping[str, Any]) -> bool:
    return str(config.get("connection_mode") or "").strip().lower() == "local_push"


DESCRIPTOR = SourceTypeDescriptor(
    source_type="github_repo",
    gene="memforge.genes.github_repo_gene:GitHubRepoGene",
    unit_noun="file",
    namespace_fields=("repo_url",),
    scope_fields=("ref", "include_paths", "exclude_paths", "include_extensions"),
    project_native=adapter.project_native,
    evidence_profiles={"file_content": MARKDOWN_STRUCTURAL_PROFILE},
    local_agent=LocalAgentCollection(
        operation="github_repo_sync",
        job_config_fields=frozenset({"repo_url", "ref", "include_paths", "exclude_paths", "include_extensions"}),
        collects=_pushed_by_local_agent,
        immutable_execution_fields=("connection_mode",),
        replay_adapter="memforge.local_agent.replay_adapter:GitHubRepoReplayAdapter",
    ),
)
