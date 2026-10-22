"""GitHub repository: files at a ref, pulled by the server or pushed by the local agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import github_repo as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.config import LocalInbox, SourceConfigPolicy
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor


def _pushed_by_local_agent(config: Mapping[str, Any]) -> bool:
    return str(config.get("connection_mode") or "").strip().lower() == "local_push"


def validate_config(config: dict[str, Any], existing_config: Mapping[str, Any] | None = None) -> None:
    from urllib.parse import urlsplit

    from memforge.genes.atlassian_auth import require_https_base_url

    existing_config = existing_config or {}
    if "repo_path" in config:
        raise ValueError("GitHub Repository reads the configured remote repository; repo_path is not a supported field")
    repo_url = str(config.get("repo_url") or existing_config.get("repo_url") or "").strip()
    require_https_base_url(repo_url, "GitHub Repository")
    parts = urlsplit(repo_url)
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) < 2:
        raise ValueError("GitHub Repository URL must include owner and repository")
    connection_mode = (
        str(config.get("connection_mode") or existing_config.get("connection_mode") or "cloud_pull").strip().lower()
    )
    if connection_mode not in {"cloud_pull", "local_push"}:
        raise ValueError("GitHub Repository Access must be Public internet or Internal network / VPN")
    max_files = config.get("max_files", existing_config.get("max_files", 500))
    try:
        if int(max_files) < 1:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("GitHub Repository Max Files must be a positive integer") from None


def _store_config(config: dict[str, Any], local_inbox: LocalInbox) -> dict[str, Any]:
    if _pushed_by_local_agent(config):
        config["documents_dir"] = local_inbox()
    return config


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
        package_submitter="memforge.local_adapter:submit_github_repo_request",
        replay_adapter="memforge.local_agent.replay_adapter:GitHubRepoReplayAdapter",
        accepts_artifact_inputs=True,
    ),
    config=SourceConfigPolicy(validate_input=validate_config, store=_store_config),
)
