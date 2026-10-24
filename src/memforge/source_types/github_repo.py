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


def _validate_input(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    if "repo_path" in config:
        raise ValueError("GitHub Repository reads the configured remote repository; repo_path is not a supported field")


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    from memforge.genes.toolkit.config import FieldIssue, SourceConfigInvalid

    try:
        if int(config.get("max_files", 500)) < 1:
            raise ValueError
    except (TypeError, ValueError):
        raise SourceConfigInvalid(
            [FieldIssue("max_files", "GitHub Repository Max Files must be a positive integer")]
        ) from None


def _facts(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    from memforge.github_repo_utils import parse_github_repo_url

    try:
        repo = parse_github_repo_url(str(config.get("repo_url") or ""))
    except ValueError:
        return []
    return [("Host", repo["host"]), ("Repository", f"{repo['owner']}/{repo['repo']}")]


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
    config=SourceConfigPolicy(
        validate_input=_validate_input,
        validate=_validate_config,
        store=_store_config,
        facts=_facts,
    ),
)
