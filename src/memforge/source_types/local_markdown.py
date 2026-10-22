"""Local markdown: the files of a folder the local agent snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import local_markdown as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.config import LocalInbox, SourceConfigPolicy
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor


def _complete_config(config: dict[str, Any], source_id: str, existing_config: Mapping[str, Any] | None) -> dict[str, Any]:
    vault_id = str(config.get("vault_id") or (existing_config or {}).get("vault_id") or "").strip()
    config["vault_id"] = vault_id or f"local-{source_id}"
    return config


def _store_config(config: dict[str, Any], local_inbox: LocalInbox) -> dict[str, Any]:
    config["documents_dir"] = local_inbox()
    return config


DESCRIPTOR = SourceTypeDescriptor(
    source_type="local_markdown",
    gene="memforge.genes.local_markdown_gene:LocalMarkdownGene",
    unit_noun="file",
    # vault_id is the portable source namespace; root is a device-local locator
    # and may change without changing file lineage.
    namespace_fields=("vault_id",),
    scope_fields=("include", "exclude"),
    project_native=adapter.project_native,
    evidence_profiles={"file_content": MARKDOWN_STRUCTURAL_PROFILE},
    local_agent=LocalAgentCollection(
        operation="local_markdown_sync",
        job_config_fields=frozenset({"root", "vault_id", "include", "exclude"}),
        package_submitter="memforge.local_adapter:submit_local_markdown_request",
        replay_adapter="memforge.local_agent.replay_adapter:LocalMarkdownReplayAdapter",
    ),
    config=SourceConfigPolicy(complete=_complete_config, store=_store_config),
)
