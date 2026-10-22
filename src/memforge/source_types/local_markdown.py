"""Local markdown: the files of a folder the local agent snapshots."""

from __future__ import annotations

from memforge.source_adapters import local_markdown as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.descriptor import LocalAgentCollection, SourceTypeDescriptor

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
        replay_adapter="memforge.local_agent.replay_adapter:LocalMarkdownReplayAdapter",
    ),
)
