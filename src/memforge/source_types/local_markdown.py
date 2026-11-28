"""Local markdown: the files of a folder on the Source's execution device."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import local_markdown as adapter
from memforge.source_adapters.contracts import FILE_ALTERNATIVE_EVIDENCE_PROFILES, MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.config import SourceConfigPolicy
from memforge.source_types.descriptor import SOURCE_EXECUTOR_DEVICE, SourceTypeDescriptor


def _complete_config(config: dict[str, Any], source_id: str, existing_config: Mapping[str, Any] | None) -> dict[str, Any]:
    vault_id = str(config.get("vault_id") or (existing_config or {}).get("vault_id") or "").strip()
    config["vault_id"] = vault_id or f"local-{source_id}"
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
    alternative_evidence_profiles={"file_content": FILE_ALTERNATIVE_EVIDENCE_PROFILES},
    acquisition_contract=3,
    # Its files exist only on the owner's machine.
    executors=frozenset({SOURCE_EXECUTOR_DEVICE}),
    device_location_fields=("root",),
    config=SourceConfigPolicy(complete=_complete_config),
)
