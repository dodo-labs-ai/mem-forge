"""Confluence: the pages of configured spaces or of one page tree."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import confluence as adapter
from memforge.source_types.config import SourceConfigPolicy
from memforge.source_types.descriptor import SourceTypeDescriptor


def _canonical_scope(scope: dict[str, object]) -> dict[str, object]:
    mode = str(scope.get("sync_mode") or "").lower()
    mode = mode if mode in {"page_tree", "space"} else ("page_tree" if scope.get("page_tree_root") else "space")
    scope["sync_mode"] = mode
    if mode == "page_tree":
        scope.pop("spaces", None)
    else:
        scope.pop("page_tree_root", None)
        scope.pop("include_children", None)
    return scope


def _facts(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    from memforge.genes.toolkit.config import list_config

    facts: list[tuple[str, str]] = []
    base_url = str(config.get("base_url") or "").strip()
    if base_url:
        facts.append(("Site", base_url))
    if str(config.get("sync_mode") or "").strip().lower() == "page_tree":
        root = str(config.get("page_tree_root") or "").strip()
        if root:
            facts.append(("Scope", f"The page tree under page {root}"))
    else:
        spaces = list_config(config.get("spaces"))
        if spaces:
            facts.append(("Scope", f"Whole spaces: {', '.join(spaces)}"))
    return facts


DESCRIPTOR = SourceTypeDescriptor(
    source_type="confluence",
    gene="memforge.genes.confluence_gene:ConfluenceGene",
    unit_noun="page",
    namespace_fields=("base_url",),
    scope_fields=("sync_mode", "spaces", "page_tree_root", "exclude_labels", "include_children"),
    canonical_scope=_canonical_scope,
    project_native=adapter.project_native,
    evidence_profiles=adapter.CURRENT_OBSERVATION_PROFILES,
    legacy_evidence_profiles=adapter.LEGACY_OBSERVATION_PROFILES,
    record_schemas=adapter.CANONICAL_RECORD_SCHEMAS,
    exports_pdf=True,
    config=SourceConfigPolicy(facts=_facts),
)
