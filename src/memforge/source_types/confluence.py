"""Confluence: the pages of configured spaces or of one page tree."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import confluence as adapter
from memforge.source_types.config import SourceConfigPolicy, list_has_value
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


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    sync_mode = str(config.get("sync_mode") or existing_config.get("sync_mode") or "").strip().lower()
    if sync_mode not in {"", "page_tree", "space"}:
        raise ValueError("Confluence Sync Scope must be Page tree or Space")
    if not sync_mode:
        sync_mode = (
            "page_tree"
            if str(config.get("page_tree_root") or existing_config.get("page_tree_root") or "").strip()
            else "space"
        )

    if sync_mode == "page_tree":
        page_tree_root = str(config.get("page_tree_root") or existing_config.get("page_tree_root") or "").strip()
        if not page_tree_root:
            raise ValueError("Confluence Page Tree Root is required when syncing a page tree")
        return

    if not list_has_value(config.get("spaces")) and not list_has_value(existing_config.get("spaces")):
        raise ValueError("Confluence Spaces to Sync is required when syncing whole spaces")


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
    config=SourceConfigPolicy(validate=_validate_config),
)
