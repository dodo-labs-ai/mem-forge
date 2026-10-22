"""Confluence: the pages of configured spaces or of one page tree."""

from __future__ import annotations

from memforge.source_adapters import confluence as adapter
from memforge.source_adapters.contracts import canonical_record_profile
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


DESCRIPTOR = SourceTypeDescriptor(
    source_type="confluence",
    gene="memforge.genes.confluence_gene:ConfluenceGene",
    unit_noun="page",
    namespace_fields=("base_url",),
    scope_fields=("sync_mode", "spaces", "page_tree_root", "exclude_labels", "include_children"),
    canonical_scope=_canonical_scope,
    project_native=adapter.project_native,
    evidence_profiles={"page_body": canonical_record_profile("confluence-page-storage")},
    legacy_evidence_profiles=adapter.LEGACY_OBSERVATION_PROFILES,
    record_schemas=adapter.CANONICAL_RECORD_SCHEMAS,
    exports_pdf=True,
)
