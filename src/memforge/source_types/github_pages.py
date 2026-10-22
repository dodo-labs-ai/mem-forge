"""GitHub Pages: rendered pages of a published site."""

from __future__ import annotations

from memforge.source_adapters import github_pages as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.descriptor import SourceTypeDescriptor

DESCRIPTOR = SourceTypeDescriptor(
    source_type="github_pages",
    gene="memforge.genes.github_pages_gene:GitHubPagesGene",
    unit_noun="page",
    scope_fields=("sync_mode", "page_url", "root_url", "pages", "max_depth", "exclude_url_patterns"),
    project_native=adapter.project_native,
    evidence_profiles={"page_content": MARKDOWN_STRUCTURAL_PROFILE},
)
