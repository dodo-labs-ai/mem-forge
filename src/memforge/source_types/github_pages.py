"""GitHub Pages: rendered pages of a published site."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import github_pages as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.config import SourceConfigPolicy
from memforge.source_types.descriptor import SourceTypeDescriptor


def _canonical_scope(scope: dict[str, object]) -> dict[str, object]:
    # A branch selects which pages a subtree lists. A single page or an
    # explicit list names its pages, so there the branch only chooses where
    # their content is read.
    if str(scope.get("sync_mode") or "single_page").strip().lower() != "subtree":
        scope.pop("branch", None)
    return scope


_SCOPE_URL_FIELDS = {
    "single_page": ("page_url", "Page URL"),
    "subtree": ("root_url", "Subtree Root URL"),
    "explicit_list": ("pages", "Explicit Page URL"),
}


def _complete_config(config: dict[str, Any], source_id: str, existing_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep the stored site when none is submitted; the Gene derives one for a new Source."""

    stored_base_url = str((existing_config or {}).get("base_url") or "").strip()
    if not str(config.get("base_url") or "").strip() and stored_base_url:
        config["base_url"] = stored_base_url
    return config


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    """Every scope URL of the configured mode stays on the site the Source reads."""

    from memforge.genes.toolkit.config import list_config

    base_url = str(config.get("base_url") or "").strip()
    sync_mode = str(config.get("sync_mode") or "single_page").strip().lower()
    key, label = _SCOPE_URL_FIELDS[sync_mode]
    # A single page or a subtree root is one URL, which may itself hold a
    # comma; only an explicit list names several.
    if key == "pages":
        urls = list_config(config.get(key))
    else:
        single_url = str(config.get(key) or "").strip()
        urls = [single_url] if single_url else []
    for url in urls:
        _validate_scope_url(base_url, url, key, label)


def _validate_scope_url(base_url: str, candidate_url: str, key: str, label: str) -> None:
    from urllib.parse import urlsplit

    from memforge.genes.toolkit.config import FieldIssue, SourceConfigInvalid, require_https_base_url

    def refuse(message: str) -> SourceConfigInvalid:
        return SourceConfigInvalid([FieldIssue(key, message)])

    try:
        require_https_base_url(candidate_url, f"GitHub Pages {label}")
    except ValueError as exc:
        raise refuse(str(exc)) from None
    base = urlsplit(base_url)
    candidate = urlsplit(candidate_url)
    if candidate.netloc.lower() != base.netloc.lower():
        raise refuse(f"GitHub Pages {label} must stay on the configured site origin")
    base_path = base.path.rstrip("/") + "/"
    candidate_path = candidate.path.rstrip("/") + "/"
    if not candidate_path.startswith(base_path):
        raise refuse(f"GitHub Pages {label} must stay under the configured site path")


DESCRIPTOR = SourceTypeDescriptor(
    source_type="github_pages",
    gene="memforge.genes.github_pages_gene:GitHubPagesGene",
    unit_noun="page",
    # ``branch`` selects the repository tree a subtree lists, and has no
    # default: without it the run reads the repository's default branch.
    # ``sitemap_authoritative``, ``auth_mode`` and ``max_pages`` only choose
    # how the same pages are listed, so they are not scope fields.
    scope_fields=("sync_mode", "page_url", "root_url", "pages", "max_depth", "exclude_url_patterns", "branch"),
    canonical_scope=_canonical_scope,
    project_native=adapter.project_native,
    evidence_profiles={"page_content": MARKDOWN_STRUCTURAL_PROFILE},
    config=SourceConfigPolicy(complete=_complete_config, validate=_validate_config),
)
