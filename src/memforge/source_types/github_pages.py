"""GitHub Pages: rendered pages of a published site."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.source_adapters import github_pages as adapter
from memforge.source_adapters.contracts import MARKDOWN_STRUCTURAL_PROFILE
from memforge.source_types.config import SourceConfigPolicy, secret_has_value
from memforge.source_types.descriptor import SourceTypeDescriptor


def _validate_config(config: dict[str, Any], existing_config: Mapping[str, Any]) -> None:
    from memforge.genes.atlassian_auth import require_https_base_url

    base_url = _ensure_base_url(config, existing_config)
    require_https_base_url(base_url, "GitHub Pages")

    auth_mode = str(config.get("auth_mode") or existing_config.get("auth_mode") or "github_pat").strip().lower()
    if auth_mode not in {"github_pat", "none"}:
        raise ValueError("GitHub Pages Authentication Method must be Personal access token or No authentication")
    if auth_mode == "github_pat" and not secret_has_value("pat", config, existing_config):
        raise ValueError("GitHub Pages Personal Access Token is required for github_pat authentication")

    sync_mode = str(config.get("sync_mode") or "single_page").strip().lower()
    if sync_mode == "single_page":
        page_url = str(config.get("page_url") or "").strip()
        if not page_url:
            raise ValueError("GitHub Pages Page URL is required for single page sync")
        _validate_scope_url(base_url, page_url, "Page URL")
        return
    if sync_mode == "subtree":
        root_url = str(config.get("root_url") or "").strip()
        if not root_url:
            raise ValueError("GitHub Pages Subtree Root URL is required for subtree sync")
        _validate_scope_url(base_url, root_url, "Subtree Root URL")
        return
    if sync_mode == "explicit_list":
        pages = config.get("pages")
        page_urls = [
            str(page).strip()
            for page in (pages if isinstance(pages, list) else str(pages or "").split(","))
            if str(page).strip()
        ]
        if page_urls:
            for page_url in page_urls:
                _validate_scope_url(base_url, page_url, "Explicit Page URL")
            return
        raise ValueError("GitHub Pages Explicit Page URLs are required for explicit list sync")
    raise ValueError("GitHub Pages Sync Mode must be Single page, Subtree, or Explicit list")


def _ensure_base_url(config: dict[str, Any], existing_config: Mapping[str, Any]) -> str:
    base_url = str(config.get("base_url") or existing_config.get("base_url") or "").strip()
    if not base_url:
        base_url = _site_root_from_url(_scope_url(config, existing_config))
    if base_url:
        config["base_url"] = base_url
    return base_url


def _scope_url(config: Mapping[str, Any], existing_config: Mapping[str, Any]) -> str:
    sync_mode = str(config.get("sync_mode") or existing_config.get("sync_mode") or "single_page").strip().lower()
    if sync_mode == "subtree":
        return str(config.get("root_url") or existing_config.get("root_url") or "").strip()
    if sync_mode == "explicit_list":
        pages = config.get("pages") or existing_config.get("pages")
        page_urls = [
            str(page).strip()
            for page in (pages if isinstance(pages, list) else str(pages or "").split(","))
            if str(page).strip()
        ]
        return page_urls[0] if page_urls else ""
    return str(config.get("page_url") or existing_config.get("page_url") or "").strip()


def _site_root_from_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    value = str(url or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "pages":
        root_path = "/" + "/".join(path_parts[:3])
        return urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
    if len(path_parts) >= 2 and (parts.hostname or "").lower().startswith("pages."):
        root_path = "/" + "/".join(path_parts[:2])
        return urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _validate_scope_url(base_url: str, candidate_url: str, label: str) -> None:
    from urllib.parse import urlsplit

    from memforge.genes.atlassian_auth import require_https_base_url

    require_https_base_url(candidate_url, f"GitHub Pages {label}")
    base = urlsplit(base_url)
    candidate = urlsplit(candidate_url)
    if candidate.netloc.lower() != base.netloc.lower():
        raise ValueError(f"GitHub Pages {label} must stay on the configured site origin")
    base_path = base.path.rstrip("/") + "/"
    candidate_path = candidate.path.rstrip("/") + "/"
    if not candidate_path.startswith(base_path):
        raise ValueError(f"GitHub Pages {label} must stay under the configured site path")


DESCRIPTOR = SourceTypeDescriptor(
    source_type="github_pages",
    gene="memforge.genes.github_pages_gene:GitHubPagesGene",
    unit_noun="page",
    # ``branch`` selects the repository tree a subtree lists, and has no
    # default: without it the run reads the repository's default branch.
    # ``sitemap_authoritative``, ``auth_mode`` and ``max_pages`` only choose
    # how the same pages are listed, so they are not scope fields.
    scope_fields=("sync_mode", "page_url", "root_url", "pages", "max_depth", "exclude_url_patterns", "branch"),
    project_native=adapter.project_native,
    evidence_profiles={"page_content": MARKDOWN_STRUCTURAL_PROFILE},
    config=SourceConfigPolicy(validate_input=_validate_config),
)
