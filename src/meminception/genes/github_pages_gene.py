"""GitHub Pages Gene -- syncs rendered documentation pages into markdown."""

from __future__ import annotations

import hashlib
import asyncio
import base64
import logging
import re
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from meminception.genes.atlassian_auth import require_https_base_url, resolve_pat, tls_verify
from meminception.genes.base import Gene
from meminception.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)
from meminception.pipeline.normalizer_utils import annotate_code_blocks, html_to_markdown, strip_boilerplate

logger = logging.getLogger(__name__)

__all__ = ["GitHubPagesGene"]

AUTH_MODE_GITHUB_PAT = "github_pat"
AUTH_MODE_NONE = "none"
SYNC_MODE_SINGLE_PAGE = "single_page"
SYNC_MODE_SUBTREE = "subtree"
SYNC_MODE_EXPLICIT_LIST = "explicit_list"
DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_PAGES = 200


@dataclass(frozen=True)
class _RepoRef:
    origin: str
    owner: str
    repo: str
    page_path: str


class GitHubPagesGene(Gene):
    """Rendered GitHub Pages documentation source."""

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="github_pages",
            display_name="GitHub Pages",
            description="Rendered documentation pages from GitHub Pages sites",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Connection", order=0),
                ConfigGroup(key="scope", label="What to Sync", order=1),
            ],
            fields=[
                ConfigField(
                    key="auth_mode",
                    label="Authentication Method",
                    field_type=ConfigFieldType.SELECT,
                    required=True,
                    options=[AUTH_MODE_GITHUB_PAT, AUTH_MODE_NONE],
                    default=AUTH_MODE_GITHUB_PAT,
                    help_text="Use a GitHub personal access token for Enterprise GitHub Pages, or no auth for public pages.",
                    group="connection",
                    order=0,
                ),
                ConfigField(
                    key="pat",
                    label="Personal Access Token",
                    field_type=ConfigFieldType.SECRET,
                    required=False,
                    help_text="Stored encrypted and sent as a bearer token when PAT mode is selected.",
                    group="connection",
                    order=1,
                ),
                ConfigField(
                    key="sync_mode",
                    label="Sync Mode",
                    field_type=ConfigFieldType.SELECT,
                    required=True,
                    options=[SYNC_MODE_SINGLE_PAGE, SYNC_MODE_SUBTREE, SYNC_MODE_EXPLICIT_LIST],
                    default=SYNC_MODE_SINGLE_PAGE,
                    help_text="Choose one page, all pages under a path, or an explicit URL list.",
                    group="scope",
                    order=0,
                ),
                ConfigField(
                    key="page_url",
                    label="Page URL",
                    field_type=ConfigFieldType.URL,
                    required=False,
                    placeholder="https://github.example.com/pages/org/repo/path/to/page/",
                    help_text="Single documentation page to sync.",
                    group="scope",
                    order=1,
                ),
                ConfigField(
                    key="root_url",
                    label="Subtree Root URL",
                    field_type=ConfigFieldType.URL,
                    required=False,
                    placeholder="https://github.example.com/pages/org/repo/cloud-native-platform/",
                    help_text="Only pages under this URL path are discovered.",
                    group="scope",
                    order=2,
                ),
                ConfigField(
                    key="pages",
                    label="Explicit Page URLs",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    placeholder="https://.../page-a/, https://.../page-b/",
                    help_text="Comma-separated documentation page URLs.",
                    group="scope",
                    order=3,
                ),
                ConfigField(
                    key="max_depth",
                    label="Max Crawl Depth",
                    field_type=ConfigFieldType.INTEGER,
                    required=False,
                    default=str(DEFAULT_MAX_DEPTH),
                    help_text="Maximum same-site link depth when sitemap discovery is unavailable.",
                    group="scope",
                    order=4,
                    advanced=True,
                ),
                ConfigField(
                    key="max_pages",
                    label="Max Pages",
                    field_type=ConfigFieldType.INTEGER,
                    required=False,
                    default=str(DEFAULT_MAX_PAGES),
                    help_text="Stop with an error if discovery reaches this many pages.",
                    group="scope",
                    order=5,
                    advanced=True,
                ),
                ConfigField(
                    key="exclude_url_patterns",
                    label="Exclude URL Patterns",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    placeholder="/blog/, /archive/",
                    help_text="Regex patterns for page URLs to exclude.",
                    group="scope",
                    order=6,
                    advanced=True,
                ),
            ],
        )

    async def authenticate(self) -> None:
        base_url = _site_root_from_pages_url(
            str(self.config.get("base_url") or _scope_url_for_config(self.config))
        )
        if not base_url:
            raise ValueError("GitHub Pages base_url is required")
        require_https_base_url(base_url, "GitHub Pages")

        auth_mode = _auth_mode(self.config)
        if auth_mode not in {AUTH_MODE_GITHUB_PAT, AUTH_MODE_NONE}:
            raise ValueError("GitHub Pages Authentication Method must be Personal access token or No authentication")

        headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        if auth_mode == AUTH_MODE_GITHUB_PAT:
            headers["Authorization"] = f"Bearer {resolve_pat(self.config, 'GitHub Pages')}"

        self._base_url = base_url
        self._auth_mode = auth_mode
        self._client = _RequestsAsyncClient(
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
            verify=tls_verify(self.config),
        )

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        mode = _sync_mode(self.config)
        if mode == SYNC_MODE_SINGLE_PAGE:
            url = str(self.config.get("page_url") or "").strip()
            item = await self._content_item_for_url(url)
            if _is_modified_since(item.last_modified, since):
                yield item
            return

        if mode == SYNC_MODE_EXPLICIT_LIST:
            for url in _list_config(self.config.get("pages")):
                item = await self._content_item_for_url(url)
                if _is_modified_since(item.last_modified, since):
                    yield item
            return

        if mode != SYNC_MODE_SUBTREE:
            raise ValueError(f"Unsupported GitHub Pages sync mode: {mode}")

        if self._auth_mode == AUTH_MODE_GITHUB_PAT:
            async for item in self._repo_subtree_items(since):
                yield item
            return

        sitemap_items = await self._sitemap_items()
        if sitemap_items is not None:
            for item in sitemap_items:
                if _is_modified_since(item.last_modified, since):
                    yield item
            return

        async for item in self._discover_bfs(since):
            yield item

    async def fetch(self, item: ContentItem) -> RawContent:
        repo_api_url = item.extra.get("repo_api_url")
        if repo_api_url:
            response = await self._client.get(str(repo_api_url))
            response.raise_for_status()
            payload = response.json()
            body = base64.b64decode(str(payload.get("content") or "").replace("\n", ""))
            return RawContent(
                item=item,
                body=body,
                content_type="text/markdown",
            )

        response = await self._client.get(item.source_url)
        response.raise_for_status()
        return RawContent(
            item=item,
            body=response.content,
            content_type=response.headers.get("content-type", item.content_type) or "text/html",
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        raw_text = raw.body.decode("utf-8", errors="replace")
        if raw.content_type in {"text/markdown", "text/x-markdown"} or raw.item.extra.get("repo_api_url"):
            body = strip_boilerplate(raw_text)
        else:
            article_html = _extract_article_html(raw_text)
            body = strip_boilerplate(html_to_markdown(article_html))
        body = annotate_code_blocks(body)
        page_url = _canonicalize_url(raw.item.source_url)
        metadata_lines = [
            "## Source Metadata",
            "- Source Type: GitHub Pages",
            f"- Site URL: {self._base_url}",
            f"- Page URL: {page_url}",
        ]
        if raw.item.version:
            metadata_lines.append(f"- Version: {raw.item.version}")
        if raw.item.extra.get("repo_path"):
            metadata_lines.append(f"- Repository Path: {raw.item.extra['repo_path']}")

        title = _title_from_markdown(body) or raw.item.title.strip() or "GitHub Pages Document"
        markdown = "\n\n".join([
            f"# {title}",
            "\n".join(metadata_lines),
            "## Document",
            body,
        ]).strip()
        return NormalizedContent(
            item=raw.item,
            markdown_body=markdown,
            source_semantics={
                "source_type": "github_pages",
                "site_url": self._base_url,
                "page_url": page_url,
                "canonical_url": page_url,
                "title": title,
            },
        )

    async def health_check(self) -> dict:
        try:
            if not hasattr(self, "_client"):
                await self.authenticate()
            response = await self._client.head(self._base_url)
            return {"healthy": response.status_code < 400, "status_code": response.status_code}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    async def _content_item_for_url(
        self,
        url: str,
        *,
        metadata_headers: dict[str, str] | None = None,
    ) -> ContentItem:
        canonical_url = self._require_in_site(url)
        if self._auth_mode == AUTH_MODE_GITHUB_PAT:
            return await self._repo_file_item_for_url(canonical_url)
        headers = metadata_headers
        if headers is None:
            headers = await self._metadata_headers(canonical_url)
        return _content_item_from_url(canonical_url, headers)

    async def _repo_content_item(
        self,
        *,
        canonical_url: str,
        ref: "_RepoRef",
        branch: str,
        repo_path: str,
        blob_sha: str,
    ) -> ContentItem:
        last_modified = await self._repo_path_last_modified(ref, branch, repo_path)
        version = blob_sha or last_modified.isoformat()
        return ContentItem(
            item_id=f"github-pages-{hashlib.sha1(canonical_url.encode('utf-8')).hexdigest()}",
            title=_title_from_url(canonical_url),
            source_url=canonical_url,
            last_modified=last_modified,
            content_type="text/markdown",
            version=version,
            space_or_project=f"{ref.owner}/{ref.repo}",
            extra={
                "canonical_url": canonical_url,
                "repo_api_url": _repo_contents_url(ref, repo_path, branch),
                "repo_path": repo_path,
                "repo_owner": ref.owner,
                "repo_name": ref.repo,
                "repo_branch": branch,
                "repo_blob_sha": blob_sha,
            },
        )

    async def _repo_file_item_for_url(self, canonical_url: str) -> ContentItem:
        ref = _repo_ref_from_pages_url(canonical_url)
        if ref is None:
            raise ValueError("GitHub Pages URL must use /pages/<owner>/<repo>/... for PAT-backed fetch")
        branch = str(self.config.get("branch") or "").strip()
        if not branch:
            branch = await self._default_branch(ref)
        tree = await self._repo_tree(ref, branch)
        repo_path, blob_sha = _resolve_repo_markdown_path(ref.page_path, tree)
        if not repo_path:
            raise RuntimeError(
                "Could not find a matching Markdown file in the GitHub repository for this Pages URL."
            )
        return await self._repo_content_item(
            canonical_url=canonical_url,
            ref=ref,
            branch=branch,
            repo_path=repo_path,
            blob_sha=blob_sha,
        )

    async def _repo_subtree_items(self, since: datetime | None) -> AsyncIterator[ContentItem]:
        root_url = self._require_in_site(str(self.config.get("root_url") or ""))
        ref = _repo_ref_from_pages_url(root_url)
        if ref is None:
            raise ValueError("GitHub Pages URL must use /pages/<owner>/<repo>/... for PAT-backed fetch")
        branch = str(self.config.get("branch") or "").strip()
        if not branch:
            branch = await self._default_branch(ref)
        tree = await self._repo_tree(ref, branch)
        entries = _repo_markdown_entries_under_page_path(ref.page_path, tree)

        page_urls: list[str] = []
        scoped_entries: list[tuple[str, str, str]] = []
        for repo_path, blob_sha in entries:
            page_url = _pages_url_for_repo_markdown(ref, repo_path)
            if self._url_is_in_scope(page_url):
                page_urls.append(page_url)
                scoped_entries.append((page_url, repo_path, blob_sha))
        _limit_urls(page_urls, _max_pages(self.config))

        for page_url, repo_path, blob_sha in scoped_entries:
            item = await self._repo_content_item(
                canonical_url=page_url,
                ref=ref,
                branch=branch,
                repo_path=repo_path,
                blob_sha=blob_sha,
            )
            if _is_modified_since(item.last_modified, since):
                yield item

    async def _default_branch(self, ref: "_RepoRef") -> str:
        response = await self._client.get(_repo_api_url(ref))
        response.raise_for_status()
        return str(response.json().get("default_branch") or "main")

    async def _repo_tree(self, ref: "_RepoRef", branch: str) -> list[dict]:
        response = await self._client.get(f"{_repo_api_url(ref)}/git/trees/{branch}?recursive=1")
        response.raise_for_status()
        payload = response.json()
        tree = payload.get("tree")
        return tree if isinstance(tree, list) else []

    async def _repo_path_last_modified(self, ref: "_RepoRef", branch: str, repo_path: str) -> datetime:
        response = await self._client.get(f"{_repo_api_url(ref)}/commits?sha={branch}&path={repo_path}&per_page=1")
        if response.status_code >= 400:
            return datetime.now(timezone.utc)
        commits = response.json()
        if not isinstance(commits, list) or not commits:
            return datetime.now(timezone.utc)
        commit = commits[0].get("commit", {}) if isinstance(commits[0], dict) else {}
        committer = commit.get("committer", {}) if isinstance(commit, dict) else {}
        parsed = _parse_datetime(str(committer.get("date") or ""))
        return parsed or datetime.now(timezone.utc)

    async def _metadata_headers(self, canonical_url: str) -> dict[str, str]:
        try:
            response = await self._client.head(canonical_url)
            if response.status_code < 400:
                return dict(response.headers)
        except requests.RequestException:
            logger.debug("GitHub Pages HEAD failed for %s", canonical_url, exc_info=True)
        response = await self._client.get(canonical_url)
        response.raise_for_status()
        return dict(response.headers)

    async def _sitemap_items(self) -> list[ContentItem] | None:
        sitemap_url = urljoin(self._base_url.rstrip("/") + "/", "sitemap.xml")
        try:
            response = await self._client.get(sitemap_url)
            if response.status_code >= 400:
                return None
            urls = _urls_from_sitemap(response.text)
        except Exception:
            logger.debug("GitHub Pages sitemap discovery failed for %s", sitemap_url, exc_info=True)
            return None

        scoped = sorted(url for url in urls if self._url_is_in_scope(url))
        items: list[ContentItem] = []
        for url in _limit_urls(scoped, _max_pages(self.config)):
            lastmod = _lastmod_for_url(response.text, url)
            if lastmod:
                item = await self._content_item_for_url(url, metadata_headers={})
                item.last_modified = lastmod
                item.version = lastmod.isoformat()
            else:
                item = await self._content_item_for_url(url)
            items.append(item)
        return items

    async def _discover_bfs(self, since: datetime | None) -> AsyncIterator[ContentItem]:
        root_url = self._require_in_site(str(self.config.get("root_url") or ""))
        max_depth = _int_config(self.config, "max_depth", DEFAULT_MAX_DEPTH)
        max_pages = _max_pages(self.config)
        queue: deque[tuple[str, int]] = deque([(root_url, 0)])
        seen: set[str] = set()
        discovered = 0
        while queue:
            url, depth = queue.popleft()
            if url in seen or not self._url_is_in_scope(url):
                continue
            seen.add(url)
            if discovered >= max_pages:
                raise RuntimeError(f"GitHub Pages discovery reached max_pages={max_pages}")
            discovered += 1
            response = await self._client.get(url)
            response.raise_for_status()
            item = _content_item_from_url(url, dict(response.headers))
            if _is_modified_since(item.last_modified, since):
                yield item
            if depth >= max_depth:
                continue
            for link in _links_from_html(response.text, base_url=url):
                canonical = _canonicalize_url(link)
                if canonical not in seen and self._url_is_in_scope(canonical):
                    queue.append((canonical, depth + 1))

    def _require_in_site(self, url: str) -> str:
        canonical_url = _canonicalize_url(url)
        if not canonical_url:
            raise ValueError("GitHub Pages URL is required")
        if (
            _origin_for_url(canonical_url) != _origin_for_url(self._base_url)
            or not _path_is_under(canonical_url, self._base_url)
        ):
            raise ValueError("GitHub Pages URLs must stay under the configured site")
        return canonical_url

    def _url_is_in_scope(self, url: str) -> bool:
        try:
            canonical_url = self._require_in_site(url)
        except ValueError:
            return False
        mode = _sync_mode(self.config)
        root = _canonicalize_url(
            str(self.config.get("root_url") or self.config.get("page_url") or self._base_url)
        )
        if mode == SYNC_MODE_SUBTREE and root:
            if not _path_is_under(canonical_url, root):
                return False
        return not _matches_any(canonical_url, _list_config(self.config.get("exclude_url_patterns")))


class _RequestsAsyncClient:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
        verify: bool | str,
    ) -> None:
        self._session = requests.Session()
        self._session.headers.update(headers)
        self._timeout = timeout
        self._follow_redirects = follow_redirects
        self._verify = verify

    async def get(self, url: str) -> requests.Response:
        return await asyncio.to_thread(self._request, "GET", url)

    async def head(self, url: str) -> requests.Response:
        return await asyncio.to_thread(self._request, "HEAD", url)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._session.close)

    def _request(self, method: str, url: str) -> requests.Response:
        response = self._session.request(
            method,
            url,
            timeout=self._timeout,
            allow_redirects=self._follow_redirects,
            verify=self._verify,
        )
        if _is_github_login_response(response):
            raise RuntimeError(
                "GitHub Pages returned the login page. Check that the PAT can access the published page, "
                "or use a public/no-auth page."
            )
        return response


def _auth_mode(config: dict) -> str:
    return str(config.get("auth_mode") or AUTH_MODE_GITHUB_PAT).strip().lower()


def _scope_url_for_config(config: dict) -> str:
    mode = _sync_mode(config)
    if mode == SYNC_MODE_SUBTREE:
        return str(config.get("root_url") or "")
    if mode == SYNC_MODE_EXPLICIT_LIST:
        pages = _list_config(config.get("pages"))
        return pages[0] if pages else ""
    return str(config.get("page_url") or "")


def _sync_mode(config: dict) -> str:
    return str(config.get("sync_mode") or SYNC_MODE_SINGLE_PAGE).strip().lower()


def _max_pages(config: dict) -> int:
    return _int_config(config, "max_pages", DEFAULT_MAX_PAGES)


def _int_config(config: dict, key: str, default: int) -> int:
    try:
        return max(int(config.get(key, default)), 1)
    except (TypeError, ValueError):
        return default


def _list_config(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _canonicalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    path = re.sub(r"/{2,}", "/", parts.path or "/").rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, "", ""))


def _origin_for_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _site_root_from_pages_url(url: str) -> str:
    canonical = _canonicalize_url(url)
    if not canonical:
        return ""
    parts = urlsplit(canonical)
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "pages":
        root_path = "/" + "/".join(path_parts[:3])
        return urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
    return canonical


def _repo_ref_from_pages_url(url: str) -> _RepoRef | None:
    canonical = _canonicalize_url(url)
    parts = urlsplit(canonical)
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) < 3 or path_parts[0] != "pages":
        return None
    return _RepoRef(
        origin=_origin_for_url(canonical),
        owner=path_parts[1],
        repo=path_parts[2],
        page_path="/".join(path_parts[3:]).strip("/"),
    )


def _repo_api_url(ref: _RepoRef) -> str:
    return f"{ref.origin}/api/v3/repos/{ref.owner}/{ref.repo}"


def _repo_contents_url(ref: _RepoRef, repo_path: str, branch: str) -> str:
    return f"{_repo_api_url(ref)}/contents/{quote(repo_path, safe='/')}?ref={quote(branch, safe='')}"


def _resolve_repo_markdown_path(page_path: str, tree: list[dict]) -> tuple[str, str]:
    blobs: dict[str, str] = {}
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        path = str(entry.get("path") or "")
        if not path.lower().endswith((".md", ".mdx")):
            continue
        blobs[path] = str(entry.get("sha") or "")

    normalized_page = _normalize_page_path(page_path)
    candidates = [
        f"{normalized_page}.md",
        f"{normalized_page}.mdx",
        f"{normalized_page}/index.md",
        f"{normalized_page}/index.mdx",
        f"docs/{normalized_page}.md",
        f"docs/{normalized_page}.mdx",
        f"docs/{normalized_page}/index.md",
        f"docs/{normalized_page}/index.mdx",
    ]
    for candidate in candidates:
        if candidate in blobs:
            return candidate, blobs[candidate]

    target_page = normalized_page.rstrip("/")
    for path, sha in blobs.items():
        if _page_path_from_repo_markdown(path) == target_page:
            return path, sha
    return "", ""


def _repo_markdown_entries_under_page_path(page_path: str, tree: list[dict]) -> list[tuple[str, str]]:
    root_page = _normalize_page_path(page_path)
    entries: list[tuple[str, str]] = []
    for entry in tree:
        if not isinstance(entry, dict) or entry.get("type") != "blob":
            continue
        repo_path = str(entry.get("path") or "")
        if not repo_path.lower().endswith((".md", ".mdx")):
            continue
        if _page_path_is_under(_page_path_from_repo_markdown(repo_path), root_page):
            entries.append((repo_path, str(entry.get("sha") or "")))
    return sorted(entries, key=lambda item: _page_path_from_repo_markdown(item[0]))


def _normalize_page_path(page_path: str) -> str:
    return unquote(page_path).strip("/")


def _page_path_is_under(page_path: str, root_page: str) -> bool:
    candidate = _normalize_page_path(page_path)
    root = _normalize_page_path(root_page)
    return not root or candidate == root or candidate.startswith(root.rstrip("/") + "/")


def _pages_url_for_repo_markdown(ref: _RepoRef, repo_path: str) -> str:
    page_path = _page_path_from_repo_markdown(repo_path)
    return f"{ref.origin}/pages/{ref.owner}/{ref.repo}/{quote(page_path, safe='/')}"


def _page_path_from_repo_markdown(repo_path: str) -> str:
    path = _normalize_page_path(repo_path)
    if path.startswith("docs/"):
        path = path[len("docs/"):]
    if path.endswith("/index.md"):
        path = path[:-len("/index.md")]
    elif path.endswith("/index.mdx"):
        path = path[:-len("/index.mdx")]
    elif path.endswith(".md"):
        path = path[:-len(".md")]
    elif path.endswith(".mdx"):
        path = path[:-len(".mdx")]
    return path.strip("/")


def _path_is_under(url: str, root_url: str) -> bool:
    url_path = unquote(urlsplit(url).path).rstrip("/") + "/"
    root_path = unquote(urlsplit(root_url).path).rstrip("/") + "/"
    return url_path.startswith(root_path)


def _content_item_from_url(url: str, headers: dict[str, str]) -> ContentItem:
    canonical_url = _canonicalize_url(url)
    last_modified = _last_modified_from_headers(headers) or datetime.now(timezone.utc)
    etag = headers.get("etag") or headers.get("ETag") or ""
    return ContentItem(
        item_id=f"github-pages-{hashlib.sha1(canonical_url.encode('utf-8')).hexdigest()}",
        title=_title_from_url(canonical_url),
        source_url=canonical_url,
        last_modified=last_modified,
        content_type=headers.get("content-type", "text/html") or "text/html",
        version=etag or last_modified.isoformat(),
        space_or_project=_site_project(canonical_url),
        extra={"canonical_url": canonical_url},
    )


def _last_modified_from_headers(headers: dict[str, str]) -> datetime | None:
    value = headers.get("last-modified") or headers.get("Last-Modified")
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _lastmod_for_url(sitemap_text: str, url: str) -> datetime | None:
    try:
        root = ET.fromstring(sitemap_text)
    except ET.ParseError:
        return None
    namespace = _xml_namespace(root.tag)
    for node in root.findall(f".//{namespace}url"):
        loc_node = node.find(f"{namespace}loc")
        if loc_node is None or _canonicalize_url(loc_node.text or "") != _canonicalize_url(url):
            continue
        lastmod_node = node.find(f"{namespace}lastmod")
        if lastmod_node is None or not lastmod_node.text:
            return None
        return _parse_datetime(lastmod_node.text.strip())
    return None


def _parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _urls_from_sitemap(sitemap_text: str) -> list[str]:
    root = ET.fromstring(sitemap_text)
    namespace = _xml_namespace(root.tag)
    urls: list[str] = []
    for loc in root.findall(f".//{namespace}loc"):
        if loc.text:
            urls.append(_canonicalize_url(loc.text))
    return urls


def _xml_namespace(tag: str) -> str:
    if tag.startswith("{"):
        return tag.split("}", 1)[0] + "}"
    return ""


def _links_from_html(html: str, *, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        links.append(urljoin(base_url, href))
    return links


def _limit_urls(urls: list[str], max_pages: int) -> list[str]:
    if len(urls) > max_pages:
        raise RuntimeError(f"GitHub Pages discovery reached max_pages={max_pages}")
    return urls


def _matches_any(value: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, value):
                return True
        except re.error:
            logger.warning("Ignoring invalid GitHub Pages exclude pattern: %s", pattern)
    return False


def _is_github_login_response(response: requests.Response) -> bool:
    final_path = urlsplit(str(response.url)).path.rstrip("/")
    if final_path == "/login":
        return True
    for previous in response.history:
        location = previous.headers.get("location", "")
        if urlsplit(location).path.rstrip("/") == "/login":
            return True
    return False


def _is_modified_since(last_modified: datetime, since: datetime | None) -> bool:
    if since is None:
        return True
    candidate = last_modified if last_modified.tzinfo else last_modified.replace(tzinfo=timezone.utc)
    baseline = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
    return candidate > baseline


def _extract_article_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "form"]):
        tag.decompose()
    for selector in [
        "aside",
        ".md-sidebar",
        ".md-header",
        ".md-search",
        ".md-nav",
        ".toc",
        ".table-of-contents",
        "#TableOfContents",
        "[role='navigation']",
    ]:
        for tag in soup.select(selector):
            tag.decompose()
    article = (
        soup.select_one("main article")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    return str(article)


def _title_from_url(url: str) -> str:
    path = urlsplit(url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    return slug.replace("-", " ").strip() or "GitHub Pages Document"


def _title_from_markdown(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _site_project(url: str) -> str:
    parts = [part for part in urlsplit(url).path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "pages":
        return "/".join(parts[1:3])
    return urlsplit(url).netloc
