"""GitHub Repository Gene -- syncs repository files as source documents."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import requests

from memforge.genes.base import Gene
from memforge.genes.local_adapter_packages import package_manifest, read_package_body
from memforge.genes.local_markdown_gene import _parse_dt, _to_markdown
from memforge.github_repo_utils import (
    DEFAULT_INCLUDE_EXTENSIONS,
    build_github_repo_doc_id,
    decode_github_base64_content,
    github_content_type,
    github_extension_allowed,
    github_include_extensions,
    github_include_paths,
    github_path_in_scope,
    normalize_github_relative_path,
    parse_github_repo_url,
)
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)

logger = logging.getLogger(__name__)

__all__ = ["GitHubRepoGene"]

CONNECTION_MODE_CLOUD_PULL = "cloud_pull"
CONNECTION_MODE_LOCAL_PUSH = "local_push"
GITHUB_REPO_PACKAGE_KIND = "github_repo_document"
GITHUB_REPO_CONTENT_ROLE = "repository_file"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
DEFAULT_MAX_FILES = 500


@dataclass(frozen=True)
class _RepoRef:
    repo_url: str
    origin: str
    host: str
    owner: str
    repo: str


class GitHubRepoGene(Gene):
    """GitHub repository file source.

    The same source type supports two delivery modes:
    - ``cloud_pull``: MemForge calls the GitHub REST API directly.
    - ``local_push``: a local CLI pushes selected files into the per-source inbox.
    """

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name=GITHUB_REPO_SOURCE_TYPE,
            display_name="GitHub Repository",
            description="Repository files from GitHub or GitHub Enterprise, scoped by folders and file types",
            default_sync_interval_minutes=1440,
            auth_method="github_repo",
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
                    key="connection_mode",
                    label="Repository Access",
                    field_type=ConfigFieldType.SELECT,
                    required=True,
                    options=[CONNECTION_MODE_CLOUD_PULL, CONNECTION_MODE_LOCAL_PUSH],
                    default=CONNECTION_MODE_CLOUD_PULL,
                    help_text="Use Public internet when MemForge Cloud can reach the repo. Use Internal network / VPN when only your machine can reach it.",
                    group="connection",
                    order=0,
                ),
                ConfigField(
                    key="repo_url",
                    label="Repository URL",
                    field_type=ConfigFieldType.URL,
                    required=True,
                    placeholder="https://github.com/org/repo",
                    help_text="GitHub or GitHub Enterprise repository URL.",
                    group="connection",
                    order=1,
                ),
                ConfigField(
                    key="pat",
                    label="Personal Access Token",
                    field_type=ConfigFieldType.SECRET,
                    required=False,
                    help_text="Optional token for private repositories when using cloud pull. Leave blank for public repositories or local push.",
                    group="connection",
                    order=2,
                ),
                ConfigField(
                    key="ref",
                    label="Branch, Tag, or Commit",
                    field_type=ConfigFieldType.STRING,
                    required=False,
                    default="main",
                    placeholder="main",
                    help_text="Git ref to sync. Leave as main unless the repository uses another branch.",
                    group="scope",
                    order=0,
                ),
                ConfigField(
                    key="include_paths",
                    label="Folders or Files",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    placeholder="Payroll Processing/, Flexible Payroll/README.md",
                    help_text="Repo-relative folders or files to sync. Empty means the whole repository, filtered by extension.",
                    group="scope",
                    order=1,
                ),
                ConfigField(
                    key="include_extensions",
                    label="File Extensions",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    default=DEFAULT_INCLUDE_EXTENSIONS,
                    placeholder=DEFAULT_INCLUDE_EXTENSIONS,
                    help_text="Text-like extensions to sync. Binary assets are skipped unless explicitly allowed.",
                    group="scope",
                    order=2,
                    advanced=True,
                ),
                ConfigField(
                    key="max_files",
                    label="Max Files",
                    field_type=ConfigFieldType.INTEGER,
                    required=False,
                    default=str(DEFAULT_MAX_FILES),
                    help_text="Stop with an error if the selected scope contains more files.",
                    group="scope",
                    order=3,
                    advanced=True,
                ),
            ],
        )

    @classmethod
    def normalize_config(cls, config: dict) -> None:
        repo_url = str(config.get("repo_url") or "").strip()
        if repo_url:
            repo_ref = _parse_repo_url(repo_url)
            config["repo_url"] = repo_ref.repo_url
            config["base_url"] = repo_ref.origin
        config["connection_mode"] = _connection_mode(config)
        if not str(config.get("ref") or "").strip():
            config["ref"] = "main"

    async def authenticate(self) -> None:
        self._repo_ref = _parse_repo_url(str(self.config.get("repo_url") or ""))
        self._connection_mode = _connection_mode(self.config)
        if self._connection_mode == CONNECTION_MODE_LOCAL_PUSH:
            if self._package_manifest():
                return
            self._documents_dir().mkdir(parents=True, exist_ok=True)
            return
        if self._connection_mode != CONNECTION_MODE_CLOUD_PULL:
            raise ValueError("GitHub Repository Access must be Public internet or Internal network / VPN")

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "memforge-github-repo-source",
        }
        pat = str(self.config.get("pat") or "").strip()
        if pat:
            headers["Authorization"] = f"Bearer {pat}"
        self._client = _RequestsAsyncClient(headers=headers, timeout=30.0)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        if _connection_mode(self.config) == CONNECTION_MODE_LOCAL_PUSH:
            async for item in self._discover_local_push(since):
                yield item
            return

        if not hasattr(self, "_client"):
            await self.authenticate()
        repo_ref = self._repo_ref
        ref = str(self.config.get("ref") or "").strip()
        if not ref:
            ref = await self._default_branch(repo_ref)
        entries = await self._repo_tree(repo_ref, ref)
        include_paths = github_include_paths(self.config)
        include_exts = github_include_extensions(self.config)
        max_files = _int_config(self.config, "max_files", DEFAULT_MAX_FILES)
        selected = [
            entry
            for entry in entries
            if entry.get("type") == "blob"
            and github_path_in_scope(str(entry.get("path") or ""), include_paths)
            and github_extension_allowed(str(entry.get("path") or ""), include_exts)
        ]
        if len(selected) > max_files:
            raise RuntimeError(f"GitHub Repository discovery matched {len(selected)} files, exceeding max_files={max_files}")
        for entry in selected:
            path = normalize_github_relative_path(str(entry.get("path") or ""))
            blob_sha = str(entry.get("sha") or "")
            yield ContentItem(
                item_id=build_github_repo_doc_id(
                    source_id=self.source_id,
                    repo_url=repo_ref.repo_url,
                    repo_ref=ref,
                    relative_path=path,
                ),
                title=_title_from_path(path),
                source_url=_file_url(repo_ref, ref, path),
                last_modified=datetime.now(timezone.utc),
                content_type=github_content_type(path),
                version=blob_sha,
                space_or_project=f"{repo_ref.owner}/{repo_ref.repo}",
                labels=["github_repo"],
                extra={
                    "connection_mode": CONNECTION_MODE_CLOUD_PULL,
                    "repo_url": repo_ref.repo_url,
                    "repo_host": repo_ref.host,
                    "repo_owner": repo_ref.owner,
                    "repo_name": repo_ref.repo,
                    "repo_ref": ref,
                    "relative_path": path,
                    "blob_sha": blob_sha,
                    "repo_contents_url": _contents_url(repo_ref, path, ref),
                },
            )

    async def fetch(self, item: ContentItem) -> RawContent:
        if item.extra.get("package_uri") or item.extra.get("package_path"):
            return RawContent(
                item=item,
                body=read_package_body(self, item, source_label="GitHub repository"),
                content_type="application/json",
            )

        response = await self._client.get(str(item.extra["repo_contents_url"]))
        response.raise_for_status()
        payload = response.json()
        try:
            raw = decode_github_base64_content(
                content=payload.get("content"),
                encoding=payload.get("encoding"),
                size=payload.get("size"),
                label=str(item.extra.get("relative_path") or item.item_id),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        return RawContent(item=item, body=raw, content_type=item.content_type or "text/plain")

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        if raw.content_type == "application/json":
            package = json.loads(raw.body.decode("utf-8"))
            markdown = _to_markdown(package.get("content_type") or "text/markdown", package.get("markdown") or "")
            semantics = _semantics_from_package(package)
            return NormalizedContent(item=raw.item, markdown_body=markdown, source_semantics=semantics)

        markdown = _to_markdown(raw.content_type, raw.body.decode("utf-8", errors="replace"))
        semantics = {
            "source_type": GITHUB_REPO_SOURCE_TYPE,
            "connection_mode": raw.item.extra.get("connection_mode"),
            "repo_url": raw.item.extra.get("repo_url"),
            "repo_host": raw.item.extra.get("repo_host"),
            "repo_owner": raw.item.extra.get("repo_owner"),
            "repo_name": raw.item.extra.get("repo_name"),
            "repo_ref": raw.item.extra.get("repo_ref"),
            "relative_path": raw.item.extra.get("relative_path"),
            "blob_sha": raw.item.extra.get("blob_sha"),
            "content_type": raw.item.content_type,
            "canonical_url": raw.item.source_url,
        }
        return NormalizedContent(item=raw.item, markdown_body=markdown, source_semantics=semantics)

    async def _default_branch(self, ref: _RepoRef) -> str:
        response = await self._client.get(_repo_api_url(ref))
        response.raise_for_status()
        return str(response.json().get("default_branch") or "main")

    async def _repo_tree(self, ref: _RepoRef, repo_ref: str) -> list[dict]:
        response = await self._client.get(f"{_repo_api_url(ref)}/git/trees/{quote(repo_ref, safe='')}?recursive=1")
        response.raise_for_status()
        payload = response.json()
        if payload.get("truncated") is True:
            raise RuntimeError(
                "GitHub tree response was truncated; narrow include_paths or use a non-recursive tree walk"
            )
        tree = payload.get("tree")
        return tree if isinstance(tree, list) else []

    async def _discover_local_push(self, since: datetime | None) -> AsyncIterator[ContentItem]:
        manifest = self._package_manifest()
        if manifest:
            async for item in self._discover_local_push_manifest(manifest, since):
                yield item
            return
        documents_dir = self._documents_dir()
        selected: list[tuple[Path, dict]] = []
        for package_path in sorted(documents_dir.rglob("*.json")):
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable GitHub repo package: %s", package_path)
                continue
            if package.get("package_kind") != GITHUB_REPO_PACKAGE_KIND:
                continue
            if not _package_matches_config(package, self.config):
                continue
            selected.append((package_path, package))
        max_files = _int_config(self.config, "max_files", DEFAULT_MAX_FILES)
        if len(selected) > max_files:
            raise RuntimeError(
                f"GitHub Repository Internal network / VPN sync matched {len(selected)} files, exceeding max_files={max_files}"
            )
        for package_path, package in selected:
            last_modified = _parse_dt(str(package.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            yield ContentItem(
                item_id=package["doc_id"],
                title=package.get("title") or package["doc_id"],
                source_url=package.get("source_url", ""),
                last_modified=last_modified,
                content_type=package.get("content_type") or "text/markdown",
                space_or_project=package.get("space_or_project") or "",
                version=package.get("version", ""),
                author=package.get("submitted_by"),
                labels=["github_repo"],
                extra={
                    "package_path": str(package_path),
                    "relative_path": package.get("relative_path"),
                },
            )

    async def _discover_local_push_manifest(
        self,
        manifest: list[dict],
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        selected = [
            entry for entry in manifest
            if str(entry.get("package_uri") or "").strip() and _package_matches_config(entry, self.config)
        ]
        max_files = _int_config(self.config, "max_files", DEFAULT_MAX_FILES)
        if len(selected) > max_files:
            raise RuntimeError(
                f"GitHub Repository Internal network / VPN sync matched {len(selected)} files, exceeding max_files={max_files}"
            )
        for entry in sorted(
            selected,
            key=lambda item: (str(item.get("last_modified") or ""), str(item.get("doc_id") or "")),
        ):
            last_modified = _parse_dt(str(entry.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            doc_id = str(entry.get("doc_id") or "")
            yield ContentItem(
                item_id=doc_id,
                title=str(entry.get("title") or doc_id),
                source_url=str(entry.get("source_url") or ""),
                last_modified=last_modified,
                content_type=str(entry.get("content_type") or "text/markdown"),
                space_or_project=str(entry.get("space_or_project") or ""),
                version=str(entry.get("version") or ""),
                author=entry.get("submitted_by"),
                labels=["github_repo"],
                extra={
                    "package_uri": str(entry.get("package_uri")),
                    "package_path": entry.get("package_path"),
                    "relative_path": entry.get("relative_path"),
                },
            )

    def _documents_dir(self) -> Path:
        configured = str(self.config.get("documents_dir") or "").strip()
        if not configured:
            raise ValueError("GitHub Repository Internal network / VPN source is missing documents_dir")
        return Path(configured).expanduser()

    def _package_manifest(self) -> list[dict]:
        return package_manifest(self.config)


class _RequestsAsyncClient:
    def __init__(self, *, headers: dict[str, str], timeout: float) -> None:
        self._session = requests.Session()
        self._session.headers.update(headers)
        self._timeout = timeout

    async def get(self, url: str) -> requests.Response:
        return await asyncio.to_thread(self._session.get, url, timeout=self._timeout)

    async def aclose(self) -> None:
        await asyncio.to_thread(self._session.close)


def _parse_repo_url(url: str) -> _RepoRef:
    value = str(url or "").strip()
    if not value:
        raise ValueError("GitHub Repository URL is required")
    try:
        parsed = parse_github_repo_url(value)
    except ValueError as exc:
        raise ValueError(str(exc).replace("repo_url", "GitHub Repository URL")) from exc
    return _RepoRef(
        repo_url=parsed["repo_url"],
        origin=parsed["origin"],
        host=parsed["host"],
        owner=parsed["owner"],
        repo=parsed["repo"],
    )


def _connection_mode(config: dict) -> str:
    return str(config.get("connection_mode") or CONNECTION_MODE_CLOUD_PULL).strip().lower()


def _repo_api_url(ref: _RepoRef) -> str:
    if ref.host == "github.com":
        return f"https://api.github.com/repos/{quote(ref.owner, safe='')}/{quote(ref.repo, safe='')}"
    return f"{ref.origin}/api/v3/repos/{quote(ref.owner, safe='')}/{quote(ref.repo, safe='')}"


def _contents_url(ref: _RepoRef, relative_path: str, repo_ref: str) -> str:
    return f"{_repo_api_url(ref)}/contents/{quote(relative_path, safe='/')}?ref={quote(repo_ref, safe='')}"


def _file_url(ref: _RepoRef, repo_ref: str, relative_path: str) -> str:
    return f"{ref.repo_url}/blob/{quote(repo_ref, safe='')}/{quote(relative_path, safe='/')}"


def _title_from_path(relative_path: str) -> str:
    filename = relative_path.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.[^.]+$", "", filename) or filename


def _int_config(config: dict, key: str, default: int) -> int:
    try:
        return max(int(config.get(key, default)), 1)
    except (TypeError, ValueError):
        return default


def _package_matches_config(package: dict, config: dict) -> bool:
    configured_repo = str(config.get("repo_url") or "").strip()
    if configured_repo and str(package.get("repo_url") or "").strip() != _parse_repo_url(configured_repo).repo_url:
        return False
    configured_ref = str(config.get("ref") or "main").strip() or "main"
    if str(package.get("repo_ref") or "").strip() != configured_ref:
        return False
    relative_path = str(package.get("relative_path") or "")
    if not relative_path:
        return False
    try:
        include_paths = github_include_paths(config)
        normalized_path = normalize_github_relative_path(relative_path)
    except ValueError:
        return False
    if not github_path_in_scope(normalized_path, include_paths):
        return False
    return github_extension_allowed(normalized_path, github_include_extensions(config))


def _semantics_from_package(package: dict) -> dict:
    return {
        "source_type": GITHUB_REPO_SOURCE_TYPE,
        "connection_mode": CONNECTION_MODE_LOCAL_PUSH,
        "repo_url": package.get("repo_url"),
        "repo_host": package.get("repo_host"),
        "repo_owner": package.get("repo_owner"),
        "repo_name": package.get("repo_name"),
        "repo_ref": package.get("repo_ref"),
        "relative_path": package.get("relative_path"),
        "blob_sha": package.get("blob_sha"),
        "content_type": package.get("content_type"),
        "canonical_url": package.get("source_url"),
    }
