"""GitHub Repository Gene -- syncs repository files as source documents."""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from memforge.genes.document_gene import DocumentGene
from memforge.genes.local_adapter_packages import (
    decode_package,
    has_package_manifest,
    open_packaged_source_artifact,
    read_package_body,
    source_artifacts_from_package,
)
from memforge.genes.local_markdown_gene import _to_markdown
from memforge.genes.toolkit.config import (
    FieldIssue,
    SourceConfigInvalid,
    int_config,
    list_config,
    resolve_pat,
    tls_verify,
)
from memforge.genes.toolkit.github_api import (
    GitHubApi,
    GitHubRepository,
    GitHubTreeSelectionExceeded,
    github_rate_limit_wait,
)
from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.genes.toolkit.local_packages import device_packages
from memforge.github_repo_utils import (
    DEFAULT_INCLUDE_EXTENSIONS,
    GITHUB_SYMLINK_MODE,
    build_github_repo_doc_id,
    GitHubBlobBuffer,
    GitHubResolvedTreeEntry,
    GitHubTreeEntryResolver,
    decode_github_text,
    github_content_paths,
    github_content_type,
    github_content_type_is_binary,
    github_exclude_paths,
    github_extension_allowed,
    github_include_extensions,
    github_include_paths,
    github_path_in_scope,
    normalize_github_scope_paths,
    normalize_github_relative_path,
    parse_github_repo_url,
    validate_github_file_mode,
)
from memforge.models import (
    ConfigCondition,
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ConfigOption,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
    ScopeListing,
    ScopeListingKind,
)
from memforge.repo_identity import normalize_repo_identifier
from memforge.source_time import SOURCE_UPDATED_AT_KEY, latest_source_time
from memforge.source_artifacts import (
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    RawSourceArtifact,
    SourceArtifactContractError,
    SourceArtifactDownload,
    parse_source_artifact_content_length,
)

logger = logging.getLogger(__name__)
# Why a selected binary file, an image or a PDF, has no text body: its content is its Artifact.
_BINARY_FILE_EMPTY_EVIDENCE = "github_binary_file_has_no_text_body"

__all__ = ["GitHubRepoGene", "github_renamed_from"]

CONNECTION_MODE_CLOUD_PULL = "cloud_pull"
CONNECTION_MODE_LOCAL_PUSH = "local_push"
GITHUB_REPO_PACKAGE_KIND = "github_repo_document"
GITHUB_REPO_CONTENT_ROLE = "repository_file"
GITHUB_REPO_SOURCE_TYPE = "github_repo"
DEFAULT_MAX_FILES = 500
# ``ContentItem.extra`` key for the file's latest commit time at the collection commit.
LAST_COMMIT_AT_KEY = "last_commit_at"


def github_renamed_from(
    *,
    source_id: str,
    repo_url: str,
    repo_ref: str,
    previous_filename: object,
    rename_evidence_authoritative: object,
) -> str | None:
    """The ``renamed_from`` of a discovered file: the Document id of the path it was renamed from.

    Only authoritative evidence names a predecessor, such as a validated
    Compare API ``renamed`` record, which reports ``previous_filename`` once.
    A matching blob SHA is never read as a rename, because a copy followed by
    a delete looks the same.
    """

    if rename_evidence_authoritative is not True or not previous_filename:
        return None
    return build_github_repo_doc_id(
        source_id=source_id,
        repo_url=repo_url,
        repo_ref=repo_ref,
        relative_path=normalize_github_relative_path(str(previous_filename)),
    )


@dataclass(frozen=True)
class _RepoRef:
    repo_url: str
    origin: str
    host: str
    owner: str
    repo: str

    @property
    def repository(self) -> GitHubRepository:
        return GitHubRepository(origin=self.origin, owner=self.owner, name=self.repo)


@dataclass(frozen=True)
class _ScopeTree:
    """The repository tree at the collected commit and the entries the configured scope selects."""

    repo_ref: _RepoRef
    ref: str
    commit_sha: str
    root_tree: str
    selected: list[dict]
    entries_by_path: dict[str, dict]
    exclude_paths: list[str]


class GitHubRepoGene(DocumentGene):
    """GitHub repository file source.

    The same source type supports two delivery modes:
    - ``cloud_pull``: MemForge calls the GitHub REST API directly.
    - ``local_push``: the user's daemon reads the remote repository with its
      local GitHub session and uploads raw file packages for server processing.
    """

    listing_kinds = frozenset({ScopeListingKind.EXISTENCE})
    # Opened by ``authenticate`` for cloud pull; a pushed repository is read from its packages.
    _client: ConnectorHttpClient | None = None

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name=GITHUB_REPO_SOURCE_TYPE,
            display_name="GitHub Repository",
            description="Repository files from GitHub or GitHub Enterprise, scoped by folders and file types",
            default_sync_interval_minutes=1440,
            auth_method="github_repo",
            data_shape="document",
            execution_kinds=("server", "local_agent"),
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Repository access", order=0, purpose="connection"),
                ConfigGroup(key="scope", label="Files to sync", order=1, purpose="content"),
            ],
            fields=[
                ConfigField(
                    key="connection_mode",
                    label="Repository Access",
                    field_type=ConfigFieldType.SELECT,
                    required=True,
                    choices=[
                        ConfigOption(CONNECTION_MODE_CLOUD_PULL, "Cloud access"),
                        ConfigOption(CONNECTION_MODE_LOCAL_PUSH, "Local sync"),
                    ],
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
                    visible_when=(ConfigCondition("connection_mode", (CONNECTION_MODE_CLOUD_PULL,)),),
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
                    widget="github_repo_paths",
                ),
                ConfigField(
                    key="exclude_paths",
                    label="Folders or Files to Exclude",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    placeholder="docs/archived/, docs/outdated.md",
                    help_text="Repository-relative folders or files to leave out. Exclusions override selected paths.",
                    group="scope",
                    order=2,
                    widget="github_repo_paths",
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
                    order=3,
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
                    order=4,
                    advanced=True,
                ),
            ],
        )

    @classmethod
    def normalize_config(cls, config: dict) -> None:
        repo_url = str(config.get("repo_url") or "").strip()
        if repo_url:
            try:
                repo_ref = _parse_repo_url(repo_url)
            except ValueError as exc:
                raise SourceConfigInvalid([FieldIssue("repo_url", str(exc))]) from None
            config["repo_url"] = repo_ref.repo_url
            config["base_url"] = repo_ref.origin
        config["connection_mode"] = _connection_mode(config)
        if not str(config.get("ref") or "").strip():
            config["ref"] = "main"
        config["include_paths"] = normalize_github_scope_paths(list_config(config.get("include_paths")))
        config["exclude_paths"] = normalize_github_scope_paths(list_config(config.get("exclude_paths")))

    async def authenticate(self) -> None:
        """Open the client for cloud pull; a pushed repository needs only its documents directory.

        The token is optional: a public repository is read without one. TLS
        trusts ``tls_ca_bundle``, else ``REQUESTS_CA_BUNDLE``, else the
        default trust store, so a GitHub Enterprise Server behind a private
        CA is reached however its users already trust that CA.
        """
        await self.close()
        self._repo_ref = _parse_repo_url(str(self.config.get("repo_url") or ""))
        self._connection_mode = _connection_mode(self.config)
        if self._connection_mode == CONNECTION_MODE_LOCAL_PUSH:
            if has_package_manifest(self.config):
                return
            self._documents_dir().mkdir(parents=True, exist_ok=True)
            return
        if self._connection_mode != CONNECTION_MODE_CLOUD_PULL:
            raise ValueError("GitHub Repository Access must be Public internet or Internal network / VPN")

        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "memforge-github-repo-source",
        }
        if str(self.config.get("pat") or "").strip() or str(self.config.get("pat_env_var") or "").strip():
            headers["Authorization"] = f"Bearer {resolve_pat(self.config, 'GitHub Repository')}"
        self._client = ConnectorHttpClient(
            product_name="GitHub",
            headers=headers,
            timeout=30.0,
            verify=tls_verify(self.config, honor_requests_ca_bundle=True),
            follow_redirects=True,
            rate_limit=github_rate_limit_wait,
        )

    def github_api(self) -> GitHubApi:
        """The configured repository's REST API, read through this Gene's client; ``authenticate`` first.

        The admin folder browse reads the tree this way, with the client and
        TLS a sync would use::

            await gene.authenticate()
            entries = await gene.github_api().tree("main", max_requests=BROWSE_TREE_REQUEST_BUDGET)
        """
        return GitHubApi(self._connected_client(), self._repo_ref.repository, label="GitHub repository")

    def _connected_client(self) -> ConnectorHttpClient:
        if self._client is None:
            raise RuntimeError("GitHub Repository is not connected: call authenticate() first")
        return self._client

    @classmethod
    def lists_scope(cls, config: Mapping[str, Any]) -> bool:
        """The tree at the collected commit, or the device's snapshot manifest, holds the whole scope."""
        return True

    async def list_scope(self) -> ScopeListing:
        """Every file the branch's tree selects at its current commit, by Document id.

        The tree at a commit is everything the repository holds in scope, so
        this is an existence listing: a file it no longer names was removed or
        renamed. Discovery reads the same tree on every run, so a run that
        discovered returns that tree's listing.
        """
        if _connection_mode(self.config) == CONNECTION_MODE_LOCAL_PUSH:
            raise ValueError(
                "A GitHub repository pushed from the internal network is listed by its device's snapshot manifest, "
                "not by the server"
            )
        if self._discovered_scope is not None:
            return self._discovered_scope
        tree = await self._scope_tree()
        return ScopeListing(kind=ScopeListingKind.EXISTENCE, doc_ids=frozenset(self._listed_doc_ids(tree)))

    async def _scope_tree(self) -> _ScopeTree:
        """Read the repository's complete tree at the ref's current commit and select the files in scope, or raise.

        The whole tree is read, not only the selected folders, because a
        selected symbolic link resolves anywhere in it. Reading stops as soon
        as more than ``max_files`` files are selected.
        """
        github = self.github_api()
        repo_ref = self._repo_ref
        ref = str(self.config.get("ref") or "").strip()
        if not ref:
            ref = await github.default_branch()
        commit = await github.commit(ref)
        include_paths = github_include_paths(self.config)
        exclude_paths = github_exclude_paths(self.config)
        include_exts = github_include_extensions(self.config)
        max_files = int_config(self.config, "max_files", default=DEFAULT_MAX_FILES)

        def selects(entry: Mapping[str, Any]) -> bool:
            path = str(entry.get("path") or "")
            return (
                (entry.get("type") in {"blob", "commit"} or entry.get("mode") == GITHUB_SYMLINK_MODE)
                and github_path_in_scope(path, include_paths, exclude_paths)
                and github_extension_allowed(path, include_exts)
            )

        try:
            entries = await github.tree(commit.tree_sha, selects=selects, max_selected=max_files)
        except GitHubTreeSelectionExceeded as exc:
            count = str(exc.selected) if exc.exact else f"at least {exc.selected}"
            raise RuntimeError(
                f"GitHub Repository discovery matched {count} files, exceeding max_files={max_files}"
            ) from exc
        return _ScopeTree(
            repo_ref=repo_ref,
            ref=ref,
            commit_sha=commit.sha,
            root_tree=commit.tree_sha,
            selected=[entry for entry in entries if selects(entry)],
            entries_by_path={
                normalize_github_relative_path(str(entry.get("path") or "")).rstrip("/"): entry
                for entry in entries
            },
            exclude_paths=exclude_paths,
        )

    def _listed_doc_ids(self, tree: _ScopeTree) -> set[str]:
        """The Document id of every file the tree selects; a file's path is part of its id."""
        return {
            build_github_repo_doc_id(
                source_id=self.source_id,
                repo_url=tree.repo_ref.repo_url,
                repo_ref=tree.ref,
                relative_path=normalize_github_relative_path(str(entry.get("path") or "")).rstrip("/"),
            )
            for entry in tree.selected
        }

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        if _connection_mode(self.config) == CONNECTION_MODE_LOCAL_PUSH:
            async for item in self._discover_local_push(since):
                yield item
            return

        tree = await self._scope_tree()
        repo_ref = tree.repo_ref
        ref = tree.ref
        commit_sha = tree.commit_sha
        root_tree = tree.root_tree
        for entry in tree.selected:
            resolved = await self._resolve_repo_entry(
                repo_ref,
                entry,
                entries_by_path=tree.entries_by_path,
                exclude_paths=tree.exclude_paths,
            )
            path = resolved.logical_path
            content_entry = resolved.content_entry
            blob_sha = str(content_entry.get("sha") or "")
            content_type = github_content_type(path)
            resolved_content_type = github_content_type(resolved.resolved_relative_path or path)
            if resolved.symlink_chain and (
                github_content_type_is_binary(content_type)
                or github_content_type_is_binary(resolved_content_type)
            ):
                raise RuntimeError(
                    f"GitHub symlink {path} selects a binary Artifact; symlinked Artifacts are unsupported"
                )
            doc_id = build_github_repo_doc_id(
                source_id=self.source_id,
                repo_url=repo_ref.repo_url,
                repo_ref=ref,
                relative_path=path,
            )
            yield ContentItem(
                item_id=doc_id,
                title=_title_from_path(path),
                source_url=_file_url(repo_ref, ref, path),
                # Discovery time. The file's source time is its commit time, read in fetch.
                last_modified=datetime.now(timezone.utc),
                content_type=content_type,
                version=blob_sha,
                space_or_project=f"{repo_ref.owner}/{repo_ref.repo}",
                labels=["github_repo"],
                extra={
                    "connection_mode": CONNECTION_MODE_CLOUD_PULL,
                    "file_identity_contract": "repository_path",
                    "repo_url": repo_ref.repo_url,
                    "repo_host": repo_ref.host,
                    "repo_owner": repo_ref.owner,
                    "repo_name": repo_ref.repo,
                    "repo_ref": ref,
                    "relative_path": path,
                    "blob_sha": blob_sha,
                    "blob_size": content_entry.get("size"),
                    "file_mode": content_entry.get("mode"),
                    "logical_file_mode": entry.get("mode"),
                    "symlink_chain": [dict(item) for item in resolved.symlink_chain],
                    "resolved_relative_path": resolved.resolved_relative_path,
                    "commit_sha": commit_sha,
                    "root_tree_sha": root_tree,
                    "repo_blob_url": repo_ref.repository.blob_url(blob_sha),
                },
            )
        self.record_scope_listing(ScopeListingKind.EXISTENCE, self._listed_doc_ids(tree))

    async def fetch(self, item: ContentItem) -> RawContent:
        if item.extra.get("package_uri") or item.extra.get("package_path"):
            body = read_package_body(self, item, source_label="GitHub repository")
            package = decode_package(body, GITHUB_REPO_PACKAGE_KIND)
            if package is None:
                raise ValueError(f"GitHub repository package {item.item_id} is not a {GITHUB_REPO_PACKAGE_KIND} package")
            return _package_raw_content(item, body, package, artifacts=source_artifacts_from_package(package))

        try:
            validate_github_file_mode(item.extra.get("file_mode"), label=str(item.extra.get("relative_path")))
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if (
            github_content_type_is_binary(item.content_type)
            and item.content_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES
        ):
            raise SourceArtifactContractError(
                f"GitHub selected file has unsupported binary media type: {item.content_type}"
            )
        # Kept with the item so reprocessing from the stored document reads it too.
        item.extra[LAST_COMMIT_AT_KEY] = await self._last_commit_at(item)

        if item.content_type in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
            relative_path = str(item.extra.get("relative_path") or "").strip()
            blob_sha = str(item.extra.get("blob_sha") or "").strip()
            blob_url = str(item.extra.get("repo_blob_url") or "").strip()
            if not relative_path or not blob_sha or not blob_url:
                raise SourceArtifactContractError("GitHub binary Artifact identity is incomplete")
            blob_size = item.extra.get("blob_size")
            declared_size = (
                blob_size if isinstance(blob_size, int) and not isinstance(blob_size, bool) and blob_size >= 0 else None
            )
            return RawContent(
                item=item,
                body=b"",
                content_type=item.content_type,
                authoritative_empty=True,
                empty_evidence=_BINARY_FILE_EMPTY_EVIDENCE,
                artifacts=(
                    RawSourceArtifact(
                        provider_key=relative_path,
                        parent_observation_type="file_content",
                        parent_provider_key="content",
                        provider_revision=blob_sha,
                        filename=Path(relative_path).name,
                        media_type=item.content_type,
                        declared_size_bytes=declared_size,
                        locator={
                            "blob_sha": blob_sha,
                            "blob_url": blob_url,
                        },
                    ),
                ),
            )

        try:
            raw = await self._read_text_blob(
                blob_url=str(item.extra["repo_blob_url"]),
                blob_sha=str(item.extra.get("blob_sha") or ""),
                blob_size=item.extra.get("blob_size"),
                label=str(item.extra.get("relative_path") or item.item_id),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        return RawContent(
            item=item,
            body=raw,
            content_type=item.content_type or "text/plain",
            authoritative_empty=not raw.strip(),
            empty_evidence=(
                "github_raw_blob_verified_empty_file"
                if not raw.strip()
                else None
            ),
        )

    @asynccontextmanager
    async def open_source_artifact(self, artifact: RawSourceArtifact):
        """Open one immutable Git blob without buffering it in the Gene."""

        if artifact.locator.get("input_uri"):
            async with open_packaged_source_artifact(self, artifact) as download:
                yield download
            return
        blob_url = str(artifact.locator.get("blob_url") or "").strip()
        blob_sha = str(artifact.locator.get("blob_sha") or "").strip()
        if not blob_url or not blob_sha or not blob_url.rstrip("/").endswith(f"/{quote(blob_sha, safe='')}"):
            raise SourceArtifactContractError("GitHub binary Artifact locator is incomplete")
        async with self._connected_client().stream(
            "GET",
            blob_url,
            headers={"Accept": "application/vnd.github.raw+json"},
        ) as response:
            response.raise_for_status()
            yield SourceArtifactDownload(
                chunks=response.aiter_bytes(),
                # GitHub raw blob responses may use application/octet-stream.
                # The immutable tree descriptor supplies the scoped media type.
                media_type=None,
                content_length=parse_source_artifact_content_length(response.headers.get("content-length")),
                content_encoding=response.headers.get("content-encoding"),
            )

    def raw_from_stored_input(self, item: ContentItem, body: bytes, content_type: str) -> RawContent:
        """Rebuild a stored file, a package of one, or a selected binary file.

        A binary file, an image or a PDF, is stored with no text body, as it was
        fetched, and its item names its media type; that is the evidence the
        fetch gave for its empty text. Its bytes come from the committed Artifact.
        """
        package = decode_package(body, GITHUB_REPO_PACKAGE_KIND)
        if package is not None:
            return _package_raw_content(item, body, package)
        if item.content_type in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES and not body.strip():
            return RawContent(
                item=item,
                body=body,
                content_type=item.content_type,
                authoritative_empty=True,
                empty_evidence=_BINARY_FILE_EMPTY_EVIDENCE,
            )
        return super().raw_from_stored_input(item, body, content_type)

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        # A local-push package carries the file and its repository; a
        # cloud-pull body is the file itself, described by its discovered item.
        package = decode_package(raw.body, GITHUB_REPO_PACKAGE_KIND)
        if package is not None:
            return NormalizedContent(
                item=raw.item,
                markdown_body=_package_markdown(package),
                source_semantics=_semantics_from_package(package),
            )

        text = decode_github_text(raw.body, label=str(raw.item.extra.get("relative_path") or raw.item.item_id))
        markdown = _to_markdown(raw.content_type, text)
        semantics = {
            SOURCE_UPDATED_AT_KEY: raw.item.extra.get(LAST_COMMIT_AT_KEY),
            "source_type": GITHUB_REPO_SOURCE_TYPE,
            "connection_mode": raw.item.extra.get("connection_mode"),
            "repo_url": raw.item.extra.get("repo_url"),
            "repo_identifier": _validated_repo_identifier(str(raw.item.extra.get("repo_url") or "")),
            "repo_host": raw.item.extra.get("repo_host"),
            "repo_owner": raw.item.extra.get("repo_owner"),
            "repo_name": raw.item.extra.get("repo_name"),
            "repo_ref": raw.item.extra.get("repo_ref"),
            "relative_path": raw.item.extra.get("relative_path"),
            "blob_sha": raw.item.extra.get("blob_sha"),
            "content_type": raw.item.content_type,
            "canonical_url": raw.item.source_url,
        }
        if raw.item.extra.get("logical_file_mode") == GITHUB_SYMLINK_MODE:
            semantics.update({
                "logical_file_mode": GITHUB_SYMLINK_MODE,
                "symlink_chain": raw.item.extra.get("symlink_chain"),
                "resolved_relative_path": raw.item.extra.get("resolved_relative_path"),
            })
        return NormalizedContent(item=raw.item, markdown_body=markdown, source_semantics=semantics)

    async def _last_commit_at(self, item: ContentItem) -> str | None:
        """The latest commit time of the file, or its symlink target, at the collection commit.

        A blob synced before keeps the time recorded then: a later commit that
        leaves the blob unchanged does not change its content. Otherwise one
        commits request per path. GitHub lists no commit only for a path with
        no history, and a refused request tells nothing about the content; both
        leave the time unknown, and the next sync asks again.
        """

        stored_time = item.stored_extra.get(LAST_COMMIT_AT_KEY)
        if stored_time and item.stored_extra.get("blob_sha") == item.extra.get("blob_sha"):
            return str(stored_time)
        commit_sha = str(item.extra.get("commit_sha") or "").strip()
        relative_path = str(item.extra.get("relative_path") or "").strip()
        if not commit_sha or not relative_path:
            raise RuntimeError("GitHub file identity is missing its collection commit or path")
        github = self.github_api()
        commit_times = []
        for path in github_content_paths(relative_path, item.extra.get("resolved_relative_path")):
            try:
                commit_times.append(await github.latest_commit_time(path, ref=commit_sha))
            except httpx.HTTPStatusError as exc:
                logger.warning("GitHub commit time for %s is unknown: %s", path, exc)
                return None
        return latest_source_time(commit_times)

    async def _resolve_repo_entry(
        self,
        repo_ref: _RepoRef,
        entry: dict,
        *,
        entries_by_path: dict[str, dict],
        exclude_paths: list[str],
    ) -> GitHubResolvedTreeEntry:
        resolver = GitHubTreeEntryResolver(
            entry,
            entries_by_path=entries_by_path,
            exclude_paths=exclude_paths,
        )
        try:
            while link := resolver.next_symlink_read():
                raw_target = await self._read_text_blob(
                    blob_url=repo_ref.repository.blob_url(link.blob_sha),
                    blob_sha=link.blob_sha,
                    blob_size=link.blob_size,
                    label=link.path,
                )
                resolver.accept_symlink_target(raw_target)
            return resolver.result()
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    async def _read_text_blob(
        self,
        *,
        blob_url: str,
        blob_sha: str,
        blob_size: object,
        label: str,
    ) -> bytes:
        async with self._connected_client().stream(
            "GET",
            blob_url,
            headers={"Accept": "application/vnd.github.raw+json", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ValueError("GitHub raw blob returned unsupported content encoding")
            length_header = response.headers.get("content-length")
            if length_header is not None and not re.fullmatch(r"[0-9]+", str(length_header).strip()):
                raise ValueError("GitHub raw blob has an invalid transport length")
            buffer = GitHubBlobBuffer(
                sha=blob_sha,
                size=blob_size,
                content_length=int(length_header) if length_header is not None else None,
                label=label,
            )
            async for chunk in response.aiter_bytes():
                buffer.write(chunk)
            return buffer.finish()

    async def _discover_local_push(self, since: datetime | None) -> AsyncIterator[ContentItem]:
        packages = [
            package
            for package in device_packages(
                self.config,
                package_kind=GITHUB_REPO_PACKAGE_KIND,
                inbox=self._documents_dir,
            )
            if _package_matches_config(package.fields, self.config)
        ]
        max_files = int_config(self.config, "max_files", default=DEFAULT_MAX_FILES)
        if len(packages) > max_files:
            raise RuntimeError(
                f"GitHub Repository Internal network / VPN sync matched {len(packages)} files, exceeding max_files={max_files}"
            )
        for package in packages:
            if not package.changed_after(since):
                continue
            doc_id = str(package.fields["doc_id"])
            yield ContentItem(
                item_id=doc_id,
                title=package.text("title", doc_id),
                source_url=package.text("source_url"),
                last_modified=package.last_modified,
                content_type=package.text("content_type", "text/markdown"),
                space_or_project=package.text("space_or_project"),
                version=package.text("version"),
                author=package.fields.get("submitted_by"),
                labels=["github_repo"],
                extra={**package.locator, "relative_path": package.fields.get("relative_path")},
            )

    def _documents_dir(self) -> Path:
        configured = str(self.config.get("documents_dir") or "").strip()
        if not configured:
            raise ValueError("GitHub Repository Internal network / VPN source is missing documents_dir")
        return Path(configured).expanduser()


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


def _file_url(ref: _RepoRef, repo_ref: str, relative_path: str) -> str:
    return f"{ref.repo_url}/blob/{quote(repo_ref, safe='')}/{quote(relative_path, safe='/')}"


def _title_from_path(relative_path: str) -> str:
    filename = relative_path.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"\.[^.]+$", "", filename) or filename


def _package_matches_config(package: Mapping[str, Any], config: dict) -> bool:
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
        exclude_paths = github_exclude_paths(config)
        normalized_path = normalize_github_relative_path(relative_path)
    except ValueError:
        return False
    if not github_path_in_scope(normalized_path, include_paths, exclude_paths):
        return False
    return github_extension_allowed(normalized_path, github_include_extensions(config))


def _package_markdown(package: dict) -> str:
    return _to_markdown(package.get("content_type") or "text/markdown", str(package.get("markdown") or ""))


def _package_raw_content(
    item: ContentItem,
    body: bytes,
    package: dict,
    *,
    artifacts: tuple[RawSourceArtifact, ...] = (),
) -> RawContent:
    """The raw content of one local-push package; the package attests an empty file."""

    authoritative_empty = not _package_markdown(package).strip()
    return RawContent(
        item=item,
        body=body,
        content_type="application/json",
        authoritative_empty=authoritative_empty,
        empty_evidence="github_repo_package_attested_empty_file" if authoritative_empty else None,
        artifacts=artifacts,
    )


def _semantics_from_package(package: dict) -> dict:
    semantics = {
        SOURCE_UPDATED_AT_KEY: package.get(SOURCE_UPDATED_AT_KEY),
        "source_type": GITHUB_REPO_SOURCE_TYPE,
        "connection_mode": CONNECTION_MODE_LOCAL_PUSH,
        "repo_url": package.get("repo_url"),
        "repo_identifier": _validated_repo_identifier(str(package.get("repo_url") or "")),
        "repo_host": package.get("repo_host"),
        "repo_owner": package.get("repo_owner"),
        "repo_name": package.get("repo_name"),
        "repo_ref": package.get("repo_ref"),
        "relative_path": package.get("relative_path"),
        "blob_sha": package.get("blob_sha"),
        "content_type": package.get("content_type"),
        "canonical_url": package.get("source_url"),
    }
    if package.get("symlink_chain"):
        semantics.update({
            "logical_file_mode": GITHUB_SYMLINK_MODE,
            "symlink_chain": package.get("symlink_chain"),
            "resolved_relative_path": package.get("resolved_relative_path"),
        })
    return semantics


def _validated_repo_identifier(repo_url: str) -> str:
    """Validate a GitHub URL and map it to the shared repository identity."""

    repo = _parse_repo_url(repo_url)
    identifier = normalize_repo_identifier(repo.repo_url)
    if identifier is None:
        raise ValueError("GitHub repository URL is required")
    return identifier
