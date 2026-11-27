"""Local Markdown Gene: the files of a folder on the Source's execution device.

A local Markdown Source runs only on its execution device (ADR 0058, design
§10.3), where the daemon runs this Gene on the folder at ``root``:

- **One walk.** :meth:`LocalMarkdownGene.discover` walks the folder once with
  the configured ``include`` and ``exclude`` and records the walk, so
  :meth:`~LocalMarkdownGene.list_scope` returns the same moment. The listing
  is an existence listing by relative path: a renamed or moved file is a new
  Document and its old id is unlisted.
- **The walk fails rather than shrinks.** A missing ``root``, a ``root`` that
  is not a directory, and any directory under it that cannot be enumerated
  fail the run, because the walk is the Source's only evidence of absence. A
  directory ``exclude`` leaves out is not read.
- **Every file is discovered.** Discovery ignores ``since``: a checkout, a
  copy that keeps modification times or a renamed folder puts files under new
  paths without a time after the cursor.
- **Listing never reads content.** :meth:`~LocalMarkdownGene.fetch` checks
  size, encoding and readability, so a file that is too large, not UTF-8,
  unreadable or deleted since the walk fails as its own Document and stays
  listed.

The stored input of a file is its bytes; the item's ``content_type`` names
the file's format, from its extension, and ``normalize`` converts it to
Markdown.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from memforge.genes.document_gene import DocumentGene
from memforge.genes.toolkit.config import list_config
from memforge.local_agent.document_identity import build_local_markdown_doc_id
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
    ScopeListing,
    ScopeListingKind,
    content_hash,
    slugify,
)
from memforge.source_time import SOURCE_UPDATED_AT_KEY, source_time_iso

__all__ = [
    "DEFAULT_EXCLUDE",
    "DEFAULT_INCLUDE",
    "LOCAL_FILE_CONTENT_TYPE",
    "LOCAL_FILE_MAX_BYTES",
    "LocalMarkdownGene",
    "fetched_local_file_item",
    "local_file_content_item",
]

DEFAULT_INCLUDE = (
    "*.md",
    "**/*.md",
    "*.markdown",
    "**/*.markdown",
    "*.txt",
    "**/*.txt",
    "*.json",
    "**/*.json",
    "*.html",
    "**/*.html",
    "*.htm",
    "**/*.htm",
)
DEFAULT_EXCLUDE = (".obsidian/**", ".trash/**", ".git/**", "**/.git/**")

LOCAL_FILE_MAX_BYTES = 1_000_000
"""The largest file a Source reads; a larger one fails as its own Document."""

LOCAL_FILE_CONTENT_TYPE = "text/plain; charset=utf-8"
"""The content type of a file's stored input: its UTF-8 text, whatever its format."""

_CONTENT_TYPE_BY_SUFFIX = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
    ".html": "text/html",
    ".htm": "text/html",
}


def _content_type_for_path(relative_path: str) -> str:
    """A file's format from its extension; an unknown one reads as plain text."""

    return _CONTENT_TYPE_BY_SUFFIX.get(PurePosixPath(relative_path).suffix.lower(), "text/plain")


def _to_markdown(content_type: str, body: str) -> str:
    """Convert one file's text into Markdown for extraction.

    Markdown and plain text pass through unchanged; HTML is converted with the
    shared ``html_to_markdown`` helper; JSON is wrapped in a fenced code block so
    its structure survives. Unknown types are treated as plain text.
    """
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype == "text/html":
        from memforge.pipeline.normalizer_utils import html_to_markdown

        return html_to_markdown(body)
    if ctype == "application/json":
        return f"```json\n{body.strip()}\n```\n"
    return body


def _markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


def local_file_content_item(
    *,
    source_id: str,
    vault_id: str,
    relative_path: str,
    last_modified: datetime,
) -> ContentItem:
    """The item of one listed file, as discovery yields it, before ``fetch`` reads the file.

    ``fetch`` adds the title, which a Markdown file's first heading gives,
    and the ``version``, the hash of the file's text.
    """

    return ContentItem(
        item_id=build_local_markdown_doc_id(source_id=source_id, vault_id=vault_id, relative_path=relative_path),
        title=relative_path,
        source_url=f"local-adapter://{slugify(source_id)}/{slugify(vault_id)}/{relative_path}",
        last_modified=last_modified,
        content_type=_content_type_for_path(relative_path),
        space_or_project=vault_id,
        labels=["local_markdown"],
        extra={"vault_id": vault_id, "relative_path": relative_path},
    )


def fetched_local_file_item(item: ContentItem, text: str) -> ContentItem:
    """``item`` as ``fetch`` leaves it once the file read as ``text``: its title and ``version``."""

    relative_path = str(item.extra["relative_path"])
    item.title = (_markdown_title(text) if item.content_type == "text/markdown" else None) or relative_path
    item.version = content_hash(text)
    return item


def _matches(relative_path: str, patterns: Sequence[str]) -> bool:
    path = PurePosixPath(relative_path)
    return any(path.match(pattern) for pattern in patterns)


def _excludes_directory(relative_dir: str, exclude: Sequence[str]) -> bool:
    """Whether an exclude pattern leaves out everything under a directory: ``.git/**`` for ``.git`` or ``docs/.git``."""

    return any(pattern.endswith("/**") and _matches(relative_dir, (pattern[: -len("/**")],)) for pattern in exclude)


@dataclass(frozen=True, slots=True)
class _ListedFile:
    relative_path: str
    modified_at: datetime


def _walk_folder(root: Path, include: Sequence[str], exclude: Sequence[str]) -> list[_ListedFile]:
    """Every regular file under ``root`` that ``include`` names and ``exclude`` leaves in, sorted by relative path.

    Symbolic links are neither listed nor followed. A directory that cannot be
    enumerated raises, as does a missing or non-directory ``root``; a file
    removed while the walk reads its time is not listed, since it is gone.
    """

    if not root.is_dir():
        raise NotADirectoryError(f"The folder {root} does not exist or is not a directory")
    listed: list[_ListedFile] = []
    pending = [""]
    while pending:
        relative_dir = pending.pop()
        with os.scandir(root / relative_dir if relative_dir else root) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            relative_path = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if not _excludes_directory(relative_path, exclude):
                    pending.append(relative_path)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            if _matches(relative_path, exclude) or not _matches(relative_path, include):
                continue
            try:
                modified = entry.stat(follow_symlinks=False).st_mtime
            except FileNotFoundError:
                continue
            listed.append(_ListedFile(relative_path, datetime.fromtimestamp(modified, timezone.utc)))
    return sorted(listed, key=lambda listed_file: listed_file.relative_path)


def _git_output(root: Path, *args: str) -> str | None:
    """Run a read-only ``git`` command in ``root``; None when Git is absent or refuses."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            # Reading status must not take the index lock from the user's own Git work.
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError:
        return None
    return result.stdout if result.returncode == 0 else None


def _uncommitted_paths(root: Path) -> frozenset[str] | None:
    """The root-relative paths Git reports as modified, staged, renamed or untracked; None outside Git."""

    prefix = _git_output(root, "rev-parse", "--show-prefix")
    status = (
        _git_output(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".")
        if prefix is not None
        else None
    )
    if prefix is None or status is None:
        return None
    prefix = prefix.strip()
    changed: set[str] = set()
    records = iter(status.split("\0"))
    for record in records:
        if not record:
            continue
        state, path = record[:2], record[3:]
        changed.add(path)
        if "R" in state or "C" in state:
            # A rename or copy record is followed by its original path.
            changed.add(next(records, ""))
    return frozenset(path.removeprefix(prefix) for path in changed if path.startswith(prefix))


def _file_times(root: Path, listed: Sequence[_ListedFile]) -> dict[str, datetime]:
    """Each listed file's own time for its current content.

    A committed file without local changes takes its latest commit time,
    because a checkout or clone sets the modification time to when it ran.
    Any other file takes its modification time, the only time the file
    system records for its content.
    """

    uncommitted = _uncommitted_paths(root) if listed else None
    times: dict[str, datetime] = {}
    for listed_file in listed:
        times[listed_file.relative_path] = listed_file.modified_at
        if uncommitted is None or listed_file.relative_path in uncommitted:
            continue
        # Empty for a file Git does not track, such as an ignored one.
        committed_at = (_git_output(root, "log", "-1", "--format=%cI", "--", listed_file.relative_path) or "").strip()
        if committed_at:
            times[listed_file.relative_path] = datetime.fromisoformat(committed_at).astimezone(timezone.utc)
    return times


class LocalMarkdownGene(DocumentGene):
    """The Markdown, text, JSON and HTML files of a folder on the Source's execution device."""

    listing_kinds = frozenset({ScopeListingKind.EXISTENCE})

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="local_markdown",
            display_name="Local Repository",
            description="Files from a folder on one of your computers, read by its MemForge daemon",
            default_sync_interval_minutes=0,
            auth_method="local_adapter",
            data_shape="document",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Folder", order=0, purpose="connection"),
                ConfigGroup(key="scope", label="Files to sync", order=1, purpose="content"),
            ],
            fields=[
                ConfigField(
                    key="root",
                    label="Folder Path",
                    field_type=ConfigFieldType.STRING,
                    required=True,
                    placeholder="/Users/me/notes",
                    help_text="Folder path on the computer that runs this Source.",
                    group="connection",
                    order=0,
                    widget="local_folder",
                ),
                ConfigField(
                    key="display_label",
                    label="Display Label",
                    field_type=ConfigFieldType.STRING,
                    required=False,
                    placeholder="Engineering notes",
                    help_text="Optional human-readable label shown in source metadata.",
                    group="scope",
                    order=0,
                ),
                ConfigField(
                    key="include",
                    label="Include Patterns",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    default=",".join(DEFAULT_INCLUDE),
                    help_text="Glob patterns relative to the folder path.",
                    group="scope",
                    order=1,
                    advanced=True,
                ),
                ConfigField(
                    key="exclude",
                    label="Exclude Patterns",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    default=",".join(DEFAULT_EXCLUDE),
                    help_text="Glob patterns to skip. A folder pattern ending in /** is not read at all.",
                    group="scope",
                    order=2,
                    advanced=True,
                ),
            ],
        )

    @classmethod
    def lists_scope(cls, config: Mapping[str, Any]) -> bool:
        """Every configuration lists: the folder walk names every file in scope."""
        return True

    async def authenticate(self) -> None:
        """A folder needs no credential; the walk checks that ``root`` is there."""

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Yield every file the walk lists, whatever ``since`` is, and record the walk as the listing."""

        root = self._root()
        vault_id = self._vault_id()
        listed = await asyncio.to_thread(_walk_folder, root, self._include(), self._exclude())
        times = await asyncio.to_thread(_file_times, root, listed)
        items = [
            local_file_content_item(
                source_id=self.source_id,
                vault_id=vault_id,
                relative_path=listed_file.relative_path,
                last_modified=times[listed_file.relative_path],
            )
            for listed_file in listed
        ]
        self.record_scope_listing(ScopeListingKind.EXISTENCE, {item.item_id for item in items})
        for item in items:
            yield item

    async def list_scope(self) -> ScopeListing:
        """The listing of this run's walk; without a discovery, a walk of its own."""

        if self._discovered_scope is None:
            _ = [item async for item in self.discover(None)]
        assert self._discovered_scope is not None
        return self._discovered_scope

    async def fetch(self, item: ContentItem) -> RawContent:
        """Read one listed file; one too large, not UTF-8, unreadable or gone fails this Document alone."""

        body = await asyncio.to_thread(self._read_file, str(item.extra["relative_path"]))
        fetched_local_file_item(item, body.decode("utf-8"))
        return self.raw_from_stored_input(item, body, LOCAL_FILE_CONTENT_TYPE)

    def raw_from_stored_input(self, item: ContentItem, body: bytes, content_type: str) -> RawContent:
        """A file's stored text; a file that reads as empty Markdown is empty at its source."""

        empty = not _to_markdown(item.content_type, body.decode("utf-8")).strip()
        return RawContent(
            item=item,
            body=body,
            content_type=content_type,
            authoritative_empty=empty,
            empty_evidence="local_markdown_file_read_empty" if empty else None,
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        content_type = raw.item.content_type
        return NormalizedContent(
            item=raw.item,
            markdown_body=_to_markdown(content_type, raw.body.decode("utf-8")),
            source_semantics={
                "source_kind": "local_markdown",
                "vault_id": raw.item.space_or_project,
                "relative_path": raw.item.extra.get("relative_path"),
                "content_type": content_type,
                SOURCE_UPDATED_AT_KEY: source_time_iso(raw.item.last_modified),
            },
        )

    async def health_check(self) -> dict:
        root = self._root()
        return {"healthy": root.is_dir(), "root": str(root), "vault_id": self._vault_id()}

    def _read_file(self, relative_path: str) -> bytes:
        path = self._root() / relative_path
        size = path.stat().st_size
        if size > LOCAL_FILE_MAX_BYTES:
            raise ValueError(f"{relative_path} is {size} bytes, more than the {LOCAL_FILE_MAX_BYTES} a file may have")
        body = path.read_bytes()
        if len(body) > LOCAL_FILE_MAX_BYTES:
            raise ValueError(f"{relative_path} grew past {LOCAL_FILE_MAX_BYTES} bytes while it was read")
        return body

    def _root(self) -> Path:
        configured = str(self.config.get("root") or "").strip()
        if not configured:
            raise ValueError("A local Markdown Source names no folder (root)")
        return Path(configured).expanduser().resolve()

    def _vault_id(self) -> str:
        vault_id = str(self.config.get("vault_id") or "").strip()
        if not vault_id:
            raise ValueError("A local Markdown Source has no vault_id")
        return vault_id

    def _include(self) -> list[str]:
        return list_config(self.config.get("include")) or list(DEFAULT_INCLUDE)

    def _exclude(self) -> list[str]:
        return list_config(self.config.get("exclude")) or list(DEFAULT_EXCLUDE)
