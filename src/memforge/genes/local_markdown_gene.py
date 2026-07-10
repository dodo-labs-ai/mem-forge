"""Local Markdown Gene — ingests user-pushed markdown from a local CLI adapter.

The gene treats a per-source inbox directory as the unit of sync. The local
``memforge adapter kb push`` command writes one JSON package per markdown file
into this inbox via the admin API. The gene discovers, fetches, and normalizes
those packages exactly the way the agent-session gene handles client-generated
session summaries.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

from memforge.genes.base import Gene
from memforge.genes.local_adapter_packages import package_manifest, read_package_body
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

__all__ = ["LocalMarkdownGene"]

logger = logging.getLogger(__name__)

LOCAL_MARKDOWN_PACKAGE_KIND = "local_markdown_document"
LOCAL_MARKDOWN_CONTENT_ROLE = "user_markdown_note"


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _to_markdown(content_type: str, body: str) -> str:
    """Convert one pushed file's raw text into markdown for extraction.

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


class LocalMarkdownGene(Gene):
    """Local markdown knowledge-base source pushed from the CLI adapter."""

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="local_markdown",
            display_name="Local Repository",
            description="Files from a local folder or repo (Markdown, text, JSON, HTML) pushed via the CLI adapter",
            default_sync_interval_minutes=0,
            auth_method="local_adapter",
            data_shape="document",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[ConfigGroup(key="vault", label="Vault", order=0)],
            fields=[
                ConfigField(
                    key="root",
                    label="Folder Path",
                    field_type=ConfigFieldType.STRING,
                    required=True,
                    placeholder="/Users/me/notes",
                    help_text="Folder path on the machine running the local daemon.",
                    group="vault",
                    order=0,
                ),
                ConfigField(
                    key="display_label",
                    label="Display Label",
                    field_type=ConfigFieldType.STRING,
                    required=False,
                    placeholder="Engineering notes",
                    help_text="Optional human-readable label shown in source metadata.",
                    group="vault",
                    order=1,
                ),
                ConfigField(
                    key="include",
                    label="Include Patterns",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    default="*.md,**/*.md,*.markdown,**/*.markdown,*.txt,**/*.txt,*.json,**/*.json,*.html,**/*.html,*.htm,**/*.htm",
                    help_text="Glob patterns relative to the folder path.",
                    group="vault",
                    order=2,
                ),
                ConfigField(
                    key="exclude",
                    label="Exclude Patterns",
                    field_type=ConfigFieldType.TAG_LIST,
                    required=False,
                    default=".obsidian/**,.trash/**,.git/**,**/.git/**",
                    help_text="Glob patterns to skip.",
                    group="vault",
                    order=3,
                ),
            ],
        )

    async def authenticate(self) -> None:
        if self._package_manifest():
            return
        documents_dir = self._documents_dir()
        documents_dir.mkdir(parents=True, exist_ok=True)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        manifest = self._package_manifest()
        if manifest:
            async for item in self._discover_package_manifest(manifest, since):
                yield item
            return
        documents_dir = self._documents_dir()
        for package_path in sorted(documents_dir.rglob("*.json")):
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable local markdown package: %s", package_path)
                continue
            if package.get("package_kind") != LOCAL_MARKDOWN_PACKAGE_KIND:
                continue
            last_modified = _parse_dt(package.get("last_modified", ""))
            if since and last_modified <= since:
                continue
            yield ContentItem(
                item_id=package["doc_id"],
                title=package.get("title") or package["doc_id"],
                source_url=package.get("source_url", ""),
                last_modified=last_modified,
                content_type=package.get("content_type") or "text/markdown",
                space_or_project=package.get("space_or_project") or package.get("vault_id") or "",
                version=package.get("version", ""),
                author=package.get("author"),
                labels=["local_markdown"],
                extra={"package_path": str(package_path)},
            )

    async def fetch(self, item: ContentItem) -> RawContent:
        return RawContent(
            item=item,
            body=read_package_body(self, item, source_label="local markdown"),
            content_type="application/json",
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        package = json.loads(raw.body.decode("utf-8"))
        content_type = package.get("content_type") or "text/markdown"
        markdown = _to_markdown(content_type, package.get("markdown", ""))
        return NormalizedContent(
            item=raw.item,
            markdown_body=markdown,
            source_semantics={
                "source_kind": "local_markdown",
                "vault_id": package.get("vault_id"),
                "relative_path": package.get("relative_path"),
                "content_type": content_type,
                "raw_hash": package.get("raw_hash"),
                "submitted_at": package.get("submitted_at"),
                "submitted_by": package.get("submitted_by"),
            },
        )

    async def health_check(self) -> dict:
        if self._package_manifest():
            return {
                "healthy": True,
                "documents_dir": self.config.get("documents_dir"),
                "package_manifest_entries": len(self._package_manifest()),
                "vault_id": self.config.get("vault_id"),
            }
        documents_dir = self._documents_dir()
        return {
            "healthy": documents_dir.exists() and documents_dir.is_dir(),
            "documents_dir": str(documents_dir),
            "vault_id": self.config.get("vault_id"),
        }

    def _documents_dir(self) -> Path:
        configured = str(self.config.get("documents_dir") or "").strip()
        if not configured:
            raise ValueError(
                "local_markdown source is missing documents_dir. The admin API "
                "fills this in when the source is created or updated."
            )
        return Path(configured).expanduser()

    def _package_manifest(self) -> list[dict]:
        return package_manifest(self.config)

    async def _discover_package_manifest(
        self,
        manifest: list[dict],
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        for entry in sorted(
            manifest,
            key=lambda item: (str(item.get("last_modified") or ""), str(item.get("doc_id") or "")),
        ):
            package_uri = str(entry.get("package_uri") or "").strip()
            if not package_uri:
                continue
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
                space_or_project=str(entry.get("space_or_project") or entry.get("vault_id") or ""),
                version=str(entry.get("version") or ""),
                author=entry.get("submitted_by"),
                labels=["local_markdown"],
                extra={
                    "package_uri": package_uri,
                    "package_path": entry.get("package_path"),
                    "relative_path": entry.get("relative_path"),
                },
            )
