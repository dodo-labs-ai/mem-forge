"""Confluence Gene — syncs wiki pages from Confluence via REST API.

Wraps the Confluence REST API v1 to discover, fetch, and normalize
wiki pages into comprehensive markdown for memory extraction.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import httpx

from meminception.genes.atlassian_auth import (
    atlassian_request_limiter,
    bearer_headers,
    get_with_rate_limit_retry,
    require_https_base_url,
    tls_verify,
)
from meminception.genes.base import Gene
from meminception.genes.confluence_pdf import export_confluence_page_pdf
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
from meminception.pipeline.normalizer_utils import html_to_markdown, strip_boilerplate

logger = logging.getLogger(__name__)
CONFLUENCE_REQUEST_INTERVAL_SECONDS = 2.0

__all__ = ["ConfluenceGene"]


class ConfluenceGene(Gene):
    """Confluence data source gene.

    Discovers and syncs wiki pages via Confluence REST API v1.
    Normalizes XHTML page content into comprehensive markdown.
    """

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="confluence",
            display_name="Confluence",
            description="Wiki pages and documentation",
            default_sync_interval_minutes=1440,  # daily
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
                    key="base_url", label="Confluence Base URL",
                    field_type=ConfigFieldType.URL, required=True,
                    placeholder="https://wiki.example.com",
                    help_text="Root URL of your Confluence instance",
                    group="connection", order=0,
                ),
                ConfigField(
                    key="spaces", label="Spaces to Sync",
                    field_type=ConfigFieldType.TAG_LIST, required=True,
                    placeholder="PAY, ARCH, DevOps",
                    help_text="Comma-separated Confluence space keys",
                    group="scope", order=0,
                ),
                ConfigField(
                    key="page_tree_root", label="Page Tree Root (optional)",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder="Page ID or URL",
                    help_text="Only sync pages under this root page",
                    group="scope", order=1,
                ),
                ConfigField(
                    key="exclude_labels", label="Exclude Labels",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="draft, archived, obsolete",
                    help_text="Pages with these labels will be skipped",
                    group="scope", order=2,
                ),
                ConfigField(
                    key="include_children", label="Include Child Pages",
                    field_type=ConfigFieldType.BOOLEAN, required=False,
                    default="true",
                    group="scope", order=3,
                ),
                ConfigField(
                    key="pat", label="Personal Access Token",
                    field_type=ConfigFieldType.SECRET, required=True,
                    help_text="Stored encrypted in MemInception and sent as a bearer token",
                    group="connection", order=1,
                ),
                ConfigField(
                    key="tls_ca_bundle", label="TLS CA Bundle",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder="/path/to/company-ca.pem",
                    help_text="Optional CA bundle path for internal HTTPS certificates",
                    group="connection", order=2, advanced=True,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Authenticate to Confluence via Personal Access Token."""
        base_url = self._normalize_base_url(self.config.get("base_url", ""))
        if not base_url:
            raise ValueError("Confluence base_url is required")
        require_https_base_url(base_url, "Confluence")

        self._base_url = base_url
        # Confluence REST API prefix — on-prem instances typically use /wiki
        self._api_prefix = "/wiki"
        self._request_limiter = atlassian_request_limiter(
            base_url,
            min_interval_seconds=CONFLUENCE_REQUEST_INTERVAL_SECONDS,
            owner_id=self.source_id,
        )

        client = httpx.AsyncClient(
            base_url=base_url,
            headers=bearer_headers(self.config, "Confluence"),
            timeout=30.0,
            follow_redirects=True,
            verify=tls_verify(self.config),
        )
        try:
            await self._get(client, f"{self._api_prefix}/rest/api/space", params={"limit": 1})
        except Exception:
            await client.aclose()
            raise
        self._client = client
        logger.info("Confluence authenticated via PAT: %s", base_url)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Return the Confluence origin used with webui and REST API paths."""
        value = base_url.strip().rstrip("/")
        if not value:
            return ""

        parts = urlsplit(value)
        path = parts.path.rstrip("/")
        if path == "/wiki" or path.endswith("/wiki"):
            path = path[:-len("/wiki")] or ""
            value = urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
        return value

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Discover wiki pages from configured spaces or page tree root."""
        page_tree_root = self.config.get("page_tree_root", "")
        include_children = self.config.get("include_children", True)

        if page_tree_root:
            # Page tree mode: discover all descendants of a root page
            async for item in self._discover_page_tree(page_tree_root, include_children, since):
                yield item
        else:
            # Space mode: discover all pages in configured spaces
            spaces = self.config.get("spaces", [])
            if isinstance(spaces, str):
                spaces = [s.strip() for s in spaces.split(",") if s.strip()]
            for space_key in spaces:
                async for item in self._discover_space(space_key, since):
                    yield item

    async def _discover_page_tree(
        self, root_id: str, include_children: bool, since: datetime | None
    ) -> AsyncIterator[ContentItem]:
        """Discover pages by traversing the child tree of a root page."""
        exclude_labels = set(self.config.get("exclude_labels", []))

        # Get the root page first
        root_item = await self._get_page_as_content_item(root_id, since, exclude_labels)
        if root_item:
            yield root_item

        if not include_children:
            return

        # BFS through children
        queue = [root_id]
        while queue:
            parent_id = queue.pop(0)
            start = 0
            limit = 50
            while True:
                try:
                    resp = await self._get(
                        f"{self._api_prefix}/rest/api/content/{parent_id}/child/page",
                        params={"start": start, "limit": limit, "expand": "version,metadata.labels"},
                    )
                    data = resp.json()
                except Exception as e:
                    logger.error("Failed to list children of page %s: %s", parent_id, e)
                    raise RuntimeError(f"Failed to list Confluence children for page {parent_id}: {e}") from e

                results = data.get("results", [])
                if not results:
                    break

                for page in results:
                    item = self._parse_page(page, since, exclude_labels)
                    if item:
                        yield item
                    if not self._has_excluded_label(page, exclude_labels):
                        page_id = page.get("id", "")
                        if page_id:
                            queue.append(page_id)

                if len(results) < limit:
                    break
                start += limit

    async def _get_page_as_content_item(
        self, page_id: str, since: datetime | None, exclude_labels: set
    ) -> ContentItem | None:
        """Fetch a single page's metadata and return as ContentItem."""
        try:
            resp = await self._get(
                f"{self._api_prefix}/rest/api/content/{page_id}",
                params={"expand": "version,metadata.labels,space"},
            )
            page = resp.json()
            return self._parse_page(page, since, exclude_labels)
        except Exception as e:
            logger.error("Failed to fetch page %s: %s", page_id, e)
            raise RuntimeError(f"Failed to fetch Confluence page {page_id}: {e}") from e

    def _parse_page(
        self, page: dict, since: datetime | None, exclude_labels: set
    ) -> ContentItem | None:
        """Parse a Confluence page JSON into a ContentItem."""
        labels = [
            label["name"]
            for label in page.get("metadata", {}).get("labels", {}).get("results", [])
        ]
        if exclude_labels and set(labels) & exclude_labels:
            return None

        version_info = page.get("version", {})
        modified_str = version_info.get("when", "")
        try:
            last_modified = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            last_modified = datetime.now(timezone.utc)

        if since and last_modified <= since:
            return None

        page_id = page.get("id", "")
        space_key = page.get("space", {}).get("key", self.config.get("spaces", [""])[0] if self.config.get("spaces") else "")

        return ContentItem(
            item_id=f"confluence-{page_id}",
            title=page.get("title", "Untitled"),
            source_url=f"{self._base_url}{page.get('_links', {}).get('webui', '')}",
            last_modified=last_modified,
            content_type="text/html",
            space_or_project=space_key,
            version=str(version_info.get("number", "1")),
            author=version_info.get("by", {}).get("displayName"),
            labels=labels,
            extra={"page_id": page_id, "space_key": space_key},
        )

    async def _discover_space(self, space_key: str, since: datetime | None) -> AsyncIterator[ContentItem]:
        """Discover all pages in a Confluence space."""
        exclude_labels = set(self.config.get("exclude_labels", []))
        if isinstance(exclude_labels, str):
            exclude_labels = {label.strip() for label in exclude_labels.split(",") if label.strip()}

        logger.info("Discovering pages in space: %s", space_key)
        start = 0
        limit = 50

        while True:
            try:
                resp = await self._get(
                    f"{self._api_prefix}/rest/api/content",
                    params={
                        "spaceKey": space_key,
                        "type": "page",
                        "start": start,
                        "limit": limit,
                        "expand": "version,metadata.labels,space",
                    },
                )
                data = resp.json()
            except Exception as e:
                logger.error("Failed to list pages in space %s: %s", space_key, e)
                raise RuntimeError(f"Failed to list Confluence pages in space {space_key}: {e}") from e

            results = data.get("results", [])
            if not results:
                break

            for page in results:
                item = self._parse_page(page, since, exclude_labels)
                if item:
                    yield item

            if len(results) < limit:
                break
            start += limit

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full page content (XHTML body)."""
        page_id = item.extra.get("page_id", item.item_id.replace("confluence-", ""))
        resp = await self._get(
            f"{self._api_prefix}/rest/api/content/{page_id}",
            params={"expand": "body.storage,version"},
        )
        data = resp.json()

        body_html = data.get("body", {}).get("storage", {}).get("value", "")

        return RawContent(
            item=item,
            body=body_html.encode("utf-8"),
            content_type="text/html",
        )

    async def fetch_pdf(self, item: ContentItem) -> bytes | None:
        """Render Confluence export HTML to a local PDF."""
        page_id = item.extra.get("page_id", item.item_id.replace("confluence-", ""))

        try:
            return await export_confluence_page_pdf(
                client=self._client,
                base_url=self._base_url,
                api_prefix=self._api_prefix,
                page_id=page_id,
                title=item.title,
                limiter=getattr(self, "_request_limiter", None),
            )
        except Exception as e:
            logger.warning("PDF export failed for %s: %s", item.title, e)
            return None

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert Confluence XHTML to comprehensive markdown."""
        html = raw.body.decode("utf-8", errors="replace")

        # Convert HTML to markdown
        markdown = html_to_markdown(html)
        markdown = strip_boilerplate(markdown)

        # Add structured header with metadata
        header_lines = [
            f"# {raw.item.title}",
            f"**Space**: {raw.item.space_or_project}",
        ]
        if raw.item.author:
            header_lines.append(f"**Author**: {raw.item.author}")
        if raw.item.labels:
            header_lines.append(f"**Labels**: {', '.join(raw.item.labels)}")
        header_lines.append(f"**Last modified**: {raw.item.last_modified.isoformat()}")
        header_lines.append("")

        full_markdown = "\n".join(header_lines) + "\n" + markdown

        return NormalizedContent(
            item=raw.item,
            markdown_body=full_markdown,
            source_semantics={
                "space_key": raw.item.space_or_project,
                "labels": raw.item.labels,
                "author": raw.item.author,
                "version": raw.item.version,
            },
        )

    @staticmethod
    def _has_excluded_label(page: dict, exclude_labels: set) -> bool:
        if not exclude_labels:
            return False
        labels = {
            label["name"]
            for label in page.get("metadata", {}).get("labels", {}).get("results", [])
        }
        return bool(labels & exclude_labels)

    async def health_check(self) -> dict:
        """Check Confluence connectivity."""
        try:
            await self._get(f"{self._api_prefix}/rest/api/space", params={"limit": 1})
            return {"healthy": True}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _get(
        self,
        url_or_client: str | httpx.AsyncClient,
        url: str | None = None,
        *,
        params: dict | None = None,
    ) -> httpx.Response:
        """GET a Confluence REST URL with bounded HTTP 429 retry handling."""
        if url is None:
            client = self._client
            request_url = str(url_or_client)
        else:
            client = url_or_client
            request_url = url

        return await get_with_rate_limit_retry(
            client,
            request_url,
            product_name="Confluence",
            params=params,
            limiter=getattr(self, "_request_limiter", None),
        )
