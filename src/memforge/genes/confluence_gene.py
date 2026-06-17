"""Confluence Gene — syncs wiki pages from Confluence via REST API.

Wraps the Confluence REST API v1 to discover, fetch, and normalize
wiki pages into comprehensive markdown for memory extraction.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx

from memforge.genes.atlassian_auth import (
    atlassian_request_limiter,
    bearer_headers,
    get_with_rate_limit_retry,
    require_https_base_url,
    tls_verify,
)
from memforge.genes.base import Gene
from memforge.genes.confluence_pdf import export_confluence_page_pdf
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
from memforge.pipeline.normalizer_utils import html_to_markdown, strip_boilerplate

logger = logging.getLogger(__name__)
CONFLUENCE_REQUEST_INTERVAL_SECONDS = 2.0
PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY = "_memforge_preview_limit"

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
                    key="base_url", label="Wiki URL",
                    field_type=ConfigFieldType.URL, required=True,
                    placeholder="https://wiki.example.com or a Confluence page URL",
                    help_text="Paste a Confluence root, space, or page URL",
                    group="connection", order=0,
                ),
                ConfigField(
                    key="sync_mode", label="Sync Scope",
                    field_type=ConfigFieldType.SELECT, required=False,
                    options=["page_tree", "space"],
                    help_text="Sync one page tree or a whole space",
                    group="scope", order=0,
                ),
                ConfigField(
                    key="spaces", label="Spaces to Sync",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="PAY, ARCH, DevOps",
                    help_text="Required when syncing whole spaces",
                    group="scope", order=1,
                ),
                ConfigField(
                    key="page_tree_root", label="Page Tree Root (optional)",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder="Page ID or URL",
                    help_text="Only sync pages under this root page",
                    group="scope", order=2,
                ),
                ConfigField(
                    key="exclude_labels", label="Exclude Labels",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="draft, archived, obsolete",
                    help_text="Pages with these labels will be skipped",
                    group="scope", order=3,
                ),
                ConfigField(
                    key="include_children", label="Include Child Pages",
                    field_type=ConfigFieldType.BOOLEAN, required=False,
                    default="true",
                    group="scope", order=4,
                ),
                ConfigField(
                    key="pat", label="Personal Access Token",
                    field_type=ConfigFieldType.SECRET, required=True,
                    help_text="Stored encrypted in MemForge and sent as a bearer token",
                    group="connection", order=1,
                ),
                ConfigField(
                    key="api_prefix", label="REST API Path",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder="/wiki",
                    help_text="Advanced override for Confluence deployments that serve REST below a path",
                    group="connection", order=2, advanced=True,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Authenticate to Confluence via Personal Access Token."""
        self.normalize_config(self.config)
        base_url = str(self.config.get("base_url") or "").strip()
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
            self._api_prefix = await self._select_api_prefix(client)
        except Exception:
            await client.aclose()
            raise
        self._client = client
        logger.info("Confluence authenticated via PAT: %s", base_url)

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Return the Confluence origin used with webui and REST API paths."""
        return ConfluenceGene._parse_wiki_url(base_url).get("base_url", "").rstrip("/")

    @classmethod
    def normalize_config(cls, config: dict) -> None:
        """Normalize user-provided Confluence URLs in-place."""
        raw_base_url = str(config.get("base_url") or "").strip()
        parsed = cls._parse_wiki_url(raw_base_url)
        if parsed.get("base_url"):
            config["base_url"] = parsed["base_url"]
        if parsed.get("api_prefix") and not str(config.get("api_prefix") or "").strip():
            config["api_prefix"] = parsed["api_prefix"]
        elif "api_prefix" in config:
            config["api_prefix"] = cls._normalize_api_prefix(config.get("api_prefix"))

        if parsed.get("space_key") and not cls._space_keys(config.get("spaces")):
            config["spaces"] = [parsed["space_key"]]
        if parsed.get("page_id") and not str(config.get("page_tree_root") or "").strip():
            config["page_tree_root"] = parsed["page_id"]

        page_tree_root = str(config.get("page_tree_root") or "").strip()
        if page_tree_root:
            config["page_tree_root"] = cls._page_id_from_url(page_tree_root) or page_tree_root
        if str(config.get("page_tree_root") or "").strip() and not str(config.get("sync_mode") or "").strip():
            config["sync_mode"] = "page_tree"

    @classmethod
    def _parse_wiki_url(cls, value: str) -> dict[str, str]:
        text = value.strip().rstrip("/")
        if not text:
            return {}

        parts = urlsplit(text)
        result: dict[str, str] = {}
        if parts.scheme and parts.netloc:
            result["base_url"] = urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")

        path_parts = [part for part in parts.path.split("/") if part]
        prefix_parts: list[str] = []
        if "spaces" in path_parts:
            space_index = path_parts.index("spaces")
            prefix_parts = path_parts[:space_index]
            if space_index + 1 < len(path_parts):
                result["space_key"] = path_parts[space_index + 1]
            page_id = cls._page_id_from_path_parts(path_parts[space_index + 2 :])
            if page_id:
                result["page_id"] = page_id
        elif len(path_parts) == 1:
            prefix_parts = path_parts

        query_page_ids = parse_qs(parts.query).get("pageId")
        if query_page_ids and query_page_ids[0].isdigit():
            result["page_id"] = query_page_ids[0]

        if prefix_parts:
            result["api_prefix"] = cls._normalize_api_prefix("/" + "/".join(prefix_parts))
        return result

    @staticmethod
    def _page_id_from_url(value: str) -> str | None:
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return text

        parts = urlsplit(text)
        query_page_ids = parse_qs(parts.query).get("pageId")
        if query_page_ids and query_page_ids[0].isdigit():
            return query_page_ids[0]

        path_parts = [part for part in parts.path.split("/") if part]
        return ConfluenceGene._page_id_from_path_parts(path_parts)

    @staticmethod
    def _page_id_from_path_parts(path_parts: list[str]) -> str | None:
        for index, part in enumerate(path_parts):
            if part == "pages" and index + 1 < len(path_parts) and path_parts[index + 1].isdigit():
                return path_parts[index + 1]
        return None

    @staticmethod
    def _normalize_api_prefix(value: object) -> str:
        text = str(value or "").strip().rstrip("/")
        if not text or text == "/":
            return ""
        return "/" + text.strip("/")

    @staticmethod
    def _space_keys(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    @classmethod
    def _effective_sync_mode(cls, config: dict) -> str:
        mode = str(config.get("sync_mode") or "").strip().lower()
        if mode in {"page_tree", "space"}:
            return mode
        return "page_tree" if str(config.get("page_tree_root") or "").strip() else "space"

    @classmethod
    def _api_prefix_candidates(cls, config: dict) -> list[str]:
        configured = cls._normalize_api_prefix(config.get("api_prefix"))
        if configured:
            return [configured]
        return ["/wiki", ""]

    async def _select_api_prefix(self, client: httpx.AsyncClient) -> str:
        """Return the REST prefix that answers with Confluence JSON."""
        last_error: Exception | None = None
        for prefix in self._api_prefix_candidates(self.config):
            try:
                resp = await self._get(client, f"{prefix}/rest/api/space", params={"limit": 1})
                self._json_response(resp, "checking Confluence REST API")
                self.config["api_prefix"] = prefix
                return prefix
            except Exception as exc:
                last_error = exc
                if not self._can_try_next_api_prefix(exc):
                    raise
        if last_error is not None:
            raise last_error
        return "/wiki"

    @staticmethod
    def _can_try_next_api_prefix(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == 404
        return isinstance(exc, RuntimeError) and "non-JSON response" in str(exc)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Discover wiki pages from configured spaces or page tree root."""
        self.normalize_config(self.config)
        sync_mode = self._effective_sync_mode(self.config)
        page_tree_root = str(self.config.get("page_tree_root") or "").strip()
        include_children = self.config.get("include_children", True)

        if sync_mode == "page_tree":
            if not page_tree_root:
                raise ValueError("Confluence Page Tree Root is required when syncing a page tree")
            async for item in self._discover_page_tree(page_tree_root, include_children, since):
                yield item
            return

        spaces = self._space_keys(self.config.get("spaces"))
        if not spaces:
            raise ValueError("Confluence Spaces to Sync is required when syncing whole spaces")
        for space_key in spaces:
            async for item in self._discover_space(space_key, since):
                yield item

    async def _discover_page_tree(
        self, root_id: str, include_children: bool, since: datetime | None
    ) -> AsyncIterator[ContentItem]:
        """Discover pages by traversing the child tree of a root page."""
        exclude_labels = set(self.config.get("exclude_labels", []))
        preview_limit = self._preview_discovery_limit()
        emitted = 0

        # Get the root page first
        root_item = await self._get_page_as_content_item(root_id, since, exclude_labels)
        if root_item:
            yield root_item
            emitted += 1
            if preview_limit is not None and emitted >= preview_limit:
                return

        if not include_children:
            return

        # BFS through children
        queue = [root_id]
        while queue:
            parent_id = queue.pop(0)
            start = 0
            limit = self._page_request_limit(preview_limit, emitted)
            while True:
                try:
                    resp = await self._get(
                        f"{self._api_prefix}/rest/api/content/{parent_id}/child/page",
                        params={"start": start, "limit": limit, "expand": "version,metadata.labels"},
                    )
                    data = self._json_response(resp, f"listing children of page {parent_id}")
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
                        emitted += 1
                        if preview_limit is not None and emitted >= preview_limit:
                            return
                    if not self._has_excluded_label(page, exclude_labels):
                        page_id = page.get("id", "")
                        if page_id:
                            queue.append(page_id)

                if len(results) < limit:
                    break
                start += limit
                limit = self._page_request_limit(preview_limit, emitted)

    async def _get_page_as_content_item(
        self, page_id: str, since: datetime | None, exclude_labels: set
    ) -> ContentItem | None:
        """Fetch a single page's metadata and return as ContentItem."""
        try:
            resp = await self._get(
                f"{self._api_prefix}/rest/api/content/{page_id}",
                params={"expand": "version,metadata.labels,space"},
            )
            page = self._json_response(resp, f"fetching page {page_id}")
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
        preview_limit = self._preview_discovery_limit()
        emitted = 0

        logger.info("Discovering pages in space: %s", space_key)
        start = 0
        limit = self._page_request_limit(preview_limit, emitted)

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
                data = self._json_response(resp, f"listing pages in space {space_key}")
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
                    emitted += 1
                    if preview_limit is not None and emitted >= preview_limit:
                        return

            if len(results) < limit:
                break
            start += limit
            limit = self._page_request_limit(preview_limit, emitted)

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full page content (XHTML body)."""
        page_id = item.extra.get("page_id", item.item_id.replace("confluence-", ""))
        resp = await self._get(
            f"{self._api_prefix}/rest/api/content/{page_id}",
            params={"expand": "body.storage,version"},
        )
        data = self._json_response(resp, f"fetching page content {page_id}")

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

    def _preview_discovery_limit(self) -> int | None:
        value = self.config.get(PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY)
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    @staticmethod
    def _page_request_limit(preview_limit: int | None, emitted: int) -> int:
        if preview_limit is None:
            return 50
        return max(1, min(50, preview_limit - emitted))

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

    @staticmethod
    def _json_response(resp: httpx.Response, action: str) -> dict:
        try:
            data = resp.json()
        except ValueError as exc:
            content_type = resp.headers.get("content-type", "unknown")
            raise RuntimeError(
                "Confluence returned a non-JSON response while "
                f"{action} (status={resp.status_code}, content-type={content_type}). "
                "Check that the Confluence URL resolves to the instance root and that the PAT can access the REST API."
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Confluence returned unexpected JSON while {action}: expected an object")
        return data
