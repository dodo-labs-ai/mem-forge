"""Confluence Gene — syncs wiki pages from Confluence via REST API.

Wraps the Confluence REST API v1 to discover, fetch, and normalize
wiki pages into comprehensive markdown for memory extraction.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import datetime
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
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_DESCRIPTORS_PER_UNIT,
    MAX_SOURCE_ARTIFACT_BYTES,
    MAX_SOURCE_ARTIFACT_BYTES_PER_UNIT,
    MAX_SOURCE_ARTIFACTS_PER_UNIT,
    RawSourceArtifact,
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    SourceArtifactContractError,
    normalize_source_artifact_media_type,
)

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
        self._discovered_page_ids: set[str] = set()
        sync_mode = self._effective_sync_mode(self.config)
        page_tree_root = str(self.config.get("page_tree_root") or "").strip()
        include_children = self.config.get("include_children", True)

        if sync_mode == "page_tree":
            if not page_tree_root:
                raise ValueError("Confluence Page Tree Root is required when syncing a page tree")
            async for item in self._discover_page_tree(page_tree_root, include_children, since):
                yield item
            if self._preview_discovery_limit() is None:
                self.attest_discovery_complete("confluence_page_tree_exhausted")
            return

        spaces = self._space_keys(self.config.get("spaces"))
        if not spaces:
            raise ValueError("Confluence Spaces to Sync is required when syncing whole spaces")
        for space_key in spaces:
            async for item in self._discover_space(space_key, since):
                yield item
        if self._preview_discovery_limit() is None:
            self.attest_discovery_complete("confluence_spaces_exhausted")

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
        traversed_parent_ids: set[str] = set()
        while queue:
            parent_id = queue.pop(0)
            if parent_id in traversed_parent_ids:
                raise RuntimeError(f"Confluence page tree contains a traversal cycle at page {parent_id}")
            traversed_parent_ids.add(parent_id)
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

                results = self._validated_page_results(
                    data,
                    context=f"children of page {parent_id}",
                    expected_start=start,
                )
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

                if not self._has_next_page(data):
                    break
                start += len(results)
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
            item = self._parse_page(page, since, exclude_labels)
            if str(page.get("id") or "").strip() != page_id:
                raise RuntimeError(f"Confluence page {page_id} response identity mismatch")
            return item
        except Exception as e:
            logger.error("Failed to fetch page %s: %s", page_id, e)
            raise RuntimeError(f"Failed to fetch Confluence page {page_id}: {e}") from e

    def _parse_page(
        self, page: dict, since: datetime | None, exclude_labels: set
    ) -> ContentItem | None:
        """Parse a Confluence page JSON into a ContentItem."""
        page_id = str(page.get("id") or "").strip()
        if not page_id:
            raise RuntimeError("Confluence page record is missing a stable page id")
        discovered_page_ids = getattr(self, "_discovered_page_ids", None)
        if discovered_page_ids is None:
            raise RuntimeError("Confluence discovery identity ledger is not initialized")
        if page_id in discovered_page_ids:
            raise RuntimeError(f"Confluence discovery returned duplicate page id {page_id}")
        discovered_page_ids.add(page_id)

        version_info = page.get("version")
        if not isinstance(version_info, dict):
            raise RuntimeError(f"Confluence page {page_id} is missing version metadata")
        version_number = version_info.get("number")
        if not isinstance(version_number, int) or isinstance(version_number, bool) or version_number <= 0:
            raise RuntimeError(f"Confluence page {page_id} is missing a stable version number")
        modified_str = version_info.get("when")
        if not isinstance(modified_str, str) or not modified_str.strip():
            raise RuntimeError(f"Confluence page {page_id} is missing a version timestamp")
        try:
            last_modified = datetime.fromisoformat(modified_str.replace("Z", "+00:00"))
        except ValueError as exc:
            raise RuntimeError(f"Confluence page {page_id} has an invalid version timestamp") from exc
        if last_modified.tzinfo is None or last_modified.utcoffset() is None:
            raise RuntimeError(f"Confluence page {page_id} version timestamp has no timezone")

        labels = [
            label["name"]
            for label in page.get("metadata", {}).get("labels", {}).get("results", [])
        ]
        if exclude_labels and set(labels) & exclude_labels:
            return None

        if since and last_modified <= since:
            return None

        space_key = page.get("space", {}).get("key", self.config.get("spaces", [""])[0] if self.config.get("spaces") else "")

        return ContentItem(
            item_id=f"confluence-{page_id}",
            title=page.get("title", "Untitled"),
            source_url=f"{self._base_url}{page.get('_links', {}).get('webui', '')}",
            last_modified=last_modified,
            content_type="text/html",
            space_or_project=space_key,
            version=str(version_number),
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

            results = self._validated_page_results(
                data,
                context=f"space {space_key}",
                expected_start=start,
            )
            if not results:
                break

            for page in results:
                item = self._parse_page(page, since, exclude_labels)
                if item:
                    yield item
                    emitted += 1
                    if preview_limit is not None and emitted >= preview_limit:
                        return

            if not self._has_next_page(data):
                break
            start += len(results)
            limit = self._page_request_limit(preview_limit, emitted)

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full page content (XHTML body)."""
        page_id = item.extra.get("page_id", item.item_id.replace("confluence-", ""))
        resp = await self._get(
            f"{self._api_prefix}/rest/api/content/{page_id}",
            params={
                "expand": (
                    "body.storage,version,ancestors,space,"
                    "children.attachment.version,children.attachment.extensions"
                )
            },
        )
        data = self._json_response(resp, f"fetching page content {page_id}")
        if str(data.get("id") or "").strip() != str(page_id):
            raise RuntimeError(f"Confluence page {page_id} content response identity mismatch")
        fetched_version = data.get("version")
        if (
            not isinstance(fetched_version, dict)
            or str(fetched_version.get("number") or "").strip() != str(item.version)
        ):
            raise RuntimeError(f"Confluence page {page_id} changed during discovery")

        body = data.get("body")
        storage = body.get("storage") if isinstance(body, dict) else None
        if not isinstance(storage, dict) or "value" not in storage or not isinstance(storage.get("value"), str):
            raise RuntimeError(f"Confluence page {page_id} response is missing body.storage.value")
        body_html = storage["value"]
        semantic_body = strip_boilerplate(html_to_markdown(body_html))
        authoritative_empty = not semantic_body.strip()
        ancestors = data.get("ancestors") if isinstance(data.get("ancestors"), list) else []
        parent = ancestors[-1] if ancestors and isinstance(ancestors[-1], dict) else {}
        item.extra["parent_page_id"] = str(parent.get("id") or "") or None
        item.extra["space_key"] = str(
            (data.get("space") or {}).get("key") or item.extra.get("space_key") or item.space_or_project
        )
        children = data.get("children") if isinstance(data.get("children"), dict) else {}
        attachment_page = children.get("attachment")
        if not isinstance(attachment_page, dict):
            raise SourceArtifactContractError(
                f"Confluence page {page_id} response is missing attachment membership"
            )
        artifacts = await self._fetch_source_artifacts(str(page_id), first_page=attachment_page)

        return RawContent(
            item=item,
            body=body_html.encode("utf-8"),
            content_type="text/html",
            authoritative_empty=authoritative_empty,
            empty_evidence=(
                "confluence_content_api_successful_semantically_empty_storage_body"
                if authoritative_empty
                else None
            ),
            artifacts=artifacts,
        )

    async def _fetch_source_artifacts(
        self,
        page_id: str,
        *,
        first_page: dict,
    ) -> tuple[RawSourceArtifact, ...]:
        """Return bounded provider attachments with exact bytes."""

        descriptors: list[dict] = []
        descriptor_count = 0
        start = 0
        payload = first_page
        while True:
            results = self._validated_attachment_results(payload, expected_start=start)
            descriptor_count += len(results)
            if descriptor_count > MAX_SOURCE_ARTIFACT_DESCRIPTORS_PER_UNIT:
                raise SourceArtifactContractError(
                    "Confluence page exceeds the Source Artifact descriptor scan limit"
                )
            descriptors.extend(
                descriptor
                for descriptor in results
                if normalize_source_artifact_media_type(
                    (
                        descriptor.get("extensions")
                        if isinstance(descriptor.get("extensions"), dict)
                        else {}
                    ).get("mediaType")
                )
                in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES
            )
            if len(descriptors) > MAX_SOURCE_ARTIFACTS_PER_UNIT:
                raise SourceArtifactContractError(
                    f"Confluence page exceeds {MAX_SOURCE_ARTIFACTS_PER_UNIT} supported Artifact limit"
                )
            if not self._has_next_page(payload):
                break
            start += len(results)
            response = await self._get(
                f"{self._api_prefix}/rest/api/content/{page_id}/child/attachment",
                params={
                    "start": start,
                    "limit": 100,
                    "expand": "version,extensions",
                },
            )
            payload = self._json_response(response, f"listing attachments for page {page_id}")

        artifacts: list[RawSourceArtifact] = []
        declared_bytes = 0
        for descriptor in descriptors:
            extensions = descriptor.get("extensions") if isinstance(descriptor.get("extensions"), dict) else {}
            media_type = normalize_source_artifact_media_type(extensions.get("mediaType"))
            size_value = extensions.get("fileSize")
            if not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
                raise SourceArtifactContractError("Confluence attachment is missing a valid file size")
            if size_value > MAX_SOURCE_ARTIFACT_BYTES:
                raise SourceArtifactContractError(
                    f"Confluence attachment exceeds {MAX_SOURCE_ARTIFACT_BYTES} byte limit"
                )
            declared_bytes += size_value
            if declared_bytes > MAX_SOURCE_ARTIFACT_BYTES_PER_UNIT:
                raise SourceArtifactContractError(
                    "Confluence attachments exceed the Source Unit byte aggregate limit"
                )
            attachment_id = str(descriptor.get("id") or "").strip()
            filename = str(descriptor.get("title") or "").strip()
            version = descriptor.get("version") if isinstance(descriptor.get("version"), dict) else {}
            provider_revision = str(version.get("number") or "").strip()
            links = descriptor.get("_links") if isinstance(descriptor.get("_links"), dict) else {}
            download_path = str(links.get("download") or "").strip()
            if not attachment_id or not filename or not provider_revision or not download_path:
                raise SourceArtifactContractError("Confluence attachment identity is incomplete")
            response = await self._get(download_path)
            response.raise_for_status()
            response_media_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            if response_media_type and response_media_type != media_type:
                raise SourceArtifactContractError("Confluence attachment media type changed during download")
            artifacts.append(
                RawSourceArtifact(
                    provider_key=attachment_id,
                    parent_observation_type="page_body",
                    parent_provider_key=f"{page_id}:body",
                    provider_revision=provider_revision,
                    filename=filename,
                    media_type=media_type,
                    body=bytes(response.content),
                    declared_size_bytes=size_value,
                    locator={"attachment_id": attachment_id},
                )
            )
        return tuple(artifacts)

    @staticmethod
    def _validated_attachment_results(data: object, *, expected_start: int) -> list[dict]:
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SourceArtifactContractError("Confluence attachment response is missing results")
        results = data["results"]
        if any(not isinstance(item, dict) for item in results):
            raise SourceArtifactContractError("Confluence attachment response contains an invalid record")
        if data.get("start") != expected_start or data.get("size") != len(results):
            raise SourceArtifactContractError("Confluence attachment pagination is inconsistent")
        if not isinstance(data.get("_links"), dict):
            raise SourceArtifactContractError("Confluence attachment response is missing pagination links")
        if not results and ConfluenceGene._has_next_page(data):
            raise SourceArtifactContractError("Confluence attachment response cannot advance pagination")
        return results

    @staticmethod
    def _has_next_page(data: dict) -> bool:
        links = data.get("_links")
        return isinstance(links, dict) and bool(str(links.get("next") or "").strip())

    @classmethod
    def _validated_page_results(
        cls,
        data: object,
        *,
        context: str,
        expected_start: int,
    ) -> list[dict]:
        if not isinstance(data, dict) or "results" not in data or not isinstance(data.get("results"), list):
            raise RuntimeError(f"Confluence {context} response is missing a results list")
        results = data["results"]
        if any(not isinstance(page, dict) for page in results):
            raise RuntimeError(f"Confluence {context} response contains an invalid page record")
        size = data.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size != len(results):
            raise RuntimeError(f"Confluence {context} response has inconsistent pagination size")
        start = data.get("start")
        if not isinstance(start, int) or isinstance(start, bool) or start != expected_start:
            raise RuntimeError(f"Confluence {context} response has inconsistent pagination start")
        links = data.get("_links")
        if not isinstance(links, dict):
            raise RuntimeError(f"Confluence {context} response is missing pagination links")
        next_link = links.get("next")
        if next_link is not None and (not isinstance(next_link, str) or not next_link.strip()):
            raise RuntimeError(f"Confluence {context} response has an invalid next link")
        if not results and cls._has_next_page(data):
            raise RuntimeError(f"Confluence {context} response has a next link without results")
        return results

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

    def requires_pdf_artifact(
        self,
        *,
        item: ContentItem,
        existing_doc: object | None,
        existing_hash: str | None,
        new_hash: str,
    ) -> bool:
        """Confluence pages require PDF provenance for user-facing source review."""
        del item
        return (
            existing_doc is None
            or existing_hash != new_hash
            or not getattr(existing_doc, "pdf_content_uri", None)
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert Confluence XHTML to comprehensive markdown."""
        html = raw.body.decode("utf-8", errors="replace")

        # Convert HTML to markdown
        markdown = html_to_markdown(html)
        markdown = strip_boilerplate(markdown)

        if raw.authoritative_empty:
            return NormalizedContent(
                item=raw.item,
                markdown_body="",
                source_semantics={
                    "semantic_markdown": "",
                    "space_key": raw.item.space_or_project,
                    "labels": raw.item.labels,
                    "author": raw.item.author,
                    "version": raw.item.version,
                },
            )

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
                # Provider-neutral projection hashes this body rather than the
                # display header, whose Last modified/author fields are operational.
                "semantic_markdown": markdown,
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
