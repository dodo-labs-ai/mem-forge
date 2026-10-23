"""Notion Gene — collects every page under configured root pages and databases.

A Notion Source is read with an internal connection token. The user shares
root pages and databases with that connection, and the Gene walks each root:
a page's child blocks, recursing through any block with children (toggles,
columns, callouts, original synced blocks), into child pages and child
databases, and through each database's data sources, whose rows are pages
too. Every page the walk reaches becomes one ``notion_page`` Document.

The walk is a query listing (:class:`~memforge.models.ScopeListingKind`): a
page moved out of the roots still exists, so an unlisted page is read by id
(:meth:`NotionGene.confirm_absent`) before it counts as gone. The walk either
completes or raises. Notion's search is never used, because it does not
promise to return everything.

What the walk follows is the physical parent chain, so :meth:`rediscover`
answers for one page by climbing its parents to a configured root:

- a synced block that shows another block's content (``synced_from`` set) is
  not descended; its children live under the original;
- a child database is followed only when the database's own parent is the
  page, block or database that named it. Its data sources are queried for
  their rows, and its own child blocks are read too, because a database
  nested in a wiki is reachable only there. A linked view is a database with
  no data sources, so it adds nothing;
- a trashed or archived page is neither listed nor descended. Notion's
  data-source queries leave archived rows out too, so both paths agree.

:meth:`NotionGene.fetch` reads a page's body from Notion's Markdown endpoint
and keeps it, with a row's properties, in a versioned JSON envelope that
:meth:`NotionGene.normalize` and the projection read::

    {"notion_raw": 1, "markdown": "…", "properties": [{"name": "Status", "type": "status", "value": "Done"}],
     "truncated": false, "unknown_not_found": 0}

``properties`` is present only for a row. Files Notion hosts are named in
the envelope by stable ``notion-file:`` references and streamed by
:meth:`NotionGene.open_source_artifact` while the item is processed, inside
the hour their signed URLs live.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any

import httpx

from memforge.genes.base import SourceConfigurationError
from memforge.genes.document_gene import DocumentGene
from memforge.genes.notion_api import (
    NOTION_API_ORIGIN,
    NOTION_REQUESTS_PER_MINUTE,
    NOTION_VERSION,
    NotionApi,
    NotionMarkdown,
    notion_blocked_integration,
    notion_error_code,
    notion_id,
    notion_link_id,
    notion_object,
    notion_parent,
    notion_time,
)
from memforge.genes.notion_files import NotionFiles, split_signed_url
from memforge.genes.notion_markdown import notion_markdown
from memforge.genes.notion_properties import render_properties
from memforge.genes.toolkit.config import list_config, tls_verify
from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.genes.toolkit.pacing import shared_rate_limiter
from memforge.models import (
    AbsenceAnswer,
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
)
from memforge.source_artifacts import (
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    RawSourceArtifact,
    SourceArtifactDownload,
    normalize_source_artifact_media_type,
    parse_source_artifact_content_length,
)
from memforge.source_time import SOURCE_UPDATED_AT_KEY, utc_minute_floor

__all__ = ["NOTION_RAW_VERSION", "NotionGene", "notion_envelope", "notion_raw_envelope", "notion_token"]

logger = logging.getLogger(__name__)

PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY = "_memforge_preview_limit"
_DOC_ID_PREFIX = "notion-"
# A climb from a page to its root reads at most this many ancestors.
_ANCESTOR_BOUND = 64
_READ_PATHS = {
    "page": "/v1/pages/{}",
    "block": "/v1/blocks/{}",
    "database": "/v1/databases/{}",
    "data_source": "/v1/data_sources/{}",
}
NOTION_RAW_VERSION = 1
"""The version of the JSON envelope :meth:`NotionGene.fetch` stores as a page's raw body."""
_EMPTY_EVIDENCE = "notion_markdown_api_successful_empty_page_body"
# A page's body reads at most this many left-out blocks again; beyond it the page is partial.
_REREAD_BOUND = 200
# Unsupported blocks whose target the block API names, rendered as links to it.
_LINK_BLOCK_TYPES = frozenset({"bookmark", "link_preview", "external_object_instance"})
# Property types a page object cuts at 25 references.
_CUT_PROPERTY_TYPES = frozenset({"title", "rich_text", "people", "relation"})
_CUT_AT = 25
_UNKNOWN_TAG = re.compile(r"<unknown\b([^>]*?)(?:/>|>.*?</unknown>)", re.DOTALL)
_TAG_ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')


def notion_token(config: Mapping[str, Any]) -> str:
    """The internal connection token from ``token``; a missing one is a configuration error."""
    token = str(config.get("token") or "").strip()
    if not token:
        raise SourceConfigurationError("Notion internal connection token is required")
    return token


@dataclass(frozen=True)
class _Visit:
    """One step of the walk.

    - ``page``: list a page (``record`` holds its object when already read)
      and queue its children;
    - ``children``: read the child blocks of a page, block or database;
    - ``database``: read a child database and queue its data sources and
      its child blocks;
    - ``data_source``: query a data source and list its rows.

    ``context`` is the nearest page or database above the step, as
    ``(id32, "page" | "database")``; ``named_by`` is the page, block or
    database whose children named a database.
    """

    kind: str
    object_id: str
    context: tuple[str | None, str | None] = (None, None)
    named_by: str | None = None
    record: dict[str, Any] | None = None


class NotionGene(DocumentGene):
    """Notion pages, collected on the server from configured root pages and databases."""

    listing_kinds = frozenset({ScopeListingKind.QUERY})

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="notion",
            display_name="Notion",
            description="Pages and database rows shared with an internal connection",
            default_sync_interval_minutes=1440,  # daily: a full walk reads every page
            auth_method="api_key",
            data_shape="document",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Connection", order=0),
                ConfigGroup(key="content", label="Content", order=1),
            ],
            fields=[
                ConfigField(
                    key="token", label="Internal connection token",
                    field_type=ConfigFieldType.SECRET, required=True,
                    group="connection", order=0,
                ),
                ConfigField(
                    key="root_pages", label="Root pages",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="Page URL or ID",
                    group="content", order=0,
                ),
                ConfigField(
                    key="databases", label="Databases",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="Database URL or ID",
                    group="content", order=1,
                ),
            ],
        )

    @classmethod
    def normalize_config(cls, config: dict) -> None:
        """Read pasted root links as ids and fix the API origin, in place.

        Each value of ``root_pages`` and ``databases`` that names a Notion
        page or database (:func:`~memforge.genes.notion_api.notion_link_id`)
        becomes its 32-hex id; any other value is kept as given, so the
        source type's check refuses it by its field. ``base_url`` is always
        Notion's API origin, which no user supplies::

            config = {"root_pages": ["https://www.notion.so/team/Roadmap-1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"]}
            NotionGene.normalize_config(config)
            # {"root_pages": ["1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"], "base_url": "https://api.notion.com"}
        """
        for field, open_page in (("root_pages", True), ("databases", False)):
            if field in config:
                config[field] = list(
                    dict.fromkeys(
                        notion_link_id(value, open_page=open_page) or value for value in list_config(config[field])
                    )
                )
        config["base_url"] = NOTION_API_ORIGIN

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._api: NotionApi | None = None
        self._file_client: ConnectorHttpClient | None = None
        self._walked_page_ids: set[str] = set()
        # file key -> the signed URL its item's fetch read; never stored
        self._signed_files: dict[str, str] = {}
        # (kind, id32) -> (reaches a configured root, nearest page or database at or above it)
        self._ancestor_verdicts: dict[tuple[str, str], tuple[bool, tuple[str, str] | None]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Open the API and file clients and check the token with ``GET /v1/users/me``.

        Every request to ``api.notion.com`` shares one request rate per
        connection token (keyed by a hash of it, never the token). Files are
        streamed from Notion's file host through a second client that sends
        no ``Authorization`` header. A token Notion rejects (401) is a
        configuration error, not retried.
        """
        await self.close()
        token = notion_token(self.config)
        verify = tls_verify(self.config)
        self._client = ConnectorHttpClient(
            product_name="Notion",
            base_url=NOTION_API_ORIGIN,
            headers={"Authorization": f"Bearer {token}", "Notion-Version": NOTION_VERSION},
            timeout=30.0,
            verify=verify,
            follow_redirects=False,
            limiter=shared_rate_limiter(
                "notion:" + sha256(token.encode()).hexdigest()[:16],
                requests_per_minute=NOTION_REQUESTS_PER_MINUTE,
            ),
            fail_fast=notion_blocked_integration,
        )
        self._file_client = ConnectorHttpClient(
            product_name="Notion files",
            headers={},
            timeout=60.0,
            verify=verify,
            follow_redirects=False,
        )
        self._api = NotionApi(self._client)
        response = await self._api.get("/v1/users/me")
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise SourceConfigurationError(
                "Notion rejected the connection token. Copy the internal connection token again "
                "and check that the connection is still installed in the workspace."
            )
        response.raise_for_status()
        logger.info("Notion connection authenticated for source %s", self.source_id)

    async def close(self) -> None:
        """Close the file client, forget the connection's signed file URLs and ancestor verdicts, then close the API client."""
        file_client, self._file_client = self._file_client, None
        self._api = None
        self._signed_files.clear()
        self._ancestor_verdicts.clear()
        if file_client is not None:
            await file_client.aclose()
        await super().close()

    async def health_check(self) -> dict:
        try:
            response = await self._notion().get("/v1/users/me")
            response.raise_for_status()
            return {"healthy": True}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Discovery and listing
    # ------------------------------------------------------------------

    @classmethod
    def lists_scope(cls, config: Mapping[str, Any]) -> bool:
        """Every configuration walks its roots completely or raises."""
        return True

    def filter_summary(self, since: datetime | None) -> str | None:
        return f"last_edited_time >= {utc_minute_floor(since).isoformat()}" if since else None

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Walk every root and yield each page edited at or after ``since``'s UTC minute.

        Notion truncates ``last_edited_time`` to the minute, so ``since`` is
        rounded down to it: with ``since`` at 09:30:10, a page Notion reports
        edited at 09:30 is yielded. The completed walk is recorded as the
        run's listing; a preview stops after its limit and records nothing.
        """
        async with aclosing(self._walk_scope(since)) as items:
            async for item in items:
                yield item
        if self._preview_discovery_limit() is None:
            self._record_walked_pages()

    async def list_scope(self) -> ScopeListing:
        """Every page under the roots, as this run's discovery walked them, or walked here."""
        if self._preview_discovery_limit() is not None:
            raise RuntimeError("A preview of a Notion Source stops its walk early, so it cannot list the scope")
        if self._discovered_scope is not None:
            return self._discovered_scope
        async for _item in self._walk_scope(None):
            pass
        self._record_walked_pages()
        return self._discovered_scope

    def _record_walked_pages(self) -> None:
        self.record_scope_listing(
            ScopeListingKind.QUERY, {_DOC_ID_PREFIX + page_id for page_id in self._walked_page_ids}
        )

    async def _walk_scope(self, since: datetime | None) -> AsyncIterator[ContentItem]:
        """Visit every page under the roots, yielding those edited at or after ``since``'s minute.

        The walk keeps an explicit stack, so a deep tree never exhausts
        Python's recursion, and reads each page, block, database and data
        source once, so a cycle ends. A child page, a container's children,
        a database or a data source that answers 404 is read again by id,
        and skipped only when that read also answers 404 or shows it trashed
        (it was deleted during the walk). Any other answer raises.

        One log line counts what the walk skipped: ``skipped_not_found``,
        ``skipped_outside_tree`` (a database shown here that lives elsewhere)
        and ``empty_databases`` (a database with no data sources, such as a
        linked view).
        """
        api = self._notion()
        root_pages, root_databases = self._roots()
        floor = utc_minute_floor(since) if since else None
        preview_limit = self._preview_discovery_limit()
        self._walked_page_ids = set()
        emitted = 0
        counts: Counter[str] = Counter()

        root_visits: list[_Visit] = []
        for configured_kind, root_id in [("page", page_id) for page_id in root_pages] + [
            ("database", database_id) for database_id in root_databases
        ]:
            kind, record = await self._read_root(configured_kind, root_id)
            if kind == "page":
                root_visits.append(_Visit("page", root_id, self._root_context(record), record=record))
            else:
                root_visits.extend(_database_visits(record, root_id, counts))
        stack = list(reversed(root_visits))

        visited: set[tuple[str, str]] = set()
        while stack:
            visit = stack.pop()
            if (visit.kind, visit.object_id) in visited:
                continue
            visited.add((visit.kind, visit.object_id))

            if visit.kind == "page":
                page = visit.record or await self._read_below_root("page", visit.object_id)
                if page is None:
                    counts["skipped_not_found"] += 1
                    continue
                item = self._list_page(page, visit.context, floor, stack)
                if item is not None:
                    yield item
                    emitted += 1
                    if preview_limit is not None and emitted >= preview_limit:
                        return

            elif visit.kind == "children":
                blocks = await self._children_below_root(visit)
                if blocks is None:
                    counts["skipped_not_found"] += 1
                    continue
                stack.extend(reversed(_child_visits(blocks, visit)))

            elif visit.kind == "database":
                database = await self._read_below_root("database", visit.object_id)
                if database is None:
                    counts["skipped_not_found"] += 1
                    continue
                if database.get("in_trash") is True:
                    continue
                if notion_parent(database, f"database {visit.object_id}")[1] != visit.named_by:
                    counts["skipped_outside_tree"] += 1
                    continue
                stack.extend(reversed(_database_visits(database, visit.object_id, counts)))

            else:  # data_source
                try:
                    async with aclosing(api.query_data_source(visit.object_id)) as results:
                        async for result in results:
                            result_id = notion_id(result["id"])
                            if result["object"] == "data_source":
                                # Notion documents a wiki's nested database as a data-source result.
                                kind, database_id = notion_parent(result, f"data source {result_id}")
                                context = (database_id, "database") if kind == "database" else (None, None)
                                stack.append(_Visit("data_source", result_id, context))
                                continue
                            if ("page", result_id) in visited:
                                continue
                            visited.add(("page", result_id))
                            item = self._list_page(result, visit.context, floor, stack)
                            if item is not None:
                                yield item
                                emitted += 1
                                if preview_limit is not None and emitted >= preview_limit:
                                    return
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != httpx.codes.NOT_FOUND:
                        raise
                    await self._confirm_not_found("data_source", visit.object_id)
                    counts["skipped_not_found"] += 1

        logger.info(
            "Notion walk for source %s listed %d pages "
            "(skipped_not_found=%d, skipped_outside_tree=%d, empty_databases=%d)",
            self.source_id,
            len(self._walked_page_ids),
            counts["skipped_not_found"],
            counts["skipped_outside_tree"],
            counts["empty_databases"],
        )

    def _list_page(
        self,
        page: dict[str, Any],
        context: tuple[str | None, str | None],
        floor: datetime | None,
        stack: list[_Visit],
    ) -> ContentItem | None:
        """List a listable page and queue its children; return its item when it changed since ``floor``."""
        page_id = notion_id(page.get("id"))
        if page_id is None:
            raise RuntimeError("Notion returned a page without an id")
        if not _listable_state(page):
            return None
        self._walked_page_ids.add(page_id)
        stack.append(_Visit("children", page_id, (page_id, "page")))
        item = self._item(page, context)
        if floor is not None and item.last_modified < floor:
            return None
        return item

    async def _read_root(self, kind: str, object_id: str) -> tuple[str, dict[str, Any]]:
        """A configured root, as ``(kind, record)``; one Notion cannot find, or one in the trash, is a configuration error.

        A root page that has since been turned into a wiki is a database now,
        and is walked as a database root. A page listed under ``databases``
        answers 400 ``validation_error`` and names the field to move it from.
        A 401 or 403 raises as an ordinary error, like one below a root.
        """
        label = "root page" if kind == "page" else "database"
        response = await self._notion().get(_READ_PATHS[kind].format(object_id))
        if response.status_code == httpx.codes.NOT_FOUND:
            raise SourceConfigurationError(
                f"Notion {label} {object_id} does not exist or is not shared with the connection. "
                "Share it with the connection (••• → Connections) or remove it from the Source."
            )
        if (
            kind == "database"
            and response.status_code == httpx.codes.BAD_REQUEST
            and notion_error_code(response) == "validation_error"
        ):
            raise SourceConfigurationError(
                f"Notion databases holds {object_id}, which Notion does not read as a database. "
                "If it is a page, move it to Root pages."
            )
        database = await self._database_behind(response, object_id) if kind == "page" else None
        if database is not None:
            kind, record = "database", database
        else:
            record = notion_object(response, kind, object_id)
        if record.get("in_trash") is True:
            raise SourceConfigurationError(
                f"Notion {label} {object_id} is in the trash. Restore it or remove it from the Source."
            )
        return kind, record

    async def _database_behind(self, page_response: httpx.Response, object_id: str) -> dict[str, Any] | None:
        """The database a page read hit, when the id is a database now; otherwise ``None``.

        A page turned into a wiki keeps its id and becomes a database, and
        reading it as a page answers 400 ``validation_error`` ("is a
        database, not a page"). That answer counts only when reading the id
        as a database succeeds; any other 400 stays an error for the caller.
        """
        if page_response.status_code != httpx.codes.BAD_REQUEST or notion_error_code(page_response) != "validation_error":
            return None
        response = await self._notion().get(_READ_PATHS["database"].format(object_id))
        if response.status_code != httpx.codes.OK:
            return None
        return notion_object(response, "database", object_id)

    async def _read_below_root(self, kind: str, object_id: str) -> dict[str, Any] | None:
        """A page or database below a root, or ``None`` once a second read confirms Notion cannot find it."""
        response = await self._notion().get(_READ_PATHS[kind].format(object_id))
        if response.status_code == httpx.codes.NOT_FOUND:
            await self._confirm_not_found(kind, object_id)
            return None
        return notion_object(response, kind, object_id)

    async def _children_below_root(self, visit: _Visit) -> list[dict[str, Any]] | None:
        """The child blocks of a page, block or database, or ``None`` once a read by id confirms the container is gone."""
        try:
            return [block async for block in self._notion().block_children(visit.object_id)]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != httpx.codes.NOT_FOUND:
                raise
        kind = visit.context[1] if visit.context[0] == visit.object_id else "block"
        await self._confirm_not_found(str(kind), visit.object_id)
        return None

    async def _confirm_not_found(self, kind: str, object_id: str) -> None:
        """Read an object that answered 404 once more by id; return only when Notion confirms it is gone.

        A 404, or a 200 showing it in the trash, confirms. Anything else
        raises: skipping a page Notion still shows would make it look deleted.
        """
        response = await self._notion().get(_READ_PATHS[kind].format(object_id))
        if response.status_code == httpx.codes.NOT_FOUND:
            return
        if response.status_code == httpx.codes.OK and notion_object(response, kind, object_id).get("in_trash") is True:
            return
        raise RuntimeError(
            f"Notion answered 404 for {kind.replace('_', ' ')} {object_id} during the walk, but reading it "
            f"by id answered {response.status_code}; the listing cannot skip it"
        )

    # ------------------------------------------------------------------
    # Absence and rediscovery
    # ------------------------------------------------------------------

    async def confirm_absent(self, items: Sequence[ContentItem]) -> Mapping[str, AbsenceAnswer]:
        """Read each unlisted page by id.

        - 404 or 410, or a page in the trash: ``absent``;
        - an archived page, or one moved out of the roots: ``exists``;
        - a page turned into a wiki, which is a database now: ``exists``
          (``absent`` once that database is in the trash);
        - 403: ``hidden``;
        - anything else (401, another 400, a rate limit or server error
          after retries) raises, so the run removes nothing.
        """
        answers: dict[str, AbsenceAnswer] = {}
        for item in items:
            page_id = _item_page_id(item)
            response = await self._notion().get(f"/v1/pages/{page_id}")
            if response.status_code in (httpx.codes.NOT_FOUND, httpx.codes.GONE):
                answers[item.item_id] = AbsenceAnswer.ABSENT
            elif response.status_code == httpx.codes.FORBIDDEN:
                answers[item.item_id] = AbsenceAnswer.HIDDEN
            else:
                record = await self._database_behind(response, page_id) or notion_object(response, "page", page_id)
                answers[item.item_id] = AbsenceAnswer.ABSENT if record.get("in_trash") is True else AbsenceAnswer.EXISTS
        return answers

    @classmethod
    def rediscovers_documents(cls, config: Mapping[str, Any]) -> bool:
        return True

    async def rediscover(self, item: ContentItem) -> ContentItem | None:
        """Read one page and return it when the walk would list it: by climbing its parents to a root.

        ``None`` when the page answers 404, is trashed or archived, has
        become a database (a page turned into a wiki), or its parent chain
        ends outside every root: at the workspace, or at an
        ancestor that is trashed, archived or answers 404 (an ancestor
        between a shared root and the page is always visible). A configured
        root is read like any other ancestor, so a page below a trashed or
        archived root is out of scope, as it is for the walk. Verdicts on ancestors are kept for the connection,
        so pages under one ancestor read it once. A chain longer than 64
        ancestors raises.
        """
        page_id = _item_page_id(item)
        response = await self._notion().get(f"/v1/pages/{page_id}")
        if response.status_code == httpx.codes.NOT_FOUND or await self._database_behind(response, page_id):
            return None
        page = notion_object(response, "page", page_id)
        if not _listable_state(page):
            return None
        root_pages, _root_databases = self._roots()
        if page_id in root_pages:
            return self._item(page, self._root_context(page))
        in_scope, nearest = await self._climb(notion_parent(page, f"page {page_id}"))
        if not in_scope:
            return None
        return self._item(page, (nearest[1], nearest[0]) if nearest else (None, None))

    async def _climb(self, start: tuple[str, str | None]) -> tuple[bool, tuple[str, str] | None]:
        """Whether a parent chain from ``start`` reaches a configured root, and the nearest page or database on it."""
        root_pages, root_databases = self._roots()
        roots = {*root_pages, *root_databases}
        chain: list[tuple[str, str]] = []
        node = start
        while True:
            kind, node_id = node
            if kind == "workspace" or node_id is None:
                verdict: tuple[bool, tuple[str, str] | None] = (False, None)
                break
            if (kind, node_id) in self._ancestor_verdicts:
                verdict = self._ancestor_verdicts[(kind, node_id)]
                break
            if len(chain) >= _ANCESTOR_BOUND:
                raise RuntimeError(f"Notion page's parent chain is longer than {_ANCESTOR_BOUND} ancestors")
            chain.append((kind, node_id))
            found = await self._ancestor(kind, node_id)
            if found is None:
                verdict = (False, None)
                break
            chain[-1] = (found[0], node_id)
            if node_id in roots:
                verdict = (True, None)
                break
            node = notion_parent(found[1], f"{found[0]} {node_id}")

        in_scope, nearest = verdict
        for key in reversed(chain):
            if key[0] in {"page", "database"}:
                nearest = key
            self._ancestor_verdicts[key] = (in_scope, nearest)
        return self._ancestor_verdicts[chain[0]] if chain else verdict

    async def _ancestor(self, kind: str, object_id: str) -> tuple[str, dict[str, Any]] | None:
        """An ancestor on a climb as ``(kind, record)``, or ``None`` when it answers 404, is trashed, or is an archived page.

        A parent named as a page that has since been turned into a wiki is
        returned as the database it is now.
        """
        response = await self._notion().get(_READ_PATHS[kind].format(object_id))
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        database = await self._database_behind(response, object_id) if kind == "page" else None
        if database is not None:
            kind, record = "database", database
        else:
            record = notion_object(response, kind, object_id)
        if record.get("in_trash") is True or (kind == "page" and record.get("is_archived") is True):
            return None
        return kind, record

    # ------------------------------------------------------------------
    # Fetch and normalize
    # ------------------------------------------------------------------

    async def fetch(self, item: ContentItem) -> RawContent:
        """Read one page: its object, its body as Markdown, a row's properties and the files they name.

        The page object decides whether the page is in the trash, never the
        Markdown endpoint, which still serves a trashed page's body; a page
        trashed since discovery fails the item, and the next run's listing
        and confirmation decide. Blocks the Markdown left out are resolved
        (:meth:`_page_body`), so ``truncated`` in the envelope means content
        is still missing. A row's properties are read in full and rendered
        as text records, never into the body.

        Each Notion-hosted file becomes one artifact, owned by the body when
        the body names it and by the properties otherwise. A blank body that
        is complete is attested empty, as a row with only properties often
        is; a truncated one never is.
        """
        page_id = _item_page_id(item)
        page = notion_object(await self._notion().get(f"/v1/pages/{page_id}"), "page", page_id)
        if page.get("in_trash") is True:
            raise RuntimeError(f"Notion page {page_id} was moved to the trash after discovery")
        markdown, truncated, not_found = await self._page_body(page_id)
        files = NotionFiles()
        envelope: dict[str, Any] = {"notion_raw": NOTION_RAW_VERSION, "markdown": files.stabilize(markdown, "page_body")}
        if notion_parent(page, f"page {page_id}")[0] == "database":
            properties = await self._full_properties(page_id, page.get("properties"))
            envelope["properties"] = render_properties(
                properties, file_text=lambda entry: _file_text(entry, files)
            )
        envelope["truncated"] = truncated
        envelope["unknown_not_found"] = not_found
        self._signed_files.update(files.signed_urls)
        empty = not truncated and not notion_markdown(envelope["markdown"]).strip()
        return RawContent(
            item=item,
            body=json.dumps(envelope, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            authoritative_empty=empty,
            empty_evidence=_EMPTY_EVIDENCE if empty else None,
            artifacts=_artifacts(page_id, files),
        )

    async def _page_body(self, page_id: str) -> tuple[str, bool, int]:
        """A page's Markdown with every left-out block resolved, as ``(markdown, truncated, not_found)``.

        Notion writes a block its answer left out as an ``<unknown>`` tag
        whose ``url`` names the block in its fragment:

        - a tag with an ``alt`` is a block type the Markdown cannot show. A
          bookmark or link preview is read through the block API and becomes
          a link to its target; any other type becomes a placeholder naming
          it. Neither is ever partial;
        - a tag without an ``alt`` is a block cut for size or permission. It
          is read again through the Markdown endpoint and spliced in place,
          and so are the cut blocks of that answer. One that answers 404 is a
          block the connection cannot see: left out and counted in
          ``not_found``, not partial.

        The page is partial (``truncated``) when a cut block is still cut
        after its re-read, more than 200 re-reads would be needed, Notion
        names a cut block it gave no tag for, or an answer says
        ``truncated`` while naming no block at all. Unsupported blocks set
        Notion's ``truncated`` on their own, so an answer whose named blocks
        are all unsupported is complete.
        """
        answer = await self._notion().page_markdown(page_id)
        if answer is None:
            raise RuntimeError(f"Notion page {page_id} answered 404 for its Markdown")
        state = _BodyState()
        markdown = await self._resolved(answer, page_id, state)
        return markdown, state.truncated, state.not_found

    async def _resolved(self, answer: NotionMarkdown, read_id: str, state: _BodyState) -> str:
        tags = [
            (match, _unknown_block_id(match, read_id), _tag_attribute(match.group(1), "alt"))
            for match in _UNKNOWN_TAG.finditer(answer.markdown)
        ]
        unsupported = {block_id for _match, block_id, alt in tags if alt}
        cut = {block_id for _match, block_id, alt in tags if not alt}
        # Unsupported blocks set Notion's flag and are listed on their own; only the rest is missing content.
        listed = set(answer.unknown_block_ids) - unsupported
        if (answer.truncated and not answer.unknown_block_ids and not cut) or listed - cut:
            state.truncated = True
        pieces: list[str] = []
        position = 0
        for match, block_id, alt in tags:
            pieces.append(answer.markdown[position : match.start()])
            if alt:
                pieces.append(await self._unsupported_block(block_id, alt, state))
            elif block_id == read_id:
                state.truncated = True  # the block's own re-read cut it again
            else:
                line_start = answer.markdown.rfind("\n", 0, match.start()) + 1
                indent = re.match(r"\t*", answer.markdown[line_start:]).group(0)
                pieces.append((await self._reread(block_id, state)).replace("\n", "\n" + indent))
            position = match.end()
        pieces.append(answer.markdown[position:])
        return "".join(pieces)

    async def _reread(self, block_id: str, state: _BodyState) -> str:
        if state.rereads >= _REREAD_BOUND:
            state.truncated = True
            return ""
        state.rereads += 1
        answer = await self._notion().page_markdown(block_id)
        if answer is None:
            state.not_found += 1
            return ""
        return (await self._resolved(answer, block_id, state)).strip("\n")

    async def _unsupported_block(self, block_id: str, block_type: str, state: _BodyState) -> str:
        """A link to a bookmark's or link preview's target, or a placeholder naming the block type."""
        if block_type not in _LINK_BLOCK_TYPES:
            return _placeholder(block_type)
        response = await self._notion().get(f"/v1/blocks/{block_id}")
        if response.status_code == httpx.codes.NOT_FOUND:
            state.not_found += 1
            return _placeholder(block_type)
        block = notion_object(response, "block", block_id)
        content = block.get(str(block.get("type")))
        content = content if isinstance(content, dict) else {}
        url = str(content.get("url") or "").strip()
        if not url:
            return _placeholder(str(content.get("block_type") or block_type))
        caption = "".join(
            str(part.get("plain_text") or "") for part in content.get("caption") or [] if isinstance(part, dict)
        ).strip()
        label = (caption or url).replace("[", "\\[").replace("]", "\\]")
        return f"[{label}]({url})"

    async def _full_properties(self, page_id: str, properties: object) -> dict[str, Any]:
        """A row's properties, each value Notion cut at 25 references read in full."""
        if not isinstance(properties, dict):
            raise RuntimeError(f"Notion row {page_id} has no properties")
        full = dict(properties)
        for name, value in properties.items():
            if not isinstance(value, dict) or value.get("type") not in _CUT_PROPERTY_TYPES:
                continue
            kind = value["type"]
            references = value.get(kind)
            if value.get("has_more") is not True and not (isinstance(references, list) and len(references) >= _CUT_AT):
                continue
            items = [item async for item in self._notion().property_items(page_id, str(value.get("id") or ""))]
            full[name] = {**value, kind: [item.get(kind) for item in items], "has_more": False}
        return full

    @asynccontextmanager
    async def open_source_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
        """Stream one Notion-hosted file from the signed URL its item's fetch read.

        The request goes through the file client, which sends no
        ``Authorization`` header (Notion's file host refuses one), and names
        the signature only in its query, so MemForge's retry logs and error
        messages, which name the address alone, never show it. A file
        this connection's fetch did not read, or one the file host refuses
        (an expired signature answers 400 or 403), fails the item, and the
        next run fetches it again with fresh URLs.
        """
        key = artifact.provider_key
        signed = self._signed_files.get(key)
        if signed is None or self._file_client is None:
            raise RuntimeError(f"Notion file {key} has no signed URL from this connection's fetch")
        address, query = split_signed_url(signed)
        async with self._file_client.stream("GET", address, params=httpx.QueryParams(query)) as response:
            if response.status_code >= httpx.codes.BAD_REQUEST:
                raise RuntimeError(f"Notion's file host answered {response.status_code} for file {key}")
            yield SourceArtifactDownload(
                chunks=response.aiter_bytes(),
                media_type=response.headers.get("content-type"),
                content_length=parse_source_artifact_content_length(response.headers.get("content-length")),
                content_encoding=response.headers.get("content-encoding"),
            )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """The page body as CommonMark, from the envelope's Markdown alone.

        Properties stay in their own Observation, so the body is the page's
        body only, and a row with no body text normalizes to ``""``.
        """
        envelope = notion_raw_envelope(raw)
        return NormalizedContent(
            item=raw.item,
            markdown_body=notion_markdown(envelope["markdown"]),
            source_semantics={SOURCE_UPDATED_AT_KEY: raw.item.last_modified.isoformat()},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notion(self) -> NotionApi:
        if self._api is None:
            raise RuntimeError("Notion Gene is not authenticated")
        return self._api

    def _roots(self) -> tuple[list[str], list[str]]:
        """The configured root page and database ids, as 32-hex ids; at least one is required."""
        roots: dict[str, list[str]] = {}
        for field in ("root_pages", "databases"):
            ids = []
            for value in list_config(self.config.get(field)):
                parsed = notion_id(value)
                if parsed is None:
                    raise SourceConfigurationError(f"Notion {field} holds {value!r}, which is not a Notion id")
                ids.append(parsed)
            roots[field] = ids
        if not roots["root_pages"] and not roots["databases"]:
            raise SourceConfigurationError("Notion Source needs at least one root page or database")
        return roots["root_pages"], roots["databases"]

    @staticmethod
    def _root_context(page: dict[str, Any]) -> tuple[str | None, str | None]:
        """A root page's nearest page or database, read from its own parent; a block or the workspace gives none."""
        kind, parent_id = notion_parent(page, f"page {page.get('id')}")
        return (parent_id, kind) if kind in {"page", "database"} else (None, None)

    def _item(self, page: dict[str, Any], context: tuple[str | None, str | None]) -> ContentItem:
        page_id = notion_id(page.get("id"))
        edited = page.get("last_edited_time")
        return ContentItem(
            item_id=_DOC_ID_PREFIX + str(page_id),
            title=_page_title(page),
            source_url=f"https://app.notion.com/p/{page_id}",
            last_modified=notion_time(edited, f"page {page_id}"),
            content_type="text/markdown",
            version=str(edited),
            extra={"page_id": page_id, "parent_page_id": context[0], "parent_kind": context[1]},
        )

    def _preview_discovery_limit(self) -> int | None:
        value = self.config.get(PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY)
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None


@dataclass
class _BodyState:
    """What resolving one page body found: whether content is still missing, the re-reads spent, and the 404s."""

    truncated: bool = False
    rereads: int = 0
    not_found: int = 0


def notion_raw_envelope(raw: RawContent) -> dict[str, Any]:
    """The envelope a Notion fetch stored as ``raw``'s body (:func:`notion_envelope`)."""
    try:
        envelope = json.loads(raw.body.decode("utf-8")) if raw.content_type == "application/json" else None
    except (UnicodeDecodeError, ValueError):
        envelope = None
    return notion_envelope(envelope, raw.item.item_id)


def notion_envelope(envelope: object, item_id: str) -> dict[str, Any]:
    """A decoded Notion raw body, checked to be a version 1 envelope in its documented shape.

    An envelope of another version or shape raises ``ValueError``, so the
    item fails rather than projects a body it cannot read.
    """
    if not isinstance(envelope, dict) or envelope.get("notion_raw") != NOTION_RAW_VERSION:
        raise ValueError(f"Notion raw body of {item_id} is not a version {NOTION_RAW_VERSION} envelope")
    properties = envelope.get("properties", [])
    if (
        not isinstance(envelope.get("markdown"), str)
        or not isinstance(envelope.get("truncated"), bool)
        or not isinstance(envelope.get("unknown_not_found"), int)
        or not isinstance(properties, list)
        or not all(
            isinstance(record, dict) and all(isinstance(record.get(key), str) for key in ("name", "type", "value"))
            for record in properties
        )
    ):
        raise ValueError(f"Notion raw body of {item_id} has an envelope of another shape")
    return envelope


def _unknown_block_id(tag: re.Match[str], read_id: str) -> str:
    """The block an ``<unknown>`` tag names: the id in its ``url``'s fragment, else the URL's last path id."""
    url = _tag_attribute(tag.group(1), "url")
    address, _hash, fragment = url.partition("#")
    block_id = notion_id(fragment) or notion_id(address.rstrip("/").rsplit("/", 1)[-1].rsplit("-", 1)[-1])
    if block_id is None:
        raise RuntimeError(f"Notion's Markdown of {read_id} names a left-out block without a readable block id")
    return block_id


def _tag_attribute(attributes: str, name: str) -> str:
    return next((value for key, value in _TAG_ATTRIBUTE.findall(attributes) if key == name), "")


def _placeholder(block_type: str) -> str:
    return f"[Notion block not shown: {block_type}]"


def _file_text(entry: Mapping[str, Any], files: NotionFiles) -> str:
    """One entry of a ``files`` property: a Notion-hosted file's stable reference, or an external file's URL."""
    target = entry.get(str(entry.get("type")))
    url = str(target.get("url") or "").strip() if isinstance(target, Mapping) else ""
    return files.stabilize(url, "properties") if url else str(entry.get("name") or "")


def _artifacts(page_id: str, files: NotionFiles) -> tuple[RawSourceArtifact, ...]:
    """One artifact per Notion-hosted file of a media type MemForge stores, under the Observation that first names it."""
    artifacts = []
    for key, owner in files.owners.items():
        filename = NotionFiles.filename(key)
        media_type = normalize_source_artifact_media_type(mimetypes.guess_type(filename)[0])
        if media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
            continue
        artifacts.append(
            RawSourceArtifact(
                provider_key=key,
                parent_observation_type=owner,
                parent_provider_key=f"{page_id}:{'body' if owner == 'page_body' else 'properties'}",
                provider_revision=key,
                filename=filename,
                media_type=media_type,
                locator={"file_key": key},
            )
        )
    return tuple(artifacts)


def _listable_state(page: dict[str, Any]) -> bool:
    """Whether a page is neither trashed (``in_trash``) nor archived (``is_archived``)."""
    return page.get("in_trash") is not True and page.get("is_archived") is not True


def _child_visits(blocks: list[dict[str, Any]], visit: _Visit) -> list[_Visit]:
    """The steps a container's child blocks add, in order."""
    visits = []
    for block in blocks:
        block_id = notion_id(block["id"])
        block_type = block["type"]
        if block_type == "child_page":
            visits.append(_Visit("page", block_id, visit.context))
        elif block_type == "child_database":
            visits.append(_Visit("database", block_id, visit.context, named_by=visit.object_id))
        elif block_type == "synced_block" and (block.get("synced_block") or {}).get("synced_from"):
            continue  # a duplicate shows children that live under its original
        elif block.get("has_children") is True:
            visits.append(_Visit("children", block_id, visit.context))
    return visits


def _database_visits(database: dict[str, Any], database_id: str, counts: Counter[str]) -> list[_Visit]:
    """The steps a database in the tree adds: each data source, then its own child blocks.

    A wiki database's child blocks hold the databases nested in it, which
    no query returns; an ordinary database has none, at the cost of one
    read. A database with no data sources (a linked view) is counted.
    """
    context = (database_id, "database")
    source_ids = _data_source_ids(database, database_id)
    if not source_ids:
        counts["empty_databases"] += 1
    return [_Visit("data_source", source_id, context) for source_id in source_ids] + [
        _Visit("children", database_id, context)
    ]


def _data_source_ids(database: dict[str, Any], database_id: str) -> list[str]:
    sources = database.get("data_sources")
    if not isinstance(sources, list):
        raise RuntimeError(f"Notion database {database_id} lists no data sources")
    ids = [notion_id(source.get("id")) if isinstance(source, dict) else None for source in sources]
    if any(source_id is None for source_id in ids):
        raise RuntimeError(f"Notion database {database_id} lists a data source without an id")
    return ids


def _page_title(page: dict[str, Any]) -> str:
    properties = page.get("properties")
    if isinstance(properties, dict):
        for value in properties.values():
            if isinstance(value, dict) and value.get("type") == "title":
                text = "".join(
                    str(part.get("plain_text") or "") for part in value.get("title") or [] if isinstance(part, dict)
                )
                return text.strip() or "Untitled"
    return "Untitled"


def _item_page_id(item: ContentItem) -> str:
    return str(item.extra.get("page_id") or item.item_id.removeprefix(_DOC_ID_PREFIX))
