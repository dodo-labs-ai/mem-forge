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
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import aclosing
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
    notion_blocked_integration,
    notion_error_code,
    notion_id,
    notion_object,
    notion_parent,
    notion_time,
)
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
from memforge.source_time import utc_minute_floor

__all__ = ["NotionGene", "notion_token"]

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

    def __init__(self, config: dict, source_id: str) -> None:
        super().__init__(config, source_id)
        self._api: NotionApi | None = None
        self._file_client: ConnectorHttpClient | None = None
        self._walked_page_ids: set[str] = set()
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
        """Close the file client and forget this connection's ancestor verdicts, then close the API client."""
        file_client, self._file_client = self._file_client, None
        self._api = None
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
        async for item in self._walk_scope(since):
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
        and is walked as a database root. A 401 or 403 raises as an ordinary
        error, like one below a root.
        """
        label = "root page" if kind == "page" else "database"
        response = await self._notion().get(_READ_PATHS[kind].format(object_id))
        if response.status_code == httpx.codes.NOT_FOUND:
            raise SourceConfigurationError(
                f"Notion {label} {object_id} does not exist or is not shared with the connection. "
                "Share it with the connection (••• → Connections) or remove it from the Source."
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
        root is in scope. Verdicts on ancestors are kept for the connection,
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
            if node_id in roots:
                verdict = (True, None)
                break
            found = await self._ancestor(kind, node_id)
            if found is None:
                verdict = (False, None)
                break
            chain[-1] = (found[0], node_id)
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
        raise NotImplementedError("NotionGene does not fetch page content yet")

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        raise NotImplementedError("NotionGene does not normalize page content yet")

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
