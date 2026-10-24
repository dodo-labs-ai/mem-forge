"""A fake Notion: a page tree, a database root, and pages around them, served by ``tests.notion_fake``.

The root page holds a toggle with a child page, a column list whose column
holds a child page edited in the cursor's minute, a synced block that shows
an original living on a page outside both roots, and a child database with
two data sources. One row of that database is archived, and one child page
is in the trash. The database root is a database of its own with one data
source; one of its rows was edited in the cursor's minute.

Outside both roots: a page shared with the connection, the page holding the
synced original, and a child page under that original, which the root page
shows only through its synced duplicate. Page ``GONE`` does not exist.

Notion writes ``last_edited_time`` truncated to the minute, so the page
edited at :data:`IN_THE_CURSOR_MINUTE` reads ``2026-03-01T09:30:00.000Z``;
discovery must round ``since`` down to the minute to yield it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime

import httpx

from memforge.genes import notion_gene
from memforge.genes.notion_gene import NotionGene
from memforge.models import ContentItem
from tests.connector_http_fixture import FakeConnectorHttp
from tests.contracts.document_fakes import (
    IN_THE_CURSOR_MINUTE,
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)
from tests.notion_fake import TOKEN, NotionFake

ROOT = "a0000000000040008000000000000001"
TOGGLE = "a0000000000040008000000000000002"
IN_TOGGLE = "a0000000000040008000000000000003"
COLUMN_LIST = "a0000000000040008000000000000004"
COLUMN = "a0000000000040008000000000000005"
IN_COLUMN = "a0000000000040008000000000000006"
DUPLICATE = "a0000000000040008000000000000007"
DATABASE = "a0000000000040008000000000000008"
ROW = "a0000000000040008000000000000009"
SECOND_SOURCE_ROW = "a000000000004000800000000000000a"
ARCHIVED_ROW = "a000000000004000800000000000000b"
TRASHED = "a000000000004000800000000000000c"
ROOT_DATABASE = "b0000000000040008000000000000001"
ROOT_DATABASE_ROW = "b0000000000040008000000000000002"
ROOT_DATABASE_RECENT_ROW = "b0000000000040008000000000000003"
OUTSIDE = "c0000000000040008000000000000001"
ORIGINAL_HOLDER = "c0000000000040008000000000000002"
ORIGINAL = "c0000000000040008000000000000003"
SHOWN_THROUGH_DUPLICATE = "c0000000000040008000000000000004"
GONE = "d0000000000040008000000000000001"

# Every page the fake holds, in scope or not, plus one that does not exist.
EVERY_PAGE = (
    ROOT, IN_TOGGLE, IN_COLUMN, ROW, SECOND_SOURCE_ROW, ARCHIVED_ROW, TRASHED,
    ROOT_DATABASE_ROW, ROOT_DATABASE_RECENT_ROW, OUTSIDE, ORIGINAL_HOLDER, SHOWN_THROUGH_DUPLICATE, GONE,
)
# Every row of the database root is created in one minute, so a capped query cannot be windowed.
ROOT_DATABASE_ROWS_CREATED = "2026-01-01T00:00:00.000Z"


def _minute(moment: datetime) -> str:
    """A time as Notion writes ``last_edited_time``: in UTC, truncated to the minute."""
    return moment.strftime("%Y-%m-%dT%H:%M:00.000Z")


def _workspace() -> NotionFake:
    fake = NotionFake()
    fake.add_page("Root", page_id=ROOT, edited=_minute(LONG_AGO))
    fake.add_block(ROOT, "toggle", block_id=TOGGLE)
    fake.add_page("In a toggle", parent=("block", TOGGLE), page_id=IN_TOGGLE, edited=_minute(RECENTLY))
    fake.add_block(ROOT, "column_list", block_id=COLUMN_LIST)
    fake.add_block(COLUMN_LIST, "column", block_id=COLUMN)
    fake.add_page(
        "In a column", parent=("block", COLUMN), page_id=IN_COLUMN, edited=_minute(IN_THE_CURSOR_MINUTE)
    )
    fake.add_page("Holds a synced original", page_id=ORIGINAL_HOLDER, edited=_minute(RECENTLY))
    fake.add_block(ORIGINAL_HOLDER, "synced_block", block_id=ORIGINAL)
    fake.add_page(
        "Shown only through a synced duplicate",
        parent=("block", ORIGINAL),
        page_id=SHOWN_THROUGH_DUPLICATE,
        edited=_minute(RECENTLY),
    )
    fake.add_block(ROOT, "synced_block", synced_from=ORIGINAL, block_id=DUPLICATE)
    _database, (first, second) = fake.add_database(ROOT, sources=2, database_id=DATABASE)
    fake.add_row(DATABASE, first, "Row", row_id=ROW, edited=_minute(LONG_AGO))
    fake.add_row(DATABASE, second, "Row of the second source", row_id=SECOND_SOURCE_ROW, edited=_minute(RECENTLY))
    fake.add_row(DATABASE, first, "Archived row", row_id=ARCHIVED_ROW, edited=_minute(RECENTLY), is_archived=True)
    fake.add_page("Trashed", parent=("page", ROOT), page_id=TRASHED, edited=_minute(RECENTLY), in_trash=True)
    _root_database, (source,) = fake.add_database(database_id=ROOT_DATABASE)
    fake.add_row(
        ROOT_DATABASE, source, "Database row", row_id=ROOT_DATABASE_ROW,
        created=ROOT_DATABASE_ROWS_CREATED, edited=_minute(LONG_AGO),
    )
    fake.add_row(
        ROOT_DATABASE, source, "Database row edited in the cursor's minute", row_id=ROOT_DATABASE_RECENT_ROW,
        created=ROOT_DATABASE_ROWS_CREATED, edited=_minute(IN_THE_CURSOR_MINUTE),
    )
    fake.add_page("Outside", page_id=OUTSIDE, edited=_minute(RECENTLY))
    return fake


def _is_listing_request(request: httpx.Request) -> bool:
    path = request.url.path
    return path.endswith("/children") or path.endswith("/query") or path.startswith("/v1/databases/")


class _Provider:
    def __init__(self, listing: ProviderListing, read_status: int | None = None) -> None:
        self.fake = _workspace()
        self.listing = listing
        self.read_status = read_status
        self.requests: list[str] = []
        self.http = FakeConnectorHttp(self.handle)
        if listing is ProviderListing.TRUNCATED:
            self.fake.result_cap = 1

    def connection(self, gene: NotionGene) -> FakeConnection:
        return FakeConnection(gene=gene, requests=self.requests, clients=self.http.clients, waits=self.http.waits)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url}")
        if self.read_status is not None and request.url.path == f"/v1/pages/{IN_TOGGLE}":
            return _status_answer(self.read_status, self.fake.handle(request))
        if self.listing is ProviderListing.FAILED and _is_listing_request(request):
            return httpx.Response(500, json={"object": "error", "code": "internal_server_error"})
        response = self.fake.handle(request)
        if self.listing is ProviderListing.TRUNCATED and request.url.path.endswith("/children"):
            # Claims more children without a cursor to read them with.
            return httpx.Response(200, json={**response.json(), "has_more": True, "next_cursor": None})
        return response


def _status_answer(status: int, page: httpx.Response) -> httpx.Response:
    """The page read answering ``status``: the page itself on 200, else a Notion error, waiting 2 s on 429 and 503."""
    if status == httpx.codes.OK:
        return page
    headers = {"Retry-After": "2"} if status in (httpx.codes.TOO_MANY_REQUESTS, httpx.codes.SERVICE_UNAVAILABLE) else {}
    return httpx.Response(status, json={"object": "error", "code": "error", "message": "fake"}, headers=headers)


class NotionFakes(DocumentConnectorFakes):
    retry_waits = {429: (2.0, 2.0), 503: (2.0, 2.0)}
    listing_configs = {
        "root_page": ListingConfig(
            {"token": TOKEN, "root_pages": [ROOT]},
            walks_scope=True,
            edited_in_the_cursor_minute=f"notion-{IN_COLUMN}",
        ),
        "database": ListingConfig(
            {"token": TOKEN, "databases": [ROOT_DATABASE]},
            walks_scope=True,
            edited_in_the_cursor_minute=f"notion-{ROOT_DATABASE_RECENT_ROW}",
        ),
    }

    @asynccontextmanager
    async def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AsyncIterator[FakeConnection]:
        provider = _Provider(listing)
        gene = await _authenticated(dict(config), provider)
        try:
            yield provider.connection(gene)
        finally:
            await gene.close()

    @asynccontextmanager
    async def reading_by_id(self, status: int) -> AsyncIterator[tuple[FakeConnection, Sequence[ContentItem]]]:
        provider = _Provider(ProviderListing.COMPLETE, read_status=status)
        gene = await _authenticated(dict(self.listing_configs["root_page"].config), provider)
        try:
            yield provider.connection(gene), [_stored_page(IN_TOGGLE)]
        finally:
            await gene.close()

    def rediscover_candidates(self, config: Mapping[str, object]) -> list[ContentItem]:
        return [_stored_page(page_id) for page_id in EVERY_PAGE]


async def _authenticated(config: dict, provider: _Provider) -> NotionGene:
    """A Gene whose API and file clients, both built in ``authenticate``, the fake answers and records."""
    gene = NotionGene(config=config, source_id="src-contract")
    with provider.http.serving(notion_gene):
        await gene.authenticate()
    provider.requests.clear()
    return gene


def _stored_page(page_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"notion-{page_id}",
        title="Stored page",
        source_url=f"https://app.notion.com/p/{page_id}",
        last_modified=LONG_AGO,
        extra={"page_id": page_id},
    )


FAKES = NotionFakes()
