"""A fake Confluence: one page tree in space PAY, a page in space OPS, and the REST API v1 over them.

Space PAY holds root 100 with children 101, 102 and 104; 102 carries the
excluded label ``draft`` and has a child 103. Discovery of the page tree
therefore visits 100, 101, 102 and 104 and yields 100, 101 and 104; discovery
of the space visits every PAY page and yields all but 102. Page 104 was edited
in the cursor's minute. Page 200 is in space OPS, outside both scopes, 150 is
a blog post in PAY, which no page listing returns, and page 300 is gone.

Each page's ``version.when`` is a full timestamp, as Confluence answers it,
and discovery compares it with ``since`` itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx

from memforge.genes import confluence_gene
from memforge.genes.confluence_gene import ConfluenceGene
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

BASE_URL = "https://wiki.example.test"

# content id -> (space, parent page id, labels, changed at)
PAGES: dict[str, tuple[str, str | None, tuple[str, ...], datetime]] = {
    "100": ("PAY", None, (), LONG_AGO),
    "101": ("PAY", "100", (), RECENTLY),
    "102": ("PAY", "100", ("draft",), LONG_AGO),
    "103": ("PAY", "102", (), RECENTLY),
    "104": ("PAY", "100", (), IN_THE_CURSOR_MINUTE),
    "150": ("PAY", "100", (), RECENTLY),
    "200": ("OPS", None, (), RECENTLY),
}
BLOG_POSTS = {"150"}
GONE_PAGE = "300"

_CONNECTION = {"base_url": BASE_URL, "pat": "token", "exclude_labels": ["draft"]}


def _page(page_id: str) -> dict:
    space, parent, labels, changed_at = PAGES[page_id]
    ancestors = []
    while parent is not None:
        ancestors.insert(0, _page_header(parent))
        parent = PAGES[parent][1]
    return {**_page_header(page_id), "ancestors": ancestors}


def _page_header(page_id: str) -> dict:
    space, _parent, labels, changed_at = PAGES[page_id]
    return {
        "id": page_id,
        "type": "blogpost" if page_id in BLOG_POSTS else "page",
        "status": "current",
        "title": f"Page {page_id}",
        "space": {"key": space},
        "version": {"number": 1, "when": changed_at.isoformat().replace("+00:00", "Z")},
        "metadata": {"labels": {"results": [{"name": label} for label in labels]}},
        "_links": {"webui": f"/pages/{page_id}"},
    }


def _results(pages: list[dict], start: int, *, more: bool = False) -> dict:
    return {
        "results": pages,
        "size": len(pages),
        "start": start,
        "limit": 50,
        "_links": {"next": "/rest/api/content?next"} if more else {},
    }


class _Provider:
    def __init__(self, listing: ProviderListing, read_status: int | None = None) -> None:
        self.listing = listing
        self.read_status = read_status
        self.requests: list[str] = []
        self.http = FakeConnectorHttp(self.handle)

    def connection(self, gene: ConfluenceGene) -> FakeConnection:
        return FakeConnection(gene=gene, requests=self.requests, clients=self.http.clients, waits=self.http.waits)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url}")
        path = request.url.path.removeprefix("/wiki/rest/api")
        params = request.url.params
        if path == "/space":
            return httpx.Response(200, json=_results([], 0))
        if path == "/content" or path.endswith("/child/page"):
            return self._listing(path, params)
        page_id = path.removeprefix("/content/")
        if self.read_status is not None:
            if self.read_status == 200:
                return httpx.Response(200, json=_page("101"))
            return httpx.Response(self.read_status, json={}, headers=_wait_headers(self.read_status))
        if page_id not in PAGES:
            return httpx.Response(404, json={"message": "No content found"})
        return httpx.Response(200, json=_page(page_id))

    def _listing(self, path: str, params: httpx.QueryParams) -> httpx.Response:
        if self.listing is ProviderListing.FAILED:
            return httpx.Response(500, json={"message": "Internal server error"})
        start = int(params.get("start", 0))
        pages = {page_id: page for page_id, page in PAGES.items() if page_id not in BLOG_POSTS}
        if path == "/content":
            assert params.get("type") == "page"
            members = [page_id for page_id, (space, *_rest) in pages.items() if space == params.get("spaceKey")]
        else:
            parent = path.removeprefix("/content/").removesuffix("/child/page")
            members = [page_id for page_id, (_space, page_parent, *_rest) in pages.items() if page_parent == parent]
        if self.listing is ProviderListing.TRUNCATED and members:
            # The first page claims more; the next one returns nothing.
            return httpx.Response(
                200, json=_results([_page_header(members[0])] if start == 0 else [], start, more=True)
            )
        return httpx.Response(200, json=_results([_page_header(page_id) for page_id in members[start:]], start))


class ConfluenceFakes(DocumentConnectorFakes):
    # 30 s in seconds on 429; an HTTP date 30 s ahead on 503, read to the whole second.
    retry_waits = {429: (30.0, 30.0), 503: (29.0, 30.0)}
    listing_configs = {
        "page-tree": ListingConfig(
            {**_CONNECTION, "page_tree_root": "100", "include_children": True},
            walks_scope=True,
            edited_in_the_cursor_minute="confluence-104",
        ),
        "space": ListingConfig(
            {**_CONNECTION, "sync_mode": "space", "spaces": ["PAY"]},
            walks_scope=True,
            edited_in_the_cursor_minute="confluence-104",
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
        provider = _Provider(ProviderListing.COMPLETE)
        gene = await _authenticated(dict(self.listing_configs["page-tree"].config), provider)
        provider.read_status = status
        try:
            yield provider.connection(gene), [_stored_page("101")]
        finally:
            await gene.close()

    def rediscover_candidates(self, config: Mapping[str, object]) -> list[ContentItem]:
        return [_stored_page(page_id) for page_id in [*PAGES, GONE_PAGE]]


async def _authenticated(config: dict, provider: _Provider) -> ConfluenceGene:
    gene = ConfluenceGene(config=config, source_id="src-contract")
    with provider.http.serving(confluence_gene):
        await gene.authenticate()
    provider.requests.clear()
    return gene


def _wait_headers(status: int) -> dict[str, str]:
    """A wait in seconds on 429 and as an HTTP date on 503; the stub sleep makes both instant."""
    if status == httpx.codes.TOO_MANY_REQUESTS:
        return {"Retry-After": "30"}
    if status == httpx.codes.SERVICE_UNAVAILABLE:
        return {"Retry-After": format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)}
    return {}


def _stored_page(page_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"confluence-{page_id}",
        title=f"Page {page_id}",
        source_url=f"{BASE_URL}/pages/{page_id}",
        last_modified=LONG_AGO,
        extra={"page_id": page_id},
    )


FAKES = ConfluenceFakes()
