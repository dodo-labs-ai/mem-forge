"""A fake GitHub Pages site, served from a repository branch and from a published sitemap.

Both read a handbook whose subtree ``guides`` holds the pages ``setup``, changed
long ago, and ``deploy``, changed in the cursor's minute; ``other/faq`` is
outside the subtree. The repository-backed site answers the GitHub REST API,
and the public site answers ``sitemap.xml`` and each page. A truncated listing
is a branch tree GitHub truncates whose root holds more folders than a walk
may read, a sitemap index instead of the urlset, and ``deploy`` no longer
exists; a failed one answers 500.

A page's time is its last commit in the repository, its sitemap ``lastmod``
with seconds and an offset, or its HTTP ``Last-Modified``, which has whole
seconds; discovery compares each with ``since`` itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from email.utils import format_datetime

import httpx

from memforge.genes import github_pages_gene
from memforge.genes.github_pages_gene import GitHubPagesGene, _page_doc_id
from memforge.genes.toolkit.github_api import TREE_REQUEST_BUDGET
from tests.contracts.document_fakes import (
    IN_THE_CURSOR_MINUTE,
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)
from tests.connector_http_fixture import FakeConnectorHttp
from tests.github_api_fixture import FakeGitHub

# page path -> changed at
PAGES: dict[str, datetime] = {"guides/setup": LONG_AGO, "guides/deploy": IN_THE_CURSOR_MINUTE, "other/faq": RECENTLY}

REPO_SITE = "https://github.example.test/pages/org/handbook"
API = "https://github.example.test/api/v3/repos/org/handbook"
PUBLIC_SITE = "https://docs.example.test/handbook"

_FILES = {f"docs/{page_path}.md": f"# {page_path}\n".encode() for page_path in PAGES} | {
    "docs/guides/diagram.png": b"\x89PNG\r\n",
}
# More root folders than one walk may read, each a complete subtree.
_BEYOND_THE_BUDGET = {f"vendor-{index:04}/README.txt": b"vendored\n" for index in range(TREE_REQUEST_BUDGET)}
_COMMIT_TIMES = {f"docs/{page_path}.md": changed_at.isoformat() for page_path, changed_at in PAGES.items()}


class _Provider:
    def __init__(self, listing: ProviderListing) -> None:
        self.listing = listing
        if listing is ProviderListing.TRUNCATED:
            files, truncated, answers = {**_FILES, **_BEYOND_THE_BUDGET}, {""}, {}
        elif listing is ProviderListing.FAILED:
            files, truncated = _FILES, set()
            answers = {"/git/trees/main?recursive=1": httpx.Response(500, json={"message": "Server Error"})}
        else:
            files, truncated, answers = _FILES, set(), {}
        self.github = FakeGitHub(API, files=files, truncated=truncated, commit_times=_COMMIT_TIMES, answers=answers)
        self.requests = self.github.requests
        self.http = FakeConnectorHttp(self.handle)

    def connection(self, gene: GitHubPagesGene) -> FakeConnection:
        return FakeConnection(gene=gene, requests=self.requests, clients=self.http.clients, waits=self.http.waits)

    def handle(self, request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(API):
            return self.github.handle(request)
        self.requests.append(f"{request.method} {request.url}")
        url = str(request.url)
        if url == f"{PUBLIC_SITE}/sitemap.xml":
            return self._sitemap()
        page_path = url.removeprefix(f"{PUBLIC_SITE}/")
        if self.listing is ProviderListing.FAILED:
            return httpx.Response(500, text="error")
        if page_path in PAGES and not (self.listing is ProviderListing.TRUNCATED and page_path == "guides/deploy"):
            return httpx.Response(
                200,
                text=f"<main><h1>{page_path}</h1><p>Body</p></main>",
                headers={"Last-Modified": format_datetime(PAGES[page_path], usegmt=True)},
            )
        return httpx.Response(404, text="missing")

    def _sitemap(self) -> httpx.Response:
        if self.listing is ProviderListing.FAILED:
            return httpx.Response(500, text="error")
        if self.listing is ProviderListing.TRUNCATED:
            # An index of further sitemaps is not the inventory itself.
            return httpx.Response(
                200,
                text=(
                    '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    f"<sitemap><loc>{PUBLIC_SITE}/sitemap-1.xml</loc></sitemap></sitemapindex>"
                ),
            )
        entries = "".join(
            f"<url><loc>{PUBLIC_SITE}/{page_path}</loc><lastmod>{changed_at.isoformat()}</lastmod></url>"
            for page_path, changed_at in PAGES.items()
        )
        return httpx.Response(200, text=f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>')


class GitHubPagesFakes(DocumentConnectorFakes):
    listing_configs = {
        "branch-tree": ListingConfig(
            {"auth_mode": "github_pat", "pat": "token", "sync_mode": "subtree", "root_url": f"{REPO_SITE}/guides/"},
            walks_scope=True,
            edited_in_the_cursor_minute=_page_doc_id(f"{REPO_SITE}/guides/deploy"),
        ),
        "authoritative-sitemap": ListingConfig(
            {
                "base_url": f"{PUBLIC_SITE}/",
                "auth_mode": "none",
                "sync_mode": "subtree",
                "root_url": f"{PUBLIC_SITE}/guides/",
                "sitemap_authoritative": True,
            },
            walks_scope=True,
            edited_in_the_cursor_minute=_page_doc_id(f"{PUBLIC_SITE}/guides/deploy"),
        ),
        "single-page": ListingConfig(
            {
                "base_url": f"{PUBLIC_SITE}/",
                "auth_mode": "none",
                "sync_mode": "single_page",
                "page_url": f"{PUBLIC_SITE}/guides/deploy",
            },
            walks_scope=True,
            edited_in_the_cursor_minute=_page_doc_id(f"{PUBLIC_SITE}/guides/deploy"),
        ),
        "explicit-list": ListingConfig(
            {
                "base_url": f"{PUBLIC_SITE}/",
                "auth_mode": "none",
                "sync_mode": "explicit_list",
                "pages": [f"{PUBLIC_SITE}/guides/setup", f"{PUBLIC_SITE}/guides/deploy"],
            },
            walks_scope=True,
            edited_in_the_cursor_minute=_page_doc_id(f"{PUBLIC_SITE}/guides/deploy"),
        ),
    }
    non_listing_configs = {
        "crawl-or-sitemap": {
            "base_url": f"{PUBLIC_SITE}/",
            "auth_mode": "none",
            "sync_mode": "subtree",
            "root_url": f"{PUBLIC_SITE}/guides/",
        },
    }

    @asynccontextmanager
    async def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AsyncIterator[FakeConnection]:
        provider = _Provider(listing)
        gene = GitHubPagesGene(config=dict(config), source_id="src-contract")
        with provider.http.serving(github_pages_gene):
            await gene.authenticate()
        try:
            yield provider.connection(gene)
        finally:
            await gene.close()


FAKES = GitHubPagesFakes()
