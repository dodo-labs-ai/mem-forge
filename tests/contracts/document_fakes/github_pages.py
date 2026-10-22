"""A fake GitHub Pages site, served from a repository branch and from a published sitemap.

Both read a handbook whose subtree ``guides`` holds the pages ``setup``, changed
long ago, and ``deploy``, changed recently; ``other/faq`` is outside the
subtree. The repository-backed site answers the GitHub REST API, and the
public site answers ``sitemap.xml`` and each page.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from memforge.genes.github_pages_gene import GitHubPagesGene
from tests.contracts.document_fakes import (
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)

# page path -> changed at
PAGES: dict[str, datetime] = {"guides/setup": LONG_AGO, "guides/deploy": RECENTLY, "other/faq": RECENTLY}

REPO_SITE = "https://github.example.test/pages/org/handbook"
API = "https://github.example.test/api/v3/repos/org/handbook"
PUBLIC_SITE = "https://docs.example.test/handbook"


class _Response:
    def __init__(self, url: str, *, payload: object = None, text: str = "", status_code: int = 200) -> None:
        self.url = url
        self._payload = payload
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.history: list[object] = []

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code} for {self.url}")


class _Provider:
    def __init__(self, listing: ProviderListing) -> None:
        self.listing = listing
        self.requests: list[str] = []

    def client(self, **_kwargs) -> _Client:
        return _Client(self)

    def get(self, url: str) -> _Response:
        self.requests.append(f"GET {url}")
        if url.startswith(API):
            return self._api(url)
        if url == f"{PUBLIC_SITE}/sitemap.xml":
            return self._sitemap(url)
        page_path = url.removeprefix(f"{PUBLIC_SITE}/")
        if page_path in PAGES:
            return _Response(url, text=f"<main><h1>{page_path}</h1><p>Body</p></main>")
        return _Response(url, text="missing", status_code=404)

    def _api(self, url: str) -> _Response:
        if url == API:
            return _Response(url, payload={"default_branch": "main"})
        if url == f"{API}/git/trees/main?recursive=1":
            if self.listing is ProviderListing.FAILED:
                return _Response(url, payload={"message": "Server Error"}, status_code=500)
            tree = [
                {"path": f"docs/{page_path}.md", "type": "blob", "sha": f"sha-{page_path}", "mode": "100644"}
                for page_path in PAGES
            ] + [{"path": "docs/guides/diagram.png", "type": "blob", "sha": "sha-diagram", "mode": "100644"}]
            return _Response(url, payload={"truncated": self.listing is ProviderListing.TRUNCATED, "tree": tree})
        if url.startswith(f"{API}/commits?"):
            page_path = parse_qs(urlsplit(url).query)["path"][0].removeprefix("docs/").removesuffix(".md")
            return _Response(url, payload=[{"commit": {"committer": {"date": PAGES[page_path].isoformat()}}}])
        return _Response(url, payload={"message": "Not Found"}, status_code=404)

    def _sitemap(self, url: str) -> _Response:
        if self.listing is ProviderListing.FAILED:
            return _Response(url, text="error", status_code=500)
        if self.listing is ProviderListing.TRUNCATED:
            # An index of further sitemaps is not the inventory itself.
            return _Response(
                url,
                text=(
                    '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                    f"<sitemap><loc>{PUBLIC_SITE}/sitemap-1.xml</loc></sitemap></sitemapindex>"
                ),
            )
        entries = "".join(
            f"<url><loc>{PUBLIC_SITE}/{page_path}</loc><lastmod>{changed_at.isoformat()}</lastmod></url>"
            for page_path, changed_at in PAGES.items()
        )
        return _Response(url, text=f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>')


class _Client:
    def __init__(self, provider: _Provider) -> None:
        self._provider = provider

    async def get(self, url: str, **_kwargs) -> _Response:
        return self._provider.get(url)

    async def aclose(self) -> None:
        return None


class GitHubPagesFakes(DocumentConnectorFakes):
    listing_configs = {
        "branch-tree": ListingConfig(
            {"auth_mode": "github_pat", "pat": "token", "sync_mode": "subtree", "root_url": f"{REPO_SITE}/guides/"},
            walks_scope=True,
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
        ),
    }
    non_listing_configs = {
        "single-page": {"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{PUBLIC_SITE}/guides/setup"},
        "explicit-list": {
            "auth_mode": "none",
            "sync_mode": "explicit_list",
            "pages": [f"{PUBLIC_SITE}/guides/setup", f"{PUBLIC_SITE}/guides/deploy"],
        },
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
        with patch("memforge.genes.github_pages_gene._RequestsAsyncClient", provider.client):
            await gene.authenticate()
        yield FakeConnection(gene=gene, requests=provider.requests)


FAKES = GitHubPagesFakes()
