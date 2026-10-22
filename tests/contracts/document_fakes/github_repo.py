"""A fake GitHub Enterprise repository: one branch whose tree holds docs and code.

The configuration selects Markdown under ``docs/``: ``docs/guide.md`` and
``docs/runbooks/restart.md``. ``src/app.py`` and ``docs/diagram.png`` are in
the tree but outside the scope.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from unittest.mock import patch

from memforge.genes.github_repo_gene import GitHubRepoGene
from tests.contracts.document_fakes import (
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)

REPO_URL = "https://github.example.test/payroll/architecture"
API = "https://github.example.test/api/v3/repos/payroll/architecture"
TREE = [
    {"path": "docs", "type": "tree", "sha": "tree-docs", "mode": "040000"},
    {"path": "docs/guide.md", "type": "blob", "sha": "sha-guide", "mode": "100644", "size": 10},
    {"path": "docs/runbooks", "type": "tree", "sha": "tree-runbooks", "mode": "040000"},
    {"path": "docs/runbooks/restart.md", "type": "blob", "sha": "sha-restart", "mode": "100644", "size": 10},
    {"path": "docs/diagram.png", "type": "blob", "sha": "sha-diagram", "mode": "100644", "size": 10},
    {"path": "src/app.py", "type": "blob", "sha": "sha-app", "mode": "100644", "size": 10},
]
_SCOPE = {"repo_url": REPO_URL, "ref": "main", "include_paths": ["docs/"], "include_extensions": ["md"]}


class _Response:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"GitHub answered HTTP {self.status_code}")


class _Provider:
    def __init__(self, listing: ProviderListing) -> None:
        self.listing = listing
        self.requests: list[str] = []

    def client(self, **_kwargs) -> _Client:
        return _Client(self)

    def get(self, url: str) -> _Response:
        self.requests.append(f"GET {url}")
        if url == API:
            return _Response({"default_branch": "main"})
        if url == f"{API}/commits/main":
            return _Response({"sha": "commit-main", "commit": {"tree": {"sha": "tree-main"}}})
        if url == f"{API}/git/trees/tree-main?recursive=1":
            if self.listing is ProviderListing.FAILED:
                return _Response({"message": "Server Error"}, status_code=500)
            return _Response({"truncated": self.listing is ProviderListing.TRUNCATED, "tree": TREE})
        return _Response({"message": "Not Found"}, status_code=404)


class _Client:
    def __init__(self, provider: _Provider) -> None:
        self._provider = provider

    async def get(self, url: str) -> _Response:
        return self._provider.get(url)

    async def aclose(self) -> None:
        return None


class GitHubRepoFakes(DocumentConnectorFakes):
    listing_configs = {"cloud-pull": ListingConfig({**_SCOPE, "connection_mode": "cloud_pull"}, walks_scope=True)}
    device_configs = {"local-push": {**_SCOPE, "connection_mode": "local_push"}}

    @asynccontextmanager
    async def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AsyncIterator[FakeConnection]:
        provider = _Provider(listing)
        gene = GitHubRepoGene(config=dict(config), source_id="src-contract")
        with patch("memforge.genes.github_repo_gene._RequestsAsyncClient", provider.client):
            await gene.authenticate()
        yield FakeConnection(gene=gene, requests=provider.requests)

    def manifest_entries(self, config: Mapping[str, object]) -> list[dict]:
        return [
            {
                "doc_id": f"github-repo-{path}",
                "repo_url": REPO_URL,
                "repo_ref": "main",
                "relative_path": path,
                "title": path,
                "version": f"sha-{path}",
                "last_modified": RECENTLY.isoformat(),
                "change_kind": "upsert",
            }
            for path in ("docs/guide.md", "docs/runbooks/restart.md")
        ]


FAKES = GitHubRepoFakes()
