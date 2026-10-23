"""A fake GitHub Enterprise repository: one branch whose tree holds docs and code.

The configuration selects Markdown under ``docs/``: ``docs/guide.md`` and
``docs/runbooks/restart.md``. ``src/app.py`` and ``docs/diagram.png`` are in
the tree but outside the scope. A truncated listing is a tree GitHub
truncates whose root holds more folders than a walk may read; a failed one
answers the tree with a server error.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

import httpx

from memforge.genes import github_repo_gene
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.toolkit.github_api import TREE_REQUEST_BUDGET
from tests.contracts.document_fakes import (
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)
from tests.github_api_fixture import FakeGitHub

REPO_URL = "https://github.example.test/payroll/architecture"
API = "https://github.example.test/api/v3/repos/payroll/architecture"
FILES = {
    "docs/guide.md": b"# Guide\n",
    "docs/runbooks/restart.md": b"# Restart\n",
    "docs/diagram.png": b"\x89PNG\r\n",
    "src/app.py": b"print('app')\n",
}
# More root folders than one walk may read, each a complete subtree.
_BEYOND_THE_BUDGET = {f"vendor-{index:04}/README.txt": b"vendored\n" for index in range(TREE_REQUEST_BUDGET)}
_SCOPE = {"repo_url": REPO_URL, "ref": "main", "include_paths": ["docs/"], "include_extensions": ["md"]}


class _Provider(FakeGitHub):
    def __init__(self, listing: ProviderListing) -> None:
        if listing is ProviderListing.TRUNCATED:
            super().__init__(API, files={**FILES, **_BEYOND_THE_BUDGET}, truncated={""})
        elif listing is ProviderListing.FAILED:
            # The tree fails whether it is asked for by its sha or by the branch.
            failure = httpx.Response(500, json={"message": "Server Error"})
            answers = {"/git/trees/tree-main?recursive=1": failure, "/git/trees/main?recursive=1": failure}
            super().__init__(API, files=FILES, answers=answers)
        else:
            super().__init__(API, files=FILES)

    def connection(self, gene: GitHubRepoGene) -> FakeConnection:
        return FakeConnection(gene=gene, requests=self.requests, clients=self.http.clients, waits=self.http.waits)


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
        with provider.http.serving(github_repo_gene):
            await gene.authenticate()
        try:
            yield provider.connection(gene)
        finally:
            await gene.close()

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
