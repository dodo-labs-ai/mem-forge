"""Every path that creates a Gene closes it when done, after a failure as after a success."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from memforge import runtime
from memforge.config import AppConfig
from memforge.storage.database import Database
from memforge.genes.base import SourceConfigurationError
from memforge.models import ContentItem, SyncState
from tests.contracts.document_fakes import ProviderListing
from tests.contracts.document_fakes import confluence as confluence_fakes
from tests.contracts.document_fakes import github_pages as github_pages_fakes
from tests.contracts.document_fakes import github_repo as github_repo_fakes
from tests.contracts.document_fakes import jira as jira_fakes
from tests.gene_fixture import FakeGene
from tests.source_type_fixture import register_test_source_type


class SpyClient:
    """A provider client that records whether it was closed."""

    def __init__(self) -> None:
        self.is_closed = False

    async def aclose(self) -> None:
        self.is_closed = True


class ClientGene(FakeGene):
    """A Gene that opens a client in ``authenticate``, and fails wherever a test says."""

    built: list[ClientGene] = []
    fail_authenticate = False
    fail_discover = False

    def __init__(self, config: dict | None = None, source_id: str = "src-fake") -> None:
        super().__init__(config, source_id)
        self.clients: list[SpyClient] = []
        ClientGene.built.append(self)

    async def authenticate(self) -> None:
        await self.close()
        self._client = SpyClient()
        self.clients.append(self._client)
        if self.fail_authenticate:
            raise RuntimeError("the provider is unreachable")

    async def discover(self, since=None):
        if self.fail_discover:
            raise RuntimeError("the provider answered 500")
        yield ContentItem(
            item_id="doc-1",
            title="Doc 1",
            source_url="https://docs.example.test/1",
            last_modified=datetime(2026, 5, 20, tzinfo=timezone.utc),
        )


@pytest.fixture(autouse=True)
def fresh_client_genes(monkeypatch):
    monkeypatch.setattr(ClientGene, "built", [])
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})


def _closed(gene: ClientGene) -> bool:
    return bool(gene.clients) and all(client.is_closed for client in gene.clients)


# ---------------------------------------------------------------------------
# Gene.close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_releases_the_client_once_and_logs_it(caplog) -> None:
    gene = ClientGene()
    await gene.close()  # before authenticate: nothing to release
    await gene.authenticate()

    with caplog.at_level(logging.DEBUG, logger="memforge.genes.base"):
        await gene.close()
        await gene.close()

    assert _closed(gene)
    assert gene._client is None
    assert [record.getMessage() for record in caplog.records] == ["Closed the provider client"]


# ---------------------------------------------------------------------------
# authenticate replaces its client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fakes", "client_class"),
    [
        (confluence_fakes, "httpx.AsyncClient"),
        (jira_fakes, "httpx.AsyncClient"),
        (github_repo_fakes, "memforge.genes.github_repo_gene._RequestsAsyncClient"),
        (github_pages_fakes, "memforge.genes.github_pages_gene._RequestsAsyncClient"),
    ],
    ids=["confluence", "jira", "github_repo", "github_pages"],
)
async def test_authenticating_again_closes_the_previous_client(fakes, client_class) -> None:
    provider = fakes._Provider(ProviderListing.COMPLETE)
    config = dict(next(iter(fakes.FAKES.listing_configs.values())).config)
    gene = runtime.create_gene(name=fakes.__name__.rsplit(".", 1)[-1], config=config, source_id="src-close")
    with (
        patch(client_class, provider.client),
        patch("memforge.genes.confluence_gene.CONFLUENCE_REQUEST_INTERVAL_SECONDS", 0.0),
    ):
        await gene.authenticate()
        await gene.authenticate()
    try:
        assert [client.is_closed for client in provider.clients] == [True, False]
    finally:
        await gene.close()
    assert all(client.is_closed for client in provider.clients)


@pytest.mark.asyncio
async def test_a_closed_github_repository_gene_authenticates_again_when_it_lists() -> None:
    provider = github_repo_fakes._Provider(ProviderListing.COMPLETE)
    config = dict(github_repo_fakes.FAKES.listing_configs["cloud-pull"].config)
    gene = runtime.create_gene(name="github_repo", config=config, source_id="src-close")
    with patch("memforge.genes.github_repo_gene._RequestsAsyncClient", provider.client):
        await gene.authenticate()
        await gene.close()
        listing = await gene.list_scope()
    await gene.close()

    assert listing.doc_ids
    assert [client.is_closed for client in provider.clients] == [True, True]


# ---------------------------------------------------------------------------
# run_source_sync
# ---------------------------------------------------------------------------


class _Orchestrator:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.genes: list = []

    def orchestrator(self):
        return self

    async def sync_gene(self, **kwargs):
        gene = kwargs["gene"]
        self.genes.append(gene)
        await gene.authenticate()
        if self.error is not None:
            raise self.error
        return SyncState(source=kwargs["source_id"], last_sync_status="success")


@pytest.fixture
async def database(tmp_path):
    db = Database(str(tmp_path / "runtime.db"))
    await db.connect()
    await db.upsert_source(
        id="src-close", type="closing", name="Closing", config_json=json.dumps({}),
        access_policy="workspace", owner_user_id="dev",
    )
    yield db
    await db.close()


async def _run(database, tmp_path, orchestrator):
    return await runtime.run_source_sync(
        db=database,
        config=AppConfig(base_dir=tmp_path / "mem"),
        source={"id": "src-close", "type": "closing", "name": "Closing", "config": {}},
        runtime=orchestrator,
    )


@pytest.mark.asyncio
async def test_run_source_sync_closes_the_gene_after_a_successful_run(database, tmp_path, monkeypatch) -> None:
    register_test_source_type(monkeypatch, "closing", ClientGene)
    orchestrator = _Orchestrator()

    state = await _run(database, tmp_path, orchestrator)

    assert state.last_sync_status == "success"
    assert [_closed(gene) for gene in orchestrator.genes] == [True]


@pytest.mark.asyncio
async def test_run_source_sync_closes_the_gene_after_a_failed_run(database, tmp_path, monkeypatch) -> None:
    register_test_source_type(monkeypatch, "closing", ClientGene)
    orchestrator = _Orchestrator(error=RuntimeError("the provider answered 500"))

    with pytest.raises(RuntimeError, match="answered 500"):
        await _run(database, tmp_path, orchestrator)

    assert [_closed(gene) for gene in orchestrator.genes] == [True]


@pytest.mark.asyncio
async def test_run_source_sync_reports_a_gene_that_cannot_be_built(database, tmp_path, monkeypatch) -> None:
    def unbuildable(*, name, config, source_id):
        raise ValueError("the configuration is incomplete")

    monkeypatch.setattr(runtime, "create_gene", unbuildable)

    with pytest.raises(ValueError, match="configuration is incomplete"):
        await _run(database, tmp_path, _Orchestrator())


# ---------------------------------------------------------------------------
# Excluded-scope Genes
# ---------------------------------------------------------------------------

_JIRA = {"id": "src-close", "type": "jira"}
_ADVANCED = {"base_url": "https://jira.example.test", "auth_mode": "pat", "query_mode": "advanced", "jql": "x"}
_EXCLUDED = {"query_mode": "simple", "projects": ["OPS"]}


@pytest.mark.asyncio
async def test_an_excluded_scope_gene_is_closed_when_its_block_exits(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "create_gene", lambda *, name, config, source_id: ClientGene(config, source_id))
    factory = runtime.excluded_scope_gene_factory(_JIRA, _ADVANCED)

    async with factory(_EXCLUDED) as gene:
        assert not _closed(gene)

    assert _closed(gene)
    assert gene.config["projects"] == ["OPS"]


@pytest.mark.asyncio
async def test_an_excluded_scope_gene_is_closed_when_authenticate_fails(monkeypatch) -> None:
    monkeypatch.setattr(ClientGene, "fail_authenticate", True)
    monkeypatch.setattr(runtime, "create_gene", lambda *, name, config, source_id: ClientGene(config, source_id))
    factory = runtime.excluded_scope_gene_factory(_JIRA, _ADVANCED)

    with pytest.raises(RuntimeError, match="unreachable") as raised:
        async with factory(_EXCLUDED):
            pytest.fail("the block runs only for an authenticated Gene")

    assert not isinstance(raised.value, SourceConfigurationError)
    assert [_closed(gene) for gene in ClientGene.built] == [True]


# ---------------------------------------------------------------------------
# Admin discovery preview and GitHub folder browse
# ---------------------------------------------------------------------------


def _admin_client(database, tmp_path) -> TestClient:
    from memforge.server.admin_api import create_admin_app

    config = AppConfig(base_dir=tmp_path / "mem")
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return TestClient(create_admin_app(db=database, config=config))


@pytest.mark.asyncio
@pytest.mark.parametrize(("fail_discover", "status"), [(False, 200), (True, 502)], ids=["success", "failure"])
async def test_the_discovery_preview_closes_its_gene(database, tmp_path, monkeypatch, fail_discover, status) -> None:
    register_test_source_type(monkeypatch, "closing", ClientGene)
    monkeypatch.setattr(ClientGene, "fail_discover", fail_discover)

    with _admin_client(database, tmp_path) as client:
        response = client.post("/api/v1/genes/closing/preview-discovery", json={"config": {}, "limit": 5})

    assert response.status_code == status
    assert [_closed(gene) for gene in ClientGene.built] == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listing", "status"), [(ProviderListing.COMPLETE, 200), (ProviderListing.FAILED, 400)], ids=["success", "failure"]
)
async def test_the_github_folder_browse_closes_its_gene(database, tmp_path, listing, status) -> None:
    provider = github_repo_fakes._Provider(listing)
    # Browsing reads the tree of the ref it is given; the fake serves the branch's tree by its sha.
    config = {**github_repo_fakes.FAKES.listing_configs["cloud-pull"].config, "ref": "tree-main"}

    with (
        patch("memforge.genes.github_repo_gene._RequestsAsyncClient", provider.client),
        _admin_client(database, tmp_path) as client,
    ):
        response = client.post("/api/v1/genes/github_repo/browse-tree", json={"config": config, "limit": 50})

    assert response.status_code == status, response.text
    assert [client.is_closed for client in provider.clients] == [True]
