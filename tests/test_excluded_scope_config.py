"""An excluded scope is listed with the Source's current configuration and credential, and only that scope's fields."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from memforge import runtime
from memforge.config import AppConfig
from memforge.genes.base import SourceConfigurationError
from memforge.genes.confluence_gene import ConfluenceGene
from memforge.genes.jira_gene import JiraGene, JiraQueryRejected
from memforge.models import SyncState
from memforge.source_projection_config import canonical_projection_scope, excluded_scope_config
from memforge.storage.database import Database


@pytest.mark.parametrize(
    ("source_type", "current", "excluded", "expected"),
    [
        (
            "confluence",
            {"base_url": "https://wiki.example", "pat": "token", "sync_mode": "page_tree", "page_tree_root": "42",
             "include_children": True, "exclude_labels": ["draft"]},
            {"sync_mode": "space", "spaces": ["PAY"]},
            {"base_url": "https://wiki.example", "pat": "token", "sync_mode": "space", "spaces": ["PAY"]},
        ),
        (
            "jira",
            {"base_url": "https://jira.example", "pat": "token", "query_mode": "advanced", "jql": "project = PAY"},
            {"query_mode": "simple", "projects": ["OPS", "PAY"], "jql_filter": "updated >= -90d"},
            {"base_url": "https://jira.example", "pat": "token", "query_mode": "simple", "projects": ["OPS", "PAY"],
             "jql_filter": "updated >= -90d"},
        ),
        (
            "jira",
            {"base_url": "https://jira.example", "auth_mode": "browser_cookie", "jira_cookie": "session=1",
             "query_mode": "simple", "projects": ["PAY"], "issue_types": ["Bug"], "jql_filter": "x"},
            {"query_mode": "advanced", "jql": "project in (PAY, OPS)"},
            {"base_url": "https://jira.example", "auth_mode": "browser_cookie", "jira_cookie": "session=1",
             "query_mode": "advanced", "jql": "project in (PAY, OPS)"},
        ),
        (
            "github_repo",
            {"repo_url": "https://github.example/org/repo", "pat": "token", "ref": "release", "include_paths": ["docs"],
             "include_extensions": ["md"]},
            {"ref": "main", "include_extensions": ["md", "txt"]},
            {"repo_url": "https://github.example/org/repo", "pat": "token", "ref": "main",
             "include_extensions": ["md", "txt"]},
        ),
        (
            "github_pages",
            {"base_url": "https://pages.example/site/", "auth_mode": "github_pat", "pat": "token",
             "sync_mode": "explicit_list", "pages": ["https://pages.example/site/a/"]},
            {"sync_mode": "subtree", "root_url": "https://pages.example/site/guides"},
            {"base_url": "https://pages.example/site/", "auth_mode": "github_pat", "pat": "token",
             "sync_mode": "subtree", "root_url": "https://pages.example/site/guides"},
        ),
    ],
    ids=["confluence-tree-to-space", "jira-advanced-to-simple", "jira-simple-to-advanced", "github-ref", "pages-mode"],
)
def test_an_excluded_scope_config_keeps_credentials_and_takes_only_the_excluded_scope_fields(
    source_type, current, excluded, expected,
):
    assert excluded_scope_config(source_type, current, excluded) == expected


def test_an_excluded_scope_stored_before_a_scope_default_reads_that_default():
    # A stored Jira scope from before ``query_mode`` existed is a simple query.
    config = excluded_scope_config(
        "jira", {"base_url": "https://jira.example", "query_mode": "advanced", "jql": "x"}, {"projects": ["PAY"]},
    )

    assert config == {"base_url": "https://jira.example", "query_mode": "simple", "projects": ["PAY"]}
    assert canonical_projection_scope("jira", config) == {"query_mode": "simple", "projects": ["PAY"]}


@pytest.mark.asyncio
async def test_the_run_builds_excluded_genes_from_its_own_configuration_and_authenticates_them(monkeypatch):
    built: list[dict] = []

    class RecordingGene:
        def __init__(self, config):
            self.config = config
            self.authenticated = False
            self.closed = False

        async def authenticate(self):
            self.authenticated = True

        async def close(self):
            self.closed = True

    def create_gene(*, name, config, source_id):
        assert (name, source_id) == ("jira", "src-jira")
        built.append(config)
        return RecordingGene(config)

    monkeypatch.setattr(runtime, "create_gene", create_gene)
    # The run's configuration is decrypted and carries the injected browser session.
    run_config = {
        "base_url": "https://jira.example",
        "auth_mode": "browser_cookie",
        "jira_cookie": "session=1",
        "query_mode": "advanced",
        "jql": "project = PAY",
    }
    factory = runtime.excluded_scope_gene_factory({"id": "src-jira", "type": "jira"}, run_config)

    async with factory({"query_mode": "simple", "projects": ["OPS"]}) as gene:
        assert gene.authenticated
        assert not gene.closed

    assert gene.closed
    assert built == [
        {
            "base_url": "https://jira.example",
            "auth_mode": "browser_cookie",
            "jira_cookie": "session=1",
            "query_mode": "simple",
            "projects": ["OPS"],
        }
    ]
    assert run_config["jql"] == "project = PAY", "the run's own configuration is untouched"


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "rejected"), [(400, True), (503, False)])
async def test_a_jql_jira_refuses_is_a_configuration_error_and_an_outage_is_not(status, rejected):
    gene = JiraGene(config={"base_url": "https://jira.example.test", "projects": ["PAY"]}, source_id="src-jira")
    gene._base_url = "https://jira.example.test"
    request = httpx.Request("POST", "https://jira.example.test/rest/api/2/search")
    gene._request = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "search failed", request=request, response=httpx.Response(status, request=request, text="bad JQL"),
        )
    )

    with pytest.raises(Exception) as raised:
        _ = [issue async for issue in gene._search_issues("project = ", fields=["summary"], expand=None, max_results=50)]

    assert isinstance(raised.value, JiraQueryRejected) is rejected
    assert isinstance(raised.value, SourceConfigurationError) is rejected


@pytest.mark.asyncio
async def test_an_excluded_scope_that_cannot_be_built_is_a_configuration_error_and_authentication_is_not(
    monkeypatch,
):
    def unbuildable(*, name, config, source_id):
        raise ValueError("Confluence Page Tree Root is required")

    monkeypatch.setattr(runtime, "create_gene", unbuildable)
    factory = runtime.excluded_scope_gene_factory({"id": "src", "type": "jira"}, {"query_mode": "advanced", "jql": "x"})
    with pytest.raises(SourceConfigurationError, match="cannot be configured"):
        async with factory({"query_mode": "simple", "projects": ["OPS"]}):
            pass

    class OfflineGene:
        async def authenticate(self):
            raise ValueError("Jira base URL must use HTTPS")

        async def close(self):
            return None

    monkeypatch.setattr(runtime, "create_gene", lambda **_kwargs: OfflineGene())
    with pytest.raises(ValueError) as raised:
        async with factory({"query_mode": "simple", "projects": ["OPS"]}):
            pass
    assert not isinstance(raised.value, SourceConfigurationError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [{"sync_mode": "page_tree"}, {"sync_mode": "space"}],
    ids=["page-tree-without-root", "space-without-spaces"],
)
async def test_a_confluence_scope_without_its_root_or_spaces_is_a_configuration_error(config):
    gene = ConfluenceGene(config={"base_url": "https://wiki.example.com", **config}, source_id="src-confluence")

    with pytest.raises(SourceConfigurationError, match="is required"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_run_source_sync_hands_the_orchestrator_a_factory_for_excluded_scope_genes(tmp_path, monkeypatch):
    database = Database(str(tmp_path / "runtime.db"))
    await database.connect()
    received: dict = {}

    class CapturingRuntime:
        def orchestrator(self):
            return self

        async def sync_gene(self, **kwargs):
            received.update(kwargs)
            return SyncState(source=kwargs["source_id"], last_sync_status="failed")

    built: list[dict] = []

    class RecordingGene:
        def __init__(self, config):
            self.config = config

        async def authenticate(self):
            return None

        async def close(self):
            return None

    def create_gene(*, name, config, source_id):
        built.append(dict(config))
        return RecordingGene(config)

    monkeypatch.setattr(runtime, "create_gene", create_gene)
    source_config = {"base_url": "https://jira.example", "auth_mode": "pat", "query_mode": "advanced", "jql": "x"}
    try:
        await database.upsert_source(
            id="src-jira", type="jira", name="Jira", config_json=json.dumps(source_config),
            access_policy="workspace", owner_user_id="dev",
        )
        await runtime.run_source_sync(
            db=database,
            config=AppConfig(base_dir=tmp_path / "mem"),
            source={"id": "src-jira", "type": "jira", "name": "Jira", "config": source_config},
            runtime=CapturingRuntime(),
        )
        async with received["excluded_scope_gene"]({"query_mode": "simple", "projects": ["OPS"]}) as excluded:
            pass
    finally:
        await database.close()

    assert excluded.config == {
        "base_url": "https://jira.example", "auth_mode": "pat", "query_mode": "simple", "projects": ["OPS"],
    }
