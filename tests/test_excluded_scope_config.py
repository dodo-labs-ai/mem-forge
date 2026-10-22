"""An excluded scope is listed with the Source's current configuration and credential, and only that scope's fields."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from memforge import runtime
from memforge.genes.base import SourceConfigurationError
from memforge.genes.jira_gene import JiraGene, JiraQueryRejected
from memforge.source_projection_config import canonical_projection_scope, excluded_scope_config


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

        async def authenticate(self):
            self.authenticated = True

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

    gene = await factory({"query_mode": "simple", "projects": ["OPS"]})

    assert gene.authenticated
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
