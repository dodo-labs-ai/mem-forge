"""``POST /genes/{name}/config-check``: save's own steps on a copy, returning derived values, facts and issues."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.genes import GENE_REGISTRY
from memforge.local_adapter import default_local_adapter_inbox
from memforge.models import ConfigFieldType, comparison_string
from memforge.source_types import source_type_descriptor
from memforge.source_types.form import effective_value
from tests.test_source_authority_api import _app, _config, _connect_database, _teams_payload
from tests.test_startup_runtime import TEST_SOURCE_KEY

WIKI_PAGE = "https://wiki.example.test/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll"
PAGES = "https://pages.example.test/org/site"

# One configuration a person could submit for each configurable type.
SUBMITTED = {
    "confluence": {"base_url": WIKI_PAGE, "pat": "wiki-pat-secret"},
    "github_pages": {"auth_mode": "none", "page_url": f"{PAGES}/guide/start/"},
    "github_repo": {"repo_url": "https://github.example.test/platform/cookbook.git", "connection_mode": "cloud_pull"},
    "jira": {"base_url": "https://jira.example.test", "projects": "PAY, OPS", "pat": "jira-pat-secret"},
    "local_markdown": {"root": "/Users/me/notes"},
    "notion": {
        "token": "notion-token-secret",
        "root_pages": ["https://www.notion.so/team/Payroll-Runbook-1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"],
        "databases": "9c0d1e2f-3a4b-4c5d-8e6f-7a8b9c0d1e2f",
    },
    "teams": {
        "region": "emea",
        "conversation_ids": "19:conversation-a@example.test",
        "initial_history_days": "14",
        "rolling_retention_days": "forever",
    },
}


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)


@pytest.fixture
def database(tmp_path):
    db = _connect_database(tmp_path)
    yield db
    asyncio.run(db.close())


@pytest.fixture
def client(tmp_path, database):
    with TestClient(_app(tmp_path, database)) as test_client:
        yield test_client


def _check(client: TestClient, source_type: str, config: dict, *, source_id: str | None = None, user: str = "owner-user", role: str = "member"):
    body: dict = {"config": config}
    if source_id is not None:
        body["source_id"] = source_id
    return client.post(
        f"/api/v1/genes/{source_type}/config-check",
        json=body,
        headers={"x-test-user": user, "x-test-workspace-role": role},
    )


def test_confluence_page_url_derives_page_tree(client):
    response = _check(client, "confluence", {"base_url": WIKI_PAGE})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["derived"] == {
        "base_url": "https://wiki.example.test",
        "sync_mode": "page_tree",
        "spaces": ["PAY"],
        "page_tree_root": "5695886009",
    }
    assert body["issues"] == [{"key": "pat", "message": "Confluence Personal Access Token is required"}]
    assert {"label": "Site", "value": "https://wiki.example.test"} in body["facts"]
    assert {"label": "Scope", "value": "The page tree under page 5695886009"} in body["facts"]


@pytest.mark.parametrize("source_type", sorted(SUBMITTED))
def test_config_check_derives_what_save_stores(client, database, source_type):
    submitted = SUBMITTED[source_type]
    checked = _check(client, source_type, submitted)
    assert checked.status_code == 200, checked.text
    assert checked.json()["issues"] == []
    created = client.post(
        "/api/v1/sources",
        json={"type": source_type, "name": "Checked", "access_policy": "workspace", "config": submitted},
    )
    assert created.status_code == 200, created.text
    stored = asyncio.run(database.get_source(created.json()["id"]))["config"]

    shown = {**submitted, **checked.json()["derived"]}
    descriptor = source_type_descriptor(source_type)
    for field in GENE_REGISTRY[source_type].config_schema().fields:
        if field.field_type == ConfigFieldType.SECRET:
            continue
        assert comparison_string(effective_value(field, shown, descriptor), field.field_type) == comparison_string(
            effective_value(field, stored, descriptor), field.field_type
        ), field.key


def test_config_check_on_create_reports_no_undeclared_key(client):
    response = _check(client, "local_markdown", {"root": "/Users/me/notes"})

    assert response.status_code == 200, response.text
    assert response.json()["derived"] == {}


@pytest.mark.parametrize("retention", ["365", "forever"])
def test_teams_retention_is_not_reported_as_derived(client, retention):
    response = _check(client, "teams", {**SUBMITTED["teams"], "rolling_retention_days": retention})

    assert response.status_code == 200, response.text
    assert response.json()["derived"] == {}


def test_config_check_maps_a_keyed_hook_refusal_to_its_field(client):
    response = _check(
        client,
        "github_pages",
        {"auth_mode": "none", "base_url": PAGES, "page_url": "https://other.example.test/org/site/guide/"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["issues"] == [
        {"key": "page_url", "message": "GitHub Pages Page URL must stay on the configured site origin"}
    ]


def test_config_check_writes_nothing(tmp_path, client, database):
    _check(client, "local_markdown", {"root": "/Users/me/notes"})
    _check(client, "github_repo", {**SUBMITTED["github_repo"], "connection_mode": "local_push"})
    _check(client, "confluence", SUBMITTED["confluence"])

    assert asyncio.run(database.list_sources()) == []
    inboxes = default_local_adapter_inbox(_config(tmp_path), "config-check").parent
    assert not Path(inboxes).exists()


def test_jira_fact_shows_effective_jql_with_order_by(client):
    response = _check(client, "jira", {"base_url": "https://jira.example.test", "projects": ["PAY"]})

    assert response.json()["facts"] == [
        {
            "label": "Effective query at sync",
            "value": (
                "project in (PAY) AND issuetype in (Epic,Story,Bug,Task) "
                "AND updated >= '<last sync>' ORDER BY updated DESC"
            ),
        }
    ]


def test_jira_issue_types_that_are_not_a_list_are_an_issue_not_an_error(client):
    submitted = {**SUBMITTED["jira"], "projects": ["PAY"], "auth_mode": "pat", "issue_types": 5}

    response = _check(client, "jira", submitted)

    assert response.status_code == 200, response.text
    assert "issue_types" in {issue["key"] for issue in response.json()["issues"]}


@pytest.mark.parametrize("issue_types", [[5], ["Bug", None]])
def test_jira_fact_reads_issue_types_of_any_item(client, issue_types):
    submitted = {**SUBMITTED["jira"], "projects": ["PAY"], "auth_mode": "pat", "issue_types": issue_types}

    response = _check(client, "jira", submitted)

    assert response.status_code == 200, response.text
    assert response.json()["facts"][0]["label"] == "Effective query at sync"


def test_form_module_imports_on_its_own():
    completed = subprocess.run(
        [sys.executable, "-c", "import memforge.source_types.form"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_github_repo_fact_names_owner_and_repo(client):
    response = _check(client, "github_repo", SUBMITTED["github_repo"])

    assert response.json()["facts"] == [
        {"label": "Host", "value": "github.example.test"},
        {"label": "Repository", "value": "platform/cookbook"},
    ]


def test_config_check_never_returns_secrets(client, database):
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Wiki", "access_policy": "workspace", "config": SUBMITTED["confluence"]},
    )
    source_id = created.json()["id"]

    for response in (
        _check(client, "confluence", SUBMITTED["confluence"]),
        _check(client, "confluence", {"base_url": WIKI_PAGE}, source_id=source_id),
    ):
        assert response.status_code == 200, response.text
        text = json.dumps(response.json())
        assert "wiki-pat-secret" not in text
        assert "_encrypted" not in text
        assert "_hint" not in text
    assert response.json()["issues"] == []


def test_config_check_with_source_id_requires_manage_access(client):
    wiki = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Wiki", "access_policy": "workspace", "config": SUBMITTED["confluence"]},
        headers={"x-test-user": "owner-a"},
    )
    teams = client.post("/api/v1/sources", json=_teams_payload(), headers={"x-test-user": "owner-a"})

    other_member = _check(client, "confluence", SUBMITTED["confluence"], source_id=wiki.json()["id"], user="member-b")
    other_admin = _check(
        client, "teams", SUBMITTED["teams"], source_id=teams.json()["id"], user="admin-b", role="workspace_admin"
    )

    assert other_member.status_code == 403
    assert other_member.json()["detail"]["error"] == "source_management_forbidden"
    assert other_admin.status_code == 403
    assert other_admin.json()["detail"]["error"] == "local_agent_source_connection_owner_forbidden"


def test_config_check_type_mismatch_is_refused(client):
    wiki = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Wiki", "access_policy": "workspace", "config": SUBMITTED["confluence"]},
    )

    response = _check(client, "jira", SUBMITTED["jira"], source_id=wiki.json()["id"])

    assert response.status_code == 400
    assert response.json()["detail"] == "Config-check source type does not match the existing source"


@pytest.mark.parametrize(
    ("submitted", "derived", "issues"),
    [
        # A new Source reads the declared default, a personal access token.
        ({}, {}, [{"key": "pat", "message": "Jira Personal Access Token is required"}]),
        ({"pat": "jira-pat-secret"}, {}, []),
        # Local sync offers only a browser session.
        ({"sync_mode": "local_agent"}, {"auth_mode": "browser_cookie"}, []),
    ],
)
def test_jira_config_check_reports_the_auth_mode_save_stores(client, submitted, derived, issues):
    response = _check(client, "jira", {"base_url": "https://jira.example.test", "projects": ["PAY"], **submitted})

    assert response.status_code == 200, response.text
    assert response.json()["derived"] == derived
    assert response.json()["issues"] == issues
