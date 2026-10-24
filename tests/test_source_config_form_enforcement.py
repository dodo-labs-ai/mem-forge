"""The shared form check at the route boundary: required while visible, choices, booleans and one refusal shape.

Every configuration refusal answers 400 with ``detail`` ``{"code":
"config_invalid", "message", "fields": [{"key", "message"}]}``. The check reads
the configuration the save stores, after the Gene normalizes it: a blank field
reads the descriptor's scope default, then the field's declared default.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from memforge.source_secrets import prepare_source_config_for_storage
from tests.test_source_authority_api import _app, _connect_database
from tests.test_startup_runtime import TEST_SOURCE_KEY

WIKI = "https://wiki.example.test"
JIRA = "https://jira.example.test"
PAGES = "https://pages.example.test/org/site"


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


def _create(client: TestClient, source_type: str, config: dict, name: str = "Source"):
    return client.post(
        "/api/v1/sources",
        json={"type": source_type, "name": name, "access_policy": "workspace", "config": config},
    )


def _store(database, source_id: str, source_type: str, config: dict, secret_fields=("pat",)) -> None:
    asyncio.run(
        database.upsert_source(
            id=source_id,
            type=source_type,
            name=source_id,
            config_json=json.dumps(prepare_source_config_for_storage(config, secret_fields=secret_fields)),
            access_policy="workspace",
            owner_user_id="owner-user",
        )
    )


def _stored_config(database, source_id: str) -> dict:
    return asyncio.run(database.get_source(source_id))["config"]


def _issues(response) -> list[tuple[str | None, str]]:
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "config_invalid"
    return [(issue["key"], issue["message"]) for issue in detail["fields"]]


def _keys(response) -> list[str | None]:
    return [key for key, _message in _issues(response)]


def test_required_applies_only_while_visible(client):
    simple = _create(client, "jira", {"base_url": JIRA, "query_mode": "simple", "jql": "project = PAY"})
    advanced = _create(client, "jira", {"base_url": JIRA, "query_mode": "advanced", "jql": "project = PAY"})

    assert _keys(simple) == ["projects"]
    assert advanced.status_code == 200, advanced.text


def test_hidden_option_value_is_refused(client):
    response = _create(
        client,
        "jira",
        {"base_url": JIRA, "sync_mode": "local_agent", "auth_mode": "pat", "pat": "jira-pat", "projects": ["PAY"]},
    )

    assert _issues(response) == [("auth_mode", "Jira Authentication Method must be one of: Browser session")]


def test_configured_secret_satisfies_required(client):
    created = _create(client, "confluence", {"base_url": WIKI, "pat": "wiki-pat", "spaces": ["PAY"]})
    source_id = created.json()["id"]

    updated = client.put(f"/api/v1/sources/{source_id}", json={"config": {"base_url": WIKI, "spaces": ["PAY"]}})

    assert updated.status_code == 200, updated.text


def test_condition_reads_declared_default_when_blank(client):
    # No sync_mode reads the declared default single_page, which shows and requires page_url.
    response = _create(client, "github_pages", {"auth_mode": "none"})

    assert _keys(response) == ["page_url"]


def test_condition_reads_scope_default_when_blank(client):
    # No query_mode reads the descriptor's scope default simple, which requires projects.
    response = _create(client, "jira", {"base_url": JIRA})

    assert _keys(response) == ["projects"]


def test_confluence_page_url_create_without_sync_mode_succeeds(client, database):
    response = _create(
        client,
        "confluence",
        {"base_url": f"{WIKI}/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll", "pat": "wiki-pat"},
    )

    assert response.status_code == 200, response.text
    stored = _stored_config(database, response.json()["id"])
    assert stored["sync_mode"] == "page_tree"
    assert stored["page_tree_root"] == "5695886009"


def test_confluence_blank_sync_mode_with_root_is_page_tree(client, database):
    response = _create(client, "confluence", {"base_url": WIKI, "pat": "wiki-pat", "sync_mode": "", "page_tree_root": "123"})

    assert response.status_code == 200, response.text
    assert _stored_config(database, response.json()["id"])["sync_mode"] == "page_tree"


def test_confluence_blank_sync_mode_without_root_is_stored_as_space(client, database):
    response = _create(client, "confluence", {"base_url": WIKI, "pat": "wiki-pat", "spaces": ["PAY"]})

    assert response.status_code == 200, response.text
    assert _stored_config(database, response.json()["id"])["sync_mode"] == "space"


def test_jira_blank_auth_mode_with_stored_token_is_stored_as_pat(client, database):
    _store(database, "src-jira-token", "jira", {"base_url": JIRA, "projects": ["PAY"], "pat": "jira-pat"})

    updated = client.put("/api/v1/sources/src-jira-token", json={"config": {"base_url": JIRA, "projects": ["PAY"]}})

    assert updated.status_code == 200, updated.text
    assert _stored_config(database, "src-jira-token")["auth_mode"] == "pat"


def test_jira_source_list_returns_the_effective_auth_mode(client, database):
    _store(database, "src-jira-token", "jira", {"base_url": JIRA, "projects": ["PAY"], "pat": "jira-pat"})
    _store(database, "src-jira-session", "jira", {"base_url": JIRA, "projects": ["OPS"]})

    sources = {source["id"]: source for source in client.get("/api/v1/sources").json()["data"]}

    assert sources["src-jira-token"]["config"]["auth_mode"] == "pat"
    assert sources["src-jira-session"]["config"]["auth_mode"] == "browser_cookie"
    assert "auth_mode" not in _stored_config(database, "src-jira-token")


def test_update_is_checked_against_the_config_it_stores(client):
    created = _create(client, "confluence", {"base_url": WIKI, "pat": "wiki-pat", "spaces": ["PAY"]})
    source_id = created.json()["id"]

    # PUT stores the submitted configuration whole, so an omitted required field is missing.
    omitted = client.put(f"/api/v1/sources/{source_id}", json={"config": {"base_url": WIKI}})
    kept_secret = client.put(
        f"/api/v1/sources/{source_id}", json={"config": {"base_url": WIKI, "spaces": ["PAY"], "pat": ""}}
    )

    assert _keys(omitted) == ["spaces"]
    assert kept_secret.status_code == 200, kept_secret.text


def test_non_boolean_value_is_refused_as_a_field_issue(client):
    # include_children is hidden for a whole-space Source, and still checked.
    response = _create(
        client,
        "confluence",
        {"base_url": WIKI, "pat": "wiki-pat", "spaces": ["PAY"], "include_children": "maybe"},
    )

    assert _keys(response) == ["include_children"]


def test_refusal_detail_carries_a_message_naming_the_first_field(client):
    response = _create(client, "confluence", {"spaces": ["PAY"]})

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "config_invalid",
        "message": "Confluence Wiki URL is required",
        "fields": [
            {"key": "base_url", "message": "Confluence Wiki URL is required"},
            {"key": "pat", "message": "Confluence Personal Access Token is required"},
        ],
    }


def test_github_pages_missing_scope_url_is_refused_on_its_field(client):
    response = _create(client, "github_pages", {"auth_mode": "none", "sync_mode": "single_page"})

    assert _issues(response) == [("page_url", "GitHub Pages Page URL is required")]


def test_github_pages_base_url_is_derived_from_the_scope_url_and_kept_on_update(client, database):
    created = _create(client, "github_pages", {"auth_mode": "none", "page_url": f"{PAGES}/guide/start/"})
    source_id = created.json()["id"]
    assert created.status_code == 200, created.text
    assert _stored_config(database, source_id)["base_url"] == PAGES

    updated = client.put(
        f"/api/v1/sources/{source_id}",
        json={"config": {"auth_mode": "none", "sync_mode": "explicit_list", "pages": [f"{PAGES}/guide/next/"]}},
    )

    assert updated.status_code == 200, updated.text
    assert _stored_config(database, source_id)["base_url"] == PAGES


def test_github_pages_single_page_url_is_read_as_one_url(client, database):
    created = _create(client, "github_pages", {"auth_mode": "none", "page_url": f"{PAGES}/guide/a,b/"})
    assert created.status_code == 200, created.text
    assert _stored_config(database, created.json()["id"])["page_url"] == f"{PAGES}/guide/a,b/"

    response = _create(client, "github_pages", {"auth_mode": "none", "page_url": 5})

    assert [key for key, _ in _issues(response)] == ["page_url"]


def test_github_repo_missing_repo_url_is_refused_on_its_field(client):
    response = _create(client, "github_repo", {"connection_mode": "cloud_pull"})

    assert _issues(response) == [("repo_url", "GitHub Repository URL is required")]


def test_secret_origin_guard_runs_after_the_form_check(client):
    missing = _create(client, "jira", {"projects": ["PAY"], "pat": "jira-pat", "auth_mode": "pat"})
    plaintext = _create(
        client, "jira", {"base_url": "http://jira.example.test", "projects": ["PAY"], "pat": "jira-pat", "auth_mode": "pat"}
    )

    assert _issues(missing) == [("base_url", "Jira Base URL is required")]
    assert _issues(plaintext) == [
        ("base_url", "Jira base_url must use HTTPS when storing authenticated source credentials")
    ]


@pytest.mark.parametrize("retention", [365, "365", None, "forever"])
def test_teams_retention_preset_and_forever_pass_the_choice_check(client, retention):
    response = _create(
        client,
        "teams",
        {"region": "emea", "conversation_ids": ["19:conversation-a@example.test"], "rolling_retention_days": retention},
    )

    assert response.status_code == 200, response.text


def test_a_plain_value_error_from_a_hook_answers_config_invalid_with_a_null_key(client):
    response = _create(client, "jira", {"base_url": JIRA, "projects": ["PAY"], "jira_cookie": "session=abc"})

    assert _issues(response) == [(None, "Jira browser sessions are managed by shared auth sessions, not source config")]


def test_a_keyed_refusal_from_a_hook_names_its_field(client):
    response = _create(
        client,
        "github_pages",
        {"auth_mode": "none", "base_url": PAGES, "page_url": "https://other.example.test/org/site/guide/"},
    )

    assert _issues(response) == [("page_url", "GitHub Pages Page URL must stay on the configured site origin")]


def test_preview_discovery_answers_config_invalid(client):
    response = client.post(
        "/api/v1/genes/confluence/preview-discovery",
        json={"config": {"base_url": WIKI, "pat": "wiki-pat", "sync_mode": "space"}},
    )

    assert _keys(response) == ["spaces"]


def test_browse_tree_answers_config_invalid(client):
    response = client.post("/api/v1/genes/github_repo/browse-tree", json={"config": {"connection_mode": "cloud_pull"}})

    assert _keys(response) == ["repo_url"]


@pytest.mark.parametrize(
    ("source_type", "config", "expected"),
    [
        (
            "confluence",
            {"base_url": WIKI, "pat": "wiki-pat", "sync_mode": "page_tree"},
            ("page_tree_root", "Confluence Page Tree Root is required"),
        ),
        (
            "confluence",
            {"base_url": WIKI, "pat": "wiki-pat", "sync_mode": "space"},
            ("spaces", "Confluence Spaces to Sync is required"),
        ),
        (
            "github_pages",
            {"auth_mode": "github_pat", "page_url": f"{PAGES}/guide/"},
            ("pat", "GitHub Pages Personal Access Token is required"),
        ),
        (
            "github_pages",
            {"auth_mode": "none", "sync_mode": "subtree"},
            ("root_url", "GitHub Pages Subtree Root URL is required"),
        ),
        (
            "github_pages",
            {"auth_mode": "none", "sync_mode": "explicit_list", "pages": []},
            ("pages", "GitHub Pages Explicit Page URLs is required"),
        ),
        (
            "github_pages",
            {"auth_mode": "basic", "page_url": f"{PAGES}/guide/"},
            ("auth_mode", "GitHub Pages Authentication Method must be one of: Personal access token, No authentication"),
        ),
        (
            "jira",
            {"base_url": JIRA, "auth_mode": "pat", "projects": ["PAY"]},
            ("pat", "Jira Personal Access Token is required"),
        ),
        (
            "jira",
            {"base_url": JIRA, "query_mode": "advanced", "jql": "  "},
            ("jql", "Jira JQL is required"),
        ),
        (
            "github_repo",
            {"repo_url": "https://github.example.test/org/repo", "connection_mode": "ssh"},
            ("connection_mode", "GitHub Repository Access must be one of: Cloud access, Local sync"),
        ),
    ],
)
def test_each_former_per_type_requirement_is_refused_by_the_shared_check(client, source_type, config, expected):
    assert _issues(_create(client, source_type, config)) == [expected]
