"""``GET /genes`` and ``GET /genes/{type}/config-schema``: the form description the admin UI draws."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from memforge.source_types import registered_source_types

from tests.test_source_authority_api import _app, _connect_database


@pytest.fixture
def client(tmp_path):
    database = _connect_database(tmp_path)
    try:
        with TestClient(_app(tmp_path, database)) as test_client:
            yield test_client
    finally:
        asyncio.run(database.close())


def _schema(client: TestClient, source_type: str) -> dict:
    response = client.get(f"/api/v1/genes/{source_type}/config-schema")
    assert response.status_code == 200, response.text
    return response.json()


def _fields(schema: dict) -> dict[str, dict]:
    return {field["key"]: field for field in schema["fields"]}


def test_the_response_carries_labelled_choices_and_conditions(client):
    fields = _fields(_schema(client, "jira"))

    assert fields["auth_mode"]["choices"] == [
        {"value": "browser_cookie", "label": "Browser session", "help_text": "", "visible_when": []},
        {
            "value": "pat",
            "label": "Personal access token",
            "help_text": "",
            "visible_when": [{"field": "sync_mode", "values": ["cloud"]}],
        },
    ]
    assert fields["pat"]["visible_when"] == [{"field": "auth_mode", "values": ["pat"]}]
    assert fields["pat"]["required"] is True


def test_the_response_derives_the_deprecated_options_from_choices(client):
    fields = _fields(_schema(client, "teams"))

    assert fields["rolling_retention_days"]["options"] == ["forever", "365", "730", "1095"]
    assert fields["conversation_ids"]["options"] == []


def test_groups_carry_their_purpose_description_and_panels(client):
    groups = {group["key"]: group for group in _schema(client, "jira")["groups"]}

    assert groups["connection"]["purpose"] == "connection"
    assert groups["connection"]["panels"] == [
        {"name": "jira_session", "visible_when": [{"field": "auth_mode", "values": ["browser_cookie"]}]}
    ]
    assert groups["scope"]["purpose"] == "content"
    assert groups["scope"]["description"] == ""


def test_widgets_are_named_on_their_fields(client):
    assert _fields(_schema(client, "github_repo"))["include_paths"]["widget"] == "github_repo_paths"
    assert _fields(_schema(client, "local_markdown"))["root"]["widget"] == "local_folder"
    assert _fields(_schema(client, "teams"))["conversation_ids"]["widget"] == "teams_conversations"
    assert _fields(_schema(client, "jira"))["base_url"]["widget"] is None


def test_scope_and_fixed_after_create_come_from_the_descriptor(client):
    jira = _fields(_schema(client, "jira"))
    repo = _fields(_schema(client, "github_repo"))

    assert jira["projects"]["scope"] is True
    assert jira["base_url"]["scope"] is False
    assert jira["sync_mode"]["fixed_after_create"] is True
    assert jira["auth_mode"]["fixed_after_create"] is False
    assert repo["connection_mode"]["fixed_after_create"] is True
    assert repo["include_paths"]["scope"] is True


def test_project_field_is_sent(client):
    schema = _schema(client, "confluence")
    assert "project_field" in schema
    assert schema["project_field"] is None


def test_gene_list_offers_every_registered_source_type(client):
    genes = client.get("/api/v1/genes").json()

    assert [gene["name"] for gene in genes] == sorted(d.source_type for d in registered_source_types())
    assert all("managed" not in gene for gene in genes)


def test_the_openapi_document_marks_options_deprecated(client):
    properties = client.get("/openapi.json").json()["components"]["schemas"]["ConfigFieldResponse"]["properties"]

    assert properties["options"]["deprecated"] is True
