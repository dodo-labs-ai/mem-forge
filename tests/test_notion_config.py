"""A Notion Source's configuration: pasted links read as ids, the rules on roots and token, and how it is saved."""

from __future__ import annotations

import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from memforge.genes.notion_api import notion_link_id
from memforge.genes.notion_gene import NotionGene
from memforge.server.admin_api import _validate_source_config
from memforge.source_projection_config import canonical_projection_scope
from memforge.source_types import source_type_descriptor
from tests.test_startup_runtime import TEST_SOURCE_KEY, _config, _project_binding

PAGE = "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
PAGE_DASHED = "1f3c2a4b-5d6e-4f70-8a9b-0c1d2e3f4a5b"
DATABASE = "9c0d1e2f3a4b4c5d8e6f7a8b9c0d1e2f"
VIEW = "5e6f7a8b9c0d4e1f8a2b3c4d5e6f7a8b"


@pytest.mark.parametrize(
    ("value", "open_page", "expected"),
    [
        (PAGE, True, PAGE),
        (PAGE_DASHED, True, PAGE),
        (PAGE.upper(), True, PAGE),
        (f"https://www.notion.so/team/Payroll-Runbook-{PAGE}", True, PAGE),
        (f"https://notion.so/Payroll-Runbook-{PAGE}", True, PAGE),
        (f"https://team.notion.site/Payroll-Runbook-{PAGE}", True, PAGE),
        (f"https://app.notion.com/p/{PAGE}", True, PAGE),
        (f"https://app.notion.com/p/{PAGE}/", True, PAGE),
        (f"www.notion.so/team/{PAGE_DASHED}", True, PAGE),
        (f"Payroll-Runbook-{PAGE}", True, PAGE),
        # A page opened over a database view names that page; a databases root reads the path.
        (f"https://www.notion.so/team/{DATABASE}?v={VIEW}&p={PAGE}&pm=s", True, PAGE),
        (f"https://www.notion.so/team/{DATABASE}?v={VIEW}&p={PAGE}&pm=s", False, DATABASE),
        (f"https://www.notion.so/team/{DATABASE}?v={VIEW}", False, DATABASE),
        # The host never decides: any host with a Notion id in its path reads alike.
        (f"https://notion.example.test/workspace/Payroll-{PAGE}", True, PAGE),
        ("Payroll runbook", True, None),
        ("https://www.notion.so/team/Payroll", True, None),
        ("", True, None),
    ],
)
def test_a_pasted_notion_link_reads_as_its_id(value, open_page, expected) -> None:
    assert notion_link_id(value, open_page=open_page) == expected


def test_notion_scope_reads_urls_and_ids_alike() -> None:
    pasted = {
        "token": "secret",
        "root_pages": [f"https://www.notion.so/team/Payroll-Runbook-{PAGE}"],
        "databases": f"https://www.notion.so/team/{DATABASE}?v={VIEW}",
    }
    plain = {"token": "other", "root_pages": [PAGE_DASHED], "databases": [DATABASE.upper()]}

    normalized = dict(pasted)
    NotionGene.normalize_config(normalized)

    assert normalized["root_pages"] == [PAGE]
    assert normalized["databases"] == [DATABASE]
    expected = {"root_pages": [PAGE], "databases": [DATABASE]}
    assert canonical_projection_scope("notion", pasted) == expected
    assert canonical_projection_scope("notion", plain) == expected
    assert canonical_projection_scope("notion", normalized) == expected


def test_normalize_keeps_a_value_it_cannot_read_for_the_check_to_refuse() -> None:
    config = {"root_pages": ["Payroll runbook", PAGE_DASHED, PAGE]}

    NotionGene.normalize_config(config)

    assert config["root_pages"] == ["Payroll runbook", PAGE]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"root_pages": [PAGE]}, "Internal connection token is required"),
        ({"token": "secret"}, "must name at least one page or database"),
        ({"token": "secret", "root_pages": [], "databases": []}, "must name at least one page or database"),
        ({"token": "secret", "root_pages": ["Payroll runbook"]}, "Notion Root pages holds 'Payroll runbook'"),
        ({"token": "secret", "databases": ["https://www.notion.so/team/Payroll"]}, "Notion Databases holds"),
    ],
)
def test_notion_config_requires_a_token_and_a_root(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_source_config("notion", dict(config))


def test_a_stored_token_satisfies_the_token_requirement() -> None:
    _validate_source_config("notion", {"root_pages": [PAGE]}, existing_config={"token_encrypted": "stored"})


def test_a_notion_id_in_both_root_lists_is_refused() -> None:
    config = {"token": "secret", "root_pages": [f"https://app.notion.com/p/{PAGE}"], "databases": [PAGE_DASHED]}

    with pytest.raises(ValueError, match=f"Root pages and Databases both hold {PAGE}"):
        _validate_source_config("notion", config)


def test_notion_config_stores_the_fixed_api_origin() -> None:
    config = {"token": "secret", "root_pages": [PAGE], "base_url": "https://notion.example.test"}

    _validate_source_config("notion", config)

    assert config["base_url"] == "https://api.notion.com"


@pytest.mark.parametrize(
    ("incoming", "changed"),
    [({"token": "new-token"}, True), ({"token": "old-token"}, False), ({"token": " "}, False), ({}, False)],
)
def test_notion_credentials_changed_when_a_new_token_is_supplied(incoming, changed) -> None:
    credentials_changed = source_type_descriptor("notion").config.credentials_changed

    assert credentials_changed({"root_pages": [PAGE], **incoming}, {"token": "old-token"}) is changed


def test_creating_a_notion_source_saves_ids_and_redacts_the_token(tmp_path, monkeypatch) -> None:
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/sources",
            json={
                "type": "notion",
                "access_policy": "private",
                "name": "Payroll Notion",
                "config": {
                    "token": "ntn_notion-secret",
                    "root_pages": [f"https://www.notion.so/team/Payroll-Runbook-{PAGE}"],
                    "databases": [DATABASE],
                },
                "project_binding": _project_binding(),
            },
        )
        without_root = client.post(
            "/api/v1/sources",
            json={"type": "notion", "access_policy": "private", "name": "Empty", "config": {"token": "t"}},
        )
        assert created.status_code == 200, created.text
        source_id = created.json()["id"]
        sources = client.get("/api/v1/sources")

    assert without_root.status_code == 400
    assert "at least one page or database" in without_root.json()["detail"]
    with sqlite3.connect(cfg.storage.db_path) as conn:
        (stored_json,) = conn.execute("SELECT config FROM sources WHERE id = ?", (source_id,)).fetchone()
    stored = json.loads(stored_json)
    assert "token" not in stored
    assert stored["token_encrypted"] != "ntn_notion-secret"
    assert stored["root_pages"] == [PAGE]
    assert stored["databases"] == [DATABASE]
    assert stored["base_url"] == "https://api.notion.com"
    payload = next(source for source in sources.json()["data"] if source["id"] == source_id)
    assert "token" not in payload["config"]
    assert "token_encrypted" not in payload["config"]
    assert payload["config"]["token_configured"] is True
    assert "ntn_notion-secret" not in sources.text
