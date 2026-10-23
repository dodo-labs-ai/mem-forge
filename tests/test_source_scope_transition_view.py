"""The scope change a Source is applying, as the Sources page reads it, and the preview of a configuration change.

The Source list carries the open transition: both scopes, the listed items
committed under it out of the latest listing, the failed items that keep it
open, and the configuration that restores the previous scope. Saving that
configuration through the update route returns to the previous scope. The
preview answers what saving a configuration would do, as the update route
decides it, and writes nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.release import current_release
from memforge.server.admin_api import create_admin_app
from memforge.source_item_failures import SourceItemFailure
from memforge.source_projection import ProjectionScopeTransition, ProjectionScopeTransitionStatus
from memforge.source_projection_config import canonical_projection_scope
from memforge.source_secrets import prepare_source_config_for_storage
from memforge.storage.database import Database
from tests.test_startup_runtime import TEST_SOURCE_KEY, _config

SOURCE_ID = "src-scope-view"
PAGES_SITE = "https://pages.example.test/org/site"
FAILED_AT = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


def _confluence(root: str) -> dict[str, object]:
    return {"base_url": "https://wiki.example", "sync_mode": "page_tree", "page_tree_root": root}


def _pages(root: str, **fields: object) -> dict[str, object]:
    return {
        "base_url": PAGES_SITE,
        "sync_mode": "subtree",
        "root_url": f"{PAGES_SITE}/{root}",
        "auth_mode": "none",
        **fields,
    }


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    database = Database(str(tmp_path / "scope-view.db"))
    await database.connect()
    yield database
    await database.close()


async def _source(db: Database, source_type: str, config: dict[str, object], **fields: object) -> None:
    await db.upsert_source(
        id=SOURCE_ID,
        type=source_type,
        name="Scoped",
        config_json=json.dumps(prepare_source_config_for_storage(config, secret_fields=("pat",))),
        access_policy="workspace",
        owner_user_id="dev",
        **fields,
    )


def _client(tmp_path: Path, db: Database) -> TestClient:
    return TestClient(create_admin_app(db=db, config=_config(tmp_path)))


def _save(tmp_path: Path, db: Database, config: dict[str, object]):
    with _client(tmp_path, db) as client:
        return client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": config})


def _preview(tmp_path: Path, db: Database, config: dict[str, object]) -> dict:
    with _client(tmp_path, db) as client:
        response = client.post(f"/api/v1/sources/{SOURCE_ID}/scope-change-preview", json={"config": config})
    assert response.status_code == 200, response.text
    return response.json()


def _row(tmp_path: Path, db: Database) -> dict:
    with _client(tmp_path, db) as client:
        response = client.get("/api/v1/sources")
    assert response.status_code == 200, response.text
    return next(row for row in response.json()["data"] if row["id"] == SOURCE_ID)


async def _commit_member(db: Database, transition_id: str, doc_id: str) -> None:
    # The commit path writes member rows; the view only counts them.
    await db.db.execute(
        "INSERT INTO projection_scope_transition_members (transition_id, doc_id, unit_revision_id) VALUES (?, ?, ?)",
        (transition_id, doc_id, f"rev-{doc_id}"),
    )
    await db.db.commit()


async def _fail_item(db: Database, doc_id: str, message: str) -> None:
    await db.upsert_source_item_failure(
        SourceItemFailure(
            source_id=SOURCE_ID,
            document_id=doc_id,
            stage="fetch",
            error_class="ValueError",
            message=message,
            retryable=True,
            release=current_release(),
            failures_in_release=1,
            first_failed_at=FAILED_AT,
            last_failed_at=FAILED_AT,
            title=doc_id.title(),
        )
    )


@pytest.mark.asyncio
async def test_the_source_list_shows_the_open_transition_its_progress_and_the_items_that_keep_it_open(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    assert _save(tmp_path, db, {**_confluence("root-b"), "pat": "secret"}).status_code == 200
    assert _row(tmp_path, db)["scope_transition"]["listed_count"] is None
    transition = await db.get_open_projection_scope_transition(SOURCE_ID)
    await db.start_projection_scope_transition(transition.id, run_id="run-1")
    await _commit_member(db, transition.id, "page-1")
    await _commit_member(db, transition.id, "page-2")
    await _fail_item(db, "page-3", "403 Forbidden")
    # A failure recorded before the item committed under the change does not keep it open.
    await _fail_item(db, "page-2", "timed out")
    await db.fail_projection_scope_transition(
        transition.id,
        run_id="run-1",
        error="1 listed Documents have not committed under the transition",
        listed_count=3,
    )

    row = _row(tmp_path, db)
    summary = row["scope_transition"]

    assert summary["id"] == transition.id
    assert summary["status"] == "failed"
    assert summary["previous_scope"] == canonical_projection_scope("confluence", _confluence("root-a"))
    assert summary["target_scope"] == canonical_projection_scope("confluence", _confluence("root-b"))
    assert (summary["members_committed"], summary["listed_count"]) == (2, 3)
    assert summary["error"] == "1 listed Documents have not committed under the transition"
    assert summary["excluded_scopes_due"] is False
    assert summary["failing_items"]["count"] == 1
    assert [(item["document_id"], item["message"]) for item in summary["failing_items"]["items"]] == [
        ("page-3", "403 Forbidden")
    ]
    assert row["failed_items"]["count"] == 2
    assert summary["created_at"] == transition.created_at
    # The restore configuration is the viewer's configuration with the previous scope, secrets redacted.
    assert summary["restore_config"]["page_tree_root"] == "root-a"
    assert {field: value for field, value in summary["restore_config"].items() if field != "page_tree_root"} == {
        field: value for field, value in row["config"].items() if field != "page_tree_root"
    }
    assert "secret" not in json.dumps(summary)


@pytest.mark.asyncio
async def test_a_source_without_an_open_change_shows_none(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    _save(tmp_path, db, {**_confluence("root-b"), "pat": "secret"})
    transition = await db.get_open_projection_scope_transition(SOURCE_ID)
    await db.start_projection_scope_transition(transition.id, run_id="run-1")
    await db.complete_projection_scope_transition(transition.id, run_id="run-1")

    assert _row(tmp_path, db)["scope_transition"] is None


@pytest.mark.asyncio
async def test_saving_the_restore_configuration_returns_a_listing_source_to_its_previous_scope(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    _save(tmp_path, db, {**_confluence("root-b"), "pat": "secret"})
    stuck = await db.get_open_projection_scope_transition(SOURCE_ID)
    stored_secret = (await db.get_source(SOURCE_ID))["config"]["pat_encrypted"]
    restore = _row(tmp_path, db)["scope_transition"]["restore_config"]

    assert _preview(tmp_path, db, restore) == {
        "outcome": "restores_previous_scope",
        "open_transition": True,
        "refusal": None,
    }
    response = _save(tmp_path, db, restore)
    back = await db.get_open_projection_scope_transition(SOURCE_ID)
    stored = await db.get_source(SOURCE_ID)

    assert response.status_code == 200, response.text
    assert back.previous_scope == back.target_scope == stuck.previous_scope
    assert (await db.get_projection_scope_transition(stuck.id)).status is ProjectionScopeTransitionStatus.SUPERSEDED
    assert stored["config"]["page_tree_root"] == "root-a"
    # The redacted secret is kept, not replaced.
    assert stored["config"]["pat_encrypted"] == stored_secret


@pytest.mark.asyncio
async def test_the_restore_configuration_keeps_how_the_source_lists(db, tmp_path):
    await _source(db, "github_pages", _pages("docs"))
    _save(tmp_path, db, _pages("guide", sitemap_authoritative=True))
    restore = _row(tmp_path, db)["scope_transition"]["restore_config"]

    response = _save(tmp_path, db, restore)
    back = await db.get_open_projection_scope_transition(SOURCE_ID)

    assert restore["root_url"] == f"{PAGES_SITE}/docs"
    assert restore["sitemap_authoritative"] is True
    assert response.status_code == 200, response.text
    assert back.previous_scope == back.target_scope == canonical_projection_scope("github_pages", _pages("docs"))


@pytest.mark.asyncio
async def test_saving_the_restore_configuration_of_a_source_that_cannot_list_closes_the_change(db, tmp_path):
    await _source(db, "github_pages", _pages("guide"))
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-into-a-crawl",
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("github_pages", _pages("docs")),
            target_scope=canonical_projection_scope("github_pages", _pages("guide")),
            created_at="2026-10-20T09:00:00+00:00",
        )
    )
    restore = _row(tmp_path, db)["scope_transition"]["restore_config"]

    assert _preview(tmp_path, db, restore)["outcome"] == "restores_previous_scope"
    response = _save(tmp_path, db, restore)

    assert response.status_code == 200, response.text
    assert await db.get_open_projection_scope_transition(SOURCE_ID) is None
    assert _row(tmp_path, db)["scope_transition"] is None


@pytest.mark.asyncio
async def test_a_conversation_source_shows_its_open_change_without_a_restore(db, tmp_path):
    await _source(db, "teams", {"conversation_ids": ["19:a@thread.v2"]}, execution_owner_user_id="dev")
    assert _save(tmp_path, db, {"conversation_ids": ["19:b@thread.v2"]}).status_code == 200

    summary = _row(tmp_path, db)["scope_transition"]

    assert summary["target_scope"]["conversation_ids"] == ["19:b@thread.v2"]
    assert summary["restore_config"] is None
    refused = _preview(tmp_path, db, {"conversation_ids": ["19:c@thread.v2"]})
    assert refused["outcome"] == "refused"
    assert refused["refusal"]["code"] == "projection_scope_transition_active"


@pytest.mark.asyncio
async def test_the_preview_decides_as_saving_would_and_writes_nothing(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})

    assert _preview(tmp_path, db, _confluence("root-a"))["outcome"] == "unchanged"
    assert _preview(tmp_path, db, _confluence("root-b")) == {
        "outcome": "opens_transition",
        "open_transition": False,
        "refusal": None,
    }
    assert await db.get_open_projection_scope_transition(SOURCE_ID) is None
    assert (await db.get_source(SOURCE_ID))["config"]["page_tree_root"] == "root-a"

    _save(tmp_path, db, _confluence("root-b"))
    opened = await db.get_open_projection_scope_transition(SOURCE_ID)

    assert _preview(tmp_path, db, _confluence("root-b")) == {
        "outcome": "unchanged",
        "open_transition": True,
        "refusal": None,
    }
    assert _preview(tmp_path, db, _confluence("root-c"))["outcome"] == "supersedes_transition"
    assert _preview(tmp_path, db, _confluence("root-a"))["outcome"] == "restores_previous_scope"
    assert [transition.id for transition in await db.list_projection_scope_transitions(SOURCE_ID)] == [opened.id]


@pytest.mark.asyncio
async def test_the_preview_names_a_mode_that_cannot_list_and_what_to_choose_instead(db, tmp_path):
    await _source(db, "github_pages", {**_pages("docs", auth_mode="github_pat"), "pat": "secret"})

    into_crawl = _preview(tmp_path, db, _pages("guide"))
    _save(tmp_path, db, _pages("guide", auth_mode="github_pat"))
    stop_listing = _preview(tmp_path, db, _pages("guide"))

    assert into_crawl["outcome"] == "refused"
    assert into_crawl["open_transition"] is False
    assert into_crawl["refusal"]["code"] == "scope_change_requires_listing"
    assert into_crawl["refusal"]["mode"] == "subtree read by link crawl or by a sitemap not declared complete"
    assert "new Source" in into_crawl["refusal"]["message"]
    assert stop_listing["outcome"] == "refused"
    assert stop_listing["open_transition"] is True
    assert stop_listing["refusal"]["code"] == "transition_requires_listing"


@pytest.mark.asyncio
async def test_a_run_without_a_complete_listing_keeps_the_latest_listed_count(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    _save(tmp_path, db, _confluence("root-b"))
    transition = await db.get_open_projection_scope_transition(SOURCE_ID)

    await db.start_projection_scope_transition(transition.id, run_id="run-1")
    await db.fail_projection_scope_transition(transition.id, run_id="run-1", error="waiting", listed_count=40)
    await db.start_projection_scope_transition(transition.id, run_id="run-2")
    await db.fail_projection_scope_transition(transition.id, run_id="run-2", error="the listing failed")

    assert (await db.get_projection_scope_transition(transition.id)).listed_count == 40
