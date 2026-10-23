"""A scope change saved through the Source route: supersession, refusal and the return to the last applied scope.

A document Source's change while a transition is open supersedes it, and the
new transition runs from the last applied scope. A change into a
configuration that cannot list is refused, except a return to exactly the
last applied scope. A conversation Source still waits for its transition, and
every Source waits for a running sync.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from memforge.models import SyncState
from memforge.server.admin_api import create_admin_app
from memforge.source_projection import ProjectionScopeTransition, ProjectionScopeTransitionStatus
from memforge.source_projection_config import canonical_projection_scope
from memforge.source_secrets import prepare_source_config_for_storage
from memforge.storage.database import Database
from tests.test_startup_runtime import TEST_SOURCE_KEY, _config

SOURCE_ID = "src-scope-change"
PAGES_SITE = "https://pages.example.test/org/site"


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
    database = Database(str(tmp_path / "scope-change.db"))
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


def _save(tmp_path: Path, db: Database, config: dict[str, object]) -> httpx.Response:
    with TestClient(create_admin_app(db=db, config=_config(tmp_path))) as client:
        return client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": config})


async def _open(db: Database) -> ProjectionScopeTransition | None:
    return await db.get_open_projection_scope_transition(SOURCE_ID)


async def _statuses(db: Database) -> dict[str, ProjectionScopeTransitionStatus]:
    return {transition.id: transition.status for transition in await db.list_projection_scope_transitions(SOURCE_ID)}


async def _cursor(db: Database) -> datetime | None:
    state = await db.get_sync_state(SOURCE_ID)
    return state.last_sync_at if state is not None else None


@pytest.mark.asyncio
async def test_a_change_during_an_open_transition_supersedes_it_from_the_last_applied_scope(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    assert _save(tmp_path, db, _confluence("root-b")).status_code == 200
    to_b = await _open(db)

    to_c = _save(tmp_path, db, _confluence("root-c"))
    superseding = await _open(db)
    assert to_c.status_code == 200, to_c.text
    back_to_b = _save(tmp_path, db, _confluence("root-b"))
    again_to_b = await _open(db)

    assert back_to_b.status_code == 200, back_to_b.text
    assert superseding.previous_scope == canonical_projection_scope("confluence", _confluence("root-a"))
    assert superseding.target_scope == canonical_projection_scope("confluence", _confluence("root-c"))
    assert superseding.predecessor_transition_id == to_b.id
    # A→B again, after A→C: a new cycle with its own identity (ADR 0005).
    assert again_to_b.previous_scope == to_b.previous_scope
    assert again_to_b.target_scope == to_b.target_scope
    assert again_to_b.predecessor_transition_id == superseding.id
    assert len({to_b.id, superseding.id, again_to_b.id}) == 3
    assert await _statuses(db) == {
        to_b.id: ProjectionScopeTransitionStatus.SUPERSEDED,
        superseding.id: ProjectionScopeTransitionStatus.SUPERSEDED,
        again_to_b.id: ProjectionScopeTransitionStatus.PENDING,
    }


@pytest.mark.asyncio
async def test_a_return_to_a_last_applied_scope_that_lists_opens_a_transition_back_to_it(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    _save(tmp_path, db, _confluence("root-b"))
    to_b = await _open(db)

    response = _save(tmp_path, db, _confluence("root-a"))
    back = await _open(db)

    assert response.status_code == 200, response.text
    assert back.previous_scope == back.target_scope == to_b.previous_scope
    assert back.predecessor_transition_id == to_b.id
    assert (await _statuses(db))[to_b.id] is ProjectionScopeTransitionStatus.SUPERSEDED


@pytest.mark.asyncio
async def test_a_save_that_changes_no_scope_field_opens_nothing(db, tmp_path):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    unchanged = _save(tmp_path, db, _confluence("root-a"))
    assert await _open(db) is None
    _save(tmp_path, db, _confluence("root-b"))
    to_b = await _open(db)

    unchanged_during_transition = _save(tmp_path, db, _confluence("root-b"))

    assert unchanged.status_code == unchanged_during_transition.status_code == 200
    assert await _statuses(db) == {to_b.id: ProjectionScopeTransitionStatus.PENDING}


@pytest.mark.asyncio
async def test_a_transition_that_changed_meanwhile_answers_409_and_writes_nothing(db, tmp_path, monkeypatch):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    _save(tmp_path, db, _confluence("root-b"))
    to_b = await _open(db)
    upsert_source = db.upsert_source

    async def completed_meanwhile(**kwargs):
        # A run completes the transition after the change read it.
        await db.start_projection_scope_transition(to_b.id, run_id="run-1")
        await db.complete_projection_scope_transition(to_b.id, run_id="run-1")
        return await upsert_source(**kwargs)

    monkeypatch.setattr(db, "upsert_source", completed_meanwhile)
    response = _save(tmp_path, db, _confluence("root-c"))

    assert response.status_code == 409
    assert response.json()["detail"] == "projection_scope_transition_stale"
    assert (await db.get_source(SOURCE_ID))["config"]["page_tree_root"] == "root-b"
    assert await _statuses(db) == {to_b.id: ProjectionScopeTransitionStatus.APPLIED}


@pytest.mark.asyncio
@pytest.mark.parametrize("open_before", [False, True], ids=["opening", "superseding"])
async def test_identical_concurrent_changes_open_one_transition(db, tmp_path, open_before):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    app = create_admin_app(db=db, config=_config(tmp_path))
    upsert_source = db.upsert_source
    both_read = asyncio.Barrier(2)

    async def after_both_read(**kwargs):
        await both_read.wait()
        return await upsert_source(**kwargs)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        if open_before:
            await client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": _confluence("root-b")})
        db.upsert_source = after_both_read
        responses = await asyncio.wait_for(
            asyncio.gather(
                *(client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": _confluence("root-c")}) for _ in range(2))
            ),
            timeout=10,
        )
        db.upsert_source = upsert_source

    assert [response.status_code for response in responses] == [200, 200]
    transitions = await db.list_projection_scope_transitions(SOURCE_ID)
    assert len(transitions) == (2 if open_before else 1)
    assert sum(transition.status is ProjectionScopeTransitionStatus.PENDING for transition in transitions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("sitemap", [None, False], ids=["link-crawl", "sitemap-not-declared-complete"])
async def test_a_change_into_a_mode_that_cannot_list_is_refused(db, tmp_path, sitemap):
    fields = {} if sitemap is None else {"sitemap_authoritative": sitemap}
    await _source(db, "github_pages", _pages("docs", **fields))

    response = _save(tmp_path, db, _pages("guide", **fields))

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "scope_change_requires_listing"
    assert detail["mode"] == "subtree read by link crawl or by a sitemap not declared complete"
    assert "personal access token" in detail["message"]
    assert "new Source" in detail["message"]
    assert (await db.get_source(SOURCE_ID))["config"]["root_url"] == f"{PAGES_SITE}/docs"
    assert await _open(db) is None


@pytest.mark.asyncio
async def test_a_change_from_a_crawl_into_a_complete_sitemap_is_allowed(db, tmp_path):
    await _source(db, "github_pages", _pages("docs"))

    response = _save(tmp_path, db, _pages("guide", sitemap_authoritative=True))
    transition = await _open(db)

    assert response.status_code == 200, response.text
    assert transition.previous_scope["root_url"] == f"{PAGES_SITE}/docs"
    assert transition.target_scope["root_url"] == f"{PAGES_SITE}/guide"


@pytest.mark.asyncio
async def test_a_return_to_a_last_applied_scope_that_cannot_list_supersedes_and_opens_nothing(db, tmp_path):
    await _source(db, "github_pages", _pages("docs"))
    cursor = datetime(2026, 10, 22, 8, 0, tzinfo=timezone.utc)
    await db.upsert_sync_state(SyncState(source=SOURCE_ID, last_sync_at=cursor, last_sync_status="success"))
    _save(tmp_path, db, _pages("guide", sitemap_authoritative=True))
    stuck = await _open(db)
    assert await _cursor(db) == cursor

    elsewhere = _save(tmp_path, db, _pages("other"))
    assert elsewhere.status_code == 409
    assert elsewhere.json()["detail"]["code"] == "scope_change_requires_listing"
    response = _save(tmp_path, db, _pages("docs"))

    assert response.status_code == 200, response.text
    assert await _open(db) is None
    assert await _statuses(db) == {stuck.id: ProjectionScopeTransitionStatus.SUPERSEDED}
    assert (await db.get_source(SOURCE_ID))["config"]["root_url"] == f"{PAGES_SITE}/docs"
    # The superseded transition's runs may have moved the cursor and settled
    # failed Documents of this scope, so the next run reads everything.
    assert await _cursor(db) is None


@pytest.mark.asyncio
async def test_github_pages_branch_is_a_scope_field_and_how_pages_are_listed_is_not(db, tmp_path):
    listing = _pages("docs", auth_mode="github_pat", sitemap_authoritative=False, max_pages=200)
    await _source(db, "github_pages", {**listing, "pat": "secret"})

    unchanged = [
        _save(tmp_path, db, {**listing, "sitemap_authoritative": True}),
        _save(tmp_path, db, {**listing, "max_pages": 500}),
        _save(tmp_path, db, {**listing, "auth_mode": "none", "sitemap_authoritative": True}),
    ]
    opened_by_none = await _open(db)
    branch = _save(tmp_path, db, {**listing, "auth_mode": "none", "sitemap_authoritative": True, "branch": "gh-pages"})
    transition = await _open(db)

    assert [response.status_code for response in unchanged] == [200, 200, 200]
    assert opened_by_none is None
    assert branch.status_code == 200, branch.text
    assert "branch" not in transition.previous_scope
    assert transition.target_scope["branch"] == "gh-pages"


@pytest.mark.asyncio
async def test_a_conversation_source_still_waits_for_its_open_transition(db, tmp_path):
    await _source(db, "teams", {"conversation_ids": ["19:a@thread.v2"]}, execution_owner_user_id="dev")
    first = _save(tmp_path, db, {"conversation_ids": ["19:b@thread.v2"]})
    to_b = await _open(db)

    response = _save(tmp_path, db, {"conversation_ids": ["19:c@thread.v2"]})

    assert first.status_code == 200, first.text
    assert response.status_code == 409
    assert response.json()["detail"] == "projection_scope_transition_active"
    assert await _statuses(db) == {to_b.id: ProjectionScopeTransitionStatus.PENDING}


@pytest.mark.asyncio
@pytest.mark.parametrize("open_before", [False, True], ids=["no-transition", "open-transition"])
async def test_a_scope_change_waits_for_a_running_sync(db, tmp_path, open_before):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    if open_before:
        _save(tmp_path, db, _confluence("root-b"))
    before = await _statuses(db)
    run = await db.enqueue_source_sync_run(source_id=SOURCE_ID, workspace_id="default")
    assert await db.lease_next_source_sync_run(worker_id="worker-a") is not None

    response = _save(tmp_path, db, _confluence("root-c"))

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert (detail["code"], detail["run_id"]) == ("source_sync_running", run.run_id)
    assert await _statuses(db) == before
    assert (await db.get_source_sync_run(run.run_id)).status == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize("open_before", [False, True], ids=["opening", "superseding"])
async def test_of_two_different_concurrent_changes_one_is_saved_and_the_other_is_stale(db, tmp_path, open_before):
    await _source(db, "confluence", {**_confluence("root-a"), "pat": "secret"})
    app = create_admin_app(db=db, config=_config(tmp_path))
    upsert_source = db.upsert_source
    both_read = asyncio.Barrier(2)

    async def after_both_read(**kwargs):
        await both_read.wait()
        return await upsert_source(**kwargs)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        if open_before:
            await client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": _confluence("root-b")})
        before = await _statuses(db)
        db.upsert_source = after_both_read
        responses = await asyncio.wait_for(
            asyncio.gather(
                *(
                    client.put(f"/api/v1/sources/{SOURCE_ID}", json={"config": _confluence(root)})
                    for root in ("root-c", "root-d")
                )
            ),
            timeout=10,
        )
        db.upsert_source = upsert_source

    saved = [response for response in responses if response.status_code == 200]
    refused = [response for response in responses if response.status_code == 409]
    assert (len(saved), len(refused)) == (1, 1)
    assert refused[0].json()["detail"] == "projection_scope_transition_stale"
    transition = await _open(db)
    root = (await db.get_source(SOURCE_ID))["config"]["page_tree_root"]
    assert transition.target_scope["page_tree_root"] == root
    assert transition.predecessor_transition_id == (next(iter(before)) if open_before else None)
    assert len(await _statuses(db)) == len(before) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("listing", "unlisting"),
    [
        ({"auth_mode": "github_pat"}, {"auth_mode": "none"}),
        ({"sitemap_authoritative": True}, {"sitemap_authoritative": False}),
    ],
    ids=["token-off", "complete-sitemap-off"],
)
async def test_a_change_that_stops_listing_while_a_transition_is_open_is_refused(db, tmp_path, listing, unlisting):
    await _source(db, "github_pages", {**_pages("docs", **listing), "pat": "secret"})
    _save(tmp_path, db, _pages("guide", **listing))
    to_guide = await _open(db)

    response = _save(tmp_path, db, _pages("guide", **unlisting))

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "transition_requires_listing"
    assert detail["mode"] == "subtree read by link crawl or by a sitemap not declared complete"
    stored = (await db.get_source(SOURCE_ID))["config"]
    assert {field: stored.get(field) for field in listing} == listing
    assert await _statuses(db) == {to_guide.id: ProjectionScopeTransitionStatus.PENDING}


@pytest.mark.asyncio
async def test_a_change_that_stops_listing_without_an_open_transition_is_saved(db, tmp_path):
    await _source(db, "github_pages", {**_pages("docs", auth_mode="github_pat"), "pat": "secret"})

    response = _save(tmp_path, db, _pages("docs"))

    assert response.status_code == 200, response.text
    assert await _open(db) is None


@pytest.mark.asyncio
async def test_a_branch_pinned_transition_stored_before_branch_was_a_scope_field_reads_the_configured_branch(
    db, tmp_path
):
    pinned = _pages("guide", auth_mode="github_pat", branch="gh-pages")
    await _source(db, "github_pages", {**pinned, "pat": "secret"})
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-before-branch",
            source_id=SOURCE_ID,
            previous_scope={"sync_mode": "subtree", "root_url": f"{PAGES_SITE}/docs"},
            target_scope={"sync_mode": "subtree", "root_url": f"{PAGES_SITE}/guide"},
            created_at="2026-10-20T09:00:00+00:00",
        )
    )

    response = _save(tmp_path, db, _pages("other", auth_mode="github_pat", branch="gh-pages"))
    superseding = await _open(db)

    assert response.status_code == 200, response.text
    assert superseding.predecessor_transition_id == "scope-transition-before-branch"
    assert superseding.previous_scope == {
        "sync_mode": "subtree",
        "root_url": f"{PAGES_SITE}/docs",
        "branch": "gh-pages",
    }
