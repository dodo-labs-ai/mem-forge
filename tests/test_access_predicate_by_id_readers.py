"""The access predicate gates by-id and list reads, not just search.

A row that the caller cannot see by the workspace-default predicate must be
returned as 404 for an id-based GET, never as 200. The list endpoints (both
the search-mode list and the simple-filter list) must apply the same
predicate, so another user's private rows never leak through pagination
either. The by-id detail route is personalized by default: it includes the
resolved principal's own private row, but still returns 404 for another user's
private row. List/search routes keep the explicit ``include_private=True``
opt-in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import (
    DocumentRecord,
    Memory,
    MemorySource,
    MemoryStatus,
    SHARED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _memory(
    mid: str,
    content: str,
    *,
    visibility: str,
    owner: str | None,
    status: str = MemoryStatus.ACTIVE.value,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=SHARED_PROJECT_KEY,
        created_at=now,
        updated_at=now,
        status=status,
    )


@pytest.fixture
async def seeded_app(tmp_path):
    """A FastAPI app seeded with one workspace row, U1's private, and U2's private."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "by_id_readers.db"))
    await database.connect()
    try:
        await database.insert_memory(
            _memory("m-shared", "team meeting notes", visibility=WORKSPACE, owner=None),
        )
        await database.insert_memory(
            _memory("m-u1-private", "u1 personal note", visibility=PRIVATE, owner=LOCAL_DEV_USER_ID),
        )
        await database.insert_memory(
            _memory("m-u2-private", "u2 secret meeting notes", visibility=PRIVATE, owner="u-2"),
        )

        app = create_admin_app(db=database, config=cfg)
        yield app, database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_get_memory_by_id_returns_404_for_other_users_private(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        # Resolved principal is LOCAL_DEV_USER_ID. U2's private row must not
        # leak through the by-id reader; the handler must build a workspace
        # default scope and apply the predicate.
        response = client.get("/api/v1/memories/m-u2-private")

    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_get_memory_by_id_includes_only_owners_private_by_default(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        own = client.get("/api/v1/memories/m-u1-private")
        other = client.get("/api/v1/memories/m-u2-private")

    assert own.status_code == 200, own.text
    assert own.json()["id"] == "m-u1-private"
    assert other.status_code == 404, other.text


@pytest.mark.asyncio
async def test_list_memories_excludes_other_users_private(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        # Simple-filter list (no ``search`` param).
        simple = client.get("/api/v1/memories", params={"limit": 50})
        assert simple.status_code == 200, simple.text
        simple_ids = {row["id"] for row in simple.json()["data"]}
        assert "m-u2-private" not in simple_ids
        # U1's own private row is also hidden from the workspace default;
        # only WORKSPACE rows survive.
        assert "m-u1-private" not in simple_ids
        assert "m-shared" in simple_ids

        # Search-mode list path.
        searched = client.get(
            "/api/v1/memories",
            params={"search": "meeting", "limit": 50},
        )
        assert searched.status_code == 200, searched.text
        searched_ids = {row["id"] for row in searched.json()["data"]}
        assert "m-u2-private" not in searched_ids
        assert "m-shared" in searched_ids


@pytest.mark.asyncio
async def test_list_memories_personalized_includes_only_owners_private(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/memories",
            params={"include_private": "true", "limit": 50},
        )

    assert response.status_code == 200, response.text
    ids = {row["id"] for row in response.json()["data"]}
    assert "m-shared" in ids
    assert "m-u1-private" in ids
    assert "m-u2-private" not in ids


OTHER_USER_ID = "u-2"
RETIRED = MemoryStatus.RETIRED.value
SUPERSEDED = MemoryStatus.SUPERSEDED.value
PENDING_REVIEW = MemoryStatus.PENDING_REVIEW.value


@pytest.fixture
async def lifecycle_app(tmp_path):
    """Workspace and private Memories in every lifecycle status, for the caller and another user."""
    from memforge.server.admin_api import create_admin_app

    database = Database(str(tmp_path / "lifecycle_readers.db"))
    await database.connect()
    try:
        for memory in (
            _memory("m-active", "current rule", visibility=WORKSPACE, owner=None),
            _memory("m-retired", "retired rule", visibility=WORKSPACE, owner=None, status=RETIRED),
            _memory("m-superseded", "old rule", visibility=WORKSPACE, owner=None, status=SUPERSEDED),
            _memory("m-pending", "proposed rule", visibility=WORKSPACE, owner=None, status=PENDING_REVIEW),
            _memory("m-own-retired", "my retired note", visibility=PRIVATE, owner=LOCAL_DEV_USER_ID, status=RETIRED),
            _memory("m-other-retired", "their retired note", visibility=PRIVATE, owner=OTHER_USER_ID, status=RETIRED),
            _memory("m-other-superseded", "their old note", visibility=PRIVATE, owner=OTHER_USER_ID, status=SUPERSEDED),
        ):
            await database.insert_memory(memory)
        yield create_admin_app(db=database, config=_config(tmp_path)), database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_list_memories_without_status_lists_active_memories_only(lifecycle_app):
    app, _database = lifecycle_app
    with TestClient(app) as client:
        response = client.get("/api/v1/memories", params={"include_private": "true", "limit": 50})

    assert response.status_code == 200, response.text
    assert [row["id"] for row in response.json()["data"]] == ["m-active"]
    assert response.json()["total"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_ids"),
    [
        (RETIRED, {"m-retired", "m-own-retired"}),
        (MemoryStatus.DECAYED.value, {"m-retired", "m-own-retired"}),
        (SUPERSEDED, {"m-superseded"}),
        (PENDING_REVIEW, {"m-pending"}),
    ],
)
async def test_list_memories_status_filter_lists_that_status_without_other_users_private_rows(
    lifecycle_app, status, expected_ids
):
    app, _database = lifecycle_app
    with TestClient(app) as client:
        listed = client.get("/api/v1/memories", params={"status": status, "include_private": "true", "limit": 50})
        searched = client.get(
            "/api/v1/memories",
            params={"status": status, "search": "rule note", "include_private": "true", "limit": 50},
        )

    assert listed.status_code == 200, listed.text
    assert {row["id"] for row in listed.json()["data"]} == expected_ids
    assert listed.json()["total"] == len(expected_ids)
    assert searched.status_code == 200, searched.text
    assert not {row["id"] for row in searched.json()["data"]} - expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_id", ["m-retired", "m-superseded", "m-pending", "m-own-retired"])
async def test_get_memory_by_id_reads_every_lifecycle_status(lifecycle_app, memory_id):
    app, _database = lifecycle_app
    with TestClient(app) as client:
        response = client.get(f"/api/v1/memories/{memory_id}")

    assert response.status_code == 200, response.text
    detail = response.json()
    assert detail["id"] == memory_id
    assert detail["relations"] == []
    assert detail["source_backed"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_id", ["m-other-retired", "m-other-superseded"])
async def test_get_memory_by_id_hides_other_users_private_memories_in_every_status(lifecycle_app, memory_id):
    app, _database = lifecycle_app
    with TestClient(app) as client:
        response = client.get(f"/api/v1/memories/{memory_id}")

    assert response.status_code == 404, response.text


async def _support_from(database: Database, memory_id: str, source_id: str) -> None:
    now = datetime.now(timezone.utc)
    doc_id = f"doc-{memory_id}-{source_id}"
    await database.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"https://example.test/{doc_id}",
            title=doc_id,
            space_or_project="PAY",
            author="A",
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=1,
            last_synced=now,
        )
    )
    await database.restore_memory_source_snapshot(
        MemorySource(memory_id=memory_id, doc_id=doc_id, source_id=source_id, source_type="jira")
    )


@pytest.mark.asyncio
async def test_list_memories_never_names_another_users_private_source(lifecycle_app):
    app, database = lifecycle_app
    await database.upsert_source(
        "src-team", "jira", "Team board", "{}", access_policy="workspace", owner_user_id=OTHER_USER_ID
    )
    await database.upsert_source(
        "src-theirs", "jira", "Their private board", "{}", access_policy="private", owner_user_id=OTHER_USER_ID
    )
    await _support_from(database, "m-active", "src-team")
    await _support_from(database, "m-active", "src-theirs")

    with TestClient(app) as client:
        listed = client.get("/api/v1/memories", params={"include_private": "true", "limit": 50})
        detail = client.get("/api/v1/memories/m-active")

    assert listed.status_code == 200, listed.text
    [row] = listed.json()["data"]
    assert row["id"] == "m-active"
    assert [source["source_id"] for source in row["sources"]] == ["src-team"]
    assert "Their private board" not in listed.text
    assert detail.status_code == 200, detail.text
    assert [source["source_id"] for source in detail.json()["sources"]] == ["src-team"]
