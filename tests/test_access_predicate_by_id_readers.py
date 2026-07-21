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
    Memory,
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


def _memory(mid: str, content: str, *, visibility: str, owner: str | None) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=SHARED_PROJECT_KEY,
        tags=[],
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
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
        response = client.get("/api/memories/m-u2-private")

    assert response.status_code == 404, response.text


@pytest.mark.asyncio
async def test_get_memory_by_id_includes_only_owners_private_by_default(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        own = client.get("/api/memories/m-u1-private")
        other = client.get("/api/memories/m-u2-private")

    assert own.status_code == 200, own.text
    assert own.json()["id"] == "m-u1-private"
    assert other.status_code == 404, other.text


@pytest.mark.asyncio
async def test_list_memories_excludes_other_users_private(seeded_app):
    app, _database = seeded_app
    with TestClient(app) as client:
        # Simple-filter list (no ``search`` param).
        simple = client.get("/api/memories", params={"limit": 50})
        assert simple.status_code == 200, simple.text
        simple_ids = {row["id"] for row in simple.json()["data"]}
        assert "m-u2-private" not in simple_ids
        # U1's own private row is also hidden from the workspace default;
        # only WORKSPACE rows survive.
        assert "m-u1-private" not in simple_ids
        assert "m-shared" in simple_ids

        # Search-mode list path.
        searched = client.get(
            "/api/memories",
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
            "/api/memories",
            params={"include_private": "true", "limit": 50},
        )

    assert response.status_code == 200, response.text
    ids = {row["id"] for row in response.json()["data"]}
    assert "m-shared" in ids
    assert "m-u1-private" in ids
    assert "m-u2-private" not in ids
