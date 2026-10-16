"""The HTTP request body is never the source of authorization identity.

A caller cannot impersonate another user by stuffing ``user_id`` into the
search request body: the server resolves the principal server-side, and the
public request schema rejects identity fields supplied by callers.
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
from tests.embedding_fixture import unconfigured_embedding


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
        created_at=now,
        updated_at=now,
        status="active",
    )


def _install_search_engine(monkeypatch, vector_ids: list[str]) -> None:
    """Serve admin searches from SQLite with a vector channel that matches ``vector_ids``."""

    async def fake_build_search_engine(_provider, db, config, *, audit_logger=None):
        from memforge.config import RetrievalConfig
        from memforge.retrieval.search import SearchEngine
        from memforge.storage.adapters.sqlite import build_sqlite_adapters

        class _Coll:
            def upsert(self, **_):
                pass

            def query(self, **_):
                return {
                    "ids": [list(vector_ids)],
                    "distances": [[0.1] * len(vector_ids)],
                }

            def delete(self, **_):
                pass

            def get(self, **_):
                return {"ids": []}

        adapters = build_sqlite_adapters(db, _Coll())
        return SearchEngine(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
            embedding=unconfigured_embedding(),
            config=RetrievalConfig(),
            embedding_provider=lambda _query: [0.1, 0.1, 0.1],
        )

    from memforge import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module.DefaultRuntimeProvider,
        "build_search_engine",
        fake_build_search_engine,
    )


@pytest.mark.asyncio
async def test_request_body_user_id_is_not_access_authority(tmp_path, monkeypatch):
    # Seed U2's private memory. Send a search whose body claims to be U2. The
    # server must derive identity from the request, not from the body, and
    # must not return U2's private memory to the resolved principal (U1).
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "principal.db"))
    await database.connect()
    try:
        await database.insert_memory(
            _memory("p-u2-private", "secret meeting notes", visibility=PRIVATE, owner="u-2"),
        )
        await database.insert_memory(
            _memory("p-shared", "team meeting notes", visibility=WORKSPACE, owner=None),
        )

        _install_search_engine(monkeypatch, ["p-u2-private", "p-shared"])

        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            rejected = client.post(
                "/api/v1/memories/search",
                json={
                    "query": "meeting notes",
                    "top_k": 10,
                    # Caller tries to claim a different identity in the body.
                    # The public schema must reject this instead of accepting
                    # user-controlled authorization identity.
                    "user_id": "u-2",
                    "include_private": True,
                },
            )
            rejected_intent = client.post(
                "/api/v1/memories/search",
                json={
                    "query": "meeting notes",
                    "intent": "semantic_lookup",
                },
            )
            response = client.post(
                "/api/v1/memories/search",
                json={
                    "query": "meeting notes",
                    "intent": "known_item",
                    "top_k": 10,
                    "include_private": True,
                },
            )

        assert rejected.status_code == 422, rejected.text
        assert rejected_intent.status_code == 422, rejected_intent.text
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["retrieval_intent"] == {
            "requested_intent": "known_item",
            "resolved_intent": "known_item",
            "intent_source": "requested",
            "fallback_reason": None,
        }
        ids = {row["memory_id"] for row in body["results"]}
        assert "p-u2-private" not in ids
        assert "p-shared" in ids
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_memory_reads_include_the_callers_own_private_memories_by_default(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "own-private.db"))
    await database.connect()
    try:
        await database.insert_memory(
            _memory("p-own-private", "release checklist notes", visibility=PRIVATE, owner=LOCAL_DEV_USER_ID),
        )
        await database.insert_memory(
            _memory("p-u2-private", "release checklist secrets", visibility=PRIVATE, owner="u-2"),
        )
        await database.insert_memory(
            _memory("p-shared", "release checklist overview", visibility=WORKSPACE, owner=None),
        )
        _install_search_engine(monkeypatch, ["p-own-private", "p-u2-private", "p-shared"])

        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            searched = client.post("/api/v1/memories/search", json={"query": "release checklist", "top_k": 10})
            searched_without_private = client.post(
                "/api/v1/memories/search",
                json={"query": "release checklist", "top_k": 10, "include_private": False},
            )
            listed = client.get("/api/v1/memories")
            listed_without_private = client.get("/api/v1/memories", params={"include_private": "false"})

        for response in (searched, searched_without_private, listed, listed_without_private):
            assert response.status_code == 200, response.text
        assert {row["memory_id"] for row in searched.json()["results"]} == {"p-own-private", "p-shared"}
        assert {row["memory_id"] for row in searched_without_private.json()["results"]} == {"p-shared"}
        assert {row["id"] for row in listed.json()["data"]} == {"p-own-private", "p-shared"}
        assert {row["id"] for row in listed_without_private.json()["data"]} == {"p-shared"}
    finally:
        await database.close()


def test_recent_memory_listing_includes_the_callers_own_private_memories_by_default(tmp_path):
    from memforge.server.admin_api import create_admin_app

    scopes = []

    class FakeSearchEngine:
        async def list_recent_memories(self, *, request_scope, **_kwargs):
            scopes.append(request_scope)
            return {"results": [], "result_kind": "current_memories", "is_changelog": False, "next_cursor": None}

    class FakeRuntimeProvider:
        async def build_search_engine(self, _db, _config, *, audit_logger=None):
            return FakeSearchEngine()

    app = create_admin_app(
        config=_config(tmp_path),
        runtime_provider=FakeRuntimeProvider(),
        principal_resolver=lambda _request: "u-1",
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/memories/recent",
            json={
                "time_range": {
                    "start_at": "2026-07-21T00:00:00+00:00",
                    "end_at": "2026-07-28T00:00:00+00:00",
                },
            },
        )

    assert response.status_code == 200, response.text
    [scope] = scopes
    assert scope.user_id == "u-1"
    assert scope.include_private is True
