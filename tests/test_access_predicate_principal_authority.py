"""The HTTP request body is never the source of authorization identity.

A caller cannot impersonate another user by stuffing ``user_id`` into the
search request body: the server resolves the principal server-side, and a
body field is ignored for the purpose of access checks.
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
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
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


@pytest.mark.asyncio
async def test_request_body_user_id_is_not_access_authority(tmp_path, monkeypatch):
    # Seed U2's private memory. Send a search whose body claims to be U2. The
    # server must derive identity from the request, not from the body, and
    # must not return U2's private memory to the resolved principal (U1).
    from memforge.server.admin_api import create_admin_app
    from memforge.retrieval import search as search_module

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "principal.db"))
    await database.connect()
    try:
        await database.insert_memory(
            _memory("p-u2-private", "secret meeting notes",
                    visibility=PRIVATE, owner="u-2"),
        )
        await database.insert_memory(
            _memory("p-shared", "team meeting notes",
                    visibility=WORKSPACE, owner=None),
        )

        async def fake_build_search_engine(db, config, *, audit_logger=None):
            from memforge.config import RetrievalConfig
            from memforge.retrieval.search import SearchEngine
            from memforge.storage.adapters.sqlite import build_sqlite_adapters

            class _Coll:
                def upsert(self, **_):
                    pass

                def query(self, **_):
                    return {
                        "ids": [["p-u2-private", "p-shared"]],
                        "distances": [[0.1, 0.1]],
                    }

                def delete(self, **_):
                    pass

                def get(self, **_):
                    return {"ids": []}

            adapters = build_sqlite_adapters(db, _Coll())
            engine = SearchEngine(
                relational=adapters.relational,
                keyword=adapters.keyword,
                vector=adapters.vector,
                embed_cfg={},
                config=RetrievalConfig(),
            )
            engine._get_or_compute_embedding = lambda query: [0.1, 0.1, 0.1]
            return engine

        from memforge import runtime as runtime_module

        monkeypatch.setattr(
            runtime_module, "build_search_engine", fake_build_search_engine
        )

        from memforge.retrieval.query_analyzer import QueryAnalysis

        async def fake_analyze_query(*args, **kwargs):
            return QueryAnalysis()

        monkeypatch.setattr(search_module, "analyze_query", fake_analyze_query)

        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/memories/search",
                json={
                    "query": "meeting notes",
                    "top_k": 10,
                    # Caller tries to claim a different identity in the body.
                    # The server must ignore this for access purposes.
                    "user_id": "u-2",
                    "include_private": True,
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        ids = {row["memory_id"] for row in body["results"]}
        assert "p-u2-private" not in ids
        assert "p-shared" in ids
    finally:
        await database.close()
