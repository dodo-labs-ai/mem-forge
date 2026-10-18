"""Every OSS read surface returns all of the single owner's Memories.

The OSS edition has one principal, ``LOCAL_DEV_USER_ID``. Agent-session
Memories, hand-made Memories and Memories from a private configured Source
are private to that owner; each read surface returns them alongside
workspace Memories and never returns another user's private Memory.

A new read surface is one more row in ``SURFACES``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from memforge import plugin_mcp_proxy, tool_client
from memforge.agent_sessions import ensure_agent_session_source
from memforge.api_target import build_target
from memforge.config import AppConfig, RetrievalConfig
from memforge.main import cli
from memforge.models import (
    UNSORTED_PROJECT_KEY,
    DocumentRecord,
    Memory,
    MemorySource,
    Visibility,
    content_hash,
)
from memforge.retrieval.search import SearchEngine
from memforge.runtime import DefaultRuntimeProvider
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from tests.embedding_fixture import unconfigured_embedding


OTHER_USER_ID = "u-other"
QUERY = "release checklist"

OWNER_AGENT_SESSION = "m-owner-agent-session"
OWNER_HAND_MADE = "m-owner-hand-made"
OWNER_PRIVATE_SOURCE = "m-owner-private-source"
WORKSPACE_VISIBLE = "m-workspace"
OTHER_USERS_PRIVATE = "m-other-private"

OWNER_READABLE = {OWNER_AGENT_SESSION, OWNER_HAND_MADE, OWNER_PRIVATE_SOURCE, WORKSPACE_VISIBLE}


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _memory(memory_id: str, *, visibility: str, owner: str | None) -> Memory:
    now = datetime.now(timezone.utc)
    content = f"The {QUERY} for {memory_id} lists the deploy gates."
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        visibility=visibility,
        owner_user_id=owner,
        project_key=UNSORTED_PROJECT_KEY,
        created_at=now,
        updated_at=now,
        status="active",
    )


async def _seed_memory(
    db: Database,
    memory: Memory,
    *,
    source_id: str,
    source_type: str,
    collection: _VectorCollection,
) -> None:
    now = datetime.now(timezone.utc)
    doc_id = f"doc-{memory.id}"
    await db.insert_memory(memory)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"https://example.test/{doc_id}",
            title=doc_id,
            space_or_project="PAY",
            author="owner",
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=1,
            last_synced=now,
        )
    )
    await db.restore_memory_source_snapshot(
        MemorySource(
            memory_id=memory.id,
            doc_id=doc_id,
            source_id=source_id,
            source_type=source_type,
            source_updated_at=now,
        )
    )
    collection.add(
        memory.id,
        {
            "memory_type": memory.memory_type,
            "project_key": memory.project_key,
            "visibility": memory.visibility,
            "owner_user_id": memory.owner_user_id or "",
            "status": memory.status,
        },
    )


class _VectorCollection:
    """A vector collection that honours the access predicate's where-filter."""

    def __init__(self) -> None:
        self._metadata: dict[str, dict[str, Any]] = {}

    def add(self, memory_id: str, metadata: dict[str, Any]) -> None:
        self._metadata[memory_id] = metadata

    def query(self, *, query_embeddings, n_results, where=None, **_):
        ids = [mid for mid, meta in self._metadata.items() if where is None or _matches(meta, where)][:n_results]
        return {
            "ids": [list(ids) for _ in query_embeddings],
            "distances": [[0.1] * len(ids) for _ in query_embeddings],
        }

    def upsert(self, **_):
        pass

    def delete(self, **_):
        pass

    def get(self, **_):
        return {"ids": []}


def _matches(metadata: dict[str, Any], clause: dict[str, Any]) -> bool:
    if "$and" in clause:
        return all(_matches(metadata, part) for part in clause["$and"])
    if "$or" in clause:
        return any(_matches(metadata, part) for part in clause["$or"])
    for key, expected in clause.items():
        if isinstance(expected, dict) and "$in" in expected:
            if metadata.get(key) not in expected["$in"]:
                return False
        elif metadata.get(key) != expected:
            return False
    return True


def _search_engine(db: Database, collection: _VectorCollection) -> SearchEngine:
    adapters = build_sqlite_adapters(db, collection)
    return SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embedding=unconfigured_embedding(),
        config=RetrievalConfig(),
        embedding_provider=lambda _query: [0.1, 0.1, 0.1],
    )


class _SingleOwnerRuntime(DefaultRuntimeProvider):
    def __init__(self, collection: _VectorCollection) -> None:
        self._collection = collection

    async def build_search_engine(self, db, config, *, audit_logger=None):
        return _search_engine(db, self._collection)


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._body if size < 0 else self._body[:size]


class _AppOpener:
    """Serve the HTTP clients' urllib requests from the in-process app."""

    def __init__(self, client: TestClient) -> None:
        self._client = client

    def open(self, request, timeout=None):
        parts = urlsplit(request.full_url)
        path = f"{parts.path}?{parts.query}" if parts.query else parts.path
        response = self._client.request(
            request.get_method(),
            path,
            content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            raise HTTPError(
                request.full_url,
                response.status_code,
                response.reason_phrase,
                response.headers,
                io.BytesIO(response.content),
            )
        return _Response(response.content)


@dataclass
class _OwnerDeployment:
    client: TestClient
    engine: SearchEngine


@pytest.fixture
async def owner_deployment(tmp_path, monkeypatch):
    """A standalone OSS deployment holding the owner's Memories and another user's."""
    from memforge.server.admin_api import create_admin_app

    for name in ("MEMFORGE_API_URL", "MEMFORGE_EDITION", "MEMFORGE_WORKSPACE_ID", "MEMFORGE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(tmp_path / "workspace-bindings.json"))
    monkeypatch.setattr(plugin_mcp_proxy, "configured_target", lambda *_args: build_target(origin=None))
    monkeypatch.setattr(plugin_mcp_proxy, "configured_api_token", lambda: "")
    monkeypatch.setattr(plugin_mcp_proxy, "_active_repo_identifier", lambda: None)

    database = Database(str(tmp_path / "single-owner.db"))
    await database.connect()
    collection = _VectorCollection()
    try:
        owner_sessions = await ensure_agent_session_source(
            database, client="claude-code", owner_user_id=LOCAL_DEV_USER_ID
        )
        other_sessions = await ensure_agent_session_source(
            database, client="claude-code", owner_user_id=OTHER_USER_ID
        )
        await database.upsert_source(
            "src-owner-notes", "local_markdown", "Owner notes", "{}",
            access_policy="private", owner_user_id=LOCAL_DEV_USER_ID,
        )
        await database.upsert_source(
            "src-team-wiki", "local_markdown", "Team wiki", "{}",
            access_policy="workspace", owner_user_id=LOCAL_DEV_USER_ID,
        )

        private, workspace = Visibility.PRIVATE.value, Visibility.WORKSPACE.value
        for memory, source_id, source_type in (
            (_memory(OWNER_AGENT_SESSION, visibility=private, owner=LOCAL_DEV_USER_ID),
             owner_sessions["id"], "agent_session"),
            (_memory(OWNER_HAND_MADE, visibility=private, owner=LOCAL_DEV_USER_ID),
             "user_memory", "user_memory"),
            (_memory(OWNER_PRIVATE_SOURCE, visibility=private, owner=LOCAL_DEV_USER_ID),
             "src-owner-notes", "local_markdown"),
            (_memory(WORKSPACE_VISIBLE, visibility=workspace, owner=None),
             "src-team-wiki", "local_markdown"),
            (_memory(OTHER_USERS_PRIVATE, visibility=private, owner=OTHER_USER_ID),
             other_sessions["id"], "agent_session"),
        ):
            await _seed_memory(database, memory, source_id=source_id, source_type=source_type, collection=collection)

        app = create_admin_app(db=database, config=_config(tmp_path), runtime_provider=_SingleOwnerRuntime(collection))
        with TestClient(app) as client:
            opener = _AppOpener(client)
            monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: opener)
            monkeypatch.setattr(plugin_mcp_proxy, "build_opener", lambda *_handlers: opener)
            yield _OwnerDeployment(client=client, engine=_search_engine(database, collection))
    finally:
        await database.close()


def _ok(response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()


def _result_ids(payload: dict[str, Any], *, key: str = "results", id_field: str = "memory_id") -> set[str]:
    assert "error" not in payload, payload
    return {row[id_field] for row in payload[key]}


def _recent_window() -> dict[str, str]:
    now = datetime.now(timezone.utc)
    return {
        "start_at": (now - timedelta(days=1)).isoformat(),
        "end_at": (now + timedelta(days=1)).isoformat(),
    }


def _cli_search(*args: str) -> set[str]:
    result = CliRunner().invoke(cli, [*args, QUERY, "--top-k", "10"])
    assert result.exit_code == 0, result.output
    return _result_ids(json.loads(result.stdout))


async def _search_api(deployment: _OwnerDeployment) -> set[str]:
    return _result_ids(_ok(deployment.client.post("/api/v1/memories/search", json={"query": QUERY, "top_k": 10})))


async def _time_range_search_api(deployment: _OwnerDeployment) -> set[str]:
    today = datetime.now(timezone.utc).date()
    time_range = {"start_date": (today - timedelta(days=1)).isoformat(), "end_date": (today + timedelta(days=1)).isoformat()}
    response = deployment.client.post("/api/v1/memories/search", json={"time_range": time_range, "top_k": 10})
    return _result_ids(_ok(response))


async def _recent_memories_api(deployment: _OwnerDeployment) -> set[str]:
    response = deployment.client.post("/api/v1/memories/recent", json={"time_range": _recent_window()})
    return _result_ids(_ok(response))


async def _memory_list_api(deployment: _OwnerDeployment) -> set[str]:
    return _result_ids(_ok(deployment.client.get("/api/v1/memories")), key="data", id_field="id")


async def _tool_client_search(_deployment: _OwnerDeployment) -> set[str]:
    client = tool_client.ToolClient(target=build_target(origin=None), api_token=None)
    return _result_ids(client.search(query=QUERY, top_k=10))


async def _cli_search_command(_deployment: _OwnerDeployment) -> set[str]:
    return _cli_search("search")


async def _cli_memory_search_command(_deployment: _OwnerDeployment) -> set[str]:
    return _cli_search("memory", "search")


async def _mcp_search_tool(_deployment: _OwnerDeployment) -> set[str]:
    return _result_ids(plugin_mcp_proxy._call_tool("search", {"query": QUERY, "top_k": 10}))


async def _mcp_recent_memories_tool(_deployment: _OwnerDeployment) -> set[str]:
    return _result_ids(plugin_mcp_proxy._call_tool("list_recent_memories", {"time_range": _recent_window()}))


async def _hook_context_search(deployment: _OwnerDeployment) -> set[str]:
    response = deployment.client.post(
        "/api/v1/hooks/context",
        json={
            "client": "claude-code",
            "hook": "SessionStart",
            "workspace": "/work/payroll",
            "prompt": QUERY,
            "max_memories": 10,
            "include_recent_changes": False,
        },
    )
    return _result_ids(_ok(response), key="memories", id_field="id")


async def _search_engine_without_request_scope(deployment: _OwnerDeployment) -> set[str]:
    return {row.memory_id for row in (await deployment.engine.search(QUERY, top_k=10))["results"]}


SURFACES: tuple[Callable[[_OwnerDeployment], Awaitable[set[str]]], ...] = (
    _search_api,
    _time_range_search_api,
    _recent_memories_api,
    _memory_list_api,
    _tool_client_search,
    _cli_search_command,
    _cli_memory_search_command,
    _mcp_search_tool,
    _mcp_recent_memories_tool,
    _hook_context_search,
    _search_engine_without_request_scope,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_surface", SURFACES, ids=lambda surface: surface.__name__.lstrip("_"))
async def test_read_surface_returns_every_owner_memory_and_no_other_users_private(owner_deployment, read_surface):
    assert await read_surface(owner_deployment) == OWNER_READABLE
