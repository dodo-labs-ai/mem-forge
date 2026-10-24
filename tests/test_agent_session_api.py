from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.agent_sessions import agent_session_source_id
from memforge.config import AppConfig
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.runtime import DefaultRuntimeProvider
from memforge.storage.database import Database
from tests.test_sync_bookkeeping import _hold_document

CAPTURE_REMOVED_BODY = {
    "detail": {
        "code": "agent_session_capture_removed",
        "message": (
            "MemForge no longer captures agent sessions automatically. Update the MemForge memory plugin."
        ),
    }
}
# What an earlier plugin release sends to each removed capture route.
EARLIER_PLUGIN_UPLOADS = (
    (
        "/api/v1/agent-sessions/windows",
        {
            "schema_version": "agent-session-window/v1",
            "plugin_version": "0.2.0-beta.1",
            "client": "codex",
            "session_id": "sess-old-plugin",
            "trigger": "REQUIRED_CAPTURE",
            "workspace": "/workspace/mem-forge",
            "events": [{"kind": "user_message", "text": "remember to run ruff"}],
            "process_now": True,
        },
    ),
    (
        "/api/v1/hooks/receipts",
        {
            "client": "claude-code",
            "session_id": "sess-old-plugin",
            "hook": "Stop",
            "workspace": "/workspace/mem-forge",
        },
    ),
)


@pytest.fixture(autouse=True)
def _stub_memory_embedding(monkeypatch):
    async def _fake_embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", _fake_embed)


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


async def _seed_source_project(
    db: Database,
    *,
    doc_id: str,
    project: str,
    last_modified: datetime,
    memory_ids: list[str],
    source_id: str = "src-agent-sessions-codex",
    client: str = "codex",
) -> None:
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"agent-session://codex/sess/{doc_id}",
            title=f"Agent Session {doc_id}",
            space_or_project=project,
            author=client,
            last_modified=last_modified,
            labels=[],
            version=f"version-{doc_id}",
            content_hash=f"hash-{doc_id}",
            token_count=100,
            last_synced=last_modified,
            client=client,
        )
    )
    await _hold_document(
        db,
        source_id=source_id,
        source_type="agent_session",
        doc_id=doc_id,
        title=f"Agent Session {doc_id}",
        markdown=f"# Agent Session {doc_id}",
        version=f"version-{doc_id}",
        source_url=f"agent-session://codex/sess/{doc_id}",
        space_or_project=project,
    )
    for memory_id in memory_ids:
        memory = Memory(
            id=memory_id,
            memory_type="fact",
            content=f"Memory {memory_id}",
            content_hash=content_hash(f"Memory {memory_id}"),
            project_key=project,
            created_at=last_modified,
            updated_at=last_modified,
            status="active",
        )
        await db.insert_memory(memory)
        await db.add_memory_source(memory.id, doc_id, "agent_session", source_updated_at=None)


def test_source_projects_endpoint_groups_agent_session_memory_by_project(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)
    source_id = agent_session_source_id("codex", "dev")

    async def _setup():
        await database.connect()
        await database.upsert_source(
            source_id,
            "agent_session",
            "Codex Session",
            json.dumps({}),
            "private",
            "dev",
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-mem-inception-1",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-agent-1"],
            source_id=source_id,
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-mem-inception-2",
            project="mem-inception",
            last_modified=base_time + timedelta(minutes=5),
            memory_ids=["mem-agent-2"],
            source_id=source_id,
        )
        await _seed_source_project(
            database,
            doc_id="agent-doc-payroll-1",
            project="payroll-processing-service",
            last_modified=base_time - timedelta(days=1),
            memory_ids=["mem-agent-3"],
            source_id=source_id,
        )

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.get(f"/api/v1/sources/{source_id}/projects")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "source_id": source_id,
            "projects": [
                {
                    "project": "mem-inception",
                    "document_count": 2,
                    "memory_count": 2,
                    "last_observed_at": "2026-06-01T10:05:00+00:00",
                },
                {
                    "project": "payroll-processing-service",
                    "document_count": 1,
                    "memory_count": 1,
                    "last_observed_at": "2026-05-31T10:00:00+00:00",
                },
            ],
        }
    finally:
        asyncio.run(database.close())


def _make_receipt(
    *,
    doc_id,
    session_id,
    outcome,
    source_id="src-agent-sessions-codex",
    source_kind="generated_agent_window_summary",
    reason=None,
    updated_at=None,
):
    from memforge.models import AgentSessionReceipt

    metadata: dict = {"outcome": outcome}
    if reason is not None:
        metadata["reason"] = reason

    return AgentSessionReceipt(
        doc_id=doc_id,
        source_id=source_id,
        client="codex",
        session_id=session_id,
        trigger="Stop",
        workspace="/workspace/mem-forge",
        repo=None,
        branch=None,
        commit_sha=None,
        history_window_kind="transcript_window",
        history_window_start=None,
        history_window_end=None,
        submitted_at="2026-05-30T00:00:00+00:00",
        document_hash="sha256:deadbeef",
        source_kind=source_kind,
        document_uri="",
        metadata=metadata,
        updated_at=updated_at,
    )


@pytest.mark.asyncio
async def test_successful_agent_session_receipt_advances_source_activity_monotonically(
    tmp_path,
):
    database = Database(str(tmp_path / "agent-session-activity.db"))
    source_id = "src-agent-sessions-codex"
    await database.connect()
    try:
        await database.upsert_source(
            id=source_id,
            type="agent_session",
            name="Codex Session",
            config_json="{}",
            access_policy="private",
            owner_user_id="owner-user",
        )

        await database.upsert_agent_session_receipt(
            _make_receipt(
                doc_id="failed-window",
                session_id="session-activity",
                source_id=source_id,
                outcome="failed",
                updated_at="2026-07-21T10:00:00Z",
            )
        )
        assert (await database.get_source(source_id))["last_sync"] is None

        await database.upsert_agent_session_receipt(
            _make_receipt(
                doc_id="successful-explicit-document",
                session_id="session-activity",
                source_id=source_id,
                source_kind="generated_agent_summary",
                outcome="knowledge_patched",
                updated_at="2026-07-21T10:30:00Z",
            )
        )
        assert (await database.get_source(source_id))["last_sync"] is None

        await database.upsert_agent_session_receipt(
            _make_receipt(
                doc_id="successful-window",
                session_id="session-activity",
                source_id=source_id,
                outcome="knowledge_patched",
                updated_at="2026-07-21T09:00:00Z",
            )
        )
        assert (await database.get_source(source_id))["last_sync"] == "2026-07-21T09:00:00+00:00"

        await database.upsert_agent_session_receipt(
            _make_receipt(
                doc_id="older-successful-window",
                session_id="session-activity",
                source_id=source_id,
                outcome="no_output",
                updated_at="2026-07-21T08:00:00Z",
            )
        )
        assert (await database.get_source(source_id))["last_sync"] == "2026-07-21T09:00:00+00:00"
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_agent_session_receipt_and_source_activity_rollback_together(tmp_path):
    database = Database(str(tmp_path / "agent-session-activity-rollback.db"))
    source_id = "src-agent-sessions-codex"
    await database.connect()
    try:
        await database.upsert_source(
            id=source_id,
            type="agent_session",
            name="Codex Session",
            config_json="{}",
            access_policy="private",
            owner_user_id="owner-user",
        )
        await database.db.execute(
            """CREATE TRIGGER reject_agent_session_activity
               BEFORE UPDATE OF last_sync ON sources
               BEGIN
                   SELECT RAISE(ABORT, 'activity update rejected');
               END"""
        )
        await database.db.commit()

        receipt = _make_receipt(
            doc_id="atomic-window",
            session_id="session-activity",
            source_id=source_id,
            outcome="knowledge_patched",
            updated_at="2026-07-21T09:00:00Z",
        )
        with pytest.raises(sqlite3.IntegrityError, match="activity update rejected"):
            await database.upsert_agent_session_receipt(receipt)

        assert await database.get_agent_session_receipt(receipt.doc_id) is None
        assert (await database.get_source(source_id))["last_sync"] is None
    finally:
        await database.close()


def test_summarize_agent_session_outcomes_counts_and_fraction(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            outcomes = ["knowledge_patched", "package_created", "no_output", "failed"]
            for index, outcome in enumerate(outcomes):
                await database.upsert_agent_session_receipt(
                    _make_receipt(doc_id=f"doc-{index}", session_id="sess-sum", outcome=outcome)
                )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-sum")
            assert summary["session_id"] == "sess-sum"
            assert summary["total"] == 4
            assert summary["processed_total"] == 3
            assert summary["counts"] == {"knowledge_patched": 2, "no_output": 1, "failed": 1}
            assert summary["no_output_fraction"] == 1 / 3
        finally:
            await database.close()

    asyncio.run(_run())


def test_summarize_empty_session_returns_zero_fraction(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            summary = await database.summarize_agent_session_outcomes(session_id="nobody")
            assert summary["total"] == 0
            assert summary["processed_total"] == 0
            assert summary["counts"] == {"knowledge_patched": 0, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
            assert summary["latest_failure"] is None
        finally:
            await database.close()

    asyncio.run(_run())


def test_summarize_agent_session_outcomes_includes_latest_failure(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="fail-1",
                    session_id="sess-fail",
                    outcome="failed",
                    reason="LLM timeout",
                    updated_at="2026-05-29T00:00:00+00:00",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="fail-2",
                    session_id="sess-fail",
                    outcome="failed",
                    reason="schema validation error",
                    updated_at="2026-05-30T12:00:00+00:00",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="ok-1",
                    session_id="sess-fail",
                    outcome="knowledge_patched",
                )
            )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-fail")
            assert summary["counts"]["failed"] == 2
            assert summary["latest_failure"] == {
                "count": 2,
                "reason": "schema validation error",
                "last_seen_at": "2026-05-30T12:00:00+00:00",
            }
        finally:
            await database.close()

    asyncio.run(_run())


def test_summarize_agent_session_outcomes_ignores_explicit_document_metadata(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    import asyncio

    async def _run():
        await database.connect()
        try:
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="window-doc",
                    session_id="sess-filter",
                    outcome="knowledge_patched",
                )
            )
            await database.upsert_agent_session_receipt(
                _make_receipt(
                    doc_id="explicit-doc",
                    session_id="sess-filter",
                    outcome="no_output",
                    source_kind="generated_agent_summary",
                )
            )
            summary = await database.summarize_agent_session_outcomes(session_id="sess-filter")
            assert summary["total"] == 1
            assert summary["processed_total"] == 1
            assert summary["counts"] == {"knowledge_patched": 1, "no_output": 0, "failed": 0}
            assert summary["no_output_fraction"] == 0.0
        finally:
            await database.close()

    asyncio.run(_run())


def test_db_migration_partitions_legacy_client_source_by_owner(tmp_path):
    """Migration 45 re-points legacy client documents to private owner Sources."""
    import asyncio
    import aiosqlite

    async def _run():
        db_path = str(tmp_path / "migration.db")
        # The current schema represents the pre-45 shape once access columns
        # exist but coding Sources are still shared per client.
        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys = ON")

        from memforge.storage.database import SCHEMA, MIGRATIONS

        await conn.executescript(SCHEMA)
        now_ts = "2026-06-01T10:00:00+00:00"
        # Record all migrations before the partition as applied. SCHEMA already
        # contains their final table shape.
        for version, description, _ in MIGRATIONS:
            if version >= 45:
                break
            await conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                (version, description, now_ts),
            )
        await conn.commit()

        # Seed one old per-client Source containing documents from two users.
        await conn.execute(
            """INSERT INTO sources (
                   id, type, name, config, created_by_user_id, owner_user_id,
                   access_policy, access_state, execution_owner_user_id
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "src-agent-sessions-codex",
                "agent_session",
                "Codex Session",
                json.dumps({"client": "codex", "documents_dir": "/tmp/agent-sessions/codex"}),
                "alice",
                "alice",
                "private",
                "active",
                "alice",
            ),
        )
        for owner, doc_id in [("alice", "doc-alice"), ("bob", "doc-bob")]:
            await conn.execute(
                """INSERT INTO documents
                   (doc_id, source, source_url, title, space_or_project, author,
                    last_modified, labels, version, content_hash, last_synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    "src-agent-sessions-codex",
                    f"agent-session://codex/sess/{doc_id}",
                    f"{owner} doc",
                    "workspace",
                    "codex",
                    now_ts,
                    "[]",
                    "v1",
                    f"hash-{doc_id}",
                    now_ts,
                ),
            )
            await conn.execute(
                """INSERT INTO agent_session_receipts
                   (doc_id, source_id, client, session_id, trigger, workspace,
                    history_window_kind, submitted_at, document_hash, source_kind,
                    document_uri, metadata, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc_id,
                    "src-agent-sessions-codex",
                    "codex",
                    "sess-x",
                    "Stop",
                    "/workspace",
                    "session",
                    now_ts,
                    f"hash-{doc_id}",
                    "generated_agent_summary",
                    "",
                    json.dumps({"user_id": owner}),
                    now_ts,
                ),
            )
        await conn.commit()
        await conn.close()

        # Opening Database runs only migration 45.
        database = Database(db_path)
        await database.connect()
        try:
            alice_source_id = agent_session_source_id("codex", "alice")
            bob_source_id = agent_session_source_id("codex", "bob")
            alice_source = await database.get_source(alice_source_id)
            bob_source = await database.get_source(bob_source_id)
            legacy_source = await database.get_source("src-agent-sessions-codex")
            alice_doc = await database.get_document("doc-alice")
            bob_doc = await database.get_document("doc-bob")
        finally:
            await database.close()

        assert alice_source is not None
        assert alice_source["owner_user_id"] == "alice"
        assert alice_source["access_policy"] == "private"
        assert bob_source is not None
        assert bob_source["owner_user_id"] == "bob"
        assert bob_source["access_policy"] == "private"
        assert legacy_source is None
        assert alice_doc is not None and alice_doc.source == alice_source_id
        assert bob_doc is not None and bob_doc.source == bob_source_id

    asyncio.run(_run())


def test_memories_endpoint_exposes_origin_client_for_agent_session_memories(tmp_path):
    """origin_client is 'codex' or 'claude-code' for agent-session memories and None for jira."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))
    base_time = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)

    async def _setup():
        await database.connect()
        # Seed a jira source and document.
        await database.upsert_source(
            "src-jira", "jira", "Jira", json.dumps({}), access_policy="workspace", owner_user_id="dev"
        )
        await database.upsert_source(
            "src-agent-sessions-codex",
            "agent_session",
            "Codex Session",
            json.dumps({"client": "codex"}),
            access_policy="private",
            owner_user_id="dev",
        )
        await database.upsert_source(
            "src-agent-sessions-claude-code",
            "agent_session",
            "Claude Code Session",
            json.dumps({"client": "claude-code"}),
            access_policy="private",
            owner_user_id="dev",
        )
        jira_doc_id = "doc-jira-1"
        await database.upsert_document(
            DocumentRecord(
                doc_id=jira_doc_id,
                source="src-jira",
                source_url="https://jira.example.com/PAY-1",
                title="PAY-1",
                space_or_project="payroll",
                author=None,
                last_modified=base_time,
                labels=[],
                version="v1",
                content_hash="hash-jira",
                token_count=50,
                last_synced=base_time,
            )
        )
        # Seed codex and claude-code agent session documents with the client column set.
        await _seed_source_project(
            database,
            doc_id="doc-codex-origin",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-codex-origin"],
            source_id="src-agent-sessions-codex",
            client="codex",
        )
        await _seed_source_project(
            database,
            doc_id="doc-cc-origin",
            project="mem-inception",
            last_modified=base_time,
            memory_ids=["mem-cc-origin"],
            source_id="src-agent-sessions-claude-code",
            client="claude-code",
        )
        # Jira memory without a client column on its document.
        jira_memory = Memory(
            id="mem-jira-origin",
            memory_type="fact",
            content="Jira memory",
            content_hash=content_hash("Jira memory"),
            project_key="payroll",
            created_at=base_time,
            updated_at=base_time,
            status="active",
        )
        await database.insert_memory(jira_memory)
        await database.add_memory_source(jira_memory.id, jira_doc_id, "jira", source_updated_at=None)

    import asyncio

    asyncio.run(_setup())
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.get("/api/v1/memories")

        assert response.status_code == 200
        data = {m["id"]: m for m in response.json()["data"]}

        assert data["mem-codex-origin"]["origin_client"] == "codex"
        assert data["mem-codex-origin"]["origin_source_type"] == "agent_session"

        assert data["mem-cc-origin"]["origin_client"] == "claude-code"
        assert data["mem-cc-origin"]["origin_source_type"] == "agent_session"

        assert data["mem-jira-origin"]["origin_client"] is None
        assert data["mem-jira-origin"]["origin_source_type"] == "jira"
    finally:
        asyncio.run(database.close())


class _RefusingRuntimeProvider(DefaultRuntimeProvider):
    def build_structured_llm_client(self, *args, **kwargs):
        raise AssertionError("a removed capture route must not build a model client")


@pytest.mark.parametrize(("path", "body"), EARLIER_PLUGIN_UPLOADS)
def test_removed_capture_routes_refuse_earlier_plugin_uploads(tmp_path, monkeypatch, path, body):
    from memforge.server.admin_api import create_admin_app

    async def refuse_activity(*_args, **_kwargs):
        raise AssertionError("a removed capture route must not take Source activity")

    monkeypatch.setattr(Database, "acquire_source_activity", refuse_activity)
    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=_config(tmp_path), runtime_provider=_RefusingRuntimeProvider())
        with TestClient(app) as client:
            response = client.post(path, json=body)
            malformed = client.post(path, content=b"not json", headers={"content-type": "application/json"})

        assert response.status_code == 410
        assert response.json() == CAPTURE_REMOVED_BODY
        assert response.headers["MemForge-Workspace"] == "local"
        assert malformed.status_code == 410
        assert malformed.json() == CAPTURE_REMOVED_BODY
        assert asyncio.run(database.list_sources()) == []
        assert asyncio.run(database.list_agent_hook_receipts(session_id="sess-old-plugin")) == []
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize(("path", "body"), EARLIER_PLUGIN_UPLOADS)
def test_removed_capture_routes_still_resolve_the_workspace(tmp_path, path, body):
    from memforge.server.admin_api import create_admin_app

    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=_config(tmp_path))
        with TestClient(app) as client:
            response = client.post(f"{path}?workspace_id=other_workspace", json=body)

        assert response.status_code == 404
        assert response.json()["code"] == "workspace_not_found_or_inaccessible"
    finally:
        asyncio.run(database.close())


def test_agent_session_completeness_route_is_removed(tmp_path):
    from memforge.server.admin_api import create_admin_app

    database = Database(str(tmp_path / "api.db"))

    import asyncio

    asyncio.run(database.connect())
    try:
        app = create_admin_app(db=database, config=_config(tmp_path))
        with TestClient(app) as client:
            response = client.get("/api/v1/agent-sessions/completeness", params={"source_id": "src-any"})

        assert response.status_code in {404, 405}
        paths = app.openapi()["paths"]
        assert "/api/v1/agent-sessions/completeness" not in paths
        assert paths["/api/v1/agent-sessions/windows"]["post"]["deprecated"] is True
        assert paths["/api/v1/hooks/receipts"]["post"]["deprecated"] is True
    finally:
        asyncio.run(database.close())
