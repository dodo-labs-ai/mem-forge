from __future__ import annotations

import asyncio
import json

import aiosqlite
import pytest

from memforge.agent_sessions import agent_session_source_id
from memforge.storage.database import Database, MIGRATIONS, SCHEMA


_LEGACY_SOURCES_SCHEMA = """
CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    config TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_sync TEXT,
    doc_count INTEGER DEFAULT 0,
    project_binding TEXT,
    created_by_user_id TEXT,
    execution_owner_user_id TEXT,
    sync_schedule_enabled INTEGER NOT NULL DEFAULT 0,
    sync_schedule_interval_minutes INTEGER NOT NULL DEFAULT 1440,
    sync_schedule_next_at TEXT,
    sync_schedule_updated_at TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


async def _legacy_database(
    path: str,
    *,
    creator: str | None = "creator",
    memory_access: tuple[tuple[str, str | None], ...] = (),
    source_id: str = "src-legacy",
    source_type: str = "confluence",
    config: str = "{}",
) -> None:
    connection = await aiosqlite.connect(path)
    await connection.executescript(_LEGACY_SOURCES_SCHEMA)
    await connection.executescript(SCHEMA)
    for version, description, _ in MIGRATIONS:
        if version >= 42:
            break
        await connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-07-13T00:00:00+00:00"),
        )
    await connection.execute(
        """INSERT INTO sources (
               id, type, name, config, created_by_user_id, execution_owner_user_id
           ) VALUES (?, ?, 'Legacy', ?, ?, ?)""",
        (source_id, source_type, config, creator, creator),
    )
    for index, (visibility, owner_user_id) in enumerate(memory_access):
        memory_id = f"mem-{index}"
        doc_id = f"doc-{index}"
        await connection.execute(
            """INSERT INTO memories (
                   id, memory_type, content, content_hash, tags, visibility,
                   owner_user_id, project_key, status
               ) VALUES (?, 'fact', ?, ?, '[]', ?, ?, 'UNSORTED', 'active')""",
            (memory_id, memory_id, f"hash-{index}", visibility, owner_user_id),
        )
        await connection.execute(
            """INSERT INTO documents (
                   doc_id, source, source_url, title, space_or_project,
                   last_modified, version, content_hash, last_synced
               ) VALUES (?, ?, ?, ?, 'UNSORTED', ?, '1', ?, ?)""",
            (
                doc_id,
                source_id,
                f"https://example.test/{doc_id}",
                doc_id,
                "2026-07-13T00:00:00+00:00",
                f"doc-hash-{index}",
                "2026-07-13T00:00:00+00:00",
            ),
        )
        await connection.execute(
            """INSERT INTO memory_sources (
               memory_id, doc_id, source_id, source_type, support_kind
               ) VALUES (?, ?, ?, ?, 'extracted')""",
            (memory_id, doc_id, source_id, source_type),
        )
    await connection.commit()
    await connection.close()


def test_source_access_migration_preserves_workspace_visibility(tmp_path) -> None:
    async def run() -> None:
        path = str(tmp_path / "workspace.db")
        await _legacy_database(path, memory_access=(("workspace", None),))
        database = Database(path)
        await database.connect()
        try:
            source = await database.get_source("src-legacy")
            assert source is not None
            assert source["access_policy"] == "workspace"
            assert source["owner_user_id"] == "creator"
            assert source["access_state"] == "active"
        finally:
            await database.close()

    asyncio.run(run())


def test_source_access_migration_preserves_single_private_owner(tmp_path) -> None:
    async def run() -> None:
        path = str(tmp_path / "private.db")
        await _legacy_database(path, creator="creator", memory_access=(("private", "alice"),))
        database = Database(path)
        await database.connect()
        try:
            source = await database.get_source("src-legacy")
            assert source is not None
            assert source["access_policy"] == "private"
            assert source["owner_user_id"] == "alice"
        finally:
            await database.close()

    asyncio.run(run())


def test_source_access_migration_keeps_empty_configured_source_workspace_visible(tmp_path) -> None:
    async def run() -> None:
        path = str(tmp_path / "empty.db")
        await _legacy_database(path)
        database = Database(path)
        await database.connect()
        try:
            source = await database.get_source("src-legacy")
            assert source is not None
            assert source["access_policy"] == "workspace"
            assert source["owner_user_id"] == "creator"
        finally:
            await database.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("creator", "memory_access", "message"),
    [
        ("creator", (("workspace", None), ("private", "alice")), "mixed or ambiguous"),
        ("creator", (("private", "alice"), ("private", "bob")), "mixed or ambiguous"),
        (None, (("workspace", None),), "without an owner"),
    ],
)
def test_source_access_migration_rejects_ambiguous_or_ownerless_sources(
    tmp_path,
    creator: str | None,
    memory_access: tuple[tuple[str, str | None], ...],
    message: str,
) -> None:
    async def run() -> None:
        path = str(tmp_path / "invalid.db")
        await _legacy_database(path, creator=creator, memory_access=memory_access)
        database = Database(path)
        with pytest.raises(RuntimeError, match=message):
            await database.connect()
        await database.close()

    asyncio.run(run())


def test_source_access_migration_partitions_ownerless_agent_source_from_receipts(
    tmp_path,
) -> None:
    async def run() -> None:
        path = str(tmp_path / "ownerless-agent.db")
        legacy_source_id = "src-agent-sessions-claude-code"
        await _legacy_database(
            path,
            creator=None,
            source_id=legacy_source_id,
            source_type="agent_session",
            config=json.dumps({"client": "claude-code"}),
        )
        connection = await aiosqlite.connect(path)
        now = "2026-07-13T00:00:00+00:00"
        for owner in ("alice", "bob"):
            doc_id = f"doc-{owner}"
            await connection.execute(
                """INSERT INTO documents (
                       doc_id, source, source_url, title, space_or_project, author,
                       last_modified, labels, version, content_hash, last_synced
                   ) VALUES (?, ?, ?, ?, 'workspace', 'claude-code', ?, '[]', 'v1', ?, ?)""",
                (
                    doc_id,
                    legacy_source_id,
                    f"agent-session://claude-code/session/{doc_id}",
                    f"{owner} session",
                    now,
                    f"hash-{owner}",
                    now,
                ),
            )
            await connection.execute(
                """INSERT INTO agent_session_receipts (
                       doc_id, source_id, client, session_id, trigger, workspace,
                       history_window_kind, submitted_at, document_hash, source_kind,
                       document_uri, metadata, updated_at
                   ) VALUES (?, ?, 'claude-code', ?, 'Stop', 'workspace', 'session',
                             ?, ?, 'generated_agent_summary', '', ?, ?)""",
                (
                    doc_id,
                    legacy_source_id,
                    f"session-{owner}",
                    now,
                    f"hash-{owner}",
                    json.dumps({"user_id": owner}),
                    now,
                ),
            )
        await connection.commit()
        await connection.close()

        database = Database(path)
        await database.connect()
        try:
            assert await database.get_source(legacy_source_id) is None
            for owner in ("alice", "bob"):
                source_id = agent_session_source_id("claude-code", owner)
                source = await database.get_source(source_id)
                document = await database.get_document(f"doc-{owner}")
                assert source is not None
                assert source["owner_user_id"] == owner
                assert source["access_policy"] == "private"
                assert document is not None and document.source == source_id
        finally:
            await database.close()

    asyncio.run(run())
