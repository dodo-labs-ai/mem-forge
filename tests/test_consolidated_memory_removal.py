from __future__ import annotations

import sqlite3

import pytest

from memforge.models import Memory
from memforge.storage.database import Database


@pytest.mark.asyncio
async def test_upgrade_purges_old_consolidated_rows_and_drops_curator_schema(tmp_path) -> None:
    db_path = tmp_path / "memforge.db"
    db = Database(str(db_path))
    await db.connect()
    await db.insert_memory(
        Memory(
            id="mem-old-consolidated",
            memory_type="fact",
            content="A materialized summary that is no longer a supported Memory kind.",
            content_hash="hash-old-consolidated",
        )
    )
    await db.close()

    with sqlite3.connect(db_path) as legacy:
        legacy.execute(
            "ALTER TABLE memories ADD COLUMN memory_level TEXT NOT NULL DEFAULT 'atomic'"
        )
        legacy.execute("ALTER TABLE memories ADD COLUMN curation_cluster_id TEXT")
        legacy.execute(
            """CREATE TABLE memory_derivations (
                parent_memory_id TEXT NOT NULL,
                child_memory_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (parent_memory_id, child_memory_id, relation)
            )"""
        )
        legacy.execute(
            """CREATE TABLE memory_curation_runs (
                id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                candidate_count INTEGER NOT NULL,
                created_memory_count INTEGER NOT NULL,
                started_at TEXT NOT NULL
            )"""
        )
        legacy.execute(
            "UPDATE memories SET memory_level = 'consolidated', curation_cluster_id = 'old-family' "
            "WHERE id = 'mem-old-consolidated'"
        )
        legacy.execute("DELETE FROM schema_migrations WHERE version = 53")

    upgraded = Database(str(db_path))
    await upgraded.connect()
    memory = await upgraded.get_memory("mem-old-consolidated")
    assert memory is None

    async with upgraded.db.execute("PRAGMA table_info(memories)") as cursor:
        columns = {str(row[1]) async for row in cursor}
    assert "memory_level" not in columns
    assert "curation_cluster_id" not in columns

    async with upgraded.db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ) as cursor:
        tables = {str(row[0]) async for row in cursor}
    assert "memory_derivations" not in tables
    assert "memory_curation_runs" not in tables
    await upgraded.close()
