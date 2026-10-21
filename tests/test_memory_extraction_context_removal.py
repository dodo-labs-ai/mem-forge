from __future__ import annotations

import sqlite3

import pytest

from memforge.models import Memory
from memforge.storage.database import Database


@pytest.mark.asyncio
async def test_upgrade_drops_the_memory_extraction_context_column_and_keeps_the_memory(tmp_path) -> None:
    db_path = tmp_path / "memforge.db"
    db = Database(str(db_path))
    await db.connect()
    await db.insert_memory(
        Memory(
            id="mem-before-upgrade",
            memory_type="fact",
            content="Payroll closes on the 25th.",
            content_hash="hash-before-upgrade",
        )
    )
    await db.close()

    with sqlite3.connect(db_path) as legacy:
        legacy.execute("ALTER TABLE memories ADD COLUMN extraction_context TEXT")
        legacy.execute(
            "UPDATE memories SET extraction_context = 'Payroll closes on the 25th.' WHERE id = 'mem-before-upgrade'"
        )
        legacy.execute("DELETE FROM schema_migrations WHERE version = 121")

    upgraded = Database(str(db_path))
    await upgraded.connect()
    try:
        async with upgraded.db.execute("PRAGMA table_info(memories)") as cursor:
            columns = {str(row[1]) async for row in cursor}
        assert "extraction_context" not in columns
        memory = await upgraded.get_memory("mem-before-upgrade")
        assert memory is not None
        assert memory.content == "Payroll closes on the 25th."
    finally:
        await upgraded.close()
