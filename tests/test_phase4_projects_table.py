"""Schema-shape coverage for the real `projects` table and the
`sources.project_binding` column added alongside it.
"""

import sqlite3

import pytest
import pytest_asyncio

from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "p4.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _columns(database: Database, table: str) -> list[str]:
    cols: list[str] = []
    async with database.db.execute(f"PRAGMA table_info({table})") as cur:
        async for row in cur:
            cols.append(row[1])
    return cols


@pytest.mark.asyncio
async def test_projects_table_has_full_schema(db):
    cols = await _columns(db, "projects")
    assert set(cols) == {"id", "key", "name", "is_shared", "created_at"}


@pytest.mark.asyncio
async def test_reserved_rows_are_seeded(db):
    rows: list[dict] = []
    async with db.db.execute("SELECT id, key, name, is_shared FROM projects ORDER BY key") as cur:
        async for row in cur:
            rows.append(dict(row))
    by_key = {r["key"]: r for r in rows}
    assert SHARED_PROJECT_KEY in by_key
    assert by_key[SHARED_PROJECT_KEY]["is_shared"] == 1
    assert UNSORTED_PROJECT_KEY in by_key
    assert by_key[UNSORTED_PROJECT_KEY]["is_shared"] == 0


@pytest.mark.asyncio
async def test_key_is_unique(db):
    with pytest.raises(sqlite3.IntegrityError):
        await db.db.execute(
            "INSERT INTO projects (id, key, name, is_shared, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
            ("proj-dup", SHARED_PROJECT_KEY, "Duplicate", 1),
        )


@pytest.mark.asyncio
async def test_sources_table_has_project_binding_column(db):
    cols = await _columns(db, "sources")
    assert "project_binding" in cols


@pytest.mark.asyncio
async def test_project_first_visibility_after_projects_table_rename(db):
    """Project-first visibility does not depend on a projects-table lookup.

    Migration 17 renames the reserved-project key column from `project_key`
    to `key`. The access predicate should keep broad project-first visibility
    without consulting either column for project-mode narrowing.
    """
    mem = Memory(
        id="m-dangle",
        memory_type="fact",
        content="x",
        content_hash=content_hash("x"),
        visibility=Visibility.WORKSPACE.value,
        owner_user_id=None,
        project_key="DANGLING",
    )
    await db.insert_memory(mem)

    adapters = build_sqlite_adapters(db, memory_collection=None)
    scope = AccessScope(
        user_id="dev",
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )
    visible = await adapters.relational.filter_visible_ids({"m-dangle"}, scope)
    assert "m-dangle" in visible
