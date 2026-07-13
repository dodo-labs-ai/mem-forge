# tests/test_visibility_migrations.py
import pytest
from memforge.storage.database import Database


async def _columns(db, table):
    cols = []
    async with db.db.execute(f"PRAGMA table_info({table})") as cur:
        async for row in cur:
            cols.append(row[1])
    return cols


async def _indexes(db):
    names = []
    async with db.db.execute("PRAGMA index_list(memories)") as cur:
        async for row in cur:
            names.append(row[1])
    return names


@pytest.mark.asyncio
async def test_fresh_schema_has_visibility_columns_and_no_scope(tmp_path):
    db = Database(str(tmp_path / "m.db"))
    await db.connect()
    try:
        cols = await _columns(db, "memories")
        assert "visibility" in cols
        assert "owner_user_id" in cols
        assert "scope" not in cols
        idx = await _indexes(db)
        assert "idx_memories_access" in idx
        assert "idx_memories_owner" in idx
        assert "idx_memories_scope" not in idx
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_agent_concept_rebuild_restores_foreign_key_enforcement(tmp_path):
    db = Database(str(tmp_path / "foreign-keys.db"))
    await db.connect()
    try:
        async with db.db.execute("PRAGMA foreign_keys") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1

        await db.upsert_source(
            "src-cascade",
            "confluence",
            "Cascade Source",
            "{}",
            "workspace",
            "dev",
        )
        run = await db.enqueue_source_sync_run(
            source_id="src-cascade",
            trigger="manual",
        )
        await db.delete_source_cascade("src-cascade")
        assert await db.get_source_sync_run(run.run_id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_backfill_maps_legacy_scope_to_project_key(tmp_path):
    from memforge.models import SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY

    db = Database(str(tmp_path / "legacy.db"))
    await db.connect()  # fresh schema + all migrations already applied
    try:
        # Reconstruct the legacy condition the backfill targets: a dormant scope
        # column carrying project-style values, project_key not yet derived. Then
        # replay Migration 15 by clearing its schema_migrations row and re-running.
        await db.db.execute("ALTER TABLE memories ADD COLUMN scope TEXT")
        await db.db.execute(
            "INSERT INTO memories (id, memory_type, content, content_hash, visibility, project_key, scope) "
            "VALUES ('a','fact','x','h1','workspace',NULL,'project:ACME'),"
            "       ('b','fact','y','h2','workspace',NULL,'team'),"
            "       ('c','fact','z','h3','workspace',NULL,'source:42')"
        )
        await db.db.execute("DELETE FROM schema_migrations WHERE version = 15")
        await db.db.commit()

        await db._run_migrations()  # re-applies Migration 15 against the legacy rows

        rows = {}
        async with db.db.execute("SELECT id, project_key FROM memories") as cur:
            async for row in cur:
                rows[row[0]] = row[1]
        assert rows["a"] == "ACME"
        assert rows["b"] == SHARED_PROJECT_KEY
        assert rows["c"] == UNSORTED_PROJECT_KEY
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upgrade_from_legacy_db_without_visibility(tmp_path):
    # A pre-visibility database: a memories table that has scope and no visibility
    # column, already at migration version 13. connect() runs the fresh SCHEMA and
    # then the migrations; it must not fail on a SCHEMA index that references a
    # column the migrations have not added yet.
    import aiosqlite

    db_path = str(tmp_path / "legacy_upgrade.db")
    async with aiosqlite.connect(db_path) as raw:
        await raw.execute(
            "CREATE TABLE memories ("
            "id TEXT PRIMARY KEY, memory_type TEXT NOT NULL, content TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', "
            "scope TEXT NOT NULL DEFAULT 'team', project_key TEXT, "
            "confidence REAL NOT NULL DEFAULT 0.7, status TEXT NOT NULL DEFAULT 'active', "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')), "
            "updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        await raw.execute("CREATE INDEX idx_memories_scope ON memories(scope)")
        await raw.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, description TEXT, applied_at TEXT)"
        )
        for version in range(1, 14):
            await raw.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) "
                "VALUES (?, 'legacy', datetime('now'))",
                (version,),
            )
        await raw.execute(
            "INSERT INTO memories (id, memory_type, content, content_hash, scope, project_key) "
            "VALUES ('leg-1','fact','c','h','team',NULL)"
        )
        await raw.commit()

    db = Database(db_path)
    await db.connect()  # the upgrade path; must not raise on a SCHEMA index
    try:
        cols = await _columns(db, "memories")
        assert "visibility" in cols
        assert "owner_user_id" in cols
        idx = await _indexes(db)
        assert "idx_memories_access" in idx
        assert "idx_memories_owner" in idx
        async with db.db.execute("SELECT visibility, project_key FROM memories WHERE id = 'leg-1'") as cur:
            row = await cur.fetchone()
        assert row[0] == "workspace"  # backfilled visibility
        assert row[1] == "SHARED"  # legacy scope 'team' maps to SHARED
    finally:
        await db.close()
