# tests/test_visibility_migrations.py
from datetime import datetime, timezone

import pytest
from memforge.models import SyncState
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
async def test_fresh_schema_has_durable_source_activity_admission(tmp_path):
    db = Database(str(tmp_path / "source-activity.db"))
    await db.connect()
    try:
        source_columns = await _columns(db, "sources")
        activity_columns = await _columns(db, "source_activity_leases")
        manifest_columns = await _columns(db, "source_sync_snapshot_manifests")
        assert "activity_epoch" not in source_columns
        assert "source_activity_epoch" not in manifest_columns
        assert activity_columns == [
            "id",
            "source_id",
            "kind",
            "capability",
            "lease_until",
            "created_at",
            "updated_at",
        ]
        async with db.db.execute(
            "SELECT description FROM schema_migrations WHERE version = 58"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "Add durable source activity admission"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upgrade_drops_the_source_activity_epoch_and_keeps_the_rows(tmp_path):
    path = str(tmp_path / "source-activity-epoch.db")
    db = Database(path)
    await db.connect()
    try:
        await db.upsert_source(
            id="src-epoch",
            type="teams",
            name="Epoch",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="dev",
        )
        # The workspace as it was before the upgrade: every fenced table
        # carries the Source activity epoch.
        await db.db.execute("ALTER TABLE sources ADD COLUMN activity_epoch INTEGER NOT NULL DEFAULT 0")
        await db.db.execute("ALTER TABLE source_activity_leases ADD COLUMN epoch INTEGER NOT NULL DEFAULT 0")
        await db.db.execute(
            "ALTER TABLE source_sync_snapshot_manifests ADD COLUMN source_activity_epoch INTEGER NOT NULL DEFAULT 0"
        )
        await db.db.execute("UPDATE sources SET activity_epoch = 3 WHERE id = 'src-epoch'")
        await db.db.execute(
            """INSERT INTO source_activity_leases (
                   id, source_id, kind, epoch, capability, lease_until, created_at, updated_at
               ) VALUES ('lease-epoch', 'src-epoch', 'sync', 3, '1',
                         '2999-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                         '2026-01-01T00:00:00+00:00')"""
        )
        await db.db.execute(
            """INSERT INTO source_sync_snapshot_manifests (
                   workspace_id, source_id, snapshot_id, coverage, item_count, manifest_sha256,
                   local_agent_job_id, local_agent_attempt_count, source_activity_epoch,
                   source_config_revision, created_at
               ) VALUES ('default', 'src-epoch', 'snapshot-epoch', 'complete_snapshot', 0, ?,
                         'job-epoch', 1, 3, 'config-revision', '2026-01-01T00:00:00+00:00')""",
            ("0" * 64,),
        )
        await db.db.execute("DELETE FROM schema_migrations WHERE version = 105")
        await db.db.commit()
    finally:
        await db.close()

    for _ in range(2):
        upgraded = Database(path)
        await upgraded.connect()
        try:
            assert "activity_epoch" not in await _columns(upgraded, "sources")
            assert "epoch" not in await _columns(upgraded, "source_activity_leases")
            assert "source_activity_epoch" not in await _columns(upgraded, "source_sync_snapshot_manifests")
            assert (await upgraded.get_source("src-epoch"))["name"] == "Epoch"
            lease_rows = await upgraded.db.execute_fetchall(
                "SELECT id, capability FROM source_activity_leases WHERE source_id = 'src-epoch'"
            )
            assert [tuple(row) for row in lease_rows] == [("lease-epoch", "1")]
            manifest_rows = await upgraded.db.execute_fetchall(
                "SELECT snapshot_id, source_config_revision FROM source_sync_snapshot_manifests"
            )
            assert [tuple(row) for row in manifest_rows] == [("snapshot-epoch", "config-revision")]
        finally:
            await upgraded.close()


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
        leased = await db.lease_next_source_sync_run(worker_id="test-worker")
        assert leased is not None and leased.run_id == run.run_id
        assert await db.complete_source_sync_run(
            run.run_id,
            worker_id="test-worker",
            lease_attempt_count=leased.lease_attempt_count,
            final_state=SyncState(
                source="src-cascade",
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            ),
        )
        await db.db.execute("DELETE FROM sources WHERE id = ?", ("src-cascade",))
        await db.db.commit()
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
