"""Dropping the snapshot manifests of device packages (migration 139, design §10.3).

A device run of a document Source records its own acquisition manifest, so
the package snapshot manifests have no writer. An upgraded database drops
them, and a device run's snapshot input still names its Document.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from memforge.storage.database import Database

SNAPSHOT_MANIFESTS_DROPPED = 139


def _tables(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


async def _rewind(path: Path) -> None:
    """A database at schema 138 whose daemon kept a package snapshot manifest."""

    db = Database(str(path))
    await db.connect()
    try:
        await db.upsert_source(
            id="src-notes",
            type="local_markdown",
            name="Notes",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="dev",
        )
    finally:
        await db.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= ?", (SNAPSHOT_MANIFESTS_DROPPED,))
        connection.executescript(
            """CREATE TABLE source_sync_snapshot_manifests (
                   workspace_id TEXT NOT NULL, source_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                   coverage TEXT NOT NULL, item_count INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL,
                   local_agent_job_id TEXT NOT NULL, local_agent_attempt_count INTEGER NOT NULL,
                   source_config_revision TEXT NOT NULL, created_at TEXT NOT NULL,
                   PRIMARY KEY (workspace_id, source_id, snapshot_id));
               CREATE TABLE source_sync_snapshot_manifest_items (
                   workspace_id TEXT NOT NULL, source_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                   doc_id TEXT NOT NULL, revision TEXT NOT NULL, change_kind TEXT NOT NULL,
                   PRIMARY KEY (workspace_id, source_id, snapshot_id, doc_id));
               CREATE INDEX idx_source_sync_manifest_item_attestation
                   ON source_sync_snapshot_manifest_items(workspace_id, source_id, doc_id, revision, change_kind, snapshot_id);
               INSERT INTO source_sync_snapshot_manifests VALUES
                   ('default', 'src-notes', 'snapshot-a', 'complete_snapshot', 1, 'sha', 'job-a', 1, 'rev', '2026-11-01');
               INSERT INTO source_sync_snapshot_manifest_items VALUES
                   ('default', 'src-notes', 'snapshot-a', 'doc-ideas', 'v1', 'upsert');"""
        )
        connection.commit()


async def test_the_upgrade_drops_the_package_snapshot_manifests_and_snapshot_inputs_name_their_document(tmp_path):
    path = tmp_path / "prior.db"
    await _rewind(path)
    assert {"source_sync_snapshot_manifests", "source_sync_snapshot_manifest_items"} <= _tables(path)

    db = Database(str(path))
    await db.connect()
    try:
        created = await db.create_source_sync_input(
            source_id="src-notes",
            raw_uri="object://input.json",
            raw_sha256="sha-input",
            raw_content_type="application/json",
            metadata={"doc_id": "doc-ideas"},
            sync_snapshot_id="snapshot-a",
        )
        listed = await db.list_source_sync_inputs(source_id="src-notes", input_snapshot_id="snapshot-a")
    finally:
        await db.close()

    tables = _tables(path)
    assert "source_sync_snapshot_manifests" not in tables
    assert "source_sync_snapshot_manifest_items" not in tables
    assert [item.input_id for item in listed] == [created.input_id]
