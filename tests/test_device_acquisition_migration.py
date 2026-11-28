"""Upgrading a database to device acquisition (migration 134, ADR 0058).

A database from before device acquisition keeps one online time per user and
the snapshot tables of device packages, and its Sources and jobs name no
executor or device. The upgrade replaces the online times with one row per
device, drops the package snapshot tables, adds the executor and device
columns, and leaves every Source on the server.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from memforge.storage.database import Database

DEVICE_ACQUISITION = 134
NEW_JOB_COLUMNS = ("device_id", "lease_device_id", "leased_at", "acquisition_contract", "uncounted_attempt_count")
NEW_SOURCE_COLUMNS = ("executor", "execution_device_id")


def _columns(path: Path, table: str) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _tables(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


async def _prior(path: Path) -> None:
    """A database at schema 133 with a Source, a user's online time and a package snapshot."""

    db = Database(str(path))
    await db.connect()
    try:
        await db.upsert_source(
            id="src-wiki",
            type="confluence",
            name="Wiki",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="dev",
        )
    finally:
        await db.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= ?", (DEVICE_ACQUISITION,))
        for column in NEW_JOB_COLUMNS:
            connection.execute(f"ALTER TABLE local_agent_jobs DROP COLUMN {column}")
        for column in NEW_SOURCE_COLUMNS:
            connection.execute(f"ALTER TABLE sources DROP COLUMN {column}")
        connection.executescript(
            """DROP TABLE local_agent_devices;
               DROP TABLE source_acquisition_manifests;
               CREATE TABLE local_agent_heartbeats (user_id TEXT PRIMARY KEY, last_seen_at TEXT NOT NULL);
               INSERT INTO local_agent_heartbeats VALUES ('dev', '2026-11-01T00:00:00+00:00');
               CREATE TABLE source_sync_snapshot_manifests (
                   workspace_id TEXT NOT NULL, source_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                   PRIMARY KEY (workspace_id, source_id, snapshot_id));
               CREATE TABLE source_sync_snapshot_manifest_items (
                   workspace_id TEXT NOT NULL, source_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                   doc_id TEXT NOT NULL, PRIMARY KEY (workspace_id, source_id, snapshot_id, doc_id));
               CREATE INDEX idx_source_sync_manifest_item_attestation
                   ON source_sync_snapshot_manifest_items(workspace_id, source_id, doc_id);
               CREATE TABLE source_sync_snapshot_confirmations (snapshot_id TEXT NOT NULL);
               CREATE TABLE source_sync_snapshot_excluded_listings (snapshot_id TEXT NOT NULL);
               INSERT INTO source_sync_snapshot_manifests VALUES ('default', 'src-wiki', 'snapshot-a');"""
        )
        connection.commit()


async def test_the_upgrade_keeps_devices_per_device_drops_package_snapshots_and_leaves_sources_on_the_server(tmp_path):
    path = tmp_path / "prior.db"
    await _prior(path)

    db = Database(str(path))
    await db.connect()
    try:
        source = await db.get_source("src-wiki")
        devices = await db.list_local_agent_devices("dev")
    finally:
        await db.close()

    assert set(NEW_JOB_COLUMNS) <= _columns(path, "local_agent_jobs")
    assert set(NEW_SOURCE_COLUMNS) <= _columns(path, "sources")
    tables = _tables(path)
    assert {"local_agent_devices", "source_acquisition_manifests"} <= tables
    assert not tables & {
        "local_agent_heartbeats",
        "source_sync_snapshot_manifests",
        "source_sync_snapshot_manifest_items",
        "source_sync_snapshot_confirmations",
        "source_sync_snapshot_excluded_listings",
    }
    assert devices == []
    assert source is not None
    assert source["executor"] == "server"
    assert source["execution_device_id"] is None
