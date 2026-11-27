"""Giving every existing device Source an execution device (migration 138, design §10.10 A5a).

Each device Source gets its execution owner's device whose latest report
named its credentials, else the owner's only device. A Source with neither
keeps none and shows Finish setup until its owner chooses one.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from memforge.storage.database import Database, latest_schema_version

EXECUTION_DEVICES = 138


async def _store_and_rewind(path: Path, sources: dict[str, tuple[str, str]], devices: list[tuple]) -> None:
    """Store device Sources by owner, then return the database to before migration 138 with ``devices`` reported.

    Each device is ``(user_id, device_id, last_seen_at, source_ids its report named)``.
    """

    db = Database(str(path))
    await db.connect()
    try:
        for source_id, (executor, owner) in sources.items():
            await db.upsert_source(
                id=source_id,
                type="confluence",
                name=source_id,
                config_json=json.dumps({"base_url": "https://wiki.example.com"}),
                access_policy="workspace",
                owner_user_id=owner,
                execution_owner_user_id=owner,
                executor=executor,
            )
    finally:
        await db.close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version >= ?", (EXECUTION_DEVICES,))
        connection.execute("ALTER TABLE local_agent_devices DROP COLUMN sources_json")
        connection.execute("ALTER TABLE local_agent_devices ADD COLUMN source_ids_json TEXT NOT NULL DEFAULT '[]'")
        for user_id, device_id, seen, source_ids in devices:
            connection.execute(
                """INSERT INTO local_agent_devices
                   (user_id, device_id, name, version, acquisition_contracts_json, source_ids_json, last_seen_at)
                   VALUES (?, ?, ?, '0.2.0', '{}', ?, ?)""",
                (user_id, device_id, device_id, json.dumps(source_ids), seen),
            )
        connection.commit()


async def test_each_device_source_gets_the_device_that_held_its_credentials_else_the_owners_only_device(
    tmp_path,
) -> None:
    path = tmp_path / "prior.db"
    await _store_and_rewind(
        path,
        {
            "src-held": ("device", "alex"),
            "src-held-twice": ("device", "alex"),
            "src-unheld": ("device", "alex"),
            "src-only-device": ("device", "sam"),
            "src-no-device": ("device", "kim"),
            "src-server": ("server", "sam"),
        },
        [
            ("alex", "dev-laptop", "2026-11-26T09:00:00+00:00", ["src-held", "src-held-twice"]),
            ("alex", "dev-desktop", "2026-11-27T09:00:00+00:00", ["src-held-twice"]),
            ("sam", "dev-sam", "2026-11-20T09:00:00+00:00", []),
        ],
    )

    db = Database(str(path))
    await db.connect()
    try:
        chosen = {source["id"]: source["execution_device_id"] for source in await db.list_sources()}
        devices = await db.list_local_agent_devices("alex")
    finally:
        await db.close()

    assert chosen == {
        "src-held": "dev-laptop",
        # The most recently seen of the devices that held it.
        "src-held-twice": "dev-desktop",
        # Two devices and neither held it: the owner chooses.
        "src-unheld": None,
        "src-only-device": "dev-sam",
        "src-no-device": None,
        "src-server": None,
    }
    # The reports keep their devices; each names its readiness with its next lease.
    assert sorted(device.report.device_id for device in devices) == ["dev-desktop", "dev-laptop"]
    assert all(dict(device.report.sources) == {} for device in devices)
    with closing(sqlite3.connect(path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(local_agent_devices)")}
        assert connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == latest_schema_version()
    assert "source_ids_json" not in columns
