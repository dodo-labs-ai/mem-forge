"""Teams Units stored before the conversation layer take their conversation Unit Types, keeping their identities."""

from __future__ import annotations

import json

import aiosqlite

from memforge.conversations.units import UnitKind
from memforge.memory.evidence import EVIDENCE_UNIT_SUPPORT_SCOPE
from memforge.storage.database import MIGRATIONS, SCHEMA, Database

THREAD_KEY = (
    "teams-thread:v1:eyJjb252ZXJzYXRpb25faWQiOiIxOTpjaGF0QHRocmVhZC52MiIsInJvb3Rfb3JfYW5jaG9yX21lc3NhZ2VfaWQiOiJyb290"
    "Iiwic291cmNlX2lkIjoic3JjIiwid2luZG93X3R5cGUiOiJ0aHJlYWQifQ"
)
BLOCK_KEY = "teams-block:v1:eyJhbmNob3IiOiJtMSJ9"
# A thread that came back after its removal keeps its incarnation in its key.
REINCARNATED_KEY = f"{THREAD_KEY}#incarnation:unit-old:3"


async def _database_before_the_conversation_layer(path: str) -> None:
    """A database at schema 120 whose Teams Units were stored before the conversation layer named them."""

    connection = await aiosqlite.connect(path)
    await connection.executescript(SCHEMA)
    # Document snapshot manifests still record projection-scope attestations.
    await connection.execute(
        "ALTER TABLE source_sync_snapshot_manifests ADD COLUMN scope_attestations_json TEXT NOT NULL DEFAULT '[]'"
    )
    for version, description, _statements in MIGRATIONS:
        if version >= 121:
            break
        await connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-10-01T00:00:00+00:00"),
        )
    await connection.execute(
        "INSERT INTO system_contract_markers (marker_key, marker_value, updated_at) VALUES (?, ?, ?)",
        ("support_scope_version", EVIDENCE_UNIT_SUPPORT_SCOPE, "2026-10-01T00:00:00+00:00"),
    )
    await connection.execute(
        "INSERT INTO sources (id, type, name, config, access_policy, owner_user_id) VALUES (?, ?, ?, ?, ?, ?)",
        ("src", "teams", "Teams", "{}", "workspace", "dev"),
    )
    for unit_id, key in (("unit-thread", THREAD_KEY), ("unit-block", BLOCK_KEY), ("unit-again", REINCARNATED_KEY)):
        await connection.execute(
            """INSERT INTO source_units (id, source_id, unit_type, provider_key, locator_json, updated_at)
               VALUES (?, 'src', 'teams_window', ?, ?, '2026-10-01T00:00:00+00:00')""",
            (unit_id, key, json.dumps({"conversation_id": "19:chat@thread.v2", "window_id": key})),
        )
    await connection.commit()
    await connection.close()


async def test_teams_units_keep_their_ids_and_keys_and_name_their_kind(tmp_path):
    path = str(tmp_path / "before.db")
    await _database_before_the_conversation_layer(path)

    database = Database(path)
    await database.connect()
    try:
        rows = {
            row["id"]: row
            for row in await database.db.execute_fetchall(
                "SELECT id, unit_type, provider_key, locator_json FROM source_units ORDER BY id"
            )
        }
        manifest_columns = {
            str(row[1]) for row in await database.db.execute_fetchall("PRAGMA table_info(source_sync_snapshot_manifests)")
        }
    finally:
        await database.close()

    thread, block, again = rows["unit-thread"], rows["unit-block"], rows["unit-again"]
    assert (again["unit_type"], again["provider_key"]) == (UnitKind.THREAD.value, REINCARNATED_KEY)
    assert json.loads(again["locator_json"])["thread_root_id"] == "root"
    assert "scope_attestations_json" not in manifest_columns
    assert (thread["unit_type"], thread["provider_key"]) == (UnitKind.THREAD.value, THREAD_KEY)
    assert json.loads(thread["locator_json"])["thread_root_id"] == "root"
    assert (block["unit_type"], block["provider_key"]) == (UnitKind.WINDOW.value, BLOCK_KEY)
    assert "thread_root_id" not in json.loads(block["locator_json"])
