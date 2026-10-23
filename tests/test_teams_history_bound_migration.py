"""Stored Teams Sources keep an Initial History the Teams Gene accepts."""

from __future__ import annotations

import json

import aiosqlite

from memforge.conversations.time_scope import MAX_INITIAL_HISTORY_DAYS
from memforge.memory.evidence import EVIDENCE_UNIT_SUPPORT_SCOPE
from memforge.storage.database import MIGRATIONS, SCHEMA, Database

CONFIGS = {
    "src-long": {"conversation_ids": ["19:a@thread.v2"], "initial_history_days": 500},
    "src-legacy": {"conversation_ids": ["19:b@thread.v2"], "max_age_days": 400},
    "src-short": {"conversation_ids": ["19:c@thread.v2"], "initial_history_days": 30},
}


async def _database_at_schema_123(path: str) -> None:
    connection = await aiosqlite.connect(path)
    await connection.executescript(SCHEMA)
    for version, description, _statements in MIGRATIONS:
        if version >= 124:
            break
        await connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-10-01T00:00:00+00:00"),
        )
    await connection.execute(
        "INSERT INTO system_contract_markers (marker_key, marker_value, updated_at) VALUES (?, ?, ?)",
        ("support_scope_version", EVIDENCE_UNIT_SUPPORT_SCOPE, "2026-10-01T00:00:00+00:00"),
    )
    for source_id, config in CONFIGS.items():
        await connection.execute(
            "INSERT INTO sources (id, type, name, config, access_policy, owner_user_id) VALUES (?, ?, ?, ?, ?, ?)",
            (source_id, "teams", source_id, json.dumps(config), "workspace", "dev"),
        )
    await connection.commit()
    await connection.close()


async def test_an_initial_history_beyond_the_bound_is_cut_to_it_and_others_are_left_alone(tmp_path):
    path = str(tmp_path / "before.db")
    await _database_at_schema_123(path)

    database = Database(path)
    await database.connect()
    try:
        stored = {
            row["id"]: json.loads(row["config"])
            for row in await database.db.execute_fetchall("SELECT id, config FROM sources")
        }
    finally:
        await database.close()

    assert stored["src-long"]["initial_history_days"] == MAX_INITIAL_HISTORY_DAYS
    assert stored["src-legacy"]["initial_history_days"] == MAX_INITIAL_HISTORY_DAYS
    assert stored["src-short"] == CONFIGS["src-short"]
