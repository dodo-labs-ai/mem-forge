"""Every local-agent sync job carries the Source's open transition; a package snapshot lists by existence, so no excluded scope."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.local_agent.source_contract import (
    LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD,
    LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS,
    LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD,
    local_agent_sync_job_payload,
)
from memforge.source_projection import ProjectionScopeTransition
from memforge.source_projection_config import canonical_projection_scope
from memforge.storage.database import Database
from tests.test_source_authority_api import _app

SOURCE = "src-device-notes"
OWNER = "owner-a"
TRANSITION = "scope-transition-docs-only"
BASE_CONFIG = {"root": "/notes", "vault_id": "notes"}
OLD_INCLUDE = ["**/*.md"]
NEW_INCLUDE = ["docs/**/*.md"]


async def _no_device_source(source):
    raise AssertionError(f"no device Source is due here: {source['id']}")


def _scope(include: list[str]) -> dict[str, object]:
    return canonical_projection_scope("local_markdown", {**BASE_CONFIG, "include": include})


async def _source(db: Database) -> dict:
    await db.upsert_source(
        id=SOURCE,
        type="local_markdown",
        name="Device Source",
        config_json=json.dumps({**BASE_CONFIG, "include": NEW_INCLUDE}),
        access_policy="workspace",
        owner_user_id=OWNER,
        execution_owner_user_id=OWNER,
        created_by_user_id=OWNER,
    )
    source = await db.get_source(SOURCE)
    assert source is not None
    return source


async def _open_transition(db: Database) -> None:
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION,
            source_id=SOURCE,
            previous_scope=_scope(OLD_INCLUDE),
            target_scope=_scope(NEW_INCLUDE),
            created_at="2026-10-23T09:00:00+00:00",
        )
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "jobs.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def test_a_job_without_an_open_transition_carries_neither_field(db):
    payload = await local_agent_sync_job_payload(db, await _source(db))

    assert LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD not in payload
    assert LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD not in payload


async def test_a_job_carries_the_open_transition_and_no_excluded_scope(db):
    source = await _source(db)
    await _open_transition(db)

    payload = await local_agent_sync_job_payload(db, source)

    assert payload[LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD] == {
        "id": TRANSITION,
        "previous_scope": _scope(OLD_INCLUDE),
        "target_scope": _scope(NEW_INCLUDE),
    }
    assert LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD not in payload


def _ready_database(path: Path) -> Database:
    database = Database(str(path))

    async def prepare() -> None:
        await database.connect()
        await _source(database)
        await _open_transition(database)

    asyncio.run(prepare())
    return database


def _job_payload(database: Database) -> dict:
    async def read() -> dict:
        rows = await database.db.execute_fetchall("SELECT payload_json FROM local_agent_jobs WHERE source_id = ?", (SOURCE,))
        [row] = rows
        return json.loads(row["payload_json"])

    payload = asyncio.run(read())
    return {key: value for key, value in payload.items() if key not in LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS}


def test_every_builder_enqueues_the_same_payload_for_the_same_source(tmp_path):
    headers = {"x-test-user": OWNER, "x-test-workspace-role": "member"}
    payloads: dict[str, dict] = {}

    manual = _ready_database(tmp_path / "manual.db")
    try:
        with TestClient(_app(tmp_path, manual)) as client:
            assert client.post(f"/api/v1/sources/{SOURCE}/sync", headers=headers).status_code == 202
        payloads["manual"] = _job_payload(manual)
        expected = asyncio.run(local_agent_sync_job_payload(manual, asyncio.run(manual.get_source(SOURCE))))
    finally:
        asyncio.run(manual.close())

    explicit = _ready_database(tmp_path / "explicit.db")
    try:
        with TestClient(_app(tmp_path, explicit)) as client:
            created = client.post(
                "/api/cloud/local-agent/jobs",
                headers=headers,
                json={"source_id": SOURCE, "source_type": "local_markdown", "operation": "local_markdown_sync", "payload": {}},
            )
            assert created.status_code == 201, created.text
        payloads["explicit"] = _job_payload(explicit)
    finally:
        asyncio.run(explicit.close())

    scheduled = _ready_database(tmp_path / "scheduled.db")
    try:

        async def schedule() -> int:
            due = datetime.now(timezone.utc) - timedelta(minutes=1)
            await scheduled.set_source_sync_schedule(SOURCE, enabled=True, interval_minutes=60, next_run_at=due)
            return await scheduled.enqueue_due_local_agent_jobs(device_acquisition_payload=_no_device_source)

        assert asyncio.run(schedule()) == 1
        payloads["scheduled"] = _job_payload(scheduled)
    finally:
        asyncio.run(scheduled.close())

    assert LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD in expected
    shared = {key: value for key, value in expected.items() if key not in LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS}
    assert payloads == {"manual": shared, "explicit": shared, "scheduled": shared}


async def test_a_configuration_saved_while_the_scheduler_reads_leaves_the_source_due(db, monkeypatch):
    import memforge.storage.database as database_module

    await _source(db)
    due = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db.set_source_sync_schedule(SOURCE, enabled=True, interval_minutes=60, next_run_at=due)
    read = database_module.local_agent_sync_job_payload

    async def read_then_save(store, row, *args):
        payload = await read(store, row, *args)
        # The owner saves a new include after the scheduler read the payload.
        await store.db.execute(
            "UPDATE sources SET config = ? WHERE id = ?",
            (json.dumps({**BASE_CONFIG, "include": OLD_INCLUDE}), SOURCE),
        )
        await store.db.commit()
        return payload

    monkeypatch.setattr(database_module, "local_agent_sync_job_payload", read_then_save)
    raced = await db.enqueue_due_local_agent_jobs(device_acquisition_payload=_no_device_source)
    monkeypatch.setattr(database_module, "local_agent_sync_job_payload", read)
    retried = await db.enqueue_due_local_agent_jobs(device_acquisition_payload=_no_device_source)

    [row] = await db.db.execute_fetchall("SELECT payload_json FROM local_agent_jobs WHERE source_id = ?", (SOURCE,))
    assert (raced, retried) == (0, 1)
    assert json.loads(row["payload_json"])["include"] == OLD_INCLUDE
