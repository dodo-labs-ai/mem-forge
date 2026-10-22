"""Every local-agent sync job carries the Source's open transition, and its excluded scopes once a run can complete it."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.local_agent.source_contract import (
    LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD,
    LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS,
    LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD,
    local_agent_job_config,
    local_agent_sync_job_payload,
)
from memforge.source_projection import ProjectionScopeTransition
from memforge.source_projection_config import canonical_projection_scope, projection_scope_fingerprint
from memforge.storage.database import Database
from tests.test_source_authority_api import _app

SOURCE = "src-device-jira"
OWNER = "owner-a"
TRANSITION = "scope-transition-pay-only"
BASE_CONFIG = {
    "base_url": "https://jira.example.test",
    "auth_mode": "browser_cookie",
    "sync_mode": "local_agent",
    "query_mode": "advanced",
}
OLD_JQL = "project in (PAY, OPS)"
NEW_JQL = "project = PAY"
SUPERSEDED_JQL = "project = OPS"


def _scope(jql: str) -> dict[str, object]:
    return canonical_projection_scope("jira", {**BASE_CONFIG, "jql": jql})


async def _source(db: Database, source_type: str = "jira", config: dict | None = None) -> dict:
    await db.upsert_source(
        id=SOURCE,
        type=source_type,
        name="Device Source",
        config_json=json.dumps(config or {**BASE_CONFIG, "jql": NEW_JQL}),
        access_policy="workspace",
        owner_user_id=OWNER,
        execution_owner_user_id=OWNER,
        created_by_user_id=OWNER,
    )
    source = await db.get_source(SOURCE)
    assert source is not None
    return source


async def _open_transition(db: Database, *, superseded: bool = False) -> None:
    """Open OLD → NEW, superseding OLD → SUPERSEDED when asked."""
    predecessor = None
    if superseded:
        predecessor = "scope-transition-ops-only"
        now = datetime.now(timezone.utc).isoformat()
        await db.db.execute(
            """INSERT INTO projection_scope_transitions (
                   id, source_id, previous_scope_json, target_scope_json, status,
                   predecessor_transition_id, created_at, updated_at, completed_at
               ) VALUES (?, ?, ?, ?, 'superseded', NULL, ?, ?, ?)""",
            (predecessor, SOURCE, json.dumps(_scope(OLD_JQL)), json.dumps(_scope(SUPERSEDED_JQL)), now, now, now),
        )
        await db.db.commit()
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION,
            source_id=SOURCE,
            previous_scope=_scope(OLD_JQL),
            target_scope=_scope(NEW_JQL),
            predecessor_transition_id=predecessor,
            created_at="2026-10-23T09:00:00+00:00",
        )
    )


async def _transition_run(db: Database, *, listed: list[str], members: list[str]) -> None:
    """A transition run bound to a device snapshot that listed ``listed``, after which ``members`` committed."""
    snapshot_id = "laj-previous:attempt:1"
    await db.attach_source_sync_inputs_to_snapshot(
        source_id=SOURCE,
        workspace_id="default",
        snapshot_id=snapshot_id,
        memberships=[],
        coverage="complete_snapshot",
        manifest_count=len(listed),
        manifest_sha256="a" * 64,
        manifest_items=[(doc_id, "r1", "upsert") for doc_id in listed],
        local_agent_job_id="laj-previous",
        local_agent_attempt_count=1,
        source_config_revision="rev-1",
    )
    for doc_id in listed:
        await db.create_source_sync_input(
            source_id=SOURCE,
            raw_uri=f"file:///packages/{doc_id}.json",
            raw_sha256=hashlib.sha256(doc_id.encode()).hexdigest(),
            raw_content_type="application/json",
            metadata={"doc_id": doc_id, "manifest_entry": {"doc_id": doc_id, "version": "r1"}},
            sync_snapshot_id=snapshot_id,
        )
    run = await db.enqueue_source_sync_run(source_id=SOURCE, trigger="local_agent", input_snapshot_id=snapshot_id)
    await db.start_projection_scope_transition(TRANSITION, run_id=run.run_id)
    await db.db.executemany(
        "INSERT INTO projection_scope_transition_members (transition_id, doc_id, unit_revision_id) VALUES (?, ?, ?)",
        [(TRANSITION, doc_id, f"rev-{doc_id}") for doc_id in members],
    )
    await db.db.commit()


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


@pytest.mark.parametrize(
    ("listed", "members"),
    [
        pytest.param(None, None, id="no transition run yet"),
        pytest.param(["jira-pay-1", "jira-pay-2"], ["jira-pay-1"], id="a listed Document waits"),
    ],
)
async def test_excluded_scopes_wait_until_the_previous_run_left_no_listed_document_waiting(db, listed, members):
    source = await _source(db)
    await _open_transition(db)
    if listed is not None:
        await _transition_run(db, listed=listed, members=members)

    payload = await local_agent_sync_job_payload(db, source)

    assert payload[LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD] == {
        "id": TRANSITION,
        "previous_scope": _scope(OLD_JQL),
        "target_scope": _scope(NEW_JQL),
    }
    assert LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD not in payload


async def test_a_job_carries_each_excluded_scope_once_the_previous_run_left_none_waiting(db):
    source = await _source(db)
    await _open_transition(db, superseded=True)
    await _transition_run(db, listed=["jira-pay-1"], members=["jira-pay-1"])

    payload = await local_agent_sync_job_payload(db, source)

    # The last applied scope and the superseded target, never the new one.
    excluded = payload[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD]
    assert [entry["scope_key"] for entry in excluded] == [
        projection_scope_fingerprint(_scope(OLD_JQL)),
        projection_scope_fingerprint(_scope(SUPERSEDED_JQL)),
    ]
    assert [entry["config"]["jql"] for entry in excluded] == [OLD_JQL, SUPERSEDED_JQL]
    assert all(
        entry["config"] == local_agent_job_config("jira", {**BASE_CONFIG, "jql": entry["config"]["jql"]})
        for entry in excluded
    )
    # The current configuration is unchanged by the excluded scopes.
    assert payload["jql"] == NEW_JQL


async def test_an_existence_listing_needs_no_excluded_scope(db):
    config = {
        "connection_mode": "local_push",
        "repo_url": "https://github.example.test/org/repo",
        "ref": "main",
        "include_paths": ["docs/"],
        "include_extensions": ["md"],
    }
    source = await _source(db, "github_repo", config)
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION,
            source_id=SOURCE,
            previous_scope=canonical_projection_scope("github_repo", {**config, "include_paths": ["notes/"]}),
            target_scope=canonical_projection_scope("github_repo", config),
            created_at="2026-10-23T09:00:00+00:00",
        )
    )
    await _transition_run(db, listed=["github-doc"], members=["github-doc"])

    payload = await local_agent_sync_job_payload(db, source)

    assert payload[LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD]["id"] == TRANSITION
    assert LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD not in payload


def _ready_database(path: Path) -> Database:
    database = Database(str(path))

    async def prepare() -> None:
        await database.connect()
        await _source(database)
        await _open_transition(database, superseded=True)
        await _transition_run(database, listed=["jira-pay-1"], members=["jira-pay-1"])

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
                json={"source_id": SOURCE, "source_type": "jira", "operation": "jira_sync", "payload": {}},
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
            return await scheduled.enqueue_due_local_agent_jobs()

        assert asyncio.run(schedule()) == 1
        payloads["scheduled"] = _job_payload(scheduled)
    finally:
        asyncio.run(scheduled.close())

    assert LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD in expected
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
        # The owner saves a new JQL after the scheduler read the payload.
        await store.db.execute(
            "UPDATE sources SET config = ? WHERE id = ?",
            (json.dumps({**BASE_CONFIG, "jql": "project = OPS"}), SOURCE),
        )
        await store.db.commit()
        return payload

    monkeypatch.setattr(database_module, "local_agent_sync_job_payload", read_then_save)
    raced = await db.enqueue_due_local_agent_jobs()
    monkeypatch.setattr(database_module, "local_agent_sync_job_payload", read)
    retried = await db.enqueue_due_local_agent_jobs()

    [row] = await db.db.execute_fetchall("SELECT payload_json FROM local_agent_jobs WHERE source_id = ?", (SOURCE,))
    assert (raced, retried) == (0, 1)
    assert json.loads(row["payload_json"])["jql"] == "project = OPS"
