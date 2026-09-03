from datetime import datetime, timedelta, timezone
import asyncio
import json
import sqlite3

import pytest

from memforge.local_agent.source_contract import local_agent_sync_job_payload
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "source-retry.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


async def _queued_retry(db):
    now = datetime.now(timezone.utc)
    await db.upsert_source(
        id="src-retry",
        type="local_markdown",
        name="Retry contract",
        config_json='{"root":"/fixture"}',
        access_policy="workspace",
        owner_user_id="owner",
        execution_owner_user_id="owner",
        created_by_user_id="owner",
    )
    payload = local_agent_sync_job_payload(await db.get_source("src-retry"))
    await db.enqueue_local_agent_job(
        job_id="laj-retry",
        source_id="src-retry",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload=payload,
        created_by_user_id="owner",
        execution_owner_user_id="owner",
    )
    [leased] = await db.lease_local_agent_jobs(user_id="owner", limit=1, lease_seconds=60, now=now)
    assert await db.complete_local_agent_job(
        job_id=leased["job_id"],
        user_id="owner",
        attempt_count=leased["attempt_count"],
        status="failed",
        result={"retryable": True},
        error="VPN unavailable",
        retryable=True,
        now=now,
    )
    return now, payload


@pytest.mark.asyncio
async def test_local_retry_waits_for_shared_delay_before_claim(db):
    now, _ = await _queued_retry(db)
    queued = await db.get_local_agent_job("laj-retry")
    assert queued["status"] == "queued"
    assert queued["next_attempt_at"] == (now + timedelta(hours=1)).isoformat()
    assert await db.lease_local_agent_jobs(user_id="owner", limit=1, lease_seconds=60, now=now) == []
    [leased] = await db.lease_local_agent_jobs(
        user_id="owner",
        limit=1,
        lease_seconds=60,
        now=now + timedelta(hours=1),
    )
    assert leased["job_id"] == "laj-retry"
    assert leased["attempt_count"] == 2
    assert leased["next_attempt_at"] is None


@pytest.mark.asyncio
async def test_manual_advances_exact_job_without_resetting_history(db):
    now, payload = await _queued_retry(db)
    before = await db.get_local_agent_job("laj-retry")
    common = dict(
        job_id="laj-must-not-be-created",
        source_id="src-retry",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload=payload,
        created_by_user_id="owner",
        execution_owner_user_id="owner",
    )
    assert await db.enqueue_local_agent_job(**common, trigger="scheduled") == ("laj-retry", False)
    assert (await db.get_local_agent_job("laj-retry"))["next_attempt_at"] == before["next_attempt_at"]
    assert await db.enqueue_local_agent_job(**common, retry_job_id="laj-retry") == ("laj-retry", False)
    advanced = await db.get_local_agent_job("laj-retry")
    assert advanced["next_attempt_at"] is None
    for field in ("payload", "attempt_count", "last_error", "result", "created_at"):
        assert advanced[field] == before[field]
    assert await db.get_local_agent_job("laj-must-not-be-created") is None
    [leased] = await db.lease_local_agent_jobs(user_id="owner", limit=1, lease_seconds=60, now=now)
    assert await db.enqueue_local_agent_job(**common, retry_job_id="laj-retry") == ("laj-retry", False)
    assert (await db.get_local_agent_job("laj-retry"))["attempt_count"] == leased["attempt_count"]


@pytest.mark.asyncio
async def test_exact_server_retry_preserves_snapshot_and_attempt_budget(db):
    await db.upsert_source(
        id="src-server",
        type="github_repo",
        name="Server retry",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner",
    )
    now = datetime.now(timezone.utc)
    run = await db.enqueue_source_sync_run(source_id="src-server", input_snapshot_id="snapshot-1")
    leased = await db.lease_next_source_sync_run(worker_id="worker", lease_seconds=60, now=now)
    assert leased is not None
    await db.fail_source_sync_run(
        run.run_id,
        worker_id="worker",
        lease_attempt_count=leased.lease_attempt_count,
        error_message="provider unavailable",
        retryable=True,
        failed_at=now,
        next_attempt_at=now + timedelta(hours=12),
    )
    before = await db.get_source_sync_run(run.run_id)
    scheduled = await db.enqueue_source_sync_run(source_id="src-server", trigger="scheduled")
    assert scheduled.run_id == run.run_id
    assert scheduled.next_attempt_at == before.next_attempt_at
    retried = await db.enqueue_source_sync_run(source_id="src-server", retry_run_id=run.run_id)
    assert retried.run_id == run.run_id
    assert retried.next_attempt_at is None
    for field in (
        "input_snapshot_id",
        "input_generation_watermark",
        "source_config_revision",
        "predecessor_activity_id",
        "lease_attempt_count",
        "error_message",
    ):
        assert getattr(retried, field) == getattr(before, field)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["manual", "force"])
async def test_ordinary_manual_run_enqueue_advances_without_new_run(db, trigger):
    await db.upsert_source(
        id="src-server",
        type="github_repo",
        name="Server",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner",
    )
    run = await db.enqueue_source_sync_run(source_id="src-server")
    await db.db.execute(
        "UPDATE source_sync_runs SET next_attempt_at = ? WHERE run_id = ?", ("2099-01-01T00:00:00+00:00", run.run_id)
    )
    await db.db.commit()
    retried = await db.enqueue_source_sync_run(
        source_id="src-server", trigger=trigger, force_full_sync=trigger == "force"
    )
    assert retried.run_id == run.run_id
    assert retried.next_attempt_at is None
    assert retried.force_full_sync == (trigger == "force")


@pytest.mark.asyncio
async def test_local_scheduler_rolls_back_schedule_when_enqueue_fails(db, monkeypatch):
    now, _ = await _queued_retry(db)
    await db.set_source_sync_schedule("src-retry", enabled=True, interval_minutes=60, next_run_at=now)
    before = await db.get_source("src-retry")

    async def fail(**kwargs):
        raise RuntimeError("injected enqueue failure")

    monkeypatch.setattr(db, "_enqueue_local_agent_job_locked", fail)
    with pytest.raises(RuntimeError, match="injected"):
        await db.enqueue_due_local_agent_jobs(now=now)
    assert not db.db.in_transaction
    assert (await db.get_source("src-retry"))["sync_schedule"] == before["sync_schedule"]


@pytest.mark.asyncio
async def test_local_scheduler_coalesces_without_advancing_backoff(db):
    now, _ = await _queued_retry(db)
    before = await db.get_local_agent_job("laj-retry")
    await db.set_source_sync_schedule("src-retry", enabled=True, interval_minutes=60, next_run_at=now)
    assert await db.enqueue_due_local_agent_jobs(now=now) == 0
    assert (await db.get_local_agent_job("laj-retry"))["next_attempt_at"] == before["next_attempt_at"]


@pytest.mark.asyncio
async def test_two_connections_retry_one_job_without_resetting_attempts(db):
    now, payload = await _queued_retry(db)
    async with db.db.execute("PRAGMA database_list") as cursor:
        row = await cursor.fetchone()
    other = Database(row[2])
    await other.connect()
    try:

        async def retry(database):
            return await database.enqueue_local_agent_job(
                job_id="must-not-create",
                source_id="src-retry",
                source_type="local_markdown",
                operation="local_markdown_sync",
                payload=payload,
                created_by_user_id="owner",
                execution_owner_user_id="owner",
                retry_job_id="laj-retry",
            )

        assert await asyncio.gather(retry(db), retry(other)) == [("laj-retry", False)] * 2
        claimed = await asyncio.gather(
            *[
                database.lease_local_agent_jobs(user_id="owner", limit=1, lease_seconds=60, now=now)
                for database in (db, other)
            ]
        )
        assert sum(map(len, claimed)) == 1
        assert (await db.get_local_agent_job("laj-retry"))["attempt_count"] == 2
    finally:
        await other.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("source_config_revision", "obsolete"), ("source_activity_epoch", 99)])
async def test_exact_local_retry_rejects_stale_payload(db, field, value):
    _, payload = await _queued_retry(db)
    await db.db.execute(
        "UPDATE local_agent_jobs SET payload_json = ? WHERE job_id = ?",
        (json.dumps({**payload, field: value}), "laj-retry"),
    )
    await db.db.commit()
    before = await db.get_local_agent_job("laj-retry")
    with pytest.raises(ValueError, match="stale"):
        await db.enqueue_local_agent_job(
            job_id="must-not-create",
            source_id="src-retry",
            source_type="local_markdown",
            operation="local_markdown_sync",
            payload=payload,
            created_by_user_id="owner",
            execution_owner_user_id="owner",
            retry_job_id="laj-retry",
        )
    assert await db.get_local_agent_job("laj-retry") == before


@pytest.mark.asyncio
async def test_old_local_job_schema_upgrade_preserves_history(tmp_path):
    path = str(tmp_path / "old-schema.db")
    database = Database(path)
    await database.connect()
    await _queued_retry(database)
    before = await database.get_local_agent_job("laj-retry")
    await database.close()
    # Materialize the prior schema, not an invented historical retry timestamp.
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE local_agent_jobs DROP COLUMN next_attempt_at")
        connection.execute("DELETE FROM schema_migrations WHERE version = 91")
    await database.connect()
    try:
        after = await database.get_local_agent_job("laj-retry")
        assert after["next_attempt_at"] is None
        for field in ("job_id", "payload", "attempt_count", "status", "last_error", "result", "created_at"):
            assert after[field] == before[field]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_current_sync_job_is_not_hidden_by_newer_setup_job(db):
    await _queued_retry(db)
    await db.insert_local_agent_job(
        job_id="setup-newer",
        source_id="src-retry",
        source_type="local_markdown",
        operation="local_markdown_pick_root",
        payload={},
        created_by_user_id="owner",
        execution_owner_user_id="owner",
    )
    jobs = await db.list_current_local_agent_jobs(workspace_id="default", user_id="owner")
    assert [job["job_id"] for job in jobs] == ["laj-retry"]
