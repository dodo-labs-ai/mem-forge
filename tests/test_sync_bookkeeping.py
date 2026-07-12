from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    DocumentMetadata,
    Entity,
    EnrichmentResult,
    GeneMetadata,
    MemoryExtractionResult,
    NormalizedContent,
    RawEntityRef,
    RawContent,
    RawMemory,
    SyncState,
    FailedDoc,
    content_hash,
)
from memforge.pipeline.sync_memory import MemorySample, SyncMemoryObserver
from memforge.pipeline.sync import (
    DocumentLifecycleAdmission,
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    summarize_failed_documents,
)
from memforge.runtime import SyncService
from memforge.config import AppConfig, SyncConfig
from memforge.storage.database import Database
from memforge.storage.database import MIGRATIONS
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.scheduler import SyncScheduler


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "sync-bookkeeping.db"))
    await database.connect()
    yield database
    await database.close()


def test_local_agent_broker_has_its_own_forward_migration() -> None:
    version, description, statements = next(
        migration for migration in MIGRATIONS if migration[0] == 37
    )

    assert version == 37
    assert description == "Add local agent job broker"
    assert any("CREATE TABLE IF NOT EXISTS local_agent_jobs" in sql for sql in statements)


@pytest.mark.asyncio
async def test_snapshot_migration_upgrades_database_that_already_recorded_migration_36(
    tmp_path,
) -> None:
    path = tmp_path / "pre-snapshot.db"
    database = Database(str(path))
    await database.connect()
    await database.close()

    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE source_sync_snapshot_items")
        conn.execute("ALTER TABLE source_sync_runs DROP COLUMN input_snapshot_id")
        conn.execute("ALTER TABLE source_sync_runs DROP COLUMN rerun_input_snapshot_id")
        conn.execute("DELETE FROM schema_migrations WHERE version = 38")
        conn.commit()

    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        columns = {
            row["name"]
            for row in await (
                await upgraded.db.execute("PRAGMA table_info(source_sync_runs)")
            ).fetchall()
        }
        table = await (
            await upgraded.db.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'source_sync_snapshot_items'"
            )
        ).fetchone()
    finally:
        await upgraded.close()

    assert {"input_snapshot_id", "rerun_input_snapshot_id"} <= columns
    assert table is not None


@pytest.mark.asyncio
async def test_latest_source_sync_run_is_scoped_to_source_and_workspace(db: Database):
    await db.upsert_source(
        id="src-latest-a",
        type="jira",
        name="Latest A",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-latest-b",
        type="jira",
        name="Latest B",
        config_json="{}",
    )
    first = await db.enqueue_source_sync_run(
        source_id="src-latest-a",
        workspace_id="ws-a",
    )
    await db.enqueue_source_sync_run(
        source_id="src-latest-b",
        workspace_id="ws-a",
    )

    latest = await db.get_latest_source_sync_run(
        source_id="src-latest-a",
        workspace_id="ws-a",
    )
    missing = await db.get_latest_source_sync_run(
        source_id="src-latest-a",
        workspace_id="ws-b",
    )

    assert latest is not None
    assert latest.run_id == first.run_id
    assert missing is None


@pytest.mark.asyncio
async def test_enqueue_source_sync_run_coalesces_by_workspace_and_source(db: Database):
    await db.upsert_source(
        id="src-sync-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )

    first = await db.enqueue_source_sync_run(
        source_id="src-sync-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    second = await db.enqueue_source_sync_run(
        source_id="src-sync-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    other_workspace = await db.enqueue_source_sync_run(
        source_id="src-sync-run",
        workspace_id="workspace-b",
        trigger="manual",
    )

    assert second.run_id == first.run_id
    assert second.coalesced is True
    assert other_workspace.run_id != first.run_id
    assert first.status == "pending"


@pytest.mark.asyncio
async def test_enqueue_source_sync_run_promotes_force_intent_on_active_run(db: Database):
    await db.upsert_source(
        id="src-force-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )

    first = await db.enqueue_source_sync_run(
        source_id="src-force-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
    )
    force = await db.enqueue_source_sync_run(
        source_id="src-force-run",
        workspace_id="workspace-a",
        trigger="force",
        force_full_sync=True,
    )

    assert force.run_id == first.run_id
    assert force.coalesced is True
    assert force.force_full_sync is True
    assert force.rerun_requested is True


@pytest.mark.asyncio
async def test_duplicate_manual_trigger_does_not_schedule_successor_for_running_run(db: Database):
    await db.upsert_source(
        id="src-running-manual",
        type="confluence",
        name="Running manual",
        config_json="{}",
    )
    first = await db.enqueue_source_sync_run(
        source_id="src-running-manual",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
    )
    duplicate = await db.enqueue_source_sync_run(
        source_id="src-running-manual",
        workspace_id="workspace-a",
        trigger="manual",
    )

    assert leased is not None
    assert duplicate.run_id == first.run_id
    assert duplicate.coalesced is True
    assert duplicate.rerun_requested is False


@pytest.mark.asyncio
async def test_duplicate_snapshot_does_not_schedule_running_successor(db: Database):
    await db.upsert_source(id="src-same-snapshot", type="github_repo", name="Repo", config_json="{}")
    first = await db.enqueue_source_sync_run(
        source_id="src-same-snapshot",
        input_snapshot_id="snapshot-a",
    )
    await db.lease_next_source_sync_run(
        worker_id="worker-a",
        lease_seconds=60,
        now=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
    )
    duplicate = await db.enqueue_source_sync_run(
        source_id="src-same-snapshot",
        trigger="local_agent",
        input_snapshot_id="snapshot-a",
    )

    assert duplicate.run_id == first.run_id
    assert duplicate.rerun_requested is False


@pytest.mark.asyncio
async def test_lease_next_source_sync_run_recovers_expired_run_without_new_run(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-lease-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-lease-run",
        workspace_id="workspace-a",
        trigger="manual",
    )

    first = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    before_expiry = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=30),
    )
    recovered = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=90),
    )

    assert first is not None
    assert first.run_id == enqueued.run_id
    assert first.status == "running"
    assert first.lease_owner == "worker-a"
    assert first.lease_attempt_count == 1
    assert first.recovery_count == 0
    assert before_expiry is None
    assert recovered is not None
    assert recovered.run_id == enqueued.run_id
    assert recovered.lease_owner == "worker-b"
    assert recovered.lease_attempt_count == 2
    assert recovered.recovery_count == 1


@pytest.mark.asyncio
async def test_source_sync_run_lease_is_compare_and_swap_across_connections(tmp_path):
    path = tmp_path / "shared-sync.db"
    first_db = Database(str(path))
    second_db = Database(str(path))
    await first_db.connect()
    await second_db.connect()
    try:
        await first_db.upsert_source(
            id="src-cas-lease",
            type="confluence",
            name="CAS Lease",
            config_json="{}",
        )
        await first_db.enqueue_source_sync_run(source_id="src-cas-lease")
        now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)

        leased = await asyncio.gather(
            first_db.lease_next_source_sync_run(
                worker_id="worker-a", lease_seconds=60, now=now
            ),
            second_db.lease_next_source_sync_run(
                worker_id="worker-b", lease_seconds=60, now=now
            ),
        )

        assert sum(run is not None for run in leased) == 1
    finally:
        await first_db.close()
        await second_db.close()


@pytest.mark.asyncio
async def test_heartbeat_source_sync_run_extends_only_current_worker_lease(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-heartbeat-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-heartbeat-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None

    wrong_worker = await db.heartbeat_source_sync_run(
        enqueued.run_id,
        worker_id="worker-b",
        lease_attempt_count=leased.lease_attempt_count,
        lease_seconds=60,
        now=now + timedelta(seconds=10),
    )
    after_wrong_worker = await db.get_source_sync_run(enqueued.run_id)
    right_worker = await db.heartbeat_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        lease_seconds=60,
        now=now + timedelta(seconds=10),
    )
    after_right_worker = await db.get_source_sync_run(enqueued.run_id)

    assert wrong_worker is False
    assert after_wrong_worker is not None
    assert after_wrong_worker.lease_expires_at == now + timedelta(seconds=60)
    assert right_worker is True
    assert after_right_worker is not None
    assert after_right_worker.lease_expires_at == now + timedelta(seconds=70)


@pytest.mark.asyncio
async def test_source_sync_run_progress_is_durable_and_fenced_by_current_lease(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-progress-run",
        type="confluence",
        name="Engineering Wiki",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-progress-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None

    stored = await db.report_source_sync_run_progress(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        progress={
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 31, "total": 86, "unit": "page"},
            "counts": {"changed": 12, "failed": 0, "memories_created": 104},
        },
        now=now + timedelta(seconds=5),
    )
    stale = await db.report_source_sync_run_progress(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count + 1,
        progress={
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 86, "total": 86, "unit": "page"},
        },
        now=now + timedelta(seconds=6),
    )
    current = await db.get_source_sync_run(enqueued.run_id)

    assert stored is True
    assert stale is False
    assert current is not None
    assert current.progress == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 31, "total": 86, "unit": "page"},
        "counts": {"changed": 12, "failed": 0, "memories_created": 104},
    }
    assert current.progress_revision == 1
    assert current.progress_updated_at == now + timedelta(seconds=5)


@pytest.mark.asyncio
async def test_complete_source_sync_run_releases_active_slot_for_next_run(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-complete-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-complete-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None

    completed_update = await db.complete_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(
            source="src-complete-run",
            last_sync_at=now,
            last_sync_status="success",
            docs_processed=3,
            docs_updated=2,
            memories_extracted=4,
        ),
        completed_at=now + timedelta(seconds=10),
    )
    completed = await db.get_source_sync_run(enqueued.run_id)
    next_run = await db.enqueue_source_sync_run(
        source_id="src-complete-run",
        workspace_id="workspace-a",
        trigger="manual",
    )

    assert completed_update is True
    assert completed is not None
    assert completed.status == "success"
    assert completed.lease_owner is None
    assert completed.completed_at == now + timedelta(seconds=10)
    assert next_run.run_id != enqueued.run_id
    assert next_run.coalesced is False


@pytest.mark.asyncio
async def test_terminal_source_sync_writes_require_current_lease(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-stale-terminal",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-stale-terminal",
        workspace_id="workspace-a",
        trigger="manual",
    )
    first = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    recovered = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=90),
    )
    assert first is not None
    assert recovered is not None

    stale_complete = await db.complete_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=first.lease_attempt_count,
        final_state=SyncState(
            source="src-stale-terminal",
            last_sync_at=now + timedelta(seconds=95),
            last_sync_status="success",
            docs_processed=99,
        ),
        completed_at=now + timedelta(seconds=95),
    )
    after_stale_complete = await db.get_source_sync_run(enqueued.run_id)
    sync_state_after_stale = await db.get_sync_state("src-stale-terminal")

    stale_fail = await db.fail_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=first.lease_attempt_count,
        error_message="stale failure",
        final_state=SyncState(
            source="src-stale-terminal",
            last_sync_at=now + timedelta(seconds=96),
            last_sync_status="failed",
            error_message="stale failure",
        ),
        retryable=False,
        failed_at=now + timedelta(seconds=96),
    )
    after_stale_fail = await db.get_source_sync_run(enqueued.run_id)
    sync_state_after_stale_fail = await db.get_sync_state("src-stale-terminal")

    assert stale_complete is False
    assert stale_fail is False
    assert after_stale_complete is not None
    assert after_stale_complete.lease_owner == "worker-b"
    assert after_stale_complete.lease_attempt_count == recovered.lease_attempt_count
    assert after_stale_complete.status == "running"
    assert after_stale_fail is not None
    assert after_stale_fail.status == "running"
    assert after_stale_fail.lease_owner == "worker-b"
    assert sync_state_after_stale is None
    assert sync_state_after_stale_fail is None


@pytest.mark.asyncio
async def test_fail_source_sync_run_requeues_retryable_failure_after_backoff(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-fail-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-fail-run",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None

    failed_update = await db.fail_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message="temporary outage",
        retryable=True,
        failed_at=now + timedelta(seconds=5),
        next_attempt_at=now + timedelta(seconds=65),
    )
    failed = await db.get_source_sync_run(enqueued.run_id)
    too_early = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=6),
    )
    retried = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=65),
    )

    assert failed is not None
    assert failed_update is True
    assert failed.status == "pending"
    assert failed.next_attempt_at == now + timedelta(seconds=65)
    assert failed.error_message == "temporary outage"
    assert failed.lease_owner is None
    assert too_early is None
    assert retried is not None
    assert retried.run_id == enqueued.run_id
    assert retried.lease_attempt_count == 2


@pytest.mark.asyncio
async def test_fail_source_sync_run_marks_terminal_after_retry_budget(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-fail-budget",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-fail-budget",
        workspace_id="workspace-a",
        trigger="manual",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None

    failed_update = await db.fail_source_sync_run(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message="permanent outage",
        retryable=False,
        failed_at=now + timedelta(seconds=5),
        next_attempt_at=now + timedelta(seconds=65),
    )
    failed = await db.get_source_sync_run(enqueued.run_id)
    retried = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=65),
    )

    assert failed is not None
    assert failed_update is True
    assert failed.status == "failed"
    assert failed.next_attempt_at is None
    assert failed.completed_at == now + timedelta(seconds=5)
    assert retried is None


@pytest.mark.asyncio
async def test_terminal_failure_preserves_coalesced_rerun(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-terminal-rerun",
        type="teams",
        name="Teams",
        config_json="{}",
    )
    active = await db.enqueue_source_sync_run(
        source_id="src-terminal-rerun",
        trigger="local_agent",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None
    await db.enqueue_source_sync_run(
        source_id="src-terminal-rerun",
        trigger="local_agent",
    )

    failed = await db.fail_source_sync_run(
        active.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message="retry budget exhausted",
        retryable=False,
        failed_at=now + timedelta(seconds=10),
    )
    successor = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        now=now + timedelta(seconds=11),
    )

    assert failed is True
    assert successor is not None
    assert successor.run_id != active.run_id
    assert successor.trigger == "rerun"


@pytest.mark.asyncio
async def test_complete_source_sync_run_creates_successor_for_coalesced_request(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-successor",
        type="github_repo",
        name="Repo",
        config_json="{}",
    )
    active = await db.enqueue_source_sync_run(
        source_id="src-successor",
        workspace_id="workspace-a",
        trigger="manual",
        input_snapshot_id="snapshot-old",
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None
    coalesced = await db.enqueue_source_sync_run(
        source_id="src-successor",
        workspace_id="workspace-a",
        trigger="force",
        force_full_sync=True,
        input_snapshot_id="snapshot-new",
    )

    completed_update = await db.complete_source_sync_run(
        active.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(
            source="src-successor",
            last_sync_at=now + timedelta(seconds=10),
            last_sync_status="success",
        ),
        completed_at=now + timedelta(seconds=10),
    )
    completed = await db.get_source_sync_run(active.run_id)
    successor = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        workspace_id="workspace-a",
        lease_seconds=60,
        now=now + timedelta(seconds=11),
    )

    assert coalesced.run_id == active.run_id
    assert coalesced.coalesced is True
    assert completed_update is True
    assert completed is not None
    assert completed.status == "success"
    assert successor is not None
    assert successor.run_id != active.run_id
    assert successor.trigger == "rerun"
    assert successor.force_full_sync is True
    assert successor.input_snapshot_id == "snapshot-new"


@pytest.mark.asyncio
async def test_source_sync_inputs_are_immutable_generations_per_workspace_source(db: Database):
    await db.upsert_source(
        id="src-input-run",
        type="teams",
        name="Teams",
        config_json="{}",
    )

    first = await db.create_source_sync_input(
        source_id="src-input-run",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-run/inputs/one.json",
        raw_sha256="sha-one",
        raw_content_type="application/json",
        metadata={"conversation_id": "chat-1"},
    )
    second = await db.create_source_sync_input(
        source_id="src-input-run",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-run/inputs/two.json",
        raw_sha256="sha-two",
        raw_content_type="application/json",
        metadata={"conversation_id": "chat-1"},
    )
    listed = await db.list_source_sync_inputs(
        source_id="src-input-run",
        workspace_id="workspace-a",
    )

    assert first.input_generation == 1
    assert second.input_generation == 2
    assert [item.input_id for item in listed] == [first.input_id, second.input_id]
    assert listed[0].metadata == {"conversation_id": "chat-1"}


@pytest.mark.asyncio
async def test_source_sync_inputs_are_idempotent_by_raw_hash(db: Database):
    await db.upsert_source(
        id="src-input-idempotent",
        type="teams",
        name="Teams",
        config_json="{}",
    )

    first = await db.create_source_sync_input(
        source_id="src-input-idempotent",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-idempotent/inputs/one.json",
        raw_sha256="sha-same",
        raw_content_type="application/json",
        metadata={"conversation_id": "chat-1"},
    )
    duplicate = await db.create_source_sync_input(
        source_id="src-input-idempotent",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-idempotent/inputs/one-copy.json",
        raw_sha256="sha-same",
        raw_content_type="application/json",
        metadata={"conversation_id": "chat-1", "submitted_at": "later"},
    )
    listed = await db.list_source_sync_inputs(
        source_id="src-input-idempotent",
        workspace_id="workspace-a",
    )

    assert duplicate.input_id == first.input_id
    assert duplicate.input_generation == first.input_generation
    assert duplicate.raw_uri == first.raw_uri
    assert [item.input_id for item in listed] == [first.input_id]


@pytest.mark.asyncio
async def test_snapshot_input_uses_top_level_doc_id_and_rejects_missing_identity(db: Database):
    await db.upsert_source(id="src-snapshot-doc", type="jira", name="Jira", config_json="{}")
    created = await db.create_source_sync_input(
        source_id="src-snapshot-doc",
        raw_uri="object://input.json",
        raw_sha256="sha-input",
        raw_content_type="application/json",
        metadata={"doc_id": "doc-top-level"},
        sync_snapshot_id="snapshot-a",
    )
    listed = await db.list_source_sync_inputs(
        source_id="src-snapshot-doc",
        input_snapshot_id="snapshot-a",
    )
    assert [item.input_id for item in listed] == [created.input_id]
    with pytest.raises(ValueError, match="requires doc_id"):
        await db.create_source_sync_input(
            source_id="src-snapshot-doc",
            raw_uri="object://missing.json",
            raw_sha256="sha-missing",
            raw_content_type="application/json",
            metadata={},
            sync_snapshot_id="snapshot-b",
        )


@pytest.mark.asyncio
async def test_source_sync_inputs_filter_by_current_snapshot_membership(db: Database):
    await db.upsert_source(
        id="src-input-snapshot",
        type="local_markdown",
        name="Local",
        config_json="{}",
    )

    old = await db.create_source_sync_input(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-snapshot/inputs/old.json",
        raw_sha256="sha-old",
        raw_content_type="application/json",
        sync_snapshot_id="snapshot-old",
        metadata={
            "manifest_entry": {"doc_id": "doc-old", "package_uri": "old"},
        },
    )
    new = await db.create_source_sync_input(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-snapshot/inputs/new.json",
        raw_sha256="sha-new",
        raw_content_type="application/json",
        sync_snapshot_id="snapshot-new",
        metadata={
            "manifest_entry": {"doc_id": "doc-new", "package_uri": "new"},
        },
    )
    repeated = await db.create_source_sync_input(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-snapshot/inputs/new-copy.json",
        raw_sha256="sha-new",
        raw_content_type="application/json",
        sync_snapshot_id="snapshot-repeat",
        metadata={
            "manifest_entry": {"doc_id": "doc-new", "package_uri": "new"},
        },
    )

    all_inputs = await db.list_source_sync_inputs(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
    )
    new_snapshot = await db.list_source_sync_inputs(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        input_snapshot_id="snapshot-new",
    )
    repeat_snapshot = await db.list_source_sync_inputs(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        input_snapshot_id="snapshot-repeat",
    )
    empty_snapshot = await db.list_source_sync_inputs(
        source_id="src-input-snapshot",
        workspace_id="workspace-a",
        input_snapshot_id="snapshot-empty",
    )

    assert [item.input_id for item in all_inputs] == [old.input_id, new.input_id]
    assert [item.input_id for item in new_snapshot] == [new.input_id]
    assert repeated.input_id == new.input_id
    assert [item.input_id for item in repeat_snapshot] == [new.input_id]
    assert empty_snapshot == []


class EmptyGene:
    def __init__(self) -> None:
        self.bound_document_store = None

    def bind_document_store(self, document_store) -> None:
        self.bound_document_store = document_store

    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        return False

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        if False:
            yield ContentItem(item_id="never", title="never", updated_at=datetime.now(timezone.utc))


def test_failed_document_summary_identifies_embedding_provider_outage():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="Embedding provider unreachable: [Errno 111] Connection refused",
            ),
        ],
    )

    assert message == (
        "1 document could not be synced. Embedding provider was unreachable for 1 document. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


def test_failed_document_summary_identifies_llm_provider_outage():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="litellm.InternalServerError: AnthropicException - Cannot connect to host provider.example:443",
            ),
        ],
    )

    assert message == (
        "1 document could not be synced. LLM provider was unreachable for 1 document. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


@pytest.mark.asyncio
async def test_extraction_work_pool_allows_one_source_to_use_all_workers():
    pool = ExtractionWorkPool(max_workers=6)
    entered = 0
    release = asyncio.Event()

    async def hold_slot() -> None:
        nonlocal entered
        async with pool.slot("src-a"):
            entered += 1
            await release.wait()

    tasks = [asyncio.create_task(hold_slot()) for _ in range(6)]
    for _ in range(20):
        if entered == 6:
            break
        await asyncio.sleep(0.01)

    assert entered == 6

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_extraction_work_pool_favors_waiting_source_over_extra_borrowed_work():
    pool = ExtractionWorkPool(max_workers=6)
    release_a = asyncio.Event()
    entered: list[str] = []

    async def hold_source_a() -> None:
        async with pool.slot("src-a"):
            entered.append("a")
            await release_a.wait()

    source_a_tasks = [asyncio.create_task(hold_source_a()) for _ in range(6)]
    for _ in range(20):
        if entered.count("a") == 6:
            break
        await asyncio.sleep(0.01)
    assert entered.count("a") == 6

    source_b_entered = asyncio.Event()
    release_b = asyncio.Event()
    extra_a_entered = asyncio.Event()

    async def wait_source_b() -> None:
        async with pool.slot("src-b"):
            entered.append("b")
            source_b_entered.set()
            await release_b.wait()

    async def wait_extra_source_a() -> None:
        async with pool.slot("src-a"):
            entered.append("extra-a")
            extra_a_entered.set()

    source_b_task = asyncio.create_task(wait_source_b())
    extra_a_task = asyncio.create_task(wait_extra_source_a())
    await asyncio.sleep(0)

    source_a_tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await source_a_tasks[0]

    await asyncio.wait_for(source_b_entered.wait(), timeout=1)
    assert not extra_a_entered.is_set()

    release_b.set()
    release_a.set()
    await asyncio.gather(*source_a_tasks[1:], source_b_task, extra_a_task)


@pytest.mark.asyncio
async def test_shared_extraction_pool_caps_orchestrator_work_across_sources(db: Database):
    for source_id in ("src-pool-a", "src-pool-b"):
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
        )

    pool = ExtractionWorkPool(max_workers=4)
    release_enrichment = asyncio.Event()
    release_fetch = asyncio.Event()
    release_fetch.set()
    enricher = BlockingEnricher(release=release_enrichment, target_entries=4)

    def make_orchestrator() -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=db,
            doc_store=StubDocumentStore(),
            enricher=enricher,
            memory_extractor=NoopMemoryExtractor(),
            memory_engine=NoopMemoryEngine(),
            memory_store=None,
            max_concurrent=4,
            extraction_pool=pool,
        )

    task_a = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=BlockingFetchGene(item_count=4, release=release_fetch),
            source_name="Source A",
            source_id="src-pool-a",
        )
    )
    task_b = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=BlockingFetchGene(item_count=4, release=release_fetch),
            source_name="Source B",
            source_id="src-pool-b",
        )
    )

    await asyncio.wait_for(enricher.target_reached.wait(), timeout=2)
    await asyncio.sleep(0.05)

    assert enricher.max_active == 4

    release_enrichment.set()
    states = await asyncio.gather(task_a, task_b)
    assert [state.last_sync_status for state in states] == ["success", "success"]


@pytest.mark.asyncio
async def test_document_lifecycle_admission_caps_fetch_across_sources(db: Database):
    for source_id in ("src-doc-a", "src-doc-b"):
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
        )

    admission = DocumentLifecycleAdmission(max_active=1)
    release_fetch = asyncio.Event()
    tracker = SharedFetchTracker(target_entries=1)

    def make_orchestrator() -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=db,
            doc_store=StubDocumentStore(),
            enricher=InstantEnricher(),
            memory_extractor=NoopMemoryExtractor(),
            memory_engine=NoopMemoryEngine(),
            memory_store=None,
            max_concurrent=2,
            document_lifecycle_admission=admission,
        )

    task_a = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=TrackedFetchGene(prefix="a", release=release_fetch, tracker=tracker),
            source_name="Source A",
            source_id="src-doc-a",
        )
    )
    task_b = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=TrackedFetchGene(prefix="b", release=release_fetch, tracker=tracker),
            source_name="Source B",
            source_id="src-doc-b",
        )
    )

    await asyncio.wait_for(tracker.target_reached.wait(), timeout=2)
    await asyncio.sleep(0.05)

    assert tracker.max_active == 1

    release_fetch.set()
    states = await asyncio.gather(task_a, task_b)
    assert [state.last_sync_status for state in states] == ["success", "success"]


def test_failed_document_summary_keeps_rate_limit_precedence_over_llm_timeout_text():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="litellm.RateLimitError: 429 rate limit after request timeout",
            ),
        ],
    )

    assert message == (
        "1 Confluence document could not be imported. Confluence rate limited 1 document. "
        "Wait a few minutes, then retry the sync."
    )


def test_failed_document_summary_preserves_mixed_failure_guidance():
    message = summarize_failed_documents(
        3,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="Embedding provider unreachable: [Errno 111] Connection refused",
            ),
            FailedDoc(doc_id="doc-2", title="Doc 2", error="Confluence rate limit 429"),
            FailedDoc(doc_id="doc-3", title="Doc 3", error="PDF export did not produce a PDF"),
        ],
    )

    assert message == (
        "3 documents could not be synced. Embedding provider was unreachable for 1 document; "
        "PDF export was unavailable for 1 document; Confluence rate limited 1 document. "
        "Wait a few minutes, then retry the sync. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


class SinceRecordingEmptyGene(EmptyGene):
    def __init__(self) -> None:
        self.seen_since = None

    async def discover(self, since=None):
        self.seen_since = since
        if False:
            yield ContentItem(item_id="never", title="never", updated_at=datetime.now(timezone.utc))


class IncrementalNewDocumentGene:
    def __init__(self) -> None:
        self.seen_since = None

    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        return False

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="agent_session",
            display_name="Agent Session",
            description="",
            default_sync_interval_minutes=0,
            auth_method="local_file",
            data_shape="message",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        self.seen_since = since
        yield ContentItem(
            item_id="doc-new",
            title="New Session",
            source_url="agent-session://new",
            last_modified=datetime.now(timezone.utc),
            content_type="application/json",
            space_or_project="sessions",
            version="new-version",
        )

    async def fetch(self, item):
        return RawContent(item=item, body=b'{"summary":"new"}', content_type="application/json")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="# New Session\n\nSummary")


class FailingAuthGene:
    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        return False

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=60,
            auth_method="pat",
            data_shape="ticket",
        )

    async def authenticate(self) -> None:
        raise RuntimeError("auth failed")


class StubDocumentStore:
    def __init__(self) -> None:
        self.normalized_content: dict[str, str] = {}

    def store_raw(self, *, source_id, title, content, content_type, extension=None):
        suffix = extension or ".raw"
        return f"file:///tmp/{source_id}/{title}{suffix}"

    def store_normalized(self, *, source_id, title, markdown):
        uri = f"stub-doc://{source_id}/{title}.md"
        self.normalized_content[uri] = markdown
        return uri

    def read_normalized(self, uri):
        return self.normalized_content.get(uri)


class RecordingSyncMemoryLogger:
    def __init__(self):
        self.records: list[tuple[str, dict]] = []

    def info(self, message):
        import json

        self.records.append(("info", json.loads(message)))

    def debug(self, message):
        import json

        self.records.append(("debug", json.loads(message)))


class FailingPdfDocumentStore(StubDocumentStore):
    def store_raw(self, *, source_id, title, content, content_type, extension=None):
        if content_type == "application/pdf":
            raise RuntimeError("disk full while storing PDF")
        return super().store_raw(
            source_id=source_id,
            title=title,
            content=content,
            content_type=content_type,
            extension=extension,
        )


@pytest.mark.asyncio
async def test_sync_gene_binds_document_store_at_pipeline_boundary(db: Database):
    source_id = "src-bind-doc-store"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json="{}",
    )
    doc_store = StubDocumentStore()
    gene = EmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Teams",
        source_id=source_id,
    )

    assert gene.bound_document_store is doc_store
    assert state.last_sync_status == "success"


class NoopMemoryEngine:
    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        return []

    async def process_memories(self, **kwargs):
        return {"inserted": 0, "corroborated": 0, "skipped": 0}

    async def reconcile_and_persist(self, **kwargs):
        return {"added": 0, "updated": 0, "superseded": 0, "deleted": 0, "noop": 0}


class RecordingSourceSupportDetector:
    async def detect_and_persist(self, **kwargs):
        return {
            "added": 1,
            "updated": 0,
            "removed_stale": 0,
        }


class FailingDocumentDeleteMemoryStore:
    async def delete_document(self, doc_id: str, **kwargs):
        raise RuntimeError("delete document failed")


class CountingMemoryEngine(NoopMemoryEngine):
    def __init__(self, inserted: int):
        self.inserted = inserted
        self.enrichment_calls = 0
        self.process_calls = 0

    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        self.enrichment_calls += 1
        return []

    async def process_memories(self, **kwargs):
        self.process_calls += 1
        return {"inserted": self.inserted, "corroborated": 0, "skipped": 0}


class RecordingMemoryEngine(NoopMemoryEngine):
    def __init__(self) -> None:
        self.process_memories_calls: list[dict] = []
        self.reconcile_calls: list[dict] = []

    async def process_memories(self, **kwargs):
        self.process_memories_calls.append(kwargs)
        return await super().process_memories(**kwargs)

    async def reconcile_and_persist(self, **kwargs):
        self.reconcile_calls.append(kwargs)
        return await super().reconcile_and_persist(**kwargs)


class FailingVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}
        self.deleted: list[str] = []

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or []) if record_id in self.upserted]
        return {
            "ids": selected,
            "metadatas": [self.upserted[record_id].get("metadata", {}) for record_id in selected],
            "embeddings": [self.upserted[record_id].get("embedding") for record_id in selected],
            "documents": [self.upserted[record_id].get("document") for record_id in selected],
        }

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = {
                "metadata": metadatas[index] if metadatas else {},
                "embedding": embeddings[index] if embeddings else None,
                "document": documents[index] if documents else None,
            }
        raise RuntimeError("document vector failed after mutation")

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for record_id in ids:
            self.upserted.pop(record_id, None)


class FalseyVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}

    def __bool__(self) -> bool:
        return False

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or []) if record_id in self.upserted]
        return {
            "ids": selected,
            "metadatas": [self.upserted[record_id].get("metadata", {}) for record_id in selected],
            "embeddings": [self.upserted[record_id].get("embedding") for record_id in selected],
            "documents": [self.upserted[record_id].get("document") for record_id in selected],
        }

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = {
                "metadata": metadatas[index] if metadatas else {},
                "embedding": embeddings[index] if embeddings else None,
                "document": documents[index] if documents else None,
            }

    def delete(self, *, ids):
        for record_id in ids:
            self.upserted.pop(record_id, None)


class FlakyFalseyVectorStore(FalseyVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("transient vector failure")
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


class NoopMemoryExtractor:
    async def extract_memories(self, **kwargs):
        return MemoryExtractionResult(memories=[])


class RecordingMemoryExtractor(NoopMemoryExtractor):
    def __init__(self) -> None:
        self.full_calls: list[dict] = []
        self.change_calls: list[dict] = []
        self.unit_calls: list[dict] = []

    async def extract_memories(self, **kwargs):
        self.full_calls.append(kwargs)
        return MemoryExtractionResult(memories=[])

    async def extract_memory_changes(self, **kwargs):
        self.change_calls.append(kwargs)
        return MemoryExtractionResult(memories=[])

    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        return MemoryExtractionResult(memories=[])


class FailingMemoryExtractor(NoopMemoryExtractor):
    async def extract_memories(self, **kwargs):
        return MemoryExtractionResult(
            memories=[],
            error_type="json_parse_error",
            error="Unterminated string starting at line 393 column 16",
        )


class PartiallyFailingUnitMemoryExtractor(RecordingMemoryExtractor):
    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        if context.unit.heading_path[-1] == "Section 2":
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error="unit failed",
            )
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    content=f"{context.unit.heading_path[-1]} contains durable design guidance.",
                    memory_type="fact",
                    extraction_context="durable design guidance",
                )
            ]
        )


class BlockingUnitMemoryExtractor(RecordingMemoryExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.started_two = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.max_active >= 2:
            self.started_two.set()
        try:
            await self.release.wait()
            return MemoryExtractionResult(memories=[])
        finally:
            self.active -= 1


class BlockingFetchGene:
    def __init__(self, item_count: int, release: asyncio.Event):
        self.item_count = item_count
        self.release = release
        self.active_fetches = 0
        self.max_active_fetches = 0

    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        return False

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=60,
            auth_method="pat",
            data_shape="ticket",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        for idx in range(self.item_count):
            yield ContentItem(
                item_id=f"jira-{idx}",
                title=f"Jira {idx}",
                source_url=f"https://jira.example/browse/{idx}",
                last_modified=datetime.now(timezone.utc),
                content_type="application/json",
                space_or_project="PAY",
                version=str(idx),
            )

    async def fetch(self, item):
        self.active_fetches += 1
        self.max_active_fetches = max(self.max_active_fetches, self.active_fetches)
        try:
            await self.release.wait()
            return RawContent(item=item, body=b'{"summary":"test"}', content_type="application/json")
        finally:
            self.active_fetches -= 1

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\nBody")


class SharedFetchTracker:
    def __init__(self, target_entries: int) -> None:
        self.active = 0
        self.max_active = 0
        self.target_entries = target_entries
        self.target_reached = asyncio.Event()

    def enter(self) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.max_active >= self.target_entries:
            self.target_reached.set()

    def exit(self) -> None:
        self.active -= 1


class TrackedFetchGene(BlockingFetchGene):
    def __init__(self, *, prefix: str, release: asyncio.Event, tracker: SharedFetchTracker):
        super().__init__(item_count=1, release=release)
        self.prefix = prefix
        self.tracker = tracker

    async def discover(self, since=None):
        yield ContentItem(
            item_id=f"jira-{self.prefix}-0",
            title=f"Jira {self.prefix}",
            source_url=f"https://jira.example/browse/{self.prefix}",
            last_modified=datetime.now(timezone.utc),
            content_type="application/json",
            space_or_project="PAY",
            version=self.prefix,
        )

    async def fetch(self, item):
        self.tracker.enter()
        try:
            await self.release.wait()
            return RawContent(item=item, body=b'{"summary":"test"}', content_type="application/json")
        finally:
            self.tracker.exit()


class PdfBackfillGene(BlockingFetchGene):
    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        del item
        return existing_doc is None or existing_hash != new_hash or not getattr(existing_doc, "pdf_content_uri", None)

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="confluence",
            display_name="Confluence",
            description="",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    async def fetch_pdf(self, item):
        return b"%PDF-1.4\n" + (b"x" * 128)


class MissingPdfGene(PdfBackfillGene):
    async def fetch_pdf(self, item):
        return None


class UpdatingDocumentGene:
    def __init__(
        self,
        markdown: str,
        version: str = "2",
        *,
        last_modified: datetime | None = None,
        source_updated_at: str | None = None,
    ) -> None:
        self.markdown = markdown
        self.version = version
        self.last_modified = last_modified or datetime.now(timezone.utc)
        self.source_updated_at = source_updated_at

    def requires_pdf_artifact(
        self,
        *,
        item,
        existing_doc,
        existing_hash,
        new_hash,
    ) -> bool:
        return False

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="docs",
            display_name="Documents",
            description="",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        yield ContentItem(
            item_id="doc-1",
            title="Design Doc",
            source_url="https://docs.example/doc-1",
            last_modified=self.last_modified,
            content_type="text/markdown",
            space_or_project="ARCH",
            version=self.version,
        )

    async def fetch(self, item):
        return RawContent(item=item, body=self.markdown.encode("utf-8"), content_type="text/markdown")

    async def normalize(self, raw):
        source_semantics = {}
        if self.source_updated_at is not None:
            source_semantics["source_updated_at"] = self.source_updated_at
        return NormalizedContent(item=raw.item, markdown_body=self.markdown, source_semantics=source_semantics)


class UpdatingTicketGene(UpdatingDocumentGene):
    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=360,
            auth_method="browser_cookie",
            data_shape="ticket",
        )


class DocumentVisibleEnricher:
    def __init__(self, db: Database, source_id: str):
        self.db = db
        self.source_id = source_id

    async def enrich_document(self, *, doc_id, content, source_type):
        async with self.db.db.execute(
            "SELECT COUNT(*) FROM documents WHERE source = ? AND doc_id = ?",
            (self.source_id, doc_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1
        return EnrichmentResult(
            summary="Summary",
            tags=[],
            entities=[],
            relationships=[],
            doc_type="jira_issue",
            complexity="low",
        )


class EntityMentioningEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        return EnrichmentResult(
            summary="Summary",
            tags=["tag-one"],
            entities=[
                RawEntityRef(
                    name="Raw Extracted Entity",
                    type="service",
                    tags=["service"],
                    aliases=["Raw Alias"],
                )
            ],
            relationships=[],
            doc_type="jira_issue",
            complexity="low",
        )


class InstantEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        del doc_id, content, source_type
        return EnrichmentResult(
            summary="Summary",
            tags=[],
            entities=[],
            relationships=[],
            doc_type="jira_issue",
            complexity="low",
        )


class BlockingEnricher:
    def __init__(self, release: asyncio.Event, target_entries: int):
        self.release = release
        self.target_entries = target_entries
        self.entered = 0
        self.active = 0
        self.max_active = 0
        self.target_reached = asyncio.Event()

    async def enrich_document(self, *, doc_id, content, source_type):
        del doc_id, content, source_type
        self.entered += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.entered >= self.target_entries:
            self.target_reached.set()
        try:
            await self.release.wait()
            return EnrichmentResult(
                summary="Summary",
                tags=[],
                entities=[],
                relationships=[],
                doc_type="jira_issue",
                complexity="low",
            )
        finally:
            self.active -= 1


class ExplodingEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        raise AssertionError("unchanged document should not be enriched")


class RaisingEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        raise RuntimeError("enrichment exploded")


class ConstantMemorySampler:
    def __init__(self):
        self.rss = 100.0

    def sample(self):
        self.rss += 1.0
        return MemorySample(rss_mb=self.rss, peak_rss_mb=self.rss)


async def _insert_source_and_doc(db: Database, source_id: str) -> None:
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json="{}",
    )
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-1", source_id, "http://example/doc-1", "Doc 1", "ARCH", now, "1", "hash-1", now),
    )
    await db.update_source_doc_count(source_id, 1)


async def _insert_source_with_docs(db: Database, source_id: str, doc_ids: list[str]) -> None:
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Agent Session Summaries",
        config_json="{}",
    )
    now = datetime.now(timezone.utc).isoformat()
    for doc_id in doc_ids:
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, source_id, f"agent-session://{doc_id}", doc_id, "sessions", now, "1", f"hash-{doc_id}", now),
        )
    await db.update_source_doc_count(source_id, len(doc_ids))


async def _insert_document_with_metadata(
    db: Database,
    *,
    source_id: str,
    doc_id: str,
    title: str,
    markdown: str,
    version: str,
    normalized_content_uri: str | None = None,
) -> None:
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    now = datetime.now(timezone.utc)
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version,
            content_hash, normalized_content_uri, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            source_id,
            f"http://example/{doc_id}",
            title,
            "ARCH",
            now.isoformat(),
            version,
            content_hash(markdown),
            normalized_content_uri,
            now.isoformat(),
        ),
    )
    await db.upsert_metadata(
        DocumentMetadata(
            doc_id=doc_id,
            summary="Existing summary",
            tags=["existing"],
            entities=[
                Entity(
                    id=1,
                    canonical_name="Existing Entity",
                    tags=[],
                    display_name="Existing Entity",
                )
            ],
            doc_type="jira_issue",
            complexity="low",
            enriched_at=now,
        )
    )
    await db.update_source_doc_count(source_id, 1)


def _audited_memory_store(db: Database) -> MemoryStore:
    adapters = build_sqlite_adapters(db, object())
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(
            db,
            default_context=AuditContext(actor_type="test", run_id="run-sync-bookkeeping"),
        ),
    )


@pytest.mark.asyncio
async def test_sync_memory_observer_records_discovery_and_document_stages(db: Database):
    source_id = "src-sync-memory"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    log = RecordingSyncMemoryLogger()
    observer = SyncMemoryObserver(
        sampler=ConstantMemorySampler(),
        logger=log,
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        source_support_detector=RecordingSourceSupportDetector(),
        max_concurrent=1,
        memory_observer=observer,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    events = [event for _level, event in log.records]
    stages = [event["stage"] for event in events]
    assert "sync_run_start" in stages
    assert "after_discovery" in stages
    assert "document_wait_start" in stages
    assert "document_lifecycle_enter" in stages
    assert "after_fetch" in stages
    assert "after_normalize" in stages
    assert "after_raw_store" in stages
    assert "after_enrich" in stages
    assert "after_extract" in stages
    assert "after_memory_engine" in stages
    assert "after_source_support" in stages
    assert "document_lifecycle_exit" in stages
    assert "sync_run_end" in stages
    discovery = next(event for event in events if event["stage"] == "after_discovery")
    assert discovery["item_count"] == 1
    assert discovery["indexed_doc_count"] == 0
    assert discovery["full_sync"] is True
    support = next(event for event in events if event["stage"] == "after_source_support")
    assert support["memory_supports_added"] == 1
    assert "title" not in support
    assert "source_url" not in support


@pytest.mark.asyncio
async def test_sync_memory_observer_records_pdf_export_stage(db: Database):
    source_id = "src-sync-memory-pdf"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Confluence Space",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    log = RecordingSyncMemoryLogger()
    observer = SyncMemoryObserver(
        sampler=ConstantMemorySampler(),
        logger=log,
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
        memory_observer=observer,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Confluence Space",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    pdf_events = [
        event
        for _level, event in log.records
        if event["stage"] == "after_pdf_export"
    ]
    assert len(pdf_events) == 1
    assert pdf_events[0]["source_id"] == source_id
    assert pdf_events[0]["doc_id"] == "jira-0"
    assert pdf_events[0]["pdf_bytes"] == 137


@pytest.mark.asyncio
async def test_sync_memory_observer_records_lifecycle_exit_when_document_fails(db: Database):
    source_id = "src-sync-memory-error"
    await db.upsert_source(id=source_id, type="jira", name="Jira Board", config_json="{}")
    release = asyncio.Event()
    release.set()
    log = RecordingSyncMemoryLogger()
    observer = SyncMemoryObserver(sampler=ConstantMemorySampler(), logger=log)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=RaisingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
        memory_observer=observer,
    )

    state = await orchestrator.sync_gene(gene=BlockingFetchGene(1, release), source_name="Jira Board", source_id=source_id)

    assert state.last_sync_status == "failed"
    exits = [event for _level, event in log.records if event["stage"] == "document_lifecycle_exit"]
    assert exits
    assert all(event["ok"] is False for event in exits)
    assert all(event["error_class"] == "RuntimeError" for event in exits)


@pytest.mark.asyncio
async def test_successful_zero_change_sync_advances_last_sync_and_keeps_doc_count(db: Database):
    source_id = "src-sync-bookkeeping"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime.now(timezone.utc) - timedelta(days=1)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
    )

    source = await db.get_source(source_id)
    assert state.last_sync_status == "success"
    assert state.docs_processed == 0
    assert state.last_sync_at is not None
    assert state.last_sync_at > previous_sync
    assert source["last_sync"] == state.last_sync_at.isoformat()
    assert source["doc_count"] == 1


@pytest.mark.asyncio
async def test_incremental_sync_uses_overlap_window_for_discovery(db: Database):
    source_id = "src-sync-overlap"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime(2026, 5, 26, 14, 55, 33, tzinfo=timezone.utc)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    gene = SinceRecordingEmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert gene.seen_since == previous_sync - timedelta(minutes=10)


@pytest.mark.asyncio
async def test_incremental_sync_does_not_delete_unchanged_documents_from_small_source(db: Database):
    source_id = "src-agent-sessions-incremental"
    await _insert_source_with_docs(db, source_id, ["doc-old-a", "doc-old-b"])
    previous_sync = datetime.now(timezone.utc) - timedelta(hours=1)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=2,
            docs_updated=2,
        ),
    )
    gene = IncrementalNewDocumentGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Agent Session Summaries",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert gene.seen_since == previous_sync - timedelta(minutes=10)
    assert await db.count_documents(source=source_id) == 3
    assert await db.get_document("doc-old-a") is not None
    assert await db.get_document("doc-old-b") is not None
    audit_rows = await db.list_memory_audit_events(event_type="document_delete_committed")
    assert audit_rows == []


@pytest.mark.asyncio
async def test_force_full_sync_ignores_incremental_cursor(db: Database):
    source_id = "src-force-full-overlap"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime(2026, 5, 26, 14, 55, 33, tzinfo=timezone.utc)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    gene = SinceRecordingEmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    await orchestrator.sync_gene(
        gene=gene,
        source_name="Architecture",
        source_id=source_id,
        force_full_sync=True,
    )

    assert gene.seen_since is None


@pytest.mark.asyncio
async def test_authoritative_snapshot_ignores_cursor_without_forcing_reprocessing(db: Database):
    source_id = "src-authoritative-snapshot"
    await _insert_source_and_doc(db, source_id)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime(2026, 5, 26, 14, 55, 33, tzinfo=timezone.utc),
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    gene = SinceRecordingEmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    await orchestrator.sync_gene(
        gene=gene,
        source_name="Architecture",
        source_id=source_id,
        authoritative_snapshot=True,
    )

    assert gene.seen_since is None


@pytest.mark.asyncio
async def test_force_full_sync_reprocesses_unchanged_document(db: Database, tmp_path):
    source_id = "src-force-reprocess"
    markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id="src-documents",
        title="Design Doc",
        markdown=markdown,
    )
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=markdown,
        version="2",
        normalized_content_uri=normalized_content_uri,
    )
    extractor = RecordingMemoryExtractor()
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="2"),
        source_name="Documents",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_updated == 1
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) == 1
    assert extractor.change_calls == []
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "full_document"


@pytest.mark.asyncio
async def test_document_last_modified_becomes_memory_source_updated_at(db: Database):
    source_id = "src-document-source-updated"
    markdown = "# Design Doc\n\nThe service keeps source timestamps."
    last_modified = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Documents",
        config_json="{}",
    )
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=memory_engine,
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, last_modified=last_modified),
        source_name="Documents",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(memory_engine.reconcile_calls) == 0
    # No source-specific metadata was supplied; document last_modified is the
    # canonical source-side update time forwarded into memory provenance.
    assert len(memory_engine.process_memories_calls) == 1
    assert memory_engine.process_memories_calls[0]["source_updated_at"] == last_modified


@pytest.mark.asyncio
async def test_explicit_source_updated_at_overrides_document_last_modified(db: Database):
    source_id = "src-explicit-source-updated"
    markdown = "# Design Doc\n\nThe source gives a separate updated time."
    last_modified = datetime(2026, 6, 10, 8, 30, tzinfo=timezone.utc)
    explicit_source_updated_at = "2026-06-11T09:45:00+00:00"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Documents",
        config_json="{}",
    )
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=memory_engine,
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(
            markdown,
            last_modified=last_modified,
            source_updated_at=explicit_source_updated_at,
        ),
        source_name="Documents",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(memory_engine.process_memories_calls) == 1
    assert memory_engine.process_memories_calls[0]["source_updated_at"].isoformat() == explicit_source_updated_at


@pytest.mark.asyncio
async def test_deletion_failure_marks_sync_failed(db: Database):
    source_id = "src-deletion-failure"
    await _insert_source_and_doc(db, source_id)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=FailingDocumentDeleteMemoryStore(),
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "delete document failed" in state.failed_docs[0].error
    assert history[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_auth_failure_records_failed_sync_state_without_secondary_error(db: Database):
    source_id = "src-auth-fail"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Auth Failure Source",
        config_json="{}",
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=FailingAuthGene(),
        source_name="Auth Failure Source",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    stored_state = await db.get_sync_state(source_id)
    assert state.last_sync_status == "failed"
    assert state.error_message == "auth failed"
    assert stored_state.last_sync_status == "failed"
    assert stored_state.error_message == "auth failed"
    assert history[0]["status"] == "failed"
    assert history[0]["error_message"] == "auth failed"


@pytest.mark.asyncio
async def test_run_all_active_sources_enqueues_durable_runs(db: Database):
    source_id = "src-scheduled-tracked"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Source",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())

    await service.run_all_active_sources()
    run = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    assert run.source_id == source_id
    assert run.status == "pending"
    assert run.coalesced is True
    assert service.tasks == {}


@pytest.mark.asyncio
async def test_source_sync_schedule_round_trips_and_claims_due_sources(db: Database):
    claim_time = datetime(2026, 6, 16, tzinfo=timezone.utc)
    due_at = claim_time - timedelta(minutes=1)
    future_at = claim_time + timedelta(hours=1)
    await db.upsert_source(
        id="src-due",
        type="jira",
        name="Due Source",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-future",
        type="jira",
        name="Future Source",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        "src-due",
        enabled=True,
        interval_minutes=30,
        next_run_at=due_at,
    )
    await db.set_source_sync_schedule(
        "src-future",
        enabled=True,
        interval_minutes=30,
        next_run_at=future_at,
    )

    stored = await db.get_source("src-due")
    assert stored is not None
    assert stored["sync_schedule"]["enabled"] is True
    assert stored["sync_schedule"]["interval_minutes"] == 30
    assert stored["sync_schedule"]["next_run_at"] == due_at.isoformat()

    await db.set_source_sync_schedule(
        "src-due",
        enabled=True,
        interval_minutes=30,
    )
    still_due_before_claim = await db.get_source("src-due")
    assert still_due_before_claim is not None
    assert still_due_before_claim["sync_schedule"]["next_run_at"] == due_at.isoformat()

    claimed_sources = await db.claim_due_scheduled_sources(now=claim_time)
    assert [source["id"] for source in claimed_sources] == ["src-due"]
    assert claimed_sources[0]["sync_schedule"]["next_run_at"] == "2026-06-16T00:30:00+00:00"

    claimed_again = await db.claim_due_scheduled_sources(now=claim_time)
    assert claimed_again == []

    await db.set_source_sync_schedule(
        "src-due",
        enabled=True,
        interval_minutes=30,
    )
    unchanged = await db.get_source("src-due")
    assert unchanged is not None
    assert unchanged["sync_schedule"]["next_run_at"] == "2026-06-16T00:30:00+00:00"

    await db.set_source_sync_schedule(
        "src-due",
        enabled=True,
        interval_minutes=30,
        next_run_at=due_at,
    )
    excluded = await db.claim_due_scheduled_sources(
        now=claim_time,
        exclude_source_ids={"src-due"},
    )
    assert excluded == []
    still_due = await db.get_source("src-due")
    assert still_due is not None
    assert still_due["sync_schedule"]["next_run_at"] == due_at.isoformat()


@pytest.mark.asyncio
async def test_enqueue_due_source_sync_runs_advances_schedule_with_run_acceptance(db: Database):
    claim_time = datetime(2026, 6, 16, tzinfo=timezone.utc)
    source_id = "src-scheduled-due"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Due Source",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=30,
        next_run_at=claim_time - timedelta(minutes=1),
    )

    runs = await db.enqueue_due_source_sync_runs(now=claim_time)

    assert [run.source_id for run in runs] == [source_id]
    assert runs[0].trigger == "schedule"
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == "2026-06-16T00:30:00+00:00"
    coalesced = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    assert coalesced.run_id == runs[0].run_id
    assert coalesced.coalesced is True


@pytest.mark.asyncio
async def test_advance_source_sync_schedule_uses_expected_due_timestamp(db: Database):
    due_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    source_id = "src-local-schedule"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Scheduled Teams",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=60,
        next_run_at=due_at,
    )

    stale = await db.advance_source_sync_schedule(
        source_id,
        expected_next_run_at=(due_at - timedelta(minutes=1)).isoformat(),
        now=due_at + timedelta(minutes=1),
    )
    advanced = await db.advance_source_sync_schedule(
        source_id,
        expected_next_run_at=due_at.isoformat(),
        now=due_at + timedelta(minutes=1),
    )

    assert stale is False
    assert advanced is True
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == "2026-07-10T01:01:00+00:00"


@pytest.mark.asyncio
async def test_due_local_source_enqueues_owner_daemon_job_not_server_run(db: Database):
    due_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    source_id = "src-local-owner-schedule"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Scheduled Teams",
        config_json=json.dumps(
            {
                "region": "emea",
                "conversation_ids": ["19:conversation-a@example.test"],
                "access_token": "must-not-reach-daemon",
            }
        ),
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=60,
        next_run_at=due_at,
    )

    local_count = await db.enqueue_due_local_agent_jobs(now=due_at)
    server_runs = await db.enqueue_due_source_sync_runs(now=due_at)
    jobs = await db.lease_local_agent_jobs(
        user_id="owner-a",
        limit=5,
        lease_seconds=60,
        now=due_at,
    )

    assert local_count == 1
    assert server_runs == []
    assert len(jobs) == 1
    assert jobs[0]["execution_owner_user_id"] == "owner-a"
    assert jobs[0]["payload"]["region"] == "emea"
    assert jobs[0]["payload"]["conversation_ids"] == ["19:conversation-a@example.test"]
    assert "access_token" not in jobs[0]["payload"]
    assert "config" not in jobs[0]["payload"]
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == "2026-07-10T01:00:00+00:00"


@pytest.mark.asyncio
async def test_due_ownerless_local_source_advances_schedule_without_job(db: Database):
    due_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    source_id = "src-ownerless-schedule"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Ownerless Scheduled Teams",
        config_json='{"conversation_ids":["19:conversation@example.test"]}',
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=60,
        next_run_at=due_at,
    )

    local_count = await db.enqueue_due_local_agent_jobs(now=due_at)

    assert local_count == 0
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == "2026-07-10T01:00:00+00:00"


@pytest.mark.asyncio
async def test_retryable_local_agent_completion_requeues_same_job(db: Database):
    now = datetime(2026, 7, 10, tzinfo=timezone.utc)
    await db.insert_local_agent_job(
        job_id="laj-retry",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_id": "src-local"},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
        now=now,
    )
    first = await db.lease_local_agent_jobs(
        user_id="owner-a",
        limit=1,
        lease_seconds=60,
        now=now,
    )

    completed = await db.complete_local_agent_job(
        job_id="laj-retry",
        user_id="owner-a",
        attempt_count=first[0]["attempt_count"],
        status="failed",
        result={
            "retryable": True,
            "progress": {
                "schema_version": 1,
                "phase": "uploading",
                "progress": {"completed": 9, "total": 10, "unit": "file"},
            },
        },
        error="one package failed",
        retryable=True,
        now=now + timedelta(seconds=1),
    )
    second = await db.lease_local_agent_jobs(
        user_id="owner-a",
        limit=1,
        lease_seconds=60,
        now=now + timedelta(seconds=2),
    )

    assert completed is True
    assert second[0]["job_id"] == "laj-retry"
    assert second[0]["attempt_count"] == 2
    assert second[0]["result"] == {}


@pytest.mark.asyncio
async def test_queued_force_request_promotes_existing_local_agent_job(db: Database):
    first_id, first_created = await db.enqueue_local_agent_job(
        job_id="laj-normal",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": False},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    promoted_id, promoted_created = await db.enqueue_local_agent_job(
        job_id="laj-force",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": True},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    jobs = await db.lease_local_agent_jobs(
        user_id="owner-a", limit=5, lease_seconds=60
    )

    assert first_created is True
    assert promoted_created is False
    assert promoted_id == first_id
    assert len(jobs) == 1
    assert jobs[0]["payload"]["force_full_sync"] is True


@pytest.mark.asyncio
async def test_leased_force_request_creates_serial_successor_job(db: Database):
    first_id, _ = await db.enqueue_local_agent_job(
        job_id="laj-leased-normal",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": False},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    leased = await db.lease_local_agent_jobs(
        user_id="owner-a", limit=1, lease_seconds=60
    )
    promoted_id, created = await db.enqueue_local_agent_job(
        job_id="laj-leased-force",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": True},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    blocked = await db.lease_local_agent_jobs(
        user_id="owner-a", limit=5, lease_seconds=60
    )
    await db.complete_local_agent_job(
        job_id=first_id,
        user_id="owner-a",
        attempt_count=leased[0]["attempt_count"],
        status="succeeded",
        result={},
        error=None,
        retryable=False,
    )
    successor = await db.lease_local_agent_jobs(
        user_id="owner-a", limit=5, lease_seconds=60
    )

    assert leased[0]["job_id"] == first_id
    assert promoted_id != first_id
    assert created is True
    assert blocked == []
    assert successor[0]["job_id"] == promoted_id
    assert successor[0]["payload"]["force_full_sync"] is True


@pytest.mark.asyncio
async def test_expired_original_and_successor_are_not_released_in_same_batch(db: Database):
    now = datetime.now(timezone.utc)
    first_id, _ = await db.enqueue_local_agent_job(
        job_id="laj-expired-original",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": False},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    await db.lease_local_agent_jobs(
        user_id="owner-a", limit=1, lease_seconds=60, now=now
    )
    successor_id, _ = await db.enqueue_local_agent_job(
        job_id="laj-expired-successor",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": True},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )

    recovered = await db.lease_local_agent_jobs(
        user_id="owner-a",
        limit=5,
        lease_seconds=60,
        now=now + timedelta(seconds=61),
    )

    assert [job["job_id"] for job in recovered] == [first_id]
    assert successor_id != first_id


@pytest.mark.asyncio
async def test_enqueue_due_source_sync_runs_rolls_back_schedule_when_run_enqueue_fails(
    db: Database,
    monkeypatch,
):
    claim_time = datetime(2026, 6, 16, tzinfo=timezone.utc)
    due_at = claim_time - timedelta(minutes=1)
    source_id = "src-scheduled-rollback"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Rollback Source",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=30,
        next_run_at=due_at,
    )

    async def fail_enqueue_locked(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("enqueue failed after schedule claim")

    monkeypatch.setattr(db, "_enqueue_source_sync_run_locked", fail_enqueue_locked)

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await db.enqueue_due_source_sync_runs(now=claim_time)

    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == due_at.isoformat()


@pytest.mark.asyncio
async def test_scheduler_enqueues_due_source_and_advances_next_run(db: Database):
    source_id = "src-scheduled-due"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Due Source",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=30,
        next_run_at=datetime.now(timezone.utc) - timedelta(minutes=1),
    )
    service = SyncService(db, AppConfig())
    scheduler = SyncScheduler(db, service)

    await scheduler._sync_due_sources()

    run = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    assert run.source_id == source_id
    assert run.coalesced is True
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] is not None
    assert datetime.fromisoformat(source["sync_schedule"]["next_run_at"]) > datetime.now(timezone.utc)


@pytest.mark.asyncio
async def test_scheduler_coalesces_active_due_source_and_advances_next_run(db: Database):
    source_id = "src-scheduled-running"
    due_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Running Source",
        config_json="{}",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=30,
        next_run_at=due_at,
    )
    service = SyncService(db, AppConfig())
    scheduler = SyncScheduler(db, service)
    active = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    await scheduler._sync_due_sources()

    scheduled = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    assert scheduled.run_id == active.run_id
    assert scheduled.coalesced is True
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] != due_at.isoformat()


@pytest.mark.asyncio
async def test_scheduler_still_scans_server_sources_when_local_job_scan_fails(db: Database, monkeypatch):
    server_scan_calls: list[int] = []

    async def fail_local_scan(*, limit):
        raise RuntimeError("local broker unavailable")

    async def record_server_scan(*, limit):
        server_scan_calls.append(limit)
        return []

    monkeypatch.setattr(db, "enqueue_due_local_agent_jobs", fail_local_scan)
    monkeypatch.setattr(db, "enqueue_due_source_sync_runs", record_server_scan)
    scheduler = SyncScheduler(db, SyncService(db, AppConfig()))

    await scheduler._sync_due_sources()

    assert server_scan_calls == [50]


@pytest.mark.asyncio
async def test_sync_service_enqueue_source_creates_durable_run_without_local_task(db: Database):
    source_id = "src-enqueue-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Enqueue Service",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())

    run = await service.enqueue_source(
        source_id,
        trigger="manual",
        force_full_sync=True,
    )

    assert run.source_id == source_id
    assert run.force_full_sync is True
    assert run.status == "pending"
    assert service.tasks == {}


@pytest.mark.asyncio
async def test_sync_service_rejects_source_while_deletion_is_in_progress(db: Database):
    source_id = "src-deleting-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Deleting Service",
        config_json="{}",
    )
    await db.db.execute("UPDATE sources SET status = 'deleting' WHERE id = ?", (source_id,))
    await db.db.commit()

    with pytest.raises(RuntimeError, match="not active.*deleting"):
        await SyncService(db, AppConfig()).enqueue_source(source_id)


@pytest.mark.asyncio
async def test_database_rejects_direct_enqueue_while_source_is_deleting(db: Database):
    source_id = "src-deleting-direct-enqueue"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Deleting Direct Enqueue",
        config_json="{}",
    )
    await db.db.execute("UPDATE sources SET status = 'deleting' WHERE id = ?", (source_id,))
    await db.db.commit()

    with pytest.raises(ValueError, match="not active.*deleting"):
        await db.enqueue_source_sync_run(source_id=source_id)


@pytest.mark.asyncio
async def test_source_sync_worker_executes_leased_run_and_completes_it(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-run"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Run",
        config_json="{}",
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.force_full_sync_values: list[bool] = []
            self.extraction_pools: list[ExtractionWorkPool | None] = []

        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, document_lifecycle_admission
            self.extraction_pools.append(extraction_pool)
            return object()

        async def run_source_sync(self, **kwargs):
            self.force_full_sync_values.append(bool(kwargs["force_full_sync"]))
            return SyncState(
                source=source_id,
                last_sync_at=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
                last_sync_status="success",
                docs_processed=1,
                docs_updated=1,
            )

    provider = CapturingRuntimeProvider()
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_extraction_workers=3)),
        runtime_provider=provider,
    )
    enqueued = await service.enqueue_source(source_id, trigger="manual", force_full_sync=True)
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(sync=SyncConfig(max_extraction_workers=3)),
        runtime_provider=provider,
        worker_id="worker-a",
    )

    leased = await worker.run_once()
    completed = await db.get_source_sync_run(enqueued.run_id)

    assert leased is not None
    assert leased.run_id == enqueued.run_id
    assert completed is not None
    assert completed.status == "success"
    assert completed.lease_owner is None
    assert provider.force_full_sync_values == [True]
    assert provider.extraction_pools == [worker._extraction_pool]


@pytest.mark.asyncio
async def test_source_sync_worker_does_not_reprocess_unchanged_complete_input_snapshot(db: Database):
    import memforge.runtime as runtime

    source_id = "src-snapshot-worker"
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Snapshot Worker",
        config_json='{"documents_dir":"/server/inbox"}',
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.force_full_sync: bool | None = None
            self.authoritative_snapshot: bool | None = None
            self.source: dict | None = None

        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            self.force_full_sync = kwargs["force_full_sync"]
            self.authoritative_snapshot = kwargs["authoritative_snapshot"]
            self.source = kwargs["source"]
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            )

    provider = CapturingRuntimeProvider()
    await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        force_full_sync=False,
        input_snapshot_id="laj-empty",
    )
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=provider,
        worker_id="worker-snapshot",
    )

    await worker.run_once()

    assert provider.force_full_sync is False
    assert provider.authoritative_snapshot is True
    assert provider.source is not None
    assert provider.source["config"]["local_agent_package_manifest"] == []


@pytest.mark.asyncio
async def test_source_sync_worker_marks_missing_source_terminal(db: Database, monkeypatch):
    import memforge.runtime as runtime

    await db.upsert_source(
        id="src-deleted-before-run",
        type="jira",
        name="Deleted Before Run",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id="src-deleted-before-run",
        trigger="manual",
    )
    async def missing_source(source_id: str):
        del source_id
        return None

    monkeypatch.setattr(db, "get_source", missing_source)
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=object(),
        worker_id="worker-a",
    )

    leased = await worker.run_once()
    completed = await db.get_source_sync_run(enqueued.run_id)

    assert leased is not None
    assert completed is not None
    assert completed.status == "failed"
    assert completed.error_message == "Source not found: src-deleted-before-run"


@pytest.mark.asyncio
async def test_source_sync_worker_delays_retryable_failure(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-fail-backoff"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Fail Backoff",
        config_json="{}",
    )

    class FailingRuntimeProvider:
        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool, document_lifecycle_admission
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            raise RuntimeError("temporary failure")

    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(sync=SyncConfig(worker_retry_base_seconds=120, worker_retry_max_seconds=120)),
        runtime_provider=FailingRuntimeProvider(),
        worker_id="worker-a",
    )

    await worker.run_once()
    failed = await db.get_source_sync_run(enqueued.run_id)
    too_early = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        now=datetime.now(timezone.utc) + timedelta(seconds=30),
    )

    assert failed is not None
    assert failed.status == "pending"
    assert failed.next_attempt_at is not None
    assert failed.next_attempt_at > datetime.now(timezone.utc) + timedelta(seconds=100)
    assert too_early is None


@pytest.mark.asyncio
async def test_source_sync_worker_retries_failed_final_state(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-final-state-failed"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Final State Failed",
        config_json="{}",
    )

    class FailedStateRuntimeProvider:
        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool, document_lifecycle_admission
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return SyncState(
                source=source_id,
                last_sync_at=None,
                last_sync_status="failed",
                error_message="temporary final-state failure",
            )

    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(sync=SyncConfig(worker_retry_base_seconds=120, worker_retry_max_seconds=120)),
        runtime_provider=FailedStateRuntimeProvider(),
        worker_id="worker-a",
    )

    await worker.run_once()
    failed = await db.get_source_sync_run(enqueued.run_id)
    sync_state = await db.get_sync_state(source_id)

    assert failed is not None
    assert failed.status == "pending"
    assert failed.next_attempt_at is not None
    assert failed.error_message == "temporary final-state failure"
    assert sync_state is not None
    assert sync_state.last_sync_status == "failed"
    assert sync_state.error_message == "temporary final-state failure"


@pytest.mark.asyncio
async def test_source_sync_worker_retries_partial_final_state(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-final-state-partial"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Partial State",
        config_json="{}",
    )

    class PartialStateRuntimeProvider:
        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="partial",
                docs_processed=2,
                docs_failed=1,
                error_message="one document failed",
            )

    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(sync=SyncConfig(worker_retry_base_seconds=120)),
        runtime_provider=PartialStateRuntimeProvider(),
        worker_id="worker-a",
    )

    await worker.run_once()
    failed = await db.get_source_sync_run(enqueued.run_id)
    sync_state = await db.get_sync_state(source_id)

    assert failed is not None
    assert failed.status == "pending"
    assert sync_state is not None
    assert sync_state.last_sync_status == "partial"


@pytest.mark.asyncio
async def test_source_sync_worker_heartbeats_while_run_is_active(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-heartbeat"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Heartbeat",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    class WaitingRuntimeProvider:
        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool, document_lifecycle_admission
            return object()

        async def run_source_sync(self, **kwargs):
            db = kwargs["db"]
            initial = await db.get_source_sync_run(enqueued.run_id)
            assert initial is not None
            assert initial.lease_expires_at is not None
            for _ in range(100):
                await asyncio.sleep(0.02)
                current = await db.get_source_sync_run(enqueued.run_id)
                assert current is not None
                if current.lease_expires_at and current.lease_expires_at > initial.lease_expires_at:
                    return SyncState(
                        source=source_id,
                        last_sync_at=datetime.now(timezone.utc),
                        last_sync_status="success",
                    )
            raise AssertionError("worker did not heartbeat the active source sync run")

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=WaitingRuntimeProvider(),
        worker_id="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.01,
    )

    await worker.run_once()
    completed = await db.get_source_sync_run(enqueued.run_id)

    assert completed is not None
    assert completed.status == "success"


@pytest.mark.asyncio
async def test_source_sync_worker_persists_pipeline_progress_while_run_is_active(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-progress"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Engineering Wiki",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    class ProgressRuntimeProvider:
        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            kwargs["progress_callback"](
                {
                    "phase": "processing",
                    "current": 31,
                    "total": 86,
                    "docs_updated": 12,
                    "memories_extracted": 104,
                }
            )
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            )

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=ProgressRuntimeProvider(),
        worker_id="worker-a",
        progress_flush_seconds=0.01,
    )

    await worker.run_once()
    completed = await db.get_source_sync_run(enqueued.run_id)

    assert completed is not None
    assert completed.status == "success"
    assert completed.progress_revision > 0
    assert completed.progress == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 31, "total": 86, "unit": "page"},
        "counts": {"changed": 12, "memories_created": 104},
    }


def test_reconciliation_without_measurable_work_is_indeterminate():
    from memforge.sync_progress import source_sync_progress_from_pipeline

    assert source_sync_progress_from_pipeline(
        {"phase": "detecting_deletions", "current": 0, "total": 0},
        source_type="confluence",
    ) == {"schema_version": 1, "phase": "reconciling"}


@pytest.mark.asyncio
async def test_source_sync_worker_does_not_complete_after_losing_lease(db: Database, monkeypatch):
    import memforge.runtime as runtime

    source_id = "src-worker-lost-lease"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Lost Lease",
        config_json="{}",
    )
    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    class SlowSuccessRuntimeProvider:
        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool, document_lifecycle_admission
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            await asyncio.sleep(0.03)
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            )

    async def lost_lease(*args, **kwargs):
        del args, kwargs
        return False

    monkeypatch.setattr(db, "heartbeat_source_sync_run", lost_lease)
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=SlowSuccessRuntimeProvider(),
        worker_id="worker-a",
        lease_seconds=1,
        heartbeat_seconds=0.01,
    )

    await worker.run_once()
    run = await db.get_source_sync_run(enqueued.run_id)

    assert run is not None
    assert run.status == "running"
    assert run.lease_owner == "worker-a"


@pytest.mark.asyncio
async def test_source_sync_worker_run_forever_polls_until_cancelled(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-loop"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Loop",
        config_json="{}",
    )

    class InstantRuntimeProvider:
        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool, document_lifecycle_admission
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return SyncState(
                source=source_id,
                last_sync_at=datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc),
                last_sync_status="success",
            )

    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=InstantRuntimeProvider(),
        worker_id="worker-loop",
    )

    task = asyncio.create_task(worker.run_forever(poll_seconds=0.01))
    for _ in range(50):
        completed = await db.get_source_sync_run(enqueued.run_id)
        if completed and completed.status == "success":
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    completed = await db.get_source_sync_run(enqueued.run_id)
    assert completed is not None
    assert completed.status == "success"


@pytest.mark.asyncio
async def test_sync_service_passes_force_full_sync_to_source_task(db: Database, monkeypatch):
    source_id = "src-force-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Force Service",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())
    captured: dict[str, object] = {}

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        captured["source_id"] = running_source_id
        captured["force_full_sync"] = force_full_sync

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task = await service.start_source(source_id, force_full_sync=True)
    await task

    assert captured == {"source_id": source_id, "force_full_sync": True}


@pytest.mark.asyncio
async def test_sync_service_limits_active_sources_without_rejecting_queued_sources(
    db: Database,
    monkeypatch,
):
    await db.upsert_source(
        id="src-a",
        type="jira",
        name="Source A",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-b",
        type="jira",
        name="Source B",
        config_json="{}",
    )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=1)),
    )
    release = asyncio.Event()
    started: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        started.append(running_source_id)
        try:
            await release.wait()
        finally:
            service.tasks.pop(running_source_id, None)
            service.progress.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task_a = await service.start_source("src-a")
    task_b = await service.start_source("src-b")
    await asyncio.sleep(0)

    assert service.is_running("src-a")
    assert service.is_running("src-b")
    assert started == ["src-a"]
    assert service.progress["src-b"]["phase"] == "queued"

    release.set()
    await asyncio.gather(task_a, task_b)
    assert started == ["src-a", "src-b"]


@pytest.mark.asyncio
async def test_sync_service_queues_ten_requested_sources_with_two_active(
    db: Database,
    monkeypatch,
):
    source_ids = [f"src-{idx}" for idx in range(10)]
    for source_id in source_ids:
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
        )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=2)),
    )
    release = asyncio.Event()
    started: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        started.append(running_source_id)
        try:
            await release.wait()
        finally:
            service.tasks.pop(running_source_id, None)
            service.progress.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    tasks = [await service.start_source(source_id) for source_id in source_ids]
    await asyncio.sleep(0)

    assert started == source_ids[:2]
    assert all(service.progress[source_id]["phase"] == "queued" for source_id in source_ids[2:])

    release.set()
    await asyncio.gather(*tasks)
    assert started == source_ids


@pytest.mark.asyncio
async def test_cancel_queued_source_clears_progress(db: Database, monkeypatch):
    await db.upsert_source(
        id="src-active",
        type="jira",
        name="Active Source",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-queued",
        type="jira",
        name="Queued Source",
        config_json="{}",
    )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=1)),
    )
    release = asyncio.Event()

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        if running_source_id == "src-active":
            await release.wait()

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task_active = await service.start_source("src-active")
    await service.start_source("src-queued")
    await asyncio.sleep(0)

    assert service.progress["src-queued"]["phase"] == "queued"

    await service.cancel_source("src-queued")

    assert "src-queued" not in service.tasks
    assert "src-queued" not in service.progress

    release.set()
    await task_active


def test_sync_max_active_sources_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_MAX_ACTIVE_SOURCES", "2")

    assert AppConfig().sync.max_active_sources == 2


def test_sync_max_extraction_workers_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_MAX_EXTRACTION_WORKERS", "6")

    assert AppConfig().sync.max_extraction_workers == 6


def test_sync_max_document_lifecycles_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_MAX_DOCUMENT_LIFECYCLES", "1")

    assert AppConfig().sync.max_document_lifecycles == 1


def test_sync_worker_config_can_be_disabled_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("MEMFORGE_SYNC_WORKER_ENABLED", "false")
    monkeypatch.setenv("MEMFORGE_SYNC_WORKER_POLL_SECONDS", "0.25")
    monkeypatch.setenv("MEMFORGE_SYNC_WORKER_RETRY_BASE_SECONDS", "2.5")
    monkeypatch.setenv("MEMFORGE_SYNC_WORKER_RETRY_MAX_SECONDS", "5")
    monkeypatch.setenv("MEMFORGE_SYNC_WORKER_MAX_ATTEMPTS", "2")

    config = AppConfig()

    assert config.sync.scheduler_enabled is False
    assert config.sync.worker_enabled is False
    assert config.sync.worker_poll_seconds == 0.25
    assert config.sync.worker_retry_base_seconds == 2.5
    assert config.sync.worker_retry_max_seconds == 5
    assert config.sync.worker_max_attempts == 2


@pytest.mark.asyncio
async def test_sync_service_passes_shared_extraction_pool_to_runtime_provider(db: Database):
    source_id = "src-shared-pool"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Shared Pool Source",
        config_json="{}",
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.extraction_pools: list[ExtractionWorkPool | None] = []

        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, document_lifecycle_admission
            self.extraction_pools.append(extraction_pool)
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return None

    provider = CapturingRuntimeProvider()
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_extraction_workers=6)),
        runtime_provider=provider,
    )

    task = await service.start_source(source_id)
    await task

    assert provider.extraction_pools == [service._extraction_pool]


@pytest.mark.asyncio
async def test_sync_service_passes_shared_document_lifecycle_admission_to_runtime_provider(db: Database):
    source_id = "src-shared-doc-admission"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Shared Document Admission Source",
        config_json="{}",
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.document_lifecycle_admissions: list[DocumentLifecycleAdmission | None] = []

        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
            document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        ):
            del db, config, extraction_pool
            self.document_lifecycle_admissions.append(document_lifecycle_admission)
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return None

    provider = CapturingRuntimeProvider()
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_document_lifecycles=1)),
        runtime_provider=provider,
    )

    task = await service.start_source(source_id)
    await task

    assert provider.document_lifecycle_admissions == [service._document_lifecycle_admission]


@pytest.mark.asyncio
async def test_sync_services_share_process_document_lifecycle_admission(db: Database):
    config = AppConfig(sync=SyncConfig(max_document_lifecycles=1))

    service_a = SyncService(db, config)
    service_b = SyncService(db, config)

    assert service_a._document_lifecycle_admission is not None
    assert service_a._document_lifecycle_admission is service_b._document_lifecycle_admission


@pytest.mark.asyncio
async def test_requested_sync_runs_after_active_source_sync_finishes(db: Database, monkeypatch):
    source_id = "src-queued-after-active"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Queued Source",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())
    first_release = asyncio.Event()
    calls: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        calls.append(running_source_id)
        try:
            await first_release.wait()
        finally:
            service.tasks.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    first_task = await service.start_source(source_id)
    await asyncio.sleep(0)

    assert await service.request_source_sync(source_id, delay_seconds=0) is True
    await asyncio.sleep(0)
    assert calls == [source_id]
    pending = await db.enqueue_source_sync_run(source_id=source_id, trigger="request")
    assert pending.coalesced is True
    assert pending.status == "pending"

    first_release.set()
    await first_task
    await service.shutdown()

    assert calls == [source_id]


@pytest.mark.asyncio
async def test_upsert_sync_state_updates_source_last_sync(db: Database):
    source_id = "src-state-bookkeeping"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Team Chat",
        config_json="{}",
    )
    sync_at = datetime.now(timezone.utc)

    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=sync_at,
            last_sync_status="success",
            docs_processed=0,
            docs_updated=0,
        ),
    )

    source = await db.get_source(source_id)
    assert source["last_sync"] == sync_at.isoformat()


@pytest.mark.asyncio
async def test_document_is_indexed_before_enrichment(db: Database):
    source_id = "src-early-document"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    gene = BlockingFetchGene(item_count=1, release=release)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert await db.count_documents(source=source_id) == 1


@pytest.mark.asyncio
async def test_full_document_extraction_failure_is_audited(db: Database):
    source_id = "src-full-extraction-failure"
    await db.upsert_source(
        id=source_id,
        type="docs",
        name="Docs",
        config_json="{}",
    )
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=FailingMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene("# Design Doc\n\nDurable content."),
        source_name="Docs",
        source_id=source_id,
    )

    rows = await db.list_memory_audit_events(event_type="memory_extraction_failed")
    assert state.last_sync_status == "failed"
    assert state.docs_updated == 0
    assert state.docs_failed == 1
    assert state.failed_docs
    assert "json_parse_error" in state.failed_docs[0].error
    assert await db.count_documents(source=source_id) == 0
    assert len(rows) == 3
    assert rows[0].doc_id == "doc-1"
    assert rows[0].source_id == source_id
    assert rows[0].reason == "json_parse_error"
    assert rows[0].error == "Unterminated string starting at line 393 column 16"
    assert rows[0].payload["extracted_count"] == 0


@pytest.mark.asyncio
async def test_document_update_uses_diff_guided_extraction_and_audits_strategy(
    db: Database,
    tmp_path,
):
    source_id = "src-diff-guided-update"
    old_markdown = "# Design Doc\n\nThe service uses PostgreSQL 14."
    new_markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id="src-documents",
        title="Design Doc",
        markdown=old_markdown,
    )
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=old_markdown,
        version="1",
        normalized_content_uri=normalized_content_uri,
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    extraction_rows = await db.list_memory_audit_events(
        event_type="memory_change_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert len(extractor.change_calls) == 1
    assert extractor.full_calls == []
    assert "PostgreSQL 14" in extractor.change_calls[0]["changed_hunks"]
    assert "PostgreSQL 15" in extractor.change_calls[0]["changed_hunks"]
    assert extractor.change_calls[0]["updated_document"] == new_markdown
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "diff_guided"
    assert "PostgreSQL 15" in memory_engine.reconcile_calls[0]["changed_hunks"]
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["data_shape"] == "document"
    assert len(audit_rows) == 1
    assert audit_rows[0].doc_id == "doc-1"
    assert audit_rows[0].source_id == source_id
    assert audit_rows[0].decision == "diff_guided"
    assert audit_rows[0].reason == "small_diff"
    assert audit_rows[0].payload["data_shape"] == "document"
    assert audit_rows[0].payload["previous_version"] == "1"
    assert audit_rows[0].payload["current_version"] == "2"
    assert audit_rows[0].payload["diff_line_count"] > 0
    assert audit_rows[0].thresholds["max_diff_lines"] > 0
    assert len(extraction_rows) == 1
    assert extraction_rows[0].doc_id == "doc-1"
    assert extraction_rows[0].decision == "diff_guided"
    assert extraction_rows[0].payload["extracted_count"] == 0
    assert extraction_rows[0].payload["diff_line_count"] > 0


@pytest.mark.asyncio
async def test_structured_source_update_uses_diff_guided_extraction_and_audits_strategy(
    db: Database,
    tmp_path,
):
    source_id = "src-jira-diff-guided-update"
    old_markdown = "# [Story] PAY-123: Cutoff flow\n\n## Source Metadata\n- Status: In Progress"
    new_markdown = "# [Story] PAY-123: Cutoff flow\n\n## Source Metadata\n- Status: Done"
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id=source_id,
        title="PAY-123",
        markdown=old_markdown,
    )
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="PAY-123",
        markdown=old_markdown,
        version="1",
        normalized_content_uri=normalized_content_uri,
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingTicketGene(new_markdown),
        source_name="Jira Board",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    extraction_rows = await db.list_memory_audit_events(
        event_type="memory_change_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert len(extractor.change_calls) == 1
    assert extractor.full_calls == []
    assert "Status: In Progress" in extractor.change_calls[0]["changed_hunks"]
    assert "Status: Done" in extractor.change_calls[0]["changed_hunks"]
    assert extractor.change_calls[0]["source_type"] == "jira"
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "diff_guided"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["data_shape"] == "ticket"
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "diff_guided"
    assert audit_rows[0].reason == "small_diff"
    assert audit_rows[0].payload["data_shape"] == "ticket"
    assert len(extraction_rows) == 1
    assert extraction_rows[0].decision == "diff_guided"


@pytest.mark.asyncio
async def test_document_update_falls_back_to_full_extraction_when_previous_content_missing(
    db: Database,
):
    source_id = "src-full-update-fallback"
    old_markdown = "# Design Doc\n\nThe service uses PostgreSQL 14."
    new_markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=old_markdown,
        version="1",
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert extractor.change_calls == []
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) == 1
    assert extractor.unit_calls[0]["context"].unit.unit_markdown == new_markdown
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "full_document"
    assert audit_rows[0].reason == "previous_content_missing"
    assert audit_rows[0].payload["fallback_from"] == "diff_guided"


@pytest.mark.asyncio
async def test_large_full_document_uses_deterministic_units(db: Database):
    source_id = "src-large-doc-full"
    await db.upsert_source(
        id=source_id,
        type="github_pages",
        name="Documents",
        config_json="{}",
    )
    markdown = "# Design Doc\n\nIntro.\n\n" + "\n\n".join(
        f"## Section {index}\n\n" + ("Durable design detail. " * 900) for index in range(8)
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="1"),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="memory_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) > 1
    assert all(call["context"].unit.unit_id for call in extractor.unit_calls)
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "full_document"
    assert audit_rows[0].payload["unitized"] is True
    assert audit_rows[0].payload["unit_count"] == len(extractor.unit_calls)
    assert audit_rows[0].payload["segmentation_version"] == "v2"


@pytest.mark.asyncio
async def test_full_document_unit_extraction_honors_orchestrator_concurrency(db: Database):
    markdown = "# Design Doc\n\nIntro.\n\n" + "\n\n".join(
        f"## Section {index}\n\n" + ("Durable design detail. " * 900) for index in range(8)
    )
    extractor = BlockingUnitMemoryExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, "src-large-doc-full"),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    task = asyncio.create_task(
        orchestrator._extract_full_document_units(
            markdown_body=markdown,
            source_type="github_pages",
            doc_type="reference",
            entity_names=[],
            existing_memories=[],
            doc_id="doc-large",
            source_id="src-large-doc-full",
            document_title="Design Doc",
            document_url="https://example.test/design",
        )
    )

    try:
        await asyncio.sleep(0.2)
        assert not extractor.started_two.is_set()
        assert extractor.max_active == 1
    finally:
        extractor.release.set()
        await task


@pytest.mark.asyncio
async def test_partial_unit_extraction_failure_skips_reconciliation(db: Database, tmp_path):
    source_id = "src-partial-unit-failure"
    markdown = "\n\n".join(
        [
            "# Design Doc",
            "Intro.",
            "## Section 1",
            " ".join(["section one durable design guidance"] * 2500),
            "## Section 2",
            " ".join(["section two durable design guidance"] * 2500),
        ]
    )
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id=source_id,
        title="Design Doc",
        markdown=markdown,
    )
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=markdown,
        version="1",
        normalized_content_uri=normalized_content_uri,
    )
    extractor = PartiallyFailingUnitMemoryExtractor()
    memory_engine = RecordingMemoryEngine()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="1"),
        source_name="Documents",
        source_id=source_id,
        force_full_sync=True,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="memory_extraction_failed",
    )
    assert state.last_sync_status == "failed"
    assert state.docs_updated == 0
    assert state.docs_failed == 1
    assert state.failed_docs
    assert "partial_unit_failure" in state.failed_docs[0].error
    assert len(memory_engine.reconcile_calls) == 0
    assert len(audit_rows) == 3
    assert audit_rows[0].reason == "partial_unit_failure"
    assert audit_rows[0].payload["failed_unit_count"] == 1
    assert audit_rows[0].payload["extracted_count"] == 0


@pytest.mark.asyncio
async def test_item_processing_is_bounded_by_max_concurrent(db: Database):
    source_id = "src-bounded-sync"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    gene = BlockingFetchGene(item_count=5, release=release)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=2,
    )

    sync_task = asyncio.create_task(
        orchestrator.sync_gene(
            gene=gene,
            source_name="Jira Board",
            source_id=source_id,
        )
    )
    await asyncio.sleep(0.05)
    release.set()
    await sync_task

    assert gene.max_active_fetches <= 2


@pytest.mark.asyncio
async def test_running_progress_reports_extracted_memories(db: Database):
    source_id = "src-running-memory-progress"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    progress_events: list[dict] = []
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=CountingMemoryEngine(inserted=3),
        memory_store=None,
        max_concurrent=1,
    )

    await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
        progress_callback=progress_events.append,
    )

    assert any(event.get("memories_extracted") == 3 for event in progress_events)
    assert [
        event["current"]
        for event in progress_events
        if event.get("phase") == "discovering"
    ] == [0, 1]
    reconciliation = [
        event for event in progress_events if event.get("phase") == "detecting_deletions"
    ]
    assert reconciliation == [{
        "phase": "detecting_deletions",
        "current": 0,
        "total": 0,
        "title": None,
    }]


@pytest.mark.asyncio
async def test_document_vector_failure_happens_before_memory_mutations(db: Database, monkeypatch):
    source_id = "src-vector-before-memory"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    memory_engine = CountingMemoryEngine(inserted=3)
    vector_store = FailingVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=memory_engine,
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert memory_engine.enrichment_calls == 0
    assert memory_engine.process_calls == 0
    assert await db.get_document("jira-0") is None
    assert "jira-0" not in vector_store.upserted


@pytest.mark.asyncio
async def test_falsey_document_collection_still_receives_vector_upsert(db: Database, monkeypatch):
    source_id = "src-falsey-vector"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"]
    assert vector_store.upserted["jira-0"]["metadata"]["version"] == "0"


@pytest.mark.asyncio
async def test_document_vector_text_is_independent_of_extracted_entity_names(db: Database, monkeypatch):
    source_id = "src-vector-text"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=EntityMentioningEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    document_text = vector_store.upserted["jira-0"]["document"]
    assert state.last_sync_status == "success"
    assert "Raw Extracted Entity" not in document_text
    assert "Raw Alias" not in document_text
    assert document_text == "Summary\ntag-one\njira_issue\nlow"


@pytest.mark.asyncio
async def test_unchanged_document_repairs_stale_vector_without_llm_reprocessing(
    db: Database,
    monkeypatch,
):
    source_id = "src-stale-vector-repair"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert history[0]["docs_updated"] == 0
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"] == content_hash(markdown)
    assert vector_store.upserted["jira-0"]["metadata"]["version"] == "0"


@pytest.mark.asyncio
async def test_unchanged_document_backfills_pdf_uri_without_llm_reprocessing(db: Database):
    source_id = "src-unchanged-pdf-backfill"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == f"file:///tmp/{source_id}/Jira 0.pdf"


@pytest.mark.asyncio
async def test_missing_pdf_uri_forces_full_sync_without_llm_reprocessing(db: Database):
    source_id = "src-missing-pdf-full-sync"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
        )
    )
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == f"file:///tmp/{source_id}/Jira 0.pdf"


@pytest.mark.asyncio
async def test_missing_required_confluence_pdf_fails_sync_without_hiding_gap(db: Database):
    source_id = "src-required-pdf-failure"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=MissingPdfGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_processed == 0
    assert state.docs_failed == 1
    assert state.error_message == (
        "1 Confluence document could not be imported. PDF export was unavailable for 1 document."
    )
    assert "Confluence PDF export did not produce a PDF" in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_confluence_pdf_storage_failure_is_not_reported_as_export_failure(db: Database):
    source_id = "src-pdf-storage-failure"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=FailingPdfDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "disk full while storing PDF" in state.failed_docs[0].error
    assert "Confluence PDF export failed" not in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_existing_confluence_pdf_uri_is_preserved_when_unchanged_export_is_unavailable(db: Database):
    source_id = "src-existing-pdf-preserved"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    await db.db.execute(
        "UPDATE documents SET pdf_content_uri = ? WHERE doc_id = ?",
        ("file:///tmp/Architecture/existing.pdf", "jira-0"),
    )
    await db.db.commit()
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=MissingPdfGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == "file:///tmp/Architecture/existing.pdf"


@pytest.mark.asyncio
async def test_unchanged_stale_vector_fails_when_embedding_config_is_incomplete(
    db: Database,
):
    source_id = "src-stale-vector-no-embed"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert "embedding config is missing" in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_embedding_connection_failure_is_reported_as_provider_unreachable(
    db: Database,
    monkeypatch,
):
    source_id = "src-embedding-refused"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()

    def fake_embed_texts(texts, *args, **kwargs):
        raise OSError("[Errno 111] Connection refused")

    async def no_retry_delay(delay):
        return None

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)
    monkeypatch.setattr("memforge.pipeline.sync.asyncio.sleep", no_retry_delay)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=FalseyVectorStore(),
        embed_cfg={"base_url": "https://embedding.example", "api_key": "test-key", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "Embedding provider unreachable" in state.failed_docs[0].error
    assert state.error_message is not None
    assert "Embedding provider was unreachable for 1 document" in state.error_message


@pytest.mark.asyncio
async def test_unchanged_document_retries_stale_vector_repair_without_reprocessing(
    db: Database,
    monkeypatch,
):
    source_id = "src-stale-vector-retry"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FlakyFalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"] == content_hash(markdown)
