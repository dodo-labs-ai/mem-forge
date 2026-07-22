from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.lifecycle_plan import (
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
)
from memforge.memory.store import MemoryStore
from memforge.local_agent.source_contract import local_agent_source_config_revision
from memforge.local_agent.document_identity import build_teams_doc_id
from memforge.local_agent.teams_ledger import build_teams_window_id
from memforge.models import (
    ContentItem,
    DocumentMetadata,
    Entity,
    EnrichmentResult,
    GeneMetadata,
    Memory,
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
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_projection import (
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
)
from memforge.source_projection_config import (
    canonical_projection_scope,
    projection_scope_fingerprint,
    projection_scope_transition_id,
)
from memforge.pipeline.sync import (
    DocumentLifecycleAdmission,
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    SourceSyncMode,
    summarize_failed_documents,
)
from memforge.runtime import SourceLifecycleMaintenanceError, SyncService
from memforge.source_activity import SourceActivityConflict, SourceActivityKind
from memforge.config import AppConfig, SyncConfig
from memforge.storage.database import Database
from memforge.storage.database import MIGRATIONS
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.source_sync_manifest import SourceSyncManifestStore
from memforge.scheduler import SyncScheduler


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "sync-bookkeeping.db"))
    await database.connect()
    yield database
    await database.close()


async def _skip_retry_delay(_delay: float) -> None:
    """Preserve retry attempts without turning contract tests into wall-clock tests."""


@pytest.mark.asyncio
async def test_expired_source_activity_can_be_reacquired_with_same_id(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-activity-retry",
        type="teams",
        name="Activity Retry",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    first = await db.acquire_source_activity(
        activity_id="job-stable-id",
        source_id="src-activity-retry",
        kind=SourceActivityKind.EXTERNAL_COLLECTION,
        capability="job-stable-id",
        expected_epoch=0,
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", first.id),
    )
    await db.db.commit()

    retried = await db.acquire_source_activity(
        activity_id="job-stable-id",
        source_id="src-activity-retry",
        kind=SourceActivityKind.EXTERNAL_COLLECTION,
        capability="job-stable-id",
        expected_epoch=0,
    )

    assert retried.id == first.id
    assert retried.epoch == first.epoch


@pytest.mark.asyncio
async def test_rebaseline_admission_atomically_cancels_active_run_and_fences_worker(
    db: Database,
) -> None:
    source_id = "src-rebaseline-cancel"
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Rebaseline cancel",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
        force_full_sync=True,
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-before-maintenance",
        lease_seconds=300,
        now=now,
    )
    assert leased is not None
    sync_activity = await db.acquire_source_activity(
        activity_id="sync-before-maintenance",
        source_id=source_id,
        kind=SourceActivityKind.SYNC,
        lease_seconds=300,
    )

    job = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="rebaseline-maintenance",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )

    cancelled = await db.get_source_sync_run(enqueued.run_id)
    assert job.status is LifecycleBackfillJobStatus.QUEUED
    assert cancelled is not None
    assert cancelled.status == "failed"
    assert cancelled.force_full_sync is True
    assert cancelled.lease_owner is None
    assert cancelled.lease_expires_at is None
    assert cancelled.next_attempt_at is None
    assert cancelled.completed_at is not None
    assert cancelled.error_message == "cancelled_by_source_lifecycle_maintenance:rebaseline-maintenance"
    assert await db.get_source_activity_epoch(source_id) == sync_activity.epoch + 1

    assert (
        await db.heartbeat_source_sync_run(
            enqueued.run_id,
            worker_id="worker-before-maintenance",
            lease_attempt_count=leased.lease_attempt_count,
            now=now + timedelta(seconds=1),
        )
        is False
    )


@pytest.mark.asyncio
async def test_rebaseline_admission_rolls_back_run_cancel_when_other_activity_owns_source(
    db: Database,
) -> None:
    source_id = "src-rebaseline-conflict-rollback"
    now = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Rebaseline conflict rollback",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
        force_full_sync=True,
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-before-conflict",
        lease_seconds=300,
        now=now,
    )
    assert leased is not None
    collection = await db.acquire_source_activity(
        activity_id="external-collection-before-maintenance",
        source_id=source_id,
        kind=SourceActivityKind.EXTERNAL_COLLECTION,
        capability="external-collection-before-maintenance",
    )

    with pytest.raises(SourceActivityConflict, match="source activity already active"):
        await db.create_source_rebaseline_job(
            LifecycleBackfillJob(
                id="rebaseline-must-roll-back",
                source_id=source_id,
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )

    still_running = await db.get_source_sync_run(enqueued.run_id)
    assert still_running is not None
    assert still_running.status == "running"
    assert still_running.lease_owner == "worker-before-conflict"
    assert still_running.lease_attempt_count == leased.lease_attempt_count
    assert await db.get_source_activity_epoch(source_id) == collection.epoch
    assert await db.list_lifecycle_backfill_jobs(source_id) == []
    assert (
        await db.report_source_sync_run_progress(
            enqueued.run_id,
            worker_id="worker-before-maintenance",
            lease_attempt_count=leased.lease_attempt_count,
            progress={"schema_version": 1, "phase": "processing"},
            now=now + timedelta(seconds=1),
        )
        is False
    )
    assert (
        await db.complete_source_sync_run(
            enqueued.run_id,
            worker_id="worker-before-maintenance",
            lease_attempt_count=leased.lease_attempt_count,
            final_state=SyncState(
                source=source_id,
                last_sync_at=now + timedelta(seconds=1),
                last_sync_status="success",
            ),
            completed_at=now + timedelta(seconds=1),
        )
        is False
    )
    assert (
        await db.fail_source_sync_run(
            enqueued.run_id,
            worker_id="worker-before-maintenance",
            lease_attempt_count=leased.lease_attempt_count,
            error_message="late worker failure",
            retryable=False,
            failed_at=now + timedelta(seconds=1),
        )
        is False
    )


def test_local_agent_broker_has_its_own_forward_migration() -> None:
    version, description, statements = next(migration for migration in MIGRATIONS if migration[0] == 37)

    assert version == 37
    assert description == "Add local agent job broker"
    assert any("CREATE TABLE IF NOT EXISTS local_agent_jobs" in sql for sql in statements)


def test_source_sync_consumed_input_boundary_has_forward_migration() -> None:
    version, description, statements = next(migration for migration in MIGRATIONS if migration[0] == 57)

    assert version == 57
    assert description == "Track source sync consumed input boundary"
    assert any("input_generation_watermark" in sql for sql in statements)
    assert any("source_config_revision" in sql for sql in statements)


@pytest.mark.asyncio
async def test_lifecycle_job_fences_source_sync_input_and_config_across_connections(tmp_path):
    path = tmp_path / "lifecycle-maintenance-fence.db"
    first = Database(str(path))
    second = Database(str(path))
    await first.connect()
    await second.connect()
    try:
        await first.upsert_source(
            id="src-fenced",
            type="teams",
            name="Teams",
            config_json='{"conversation_ids":["conversation-a"]}',
            access_policy="workspace",
            owner_user_id="user-a",
        )
        await first.create_lifecycle_backfill_job(
            LifecycleBackfillJob(
                id="lifecycle-fence",
                source_id="src-fenced",
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )

        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.enqueue_source_sync_run(source_id="src-fenced")
        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.create_source_sync_input(
                source_id="src-fenced",
                raw_uri="object://src-fenced/input.json",
                raw_sha256="sha-input",
                raw_content_type="application/json",
            )
        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.enqueue_local_agent_job(
                job_id="local-job-during-lifecycle",
                source_id="src-fenced",
                source_type="teams",
                operation="teams_sync",
                payload={"source_config_revision": "revision-a"},
                created_by_user_id="user-a",
                execution_owner_user_id="user-a",
            )
        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.upsert_source(
                id="src-fenced",
                type="teams",
                name="Changed Teams",
                config_json='{"conversation_ids":["conversation-b"]}',
                access_policy="workspace",
                owner_user_id="user-a",
            )
        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.create_source_access_transition(
                operation_id="access-during-lifecycle",
                source_id="src-fenced",
                idempotency_key="access-during-lifecycle",
                actor_user_id="user-a",
                target_policy="private",
            )
        with pytest.raises(ValueError, match="source lifecycle maintenance active"):
            await second.delete_source_cascade("src-fenced")

        await first.fail_lifecycle_backfill_job(
            "lifecycle-fence",
            error="test release",
        )
        run = await second.enqueue_source_sync_run(source_id="src-fenced")
        assert run.status == "pending"
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_direct_sync_service_rejects_active_lifecycle_maintenance(
    db: Database,
) -> None:
    source_id = "src-direct-maintenance-fence"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Direct fence",
        config_json='{"base_url":"https://wiki.example"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="lifecycle-direct-fence",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    service = SyncService(db, AppConfig())

    with pytest.raises(
        SourceLifecycleMaintenanceError,
        match="source lifecycle maintenance active: lifecycle-direct-fence",
    ):
        await service.start_source(source_id)

    assert service.tasks == {}


@pytest.mark.asyncio
async def test_lifecycle_job_acquire_rejects_running_direct_sync_activity(
    db: Database,
    monkeypatch,
    tmp_path,
) -> None:
    from memforge import runtime as runtime_module

    source_id = "src-direct-activity"
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Direct activity",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingRuntime:
        def orchestrator(self):
            return self

        async def sync_gene(self, **kwargs):
            entered.set()
            await release.wait()
            return SyncState(source=source_id, last_sync_status="success")

    monkeypatch.setattr(runtime_module, "create_gene", lambda **kwargs: object())
    task = asyncio.create_task(
        runtime_module.run_source_sync(
            db=db,
            config=AppConfig(base_dir=tmp_path / "mem"),
            source={
                "id": source_id,
                "type": "agent_session",
                "name": "Direct activity",
                "config": {},
            },
            runtime=BlockingRuntime(),
        )
    )
    await entered.wait()

    with pytest.raises(SourceActivityConflict, match="source activity already active"):
        await db.create_lifecycle_backfill_job(
            LifecycleBackfillJob(
                id="lifecycle-during-direct-sync",
                source_id=source_id,
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )

    release.set()
    await task
    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="lifecycle-after-direct-sync",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    assert job.id == "lifecycle-after-direct-sync"


@pytest.mark.asyncio
async def test_lifecycle_job_acquire_rejects_active_sync_across_connections(tmp_path):
    path = tmp_path / "lifecycle-maintenance-acquire.db"
    first = Database(str(path))
    second = Database(str(path))
    await first.connect()
    await second.connect()
    try:
        await first.upsert_source(
            id="src-syncing",
            type="confluence",
            name="Confluence",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )
        run = await first.enqueue_source_sync_run(source_id="src-syncing")

        with pytest.raises(ValueError, match=f"source sync run already active: {run.run_id}"):
            await second.create_lifecycle_backfill_job(
                LifecycleBackfillJob(
                    id="lifecycle-blocked",
                    source_id="src-syncing",
                    status=LifecycleBackfillJobStatus.QUEUED,
                )
            )
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
async def test_lifecycle_job_acquire_rejects_active_local_agent_job(
    db: Database,
) -> None:
    source_id = "src-local-agent-active"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["conversation-a"]}',
        access_policy="workspace",
        owner_user_id="user-a",
    )
    local_job_id, created = await db.enqueue_local_agent_job(
        job_id="local-agent-active",
        source_id=source_id,
        source_type="teams",
        operation="teams_sync",
        payload={"source_config_revision": "revision-a"},
        created_by_user_id="user-a",
        execution_owner_user_id="user-a",
    )
    assert created is True

    with pytest.raises(
        ValueError,
        match=f"local agent job already active: {local_job_id}",
    ):
        await db.create_lifecycle_backfill_job(
            LifecycleBackfillJob(
                id="lifecycle-blocked-by-local-job",
                source_id=source_id,
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )


@pytest.mark.asyncio
async def test_active_sync_fences_source_access_and_deletion(db: Database) -> None:
    await db.upsert_source(
        id="src-active-control-plane",
        type="confluence",
        name="Confluence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="user-a",
    )
    run = await db.enqueue_source_sync_run(source_id="src-active-control-plane")

    with pytest.raises(
        ValueError,
        match=f"source sync run already active: {run.run_id}",
    ):
        await db.create_source_access_transition(
            operation_id="access-during-sync",
            source_id="src-active-control-plane",
            idempotency_key="access-during-sync",
            actor_user_id="user-a",
            target_policy="private",
        )
    with pytest.raises(
        ValueError,
        match=f"source sync run already active: {run.run_id}",
    ):
        await db.delete_source_cascade("src-active-control-plane")


@pytest.mark.asyncio
async def test_lifecycle_job_acquire_rejects_active_source_access_transition(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-access-changing",
        type="confluence",
        name="Confluence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="user-a",
    )
    transition = await db.create_source_access_transition(
        operation_id="access-changing",
        source_id="src-access-changing",
        idempotency_key="access-changing",
        actor_user_id="user-a",
        target_policy="private",
    )

    with pytest.raises(
        ValueError,
        match=f"source access transition already active: {transition['operation_id']}",
    ):
        await db.create_lifecycle_backfill_job(
            LifecycleBackfillJob(
                id="lifecycle-blocked-by-access",
                source_id="src-access-changing",
                status=LifecycleBackfillJobStatus.QUEUED,
            )
        )


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
            row["name"] for row in await (await upgraded.db.execute("PRAGMA table_info(source_sync_runs)")).fetchall()
        }
        table = await (
            await upgraded.db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'source_sync_snapshot_items'"
            )
        ).fetchone()
    finally:
        await upgraded.close()

    assert {"input_snapshot_id", "rerun_input_snapshot_id"} <= columns
    assert table is not None


@pytest.mark.asyncio
async def test_predecessor_activity_migration_upgrades_existing_sync_run_table(
    tmp_path,
) -> None:
    path = tmp_path / "pre-handoff.db"
    database = Database(str(path))
    await database.connect()
    await database.close()

    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE source_sync_runs DROP COLUMN predecessor_activity_id")
        conn.execute("ALTER TABLE source_sync_runs DROP COLUMN rerun_predecessor_activity_id")
        conn.execute("DELETE FROM schema_migrations WHERE version = 59")
        conn.commit()

    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        columns = {
            row["name"] for row in await (await upgraded.db.execute("PRAGMA table_info(source_sync_runs)")).fetchall()
        }
    finally:
        await upgraded.close()

    assert {
        "predecessor_activity_id",
        "rerun_predecessor_activity_id",
    } <= columns


@pytest.mark.asyncio
async def test_latest_source_sync_run_is_scoped_to_source_and_workspace(db: Database):
    await db.upsert_source(
        id="src-latest-a",
        type="jira",
        name="Latest A",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.upsert_source(
        id="src-latest-b",
        type="jira",
        name="Latest B",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_source_sync_run_persists_consumed_input_boundary_and_rerun_revision(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-input-boundary",
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["conversation-a"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    first_input = await db.create_source_sync_input(
        source_id="src-input-boundary",
        raw_uri="object://first",
        raw_sha256="sha-first",
        raw_content_type="application/json",
    )
    source = await db.get_source("src-input-boundary")
    assert source is not None
    config_revision = local_agent_source_config_revision(source)
    first = await db.enqueue_source_sync_run(
        source_id="src-input-boundary",
        trigger="local_agent",
        source_config_revision=config_revision,
        predecessor_activity_id="laj-first-boundary",
    )

    assert first.input_generation_watermark == first_input.input_generation
    assert first.source_config_revision == config_revision
    assert first.predecessor_activity_id == "laj-first-boundary"

    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        workspace_id="default",
        lease_seconds=60,
    )
    assert leased is not None
    second_input = await db.create_source_sync_input(
        source_id="src-input-boundary",
        raw_uri="object://second",
        raw_sha256="sha-second",
        raw_content_type="application/json",
    )
    coalesced = await db.enqueue_source_sync_run(
        source_id="src-input-boundary",
        trigger="local_agent",
        source_config_revision=config_revision,
        predecessor_activity_id="laj-second-boundary",
    )

    assert coalesced.rerun_input_generation_watermark == second_input.input_generation
    assert coalesced.rerun_source_config_revision == config_revision
    assert coalesced.rerun_predecessor_activity_id == "laj-second-boundary"

    completed = await db.complete_source_sync_run(
        leased.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(
            source="src-input-boundary",
            last_sync_status="success",
        ),
    )
    assert completed is True
    successor = await db.get_latest_source_sync_run(source_id="src-input-boundary")
    assert successor is not None
    assert successor.status == "pending"
    assert successor.input_generation_watermark == second_input.input_generation
    assert successor.source_config_revision == config_revision
    assert successor.predecessor_activity_id == "laj-second-boundary"


@pytest.mark.asyncio
async def test_source_sync_enqueue_rejects_stale_config_revision_and_fences_updates(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-config-fenced",
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["conversation-a"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )

    with pytest.raises(ValueError, match="source config revision changed"):
        await db.enqueue_source_sync_run(
            source_id="src-config-fenced",
            trigger="local_agent",
            source_config_revision="stale-revision",
        )

    source = await db.get_source("src-config-fenced")
    assert source is not None
    run = await db.enqueue_source_sync_run(
        source_id="src-config-fenced",
        trigger="local_agent",
        source_config_revision=local_agent_source_config_revision(source),
    )
    with pytest.raises(ValueError, match=f"source sync run already active: {run.run_id}"):
        await db.upsert_source(
            id="src-config-fenced",
            type="teams",
            name="Changed Teams",
            config_json='{"conversation_ids":["conversation-b"]}',
            access_policy="workspace",
            owner_user_id="dev",
        )

    await db.upsert_source(
        id="src-config-fenced",
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["conversation-a"]}',
        status="paused",
        access_policy="workspace",
        owner_user_id="dev",
    )
    paused = await db.get_source("src-config-fenced")
    assert paused is not None
    assert paused["status"] == "paused"

    with pytest.raises(ValueError, match=f"source sync run already active: {run.run_id}"):
        await db.upsert_source(
            id="src-config-fenced",
            type="teams",
            name="Teams",
            config_json='{"conversation_ids":["conversation-b"]}',
            status="paused",
            access_policy="workspace",
            owner_user_id="dev",
        )


@pytest.mark.asyncio
async def test_active_sync_pause_only_update_compares_identity_fields_exactly(
    db: Database,
) -> None:
    await db.upsert_source(
        id="src-pause-only-exact",
        type="confluence",
        name="1",
        config_json='{"spaces":["PAY"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    run = await db.enqueue_source_sync_run(source_id="src-pause-only-exact")

    with pytest.raises(ValueError, match=f"source sync run already active: {run.run_id}"):
        await db.upsert_source(
            id="src-pause-only-exact",
            type="confluence",
            name="1.0",
            config_json='{"spaces":["PAY"]}',
            status="paused",
            access_policy="workspace",
            owner_user_id="dev",
        )


@pytest.mark.asyncio
async def test_enqueue_source_sync_run_promotes_force_intent_on_active_run(db: Database):
    await db.upsert_source(
        id="src-force-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
    await db.upsert_source(
        id="src-same-snapshot",
        type="github_repo",
        name="Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
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
async def test_source_sync_run_waits_for_exact_predecessor_activity(db: Database):
    source_id = "src-local-handoff"
    job_id = "laj-local-handoff"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Local handoff",
        config_json='{"repo_url":"https://github.example/repo"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.acquire_source_activity(
        activity_id=job_id,
        source_id=source_id,
        kind=SourceActivityKind.EXTERNAL_COLLECTION,
        capability=job_id,
        lease_seconds=300,
    )
    run = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        input_snapshot_id=f"{job_id}:attempt:1",
        predecessor_activity_id=job_id,
    )
    coalesced = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
    )

    blocked = await db.lease_next_source_sync_run(
        worker_id="worker-before-release",
        lease_seconds=60,
    )
    pending = await db.get_source_sync_run(run.run_id)

    assert blocked is None
    assert pending is not None
    assert pending.status == "pending"
    assert coalesced.run_id == run.run_id
    assert coalesced.predecessor_activity_id == job_id
    assert pending.predecessor_activity_id == job_id
    assert pending.lease_attempt_count == 0

    assert await db.release_source_activity(
        activity_id=job_id,
        capability=job_id,
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-after-release",
        lease_seconds=60,
    )

    assert leased is not None
    assert leased.run_id == run.run_id
    assert leased.lease_attempt_count == 1


@pytest.mark.asyncio
async def test_source_sync_run_does_not_wait_for_unrelated_activity(db: Database):
    source_id = "src-unrelated-handoff"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Unrelated handoff",
        config_json='{"repo_url":"https://github.example/repo"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.acquire_source_activity(
        activity_id="laj-other",
        source_id=source_id,
        kind=SourceActivityKind.EXTERNAL_COLLECTION,
        capability="laj-other",
        lease_seconds=300,
    )
    run = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        input_snapshot_id="laj-expected:attempt:1",
        predecessor_activity_id="laj-expected",
    )

    leased = await db.lease_next_source_sync_run(
        worker_id="worker-unrelated",
        lease_seconds=60,
    )

    assert leased is not None
    assert leased.run_id == run.run_id


@pytest.mark.asyncio
async def test_lease_next_source_sync_run_recovers_expired_run_without_new_run(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-lease-run",
        type="github_repo",
        name="Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
            access_policy="workspace",
            owner_user_id="dev",
        )
        await first_db.enqueue_source_sync_run(source_id="src-cas-lease")
        now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)

        leased = await asyncio.gather(
            first_db.lease_next_source_sync_run(worker_id="worker-a", lease_seconds=60, now=now),
            second_db.lease_next_source_sync_run(worker_id="worker-b", lease_seconds=60, now=now),
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_retryable_failure_folds_running_rerun_intent_into_same_run(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    source_id = "src-retry-fold-rerun"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["chat-a"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    source = await db.get_source(source_id)
    assert source is not None
    revision = local_agent_source_config_revision(source)
    first_input = await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://first",
        raw_sha256="sha-first",
        raw_content_type="application/json",
    )
    active = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        source_config_revision=revision,
    )
    leased = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None
    second_input = await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://second",
        raw_sha256="sha-second",
        raw_content_type="application/json",
    )
    coalesced = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        source_config_revision=revision,
    )
    assert active.input_generation_watermark == first_input.input_generation
    assert coalesced.rerun_requested is True
    assert coalesced.rerun_input_generation_watermark == second_input.input_generation

    assert await db.fail_source_sync_run(
        active.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message="temporary outage",
        retryable=True,
        failed_at=now + timedelta(seconds=5),
        next_attempt_at=now + timedelta(seconds=10),
    )
    pending = await db.get_source_sync_run(active.run_id)

    assert pending is not None
    assert pending.status == "pending"
    assert pending.input_generation_watermark == second_input.input_generation
    assert pending.source_config_revision == revision
    assert pending.rerun_requested is False
    assert pending.rerun_input_generation_watermark is None
    assert pending.rerun_source_config_revision is None

    retried = await db.lease_next_source_sync_run(
        worker_id="worker-b",
        lease_seconds=60,
        now=now + timedelta(seconds=10),
    )
    assert retried is not None
    assert await db.complete_source_sync_run(
        active.run_id,
        worker_id="worker-b",
        lease_attempt_count=retried.lease_attempt_count,
        final_state=SyncState(source=source_id, last_sync_status="success"),
        completed_at=now + timedelta(seconds=20),
    )
    latest = await db.get_latest_source_sync_run(source_id=source_id)
    assert latest is not None
    assert latest.run_id == active.run_id
    assert latest.status == "success"


@pytest.mark.asyncio
async def test_fail_source_sync_run_marks_terminal_after_retry_budget(db: Database):
    now = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    await db.upsert_source(
        id="src-fail-budget",
        type="github_repo",
        name="Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_source_sync_input_artifact_attestation_fills_legacy_metadata_idempotently(
    db: Database,
):
    source_id = "src-input-attestation"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    created = await db.create_source_sync_input(
        source_id=source_id,
        workspace_id="workspace-a",
        raw_uri="object://workspace-a/src-input-attestation/legacy.json",
        raw_sha256="semantic-sha",
        raw_content_type="application/json",
        metadata={
            "doc_id": "doc-a",
            "submitted_by": "legacy-daemon",
            "manifest_entry": {
                "doc_id": "doc-a",
                "version": "v1",
                "provider_field": "preserved",
            },
        },
    )

    attested = await db.attest_source_sync_input_artifact(
        source_id=source_id,
        input_id=created.input_id,
        package_sha256="package-sha",
        expected_activity_epoch=0,
    )
    repeated = await db.attest_source_sync_input_artifact(
        source_id=source_id,
        input_id=created.input_id,
        package_sha256="package-sha",
        expected_activity_epoch=0,
    )

    assert repeated == attested
    assert attested.input_id == created.input_id
    assert attested.input_generation == created.input_generation
    assert attested.raw_uri == created.raw_uri
    assert attested.raw_sha256 == created.raw_sha256
    assert attested.metadata["submitted_by"] == "legacy-daemon"
    assert attested.metadata["package_sha256"] == "package-sha"
    assert attested.metadata["manifest_entry"] == {
        "doc_id": "doc-a",
        "version": "v1",
        "provider_field": "preserved",
        "package_sha256": "package-sha",
    }

    with pytest.raises(ValueError, match="artifact attestation conflict"):
        await db.attest_source_sync_input_artifact(
            source_id=source_id,
            input_id=created.input_id,
            package_sha256="different-package-sha",
            expected_activity_epoch=0,
        )

    listed = await db.list_source_sync_inputs(
        source_id=source_id,
        workspace_id="workspace-a",
    )
    assert listed == [attested]


@pytest.mark.asyncio
async def test_source_sync_input_artifact_attestation_is_epoch_fenced(db: Database):
    source_id = "src-input-attestation-fence"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    created = await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://legacy.json",
        raw_sha256="semantic-sha",
        raw_content_type="application/json",
        metadata={
            "doc_id": "doc-a",
            "manifest_entry": {"doc_id": "doc-a", "version": "v1"},
        },
    )
    await db.db.execute(
        "UPDATE sources SET activity_epoch = activity_epoch + 1 WHERE id = ?",
        (source_id,),
    )
    await db.db.commit()

    with pytest.raises(SourceActivityConflict, match="source activity epoch changed"):
        await db.attest_source_sync_input_artifact(
            source_id=source_id,
            input_id=created.input_id,
            package_sha256="package-sha",
            expected_activity_epoch=0,
        )

    [unchanged] = await db.list_source_sync_inputs(source_id=source_id)
    assert "package_sha256" not in unchanged.metadata
    assert "package_sha256" not in unchanged.metadata["manifest_entry"]


@pytest.mark.asyncio
async def test_source_sync_input_artifact_attestation_is_maintenance_fenced(
    db: Database,
):
    source_id = "src-input-attestation-maintenance"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    created = await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://legacy.json",
        raw_sha256="semantic-sha",
        raw_content_type="application/json",
        metadata={
            "doc_id": "doc-a",
            "manifest_entry": {"doc_id": "doc-a", "version": "v1"},
        },
    )
    await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="lifecycle-attestation-fence",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )

    with pytest.raises(SourceActivityConflict, match="lifecycle maintenance active"):
        await db.attest_source_sync_input_artifact(
            source_id=source_id,
            input_id=created.input_id,
            package_sha256="package-sha",
            expected_activity_epoch=0,
        )

    [unchanged] = await db.list_source_sync_inputs(source_id=source_id)
    assert "package_sha256" not in unchanged.metadata
    assert "package_sha256" not in unchanged.metadata["manifest_entry"]


@pytest.mark.asyncio
async def test_snapshot_input_uses_top_level_doc_id_and_rejects_missing_identity(db: Database):
    await db.upsert_source(
        id="src-snapshot-doc",
        type="jira",
        name="Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
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
        access_policy="workspace",
        owner_user_id="dev",
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


@pytest.mark.asyncio
async def test_source_sync_input_attestation_lookup_is_exact_and_history_bounded(
    db: Database,
):
    source_id = "src-manifest-attestations"
    workspace_id = "workspace-a"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Repository",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    inputs = []
    for revision in ("v1", "v2", "unrelated"):
        inputs.append(
            await db.create_source_sync_input(
                source_id=source_id,
                workspace_id=workspace_id,
                raw_uri=f"object://{revision}.json",
                raw_sha256=f"semantic-{revision}",
                raw_content_type="application/json",
                metadata={
                    "package_sha256": f"package-{revision}",
                    "manifest_entry": {
                        "doc_id": "doc-a" if revision != "unrelated" else "doc-other",
                        "provider_revision": revision,
                        "change_kind": "upsert",
                        "package_sha256": f"package-{revision}",
                    },
                },
            )
        )
    for index, (revision, item) in enumerate(zip(("v1", "v2", "unrelated"), inputs)):
        doc_id = "doc-a" if revision != "unrelated" else "doc-other"
        snapshot_id = f"snapshot-{index}"
        await db.db.execute(
            """INSERT INTO source_sync_snapshot_manifests (
                   workspace_id, source_id, snapshot_id, coverage, item_count,
                   manifest_sha256, local_agent_job_id, local_agent_attempt_count,
                   source_activity_epoch, source_config_revision, created_at
               ) VALUES (?, ?, ?, 'complete_snapshot', 1, ?, 'job', 1, 0, 'cfg', ?)""",
            (workspace_id, source_id, snapshot_id, "a" * 64, "2026-07-22T00:00:00+00:00"),
        )
        await db.db.execute(
            """INSERT INTO source_sync_snapshot_manifest_items (
                   workspace_id, source_id, snapshot_id, doc_id, revision, change_kind
               ) VALUES (?, ?, ?, ?, ?, 'upsert')""",
            (workspace_id, source_id, snapshot_id, doc_id, revision),
        )
        await db.db.execute(
            """INSERT INTO source_sync_snapshot_items (
                   workspace_id, source_id, snapshot_id, doc_id, input_id, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                workspace_id,
                source_id,
                snapshot_id,
                doc_id,
                item.input_id,
                "2026-07-22T00:00:00+00:00",
            ),
        )
    await db.db.commit()

    matched = await db.find_source_sync_input_attestations(
        source_id=source_id,
        workspace_id=workspace_id,
        manifest_items=[
            ("doc-a", "v1", "upsert"),
            ("doc-missing", "v1", "upsert"),
        ],
    )

    indexes = await db.db.execute_fetchall(
        "PRAGMA index_list(source_sync_snapshot_manifest_items)"
    )
    assert isinstance(db, SourceSyncManifestStore)
    assert "idx_source_sync_manifest_item_attestation" in {
        str(row["name"]) for row in indexes
    }
    assert [item.input_id for item in matched] == [inputs[0].input_id]


class EmptyGene:
    discovery_complete = True

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


class IncompleteEmptyGene(EmptyGene):
    discovery_complete = False


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
            access_policy="workspace",
            owner_user_id="dev",
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
            access_policy="workspace",
            owner_user_id="dev",
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


@pytest.mark.asyncio
async def test_document_lifecycle_admission_does_not_prequeue_one_source_ahead_of_another(
    db: Database,
):
    for source_id in ("src-fair-a", "src-fair-b"):
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="dev",
        )

    admission = DocumentLifecycleAdmission(max_active=1)
    entered: asyncio.Queue[str] = asyncio.Queue()
    releases = {item_id: asyncio.Event() for item_id in ("jira-a-0", "jira-a-1", "jira-a-2", "jira-b-0")}

    def make_orchestrator() -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=db,
            doc_store=StubDocumentStore(),
            enricher=InstantEnricher(),
            memory_extractor=NoopMemoryExtractor(),
            memory_engine=NoopMemoryEngine(),
            memory_store=None,
            max_concurrent=3,
            document_lifecycle_admission=admission,
        )

    task_a = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=OrderedBlockingFetchGene(
                prefix="a",
                item_count=3,
                entered=entered,
                releases=releases,
            ),
            source_name="Source A",
            source_id="src-fair-a",
        )
    )
    first = await asyncio.wait_for(entered.get(), timeout=2)
    assert first == "jira-a-0"

    task_b = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=OrderedBlockingFetchGene(
                prefix="b",
                item_count=1,
                entered=entered,
                releases=releases,
            ),
            source_name="Source B",
            source_id="src-fair-b",
        )
    )
    await asyncio.sleep(0.05)
    releases[first].set()
    second = await asyncio.wait_for(entered.get(), timeout=2)

    for release in releases.values():
        release.set()
    states = await asyncio.gather(task_a, task_b)

    assert second == "jira-b-0"
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


class TeamsScopeAttestationGene(EmptyGene):
    def __init__(
        self,
        *,
        transition_id: str,
        target_scope: dict[str, object],
        collection_attempt_id: str = "job-scope:attempt:1",
    ) -> None:
        super().__init__()
        self._transition_id = transition_id
        self._target_scope = target_scope
        self._collection_attempt_id = collection_attempt_id
        self._payloads: dict[str, dict[str, object]] = {}

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="teams",
            display_name="Teams",
            description="test scope attestation",
            default_sync_interval_minutes=60,
            auth_method="none",
            data_shape="message",
        )

    async def discover(self, since=None):
        del since
        conversations = sorted(str(value) for value in self._target_scope.get("conversation_ids", []))
        for conversation_id in conversations:
            item_id = f"scope-{conversation_id}"
            window_id = f"teams-scope:v1:{conversation_id}"
            self._payloads[item_id] = {
                "_scope_attestation": True,
                "conversation_id": conversation_id,
                "window_id": window_id,
                "messages": [],
                "transition_id": self._transition_id,
                "target_scope_fingerprint": projection_scope_fingerprint(self._target_scope),
                "target_conversation_ids": conversations,
                "collection_attempt_id": self._collection_attempt_id,
                "poll": {
                    "raw_conversation_id": conversation_id,
                    "access_probe_status": "ok",
                    "pagination_complete": True,
                    "stop_reason": "no_backward_link",
                },
            }
            yield ContentItem(
                item_id=item_id,
                title=f"Scope {conversation_id}",
                source_url="",
                last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
                version=f"scope-{self._transition_id}",
                extra={
                    "conversation_id": conversation_id,
                    "window_id": window_id,
                },
            )

    async def fetch(self, item: ContentItem) -> RawContent:
        return RawContent(
            item=item,
            body=json.dumps(self._payloads[item.item_id]).encode(),
            content_type="application/json",
            authoritative_empty=True,
            empty_evidence="teams_current_collection_scope_attestation",
        )

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        return NormalizedContent(item=raw.item, markdown_body="")


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


class NoArtifactRewriteDocumentStore(StubDocumentStore):
    def store_raw(self, **kwargs):
        raise AssertionError("unchanged document artifacts must not be rewritten")

    def store_normalized(self, **kwargs):
        raise AssertionError("unchanged document artifacts must not be rewritten")


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
        access_policy="workspace",
        owner_user_id="dev",
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
    inserted = 0

    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        return []

    async def apply_projected_lifecycle(self, **kwargs):
        is_update = kwargs["projection"].deltas[0].previous_unit_revision_id is not None
        if not is_update:
            return {
                "added": self.inserted,
                "updated": 0,
                "superseded": 0,
                "deleted": 0,
                "noop": 0,
            }
        return {"added": 0, "updated": 0, "superseded": 0, "deleted": 0, "noop": 0}

    async def apply_projected_tombstone(self, **kwargs):
        return {"retired": 0, "pending_review": 0, "can_delete_document": True}


class RecordingSourceSupportDetector:
    async def detect_and_persist(self, **kwargs):
        return {
            "added": 1,
            "updated": 0,
            "removed_stale": 0,
        }


class FailingDocumentDeleteMemoryStore:
    async def delete_projected_document(self, doc_id: str, **kwargs):
        raise RuntimeError("delete document failed")


class RecordingDocumentDeleteMemoryStore:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.calls: list[tuple[str, dict]] = []

    async def delete_projected_document(self, doc_id: str, **kwargs):
        self.calls.append((doc_id, kwargs))
        await self.db.delete_projected_document(doc_id)


class CountingMemoryEngine(NoopMemoryEngine):
    def __init__(self, inserted: int):
        self.inserted = inserted
        self.enrichment_calls = 0
        self.projected_lifecycle_calls = 0

    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        self.enrichment_calls += 1
        return []

    async def apply_projected_lifecycle(self, **kwargs):
        self.projected_lifecycle_calls += 1
        return await super().apply_projected_lifecycle(**kwargs)


class RecordingMemoryEngine(NoopMemoryEngine):
    def __init__(self) -> None:
        self.projected_lifecycle_calls: list[dict] = []

    async def apply_projected_lifecycle(self, **kwargs):
        self.projected_lifecycle_calls.append(kwargs)
        return await super().apply_projected_lifecycle(**kwargs)


class FailingProjectedMemoryEngine(NoopMemoryEngine):
    def __init__(self) -> None:
        self.calls = 0

    async def apply_projected_lifecycle(self, **kwargs):
        self.calls += 1
        raise RuntimeError("lifecycle apply failed")


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


class FailingLifecycleOutboxMemoryStore:
    def __init__(self) -> None:
        self.drain_calls = 0

    async def attempt_lifecycle_vector_delivery(self, *args, **kwargs):
        from memforge.memory.lifecycle_plan import (
            LifecycleVectorDeliveryResult,
            LifecycleVectorDeliveryState,
        )

        del args, kwargs
        self.drain_calls += 1
        return LifecycleVectorDeliveryResult(
            state=LifecycleVectorDeliveryState.PENDING,
            attempted_tasks=1,
            failed_tasks=1,
            error_types=("RuntimeError",),
        )


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


class DiffBoundaryViolatingMemoryExtractor(RecordingMemoryExtractor):
    async def extract_memory_changes(self, **kwargs):
        self.change_calls.append(kwargs)
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    content="The payrollTaskExecutor thread group has five threads.",
                    memory_type="fact",
                    extraction_context="| payrollTaskExecutor | 5 | 5 |",
                ),
                RawMemory(
                    content="The document now uses the repository-owned thread-list asset.",
                    memory_type="fact",
                    extraction_context="![](assets/list-of-threads.png)",
                ),
            ]
        )


class ProjectionBatchRecordingExtractor(RecordingMemoryExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.projection_calls: list[object] = []

    async def extract_projection_batch_memories(self, batch, **kwargs):
        del kwargs
        self.projection_calls.append(batch)
        return MemoryExtractionResult(memories=[])


class ProjectionBatchCandidateExtractor(ProjectionBatchRecordingExtractor):
    async def extract_projection_batch_memories(self, batch, **kwargs):
        del kwargs
        self.projection_calls.append(batch)
        index = len(self.projection_calls) - 1
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    content=f"Durable claim from projection batch {index}.",
                    memory_type="fact",
                    source_observation_id=next(iter(dict(batch.primary_content_by_observation_id))),
                )
            ]
        )


@pytest.mark.asyncio
async def test_unchanged_multi_observation_projection_skips_full_document_extraction(
    db: Database,
) -> None:
    source_id = "src-teams-unchanged"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams Unchanged",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    item = ContentItem(
        item_id="teams-window-1",
        title="Teams window",
        source_url="https://teams.example.test/conversations/conv-1",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        version="1",
        extra={"conversation_id": "conv-1", "window_id": "window-1"},
    )
    native = {
        "messages": [
            {"id": "msg-1", "content": "Keep A7.", "time": "2026-07-16T10:00:00Z"},
            {"id": "msg-2", "content": "Agreed.", "time": "2026-07-16T10:01:00Z"},
        ]
    }
    raw = RawContent(
        item=item,
        body=json.dumps(native).encode(),
        content_type="application/json",
    )
    normalized = NormalizedContent(item=item, markdown_body="Keep A7.\n\nAgreed.")
    initial = project_source_item(
        source_id=source_id,
        source_type="teams",
        run_id="teams-unchanged-initial",
        item=item,
        raw=raw,
        normalized=normalized,
    )
    unchanged = project_source_item(
        source_id=source_id,
        source_type="teams",
        run_id="teams-unchanged-replay",
        item=item,
        raw=raw,
        normalized=normalized,
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
    )
    assert unchanged.deltas[0].changed_anchors == ()
    assert unchanged.deltas[0].added_observation_ids == ()
    extractor = ProjectionBatchRecordingExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=2,
    )

    result = await orchestrator._extract_for_document_update(
        projection=unchanged,
        update_plan=None,
        markdown_body=normalized.markdown_body,
        source_type="teams",
        doc_type="conversation",
        entity_names=[],
        existing_memories=[],
        doc_id=item.item_id,
        source_id=source_id,
        run_id=unchanged.run_id,
        document_title=item.title,
        document_url=item.source_url,
    )

    assert result.memories == []
    assert result.metadata == {"projection_changed_observation_count": 0}
    assert extractor.projection_calls == []
    assert extractor.full_calls == []
    assert extractor.unit_calls == []


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


def _jira_raw_content(item: ContentItem) -> RawContent:
    payload = {
        "id": str(item.extra["issue_id"]),
        "key": str(item.extra["issue_key"]),
        "fields": {
            "summary": item.title,
            "description": "Body",
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": item.last_modified.isoformat(),
        },
        "_comments": [],
        "_comments_included": True,
        "_comments_total": 0,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    return RawContent(
        item=item,
        body=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )


class BlockingFetchGene:
    discovery_complete = True

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
                extra={"issue_id": str(100000 + idx), "issue_key": f"PAY-{idx}"},
            )

    async def fetch(self, item):
        self.active_fetches += 1
        self.max_active_fetches = max(self.max_active_fetches, self.active_fetches)
        try:
            await self.release.wait()
            return _jira_raw_content(item)
        finally:
            self.active_fetches -= 1

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\nBody")


class EmptyNormalizedBlockingFetchGene(BlockingFetchGene):
    async def fetch(self, item):
        return RawContent(item=item, body=b"", content_type="text/plain")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="")


class GitHubPagesBlockingFetchGene(BlockingFetchGene):
    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="github_pages",
            display_name="GitHub Pages",
            description="",
            default_sync_interval_minutes=60,
            auth_method="none",
            data_shape="document",
        )


class EmptyGitHubPagesBlockingFetchGene(GitHubPagesBlockingFetchGene):
    async def fetch(self, item):
        return RawContent(
            item=item,
            body=b'{"provider_status":"successful_empty"}',
            content_type="application/json",
            authoritative_empty=True,
            empty_evidence="test_provider_successful_empty_page",
        )

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="")


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
            extra={"issue_id": "200000", "issue_key": f"PAY-{self.prefix}"},
        )

    async def fetch(self, item):
        self.tracker.enter()
        try:
            await self.release.wait()
            return _jira_raw_content(item)
        finally:
            self.tracker.exit()


class OrderedBlockingFetchGene(BlockingFetchGene):
    def __init__(
        self,
        *,
        prefix: str,
        item_count: int,
        entered: asyncio.Queue[str],
        releases: dict[str, asyncio.Event],
    ) -> None:
        super().__init__(item_count=item_count, release=asyncio.Event())
        self.prefix = prefix
        self.entered = entered
        self.releases = releases

    async def discover(self, since=None):
        for idx in range(self.item_count):
            issue_number = (300000 if self.prefix == "a" else 400000) + idx
            yield ContentItem(
                item_id=f"jira-{self.prefix}-{idx}",
                title=f"Jira {self.prefix} {idx}",
                source_url=f"https://jira.example/browse/{self.prefix}-{idx}",
                last_modified=datetime.now(timezone.utc),
                content_type="application/json",
                space_or_project="PAY",
                version=f"{self.prefix}-{idx}",
                extra={
                    "issue_id": str(issue_number),
                    "issue_key": f"PAY-{issue_number}",
                },
            )

    async def fetch(self, item):
        await self.entered.put(item.item_id)
        await self.releases[item.item_id].wait()
        return _jira_raw_content(item)


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


class AuthoritativeEmptyRequiredPdfGene(PdfBackfillGene):
    async def fetch(self, item):
        return RawContent(
            item=item,
            body=b"",
            content_type="text/html",
            authoritative_empty=True,
            empty_evidence="test_provider_successful_empty_page",
        )

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="")

    async def fetch_pdf(self, item):
        raise AssertionError("authoritative empty evidence must not request a PDF")


class UnexpectedPdfExportGene(PdfBackfillGene):
    async def fetch_pdf(self, item):
        raise AssertionError("unchanged document with a stored PDF must not export again")


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
            extra={"issue_id": "300001", "issue_key": "PAY-123"},
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

    async def fetch(self, item):
        return RawContent(
            item=item,
            body=json.dumps(
                {
                    "id": str(item.extra["issue_id"]),
                    "key": str(item.extra["issue_key"]),
                    "fields": {
                        "summary": item.title,
                        "description": self.markdown,
                        "status": None,
                        "priority": None,
                        "assignee": None,
                        "labels": [],
                        "resolution": None,
                        "updated": item.last_modified.isoformat(),
                    },
                    "_comments": [],
                    "_comments_included": True,
                    "_comments_total": 0,
                    "changelog": {"startAt": 0, "histories": [], "total": 0},
                }
            ).encode("utf-8"),
            content_type="application/json",
        )


class LargeConfluenceGene(UpdatingDocumentGene):
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


class MovingGithubFileGene(UpdatingDocumentGene):
    def __init__(
        self,
        *,
        item_id: str,
        relative_path: str,
        previous_filename: str | None = None,
        previous_document_id: str | None = None,
        file_lineage_id: str | None = "file-lineage-77",
        version: str = "blob-v1",
    ) -> None:
        super().__init__("# Design\n\nKeep A7.", version=version)
        self.item_id = item_id
        self.relative_path = relative_path
        self.previous_filename = previous_filename
        self.previous_document_id = previous_document_id
        self.file_lineage_id = file_lineage_id

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="github_repo",
            display_name="GitHub Repository",
            description="",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    async def discover(self, since=None):
        extra = {
            "relative_path": self.relative_path,
            "repo_owner": "acme",
            "repo_name": "payroll",
            "repo_ref": "main",
        }
        if self.file_lineage_id is not None:
            extra["file_lineage_id"] = self.file_lineage_id
        if self.previous_filename is not None:
            extra["previous_filename"] = self.previous_filename
            extra["rename_evidence_authoritative"] = True
        if self.previous_document_id is not None:
            extra["previous_document_id"] = self.previous_document_id
        yield ContentItem(
            item_id=self.item_id,
            title="Design",
            source_url=f"https://github.example/acme/payroll/{self.relative_path}",
            last_modified=self.last_modified,
            content_type="text/markdown",
            space_or_project="payroll",
            version=self.version,
            extra=extra,
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


class RecordingMemoryReclaimer:
    def __init__(self):
        self.calls = 0

    def reclaim(self):
        self.calls += 1
        return {
            "collected_objects": 3,
            "heap_trim_supported": True,
            "heap_trimmed": True,
        }


async def _insert_source_and_doc(db: Database, source_id: str) -> None:
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="private",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_large_single_observation_uses_bounded_projection_batches(db: Database) -> None:
    source_id = "src-large-confluence"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Large Confluence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    body = "\n".join(f"design-line-{index:05d}" for index in range(9_000))
    extractor = ProjectionBatchRecordingExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=2,
    )

    state = await orchestrator.sync_gene(
        gene=LargeConfluenceGene(body),
        source_name="Large Confluence",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(extractor.projection_calls) > 1
    assert all(len(batch.primary_markdown) <= 60_000 for batch in extractor.projection_calls)
    assert extractor.full_calls == []
    assert extractor.unit_calls == []


@pytest.mark.asyncio
async def test_projection_batch_candidates_are_aggregated_before_projected_lifecycle(
    db: Database,
) -> None:
    source_id = "src-large-confluence-ledger"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Large Confluence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    body = "\n".join(f"design-line-{index:05d}" for index in range(9_000))
    extractor = ProjectionBatchCandidateExtractor()
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=None,
        max_concurrent=2,
    )

    state = await orchestrator.sync_gene(
        gene=LargeConfluenceGene(body),
        source_name="Large Confluence",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(extractor.projection_calls) > 1
    assert len(memory_engine.projected_lifecycle_calls) == 1
    [lifecycle_call] = memory_engine.projected_lifecycle_calls
    assert len(lifecycle_call["raw_memories"]) == len(extractor.projection_calls)


@pytest.mark.asyncio
async def test_sync_memory_observer_records_discovery_and_document_stages(db: Database):
    source_id = "src-sync-memory"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
    assert "document_lifecycle_exit" in stages
    assert "sync_run_end" in stages
    discovery = next(event for event in events if event["stage"] == "after_discovery")
    assert discovery["item_count"] == 1
    assert discovery["indexed_doc_count"] == 0
    assert discovery["full_sync"] is True
    assert "after_source_support" not in stages


@pytest.mark.asyncio
async def test_document_lifecycle_reclaims_process_memory_after_each_document(
    db: Database,
):
    source_id = "src-sync-memory-reclaim"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    log = RecordingSyncMemoryLogger()
    observer = SyncMemoryObserver(
        sampler=ConstantMemorySampler(),
        logger=log,
    )
    reclaimer = RecordingMemoryReclaimer()
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
        memory_reclaimer=reclaimer,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=2, release=release),
        source_name="Jira Board",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    assert reclaimer.calls == 2
    reclaimed = [event for _level, event in log.records if event["stage"] == "document_memory_reclaimed"]
    assert len(reclaimed) == 2
    assert all(event["collected_objects"] == 3 for event in reclaimed)
    assert all(event["heap_trim_supported"] is True for event in reclaimed)
    assert all(event["heap_trimmed"] is True for event in reclaimed)


@pytest.mark.asyncio
async def test_sync_memory_observer_records_pdf_export_stage(db: Database):
    source_id = "src-sync-memory-pdf"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Confluence Space",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
    pdf_events = [event for _level, event in log.records if event["stage"] == "after_pdf_export"]
    assert len(pdf_events) == 1
    assert pdf_events[0]["source_id"] == source_id
    assert pdf_events[0]["doc_id"] == "jira-0"
    assert pdf_events[0]["pdf_bytes"] == 137


@pytest.mark.asyncio
async def test_sync_memory_observer_records_lifecycle_exit_when_document_fails(db: Database):
    source_id = "src-sync-memory-error"
    await db.upsert_source(
        id=source_id, type="jira", name="Jira Board", config_json="{}", access_policy="workspace", owner_user_id="dev"
    )
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
        retry_sleep=_skip_retry_delay,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(1, release), source_name="Jira Board", source_id=source_id
    )

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
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
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
@pytest.mark.parametrize("source_type", ["confluence", "jira", "github_repo", "github_pages"])
async def test_full_discovery_without_completion_evidence_never_deletes_existing_documents(
    db: Database,
    source_type: str,
) -> None:
    source_id = f"src-incomplete-{source_type}"
    await db.upsert_source(
        id=source_id,
        type=source_type,
        name="Incomplete provider response",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-existing", source_id, "https://example/doc", "Existing", "ENG", now, "1", "hash", now),
    )
    await db.db.commit()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
        memory_store=FailingDocumentDeleteMemoryStore(),
    )

    state = await orchestrator.sync_gene(
        gene=IncompleteEmptyGene(),
        source_name="Incomplete provider response",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    assert await db.list_indexed_doc_ids(source_id) == {"doc-existing"}


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
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
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
async def test_complete_source_scope_transition_forces_snapshot_and_applies(db: Database):
    source_id = "src-scope-transition-complete"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json=json.dumps({"spaces": ["NEW"]}),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-complete",
            source_id=source_id,
            previous_scope={"sync_mode": "space", "spaces": ["OLD"]},
            target_scope={"sync_mode": "space", "spaces": ["NEW"]},
        )
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

    state = await orchestrator.sync_gene(gene, "Architecture", source_id)
    transition = (await db.list_projection_scope_transitions(source_id))[0]

    assert state.last_sync_status == "success"
    assert gene.seen_since is None
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_scope_transition_reenters_same_durable_run_on_worker_retry(db: Database):
    source_id = "src-scope-transition-worker-retry"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json=json.dumps({"spaces": ["NEW"]}),
        access_policy="workspace",
        owner_user_id="dev",
    )
    created = await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-worker-retry",
            source_id=source_id,
            previous_scope={"sync_mode": "space", "spaces": ["OLD"]},
            target_scope={"sync_mode": "space", "spaces": ["NEW"]},
        )
    )
    await db.start_projection_scope_transition(
        created.id,
        run_id="source-sync-run",
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
        SinceRecordingEmptyGene(),
        "Architecture",
        source_id,
        lifecycle_cycle_id="source-sync-run:attempt:2",
        scope_transition_run_id="source-sync-run",
    )
    transition = (await db.list_projection_scope_transitions(source_id))[0]

    assert state.last_sync_status == "success"
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert transition.run_id == "source-sync-run"


@pytest.mark.asyncio
async def test_partial_conversation_scope_transition_preserves_old_membership(db: Database):
    source_id = "src-scope-transition-partial"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json=json.dumps({"conversation_ids": ["new-thread"]}),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-partial",
            source_id=source_id,
            previous_scope={"conversation_ids": ["old-thread"]},
            target_scope={"conversation_ids": ["new-thread"]},
        )
    )
    old_item = ContentItem(
        item_id="old-window",
        title="Old Teams window",
        source_url="https://teams.example.test/old-window",
        last_modified=datetime(2026, 7, 15, tzinfo=timezone.utc),
        version="revision-old",
        extra={
            "conversation_id": "old-thread",
            "window_id": "old-window",
            "root_message_id": "old-message",
        },
    )
    old_payload = {
        "messages": [
            {
                "id": "old-message",
                "content": "Old scoped answer",
                "time": "2026-07-15T09:00:00Z",
            }
        ]
    }
    await db.record_source_projection(
        project_source_item(
            source_id=source_id,
            source_type="teams",
            run_id="projection-old-scope",
            item=old_item,
            raw=RawContent(
                item=old_item,
                body=json.dumps(old_payload).encode(),
                content_type="application/json",
            ),
            normalized=NormalizedContent(
                item=old_item,
                markdown_body="Old scoped answer",
            ),
        )
    )
    gene = SinceRecordingEmptyGene()
    gene.discovery_complete = False
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(gene, "Teams", source_id)
    transition = (await db.list_projection_scope_transitions(source_id))[0]

    assert state.last_sync_status == "success"
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.coverage.value == "partial_projection"
    assert "complete successful snapshot" in (transition.error or "")


@pytest.mark.asyncio
async def test_teams_scope_transition_applies_after_removed_units_are_tombstoned(
    db: Database,
):
    source_id = "src-scope-transition-reconciled"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json=json.dumps({"conversation_ids": ["19:retained-thread@example.test"]}),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-reconciled",
            source_id=source_id,
            previous_scope={
                "conversation_ids": [
                    "19:retained-thread@example.test",
                    "19:removed-thread@example.test",
                ]
            },
            target_scope={"conversation_ids": ["19:retained-thread@example.test"]},
        )
    )
    gene = TeamsScopeAttestationGene(
        transition_id="scope-transition-reconciled",
        target_scope={"conversation_ids": ["19:retained-thread@example.test"]},
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene,
        "Teams",
        source_id,
    )
    transition = (await db.list_projection_scope_transitions(source_id))[0]

    assert state.last_sync_status == "success"
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert transition.coverage.value == "tombstoned_delta"
    assert await db.list_indexed_doc_ids(source_id) == set()
    assert not any(
        unit.unit_type == "teams_scope_attestation" for unit in await db.list_current_source_units(source_id)
    )


@pytest.mark.asyncio
async def test_teams_max_age_transition_applies_only_with_target_time_attestation(
    db: Database,
):
    source_id = "src-scope-transition-time"
    target_scope = {
        "conversation_ids": ["19:conversation-a@example.test"],
        "max_age_days": 30,
    }
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json=json.dumps(target_scope),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-time",
            source_id=source_id,
            previous_scope={
                "conversation_ids": ["19:conversation-a@example.test"],
                "max_age_days": 365,
            },
            target_scope=target_scope,
        )
    )
    item = ContentItem(
        item_id="window-time",
        title="Recent Teams window",
        source_url="https://teams.example.test/window-time",
        last_modified=datetime(2026, 7, 10, tzinfo=timezone.utc),
        version="revision-time",
        extra={
            "conversation_id": "19:conversation-a@example.test",
            "window_id": "window-time",
            "root_message_id": "message-time",
        },
    )
    payload = {
        "_scope_coverage_from": "2026-07-01T00:00:00+00:00",
        "_scope_coverage_to": "2026-07-16T00:00:00+00:00",
        "messages": [
            {
                "id": "message-time",
                "content": "Current scoped answer",
                "time": "2026-07-10T09:00:00+00:00",
            }
        ],
    }
    await db.record_source_projection(
        project_source_item(
            source_id=source_id,
            source_type="teams",
            run_id="projection-time-scope",
            item=item,
            raw=RawContent(
                item=item,
                body=json.dumps(payload).encode(),
                content_type="application/json",
            ),
            normalized=NormalizedContent(
                item=item,
                markdown_body="Current scoped answer",
            ),
            scope={"configured_scope": target_scope},
        )
    )
    gene = TeamsScopeAttestationGene(
        transition_id="scope-transition-time",
        target_scope=target_scope,
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene,
        "Teams",
        source_id,
    )
    transition = (await db.list_projection_scope_transitions(source_id))[0]

    assert state.last_sync_status == "success"
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert transition.coverage.value == "tombstoned_delta"


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
    assert len(memory_engine.projected_lifecycle_calls) == 1
    assert memory_engine.projected_lifecycle_calls[0]["update_mode"] == "full_document"


@pytest.mark.asyncio
async def test_targeted_recovery_skips_unchanged_documents_outside_finding_scope(
    db: Database,
) -> None:
    source_id = "src-targeted-recovery"
    markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
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
        reprocess_doc_ids=frozenset({"another-doc"}),
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_updated == 0
    assert extractor.full_calls == []
    assert extractor.unit_calls == []
    assert memory_engine.projected_lifecycle_calls == []


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
        access_policy="workspace",
        owner_user_id="dev",
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
    assert len(memory_engine.projected_lifecycle_calls) == 1
    # No source-specific metadata was supplied; document last_modified is the
    # canonical source-side update time forwarded into memory provenance.
    assert len(memory_engine.projected_lifecycle_calls) == 1
    assert memory_engine.projected_lifecycle_calls[0]["source_updated_at"] == last_modified


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
        access_policy="workspace",
        owner_user_id="dev",
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
    assert len(memory_engine.projected_lifecycle_calls) == 1
    assert memory_engine.projected_lifecycle_calls[0]["source_updated_at"].isoformat() == explicit_source_updated_at


@pytest.mark.asyncio
async def test_deletion_failure_marks_sync_failed(db: Database):
    source_id = "src-deletion-failure"
    await _insert_source_and_doc(db, source_id)
    item = ContentItem(
        item_id="doc-1",
        title="Doc 1",
        source_url="http://example/doc-1",
        last_modified=datetime.now(timezone.utc),
        version="1",
        extra={"page_id": "doc-1", "space_key": "ARCH"},
    )
    raw = RawContent(item=item, body=b"Document body", content_type="text/html")
    projection = project_source_item(
        source_id=source_id,
        source_type="confluence",
        run_id="projection-before-delete-failure",
        item=item,
        raw=raw,
        normalized=NormalizedContent(item=item, markdown_body="Document body"),
    )
    await db.record_source_projection(projection)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
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
async def test_rebaseline_replay_removes_legacy_document_without_source_unit(
    db: Database,
) -> None:
    source_id = "src-rebaseline-legacy-document"
    await _insert_source_and_doc(db, source_id)
    memory_store = RecordingDocumentDeleteMemoryStore(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_REPLAY,
    )

    assert state.last_sync_status == "success"
    assert await db.get_document("doc-1") is None
    assert memory_store.calls == [
        (
            "doc-1",
            {
                "deletion_context": {
                    "deletion_kind": "rebaseline_legacy_absence",
                    "reason": "not_returned_by_complete_rebaseline_replay",
                }
            },
        )
    ]


@pytest.mark.asyncio
async def test_normal_sync_keeps_legacy_document_without_source_unit_fail_closed(
    db: Database,
) -> None:
    source_id = "src-normal-legacy-document"
    await _insert_source_and_doc(db, source_id)
    memory_store = RecordingDocumentDeleteMemoryStore(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "without persisted Source Unit lineage" in state.failed_docs[0].error
    assert await db.get_document("doc-1") is not None
    assert memory_store.calls == []


@pytest.mark.asyncio
async def test_auth_failure_records_failed_sync_state_without_secondary_error(db: Database):
    source_id = "src-auth-fail"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Auth Failure Source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
    )
    service = SyncService(db, AppConfig())

    await service.run_all_active_sources()
    run = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    assert run.source_id == source_id
    assert run.status == "pending"
    assert run.coalesced is True
    assert service.tasks == {}


@pytest.mark.asyncio
async def test_run_all_active_sources_skips_sources_without_sync_execution(db: Database):
    await db.upsert_source(
        id="src-agent-session-no-sync",
        type="agent_session",
        name="Codex Session",
        config_json="{}",
        access_policy="private",
        owner_user_id="dev",
    )
    await db.upsert_source(
        id="src-server-sync",
        type="jira",
        name="Jira",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )

    await SyncService(db, AppConfig()).run_all_active_sources()

    assert (await db.get_latest_source_sync_run(source_id="src-agent-session-no-sync")) is None
    run = await db.get_latest_source_sync_run(source_id="src-server-sync")
    assert run is not None
    assert run.status == "pending"


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
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.upsert_source(
        id="src-future",
        type="jira",
        name="Future Source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="owner-a",
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
async def test_due_local_source_remains_due_during_lifecycle_maintenance(
    db: Database,
) -> None:
    due_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    source_id = "src-local-maintenance-schedule"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Scheduled Teams",
        config_json='{"conversation_ids":["19:conversation@example.test"]}',
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
        access_policy="workspace",
        owner_user_id="owner-a",
    )
    await db.set_source_sync_schedule(
        source_id,
        enabled=True,
        interval_minutes=60,
        next_run_at=due_at,
    )
    await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id="lifecycle-schedule-fence",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )

    assert await db.enqueue_due_local_agent_jobs(now=due_at) == 0
    source = await db.get_source(source_id)
    assert source is not None
    assert source["sync_schedule"]["next_run_at"] == due_at.isoformat()


@pytest.mark.asyncio
async def test_due_ownerless_local_source_advances_schedule_without_job(db: Database):
    due_at = datetime(2026, 7, 10, tzinfo=timezone.utc)
    source_id = "src-ownerless-schedule"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Ownerless Scheduled Teams",
        config_json='{"conversation_ids":["19:conversation@example.test"]}',
        access_policy="workspace",
        owner_user_id="dev",
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
    await db.upsert_source(
        id="src-local",
        type="local_markdown",
        name="Local markdown",
        config_json='{"root":"/vault","vault_id":"vault-a"}',
        access_policy="workspace",
        owner_user_id="owner-a",
    )
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
    jobs = await db.lease_local_agent_jobs(user_id="owner-a", limit=5, lease_seconds=60)

    assert first_created is True
    assert promoted_created is False
    assert promoted_id == first_id
    assert len(jobs) == 1
    assert jobs[0]["payload"]["force_full_sync"] is True


@pytest.mark.asyncio
async def test_leased_force_request_creates_serial_successor_job(db: Database):
    await db.upsert_source(
        id="src-local",
        type="local_markdown",
        name="Local markdown",
        config_json='{"root":"/vault","vault_id":"vault-a"}',
        access_policy="workspace",
        owner_user_id="owner-a",
    )
    first_id, _ = await db.enqueue_local_agent_job(
        job_id="laj-leased-normal",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": False},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    leased = await db.lease_local_agent_jobs(user_id="owner-a", limit=1, lease_seconds=60)
    promoted_id, created = await db.enqueue_local_agent_job(
        job_id="laj-leased-force",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": True},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    blocked = await db.lease_local_agent_jobs(user_id="owner-a", limit=5, lease_seconds=60)
    await db.complete_local_agent_job(
        job_id=first_id,
        user_id="owner-a",
        attempt_count=leased[0]["attempt_count"],
        status="succeeded",
        result={},
        error=None,
        retryable=False,
    )
    successor = await db.lease_local_agent_jobs(user_id="owner-a", limit=5, lease_seconds=60)

    assert leased[0]["job_id"] == first_id
    assert promoted_id != first_id
    assert created is True
    assert blocked == []
    assert successor[0]["job_id"] == promoted_id
    assert successor[0]["payload"]["force_full_sync"] is True


@pytest.mark.asyncio
async def test_expired_original_and_successor_are_not_released_in_same_batch(db: Database):
    now = datetime.now(timezone.utc)
    await db.upsert_source(
        id="src-local",
        type="local_markdown",
        name="Local markdown",
        config_json='{"root":"/vault","vault_id":"vault-a"}',
        access_policy="workspace",
        owner_user_id="owner-a",
    )
    first_id, _ = await db.enqueue_local_agent_job(
        job_id="laj-expired-original",
        source_id="src-local",
        source_type="local_markdown",
        operation="local_markdown_sync",
        payload={"source_config_revision": "rev-a", "force_full_sync": False},
        created_by_user_id="owner-a",
        execution_owner_user_id="owner-a",
    )
    await db.lease_local_agent_jobs(user_id="owner-a", limit=1, lease_seconds=60, now=now)
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
    server_scan_calls: list[tuple[int, str, set[str]]] = []

    async def fail_local_scan(*, limit):
        raise RuntimeError("local broker unavailable")

    async def record_server_scan(*, limit, workspace_id, exclude_source_ids):
        server_scan_calls.append((limit, workspace_id, exclude_source_ids))
        return []

    monkeypatch.setattr(db, "enqueue_due_local_agent_jobs", fail_local_scan)
    monkeypatch.setattr(db, "enqueue_due_source_sync_runs", record_server_scan)
    scheduler = SyncScheduler(
        db,
        SyncService(db, AppConfig(), workspace_id="workspace-a"),
    )

    await scheduler._sync_due_sources()

    assert server_scan_calls == [(50, "workspace-a", set())]


@pytest.mark.asyncio
async def test_scheduler_excludes_sources_without_sync_execution(
    db: Database,
    monkeypatch,
):
    await db.upsert_source(
        id="src-agent-session-schedule-policy",
        type="agent_session",
        name="Codex Session",
        config_json="{}",
        access_policy="private",
        owner_user_id="dev",
    )
    server_scan_calls: list[tuple[str, set[str]]] = []

    async def record_server_scan(*, limit, workspace_id, exclude_source_ids):
        assert limit == 50
        server_scan_calls.append((workspace_id, exclude_source_ids))
        return []

    monkeypatch.setattr(db, "enqueue_due_source_sync_runs", record_server_scan)

    await SyncScheduler(db, SyncService(db, AppConfig()))._sync_due_sources()

    assert server_scan_calls == [("default", {"src-agent-session-schedule-policy"})]


@pytest.mark.asyncio
async def test_sync_service_enqueue_source_creates_durable_run_without_local_task(db: Database):
    source_id = "src-enqueue-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Enqueue Service",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_sync_service_attributes_durable_run_to_its_workspace(db: Database):
    source_id = "src-workspace-attribution"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Workspace Attribution",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    service = SyncService(db, AppConfig(), workspace_id="workspace-a")

    run = await service.enqueue_source(source_id, trigger="manual")

    assert run.workspace_id == "workspace-a"
    persisted = await db.get_source_sync_run(run.run_id)
    assert persisted is not None
    assert persisted.workspace_id == "workspace-a"


@pytest.mark.asyncio
async def test_sync_service_rejects_source_without_sync_execution(db: Database):
    source_id = "src-agent-session-service-policy"
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Codex Session",
        config_json="{}",
        access_policy="private",
        owner_user_id="dev",
    )

    with pytest.raises(RuntimeError, match="does not support ordinary sync"):
        await SyncService(db, AppConfig()).enqueue_source(source_id)

    assert await db.get_latest_source_sync_run(source_id=source_id) is None


@pytest.mark.asyncio
async def test_sync_service_captures_local_source_config_revision(db: Database) -> None:
    source_id = "src-local-revision"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Local Teams",
        config_json='{"conversation_ids":["conversation-a"],"sync_mode":"local_agent"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    source = await db.get_source(source_id)
    assert source is not None

    run = await SyncService(db, AppConfig()).enqueue_source(
        source_id,
        trigger="local_agent",
    )

    assert run.source_config_revision == local_agent_source_config_revision(source)

    with pytest.raises(ValueError, match="source config revision changed"):
        await SyncService(db, AppConfig()).enqueue_source(
            source_id,
            trigger="local_agent",
            source_config_revision="stale-revision",
        )


@pytest.mark.asyncio
async def test_sync_service_rejects_source_while_deletion_is_in_progress(db: Database):
    source_id = "src-deleting-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Deleting Service",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.force_full_sync_values: list[bool] = []
            self.extraction_pools: list[ExtractionWorkPool | None] = []
            self.lifecycle_cycle_ids: list[str] = []
            self.vector_delivery_source_ids: list[str] = []
            self.memory_store = self

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
            return self

        async def attempt_lifecycle_vector_delivery(
            self,
            *,
            source_id: str,
        ) -> None:
            self.vector_delivery_source_ids.append(source_id)

        async def run_source_sync(self, **kwargs):
            self.force_full_sync_values.append(bool(kwargs["force_full_sync"]))
            self.lifecycle_cycle_ids.append(str(kwargs["lifecycle_cycle_id"]))
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
    assert provider.lifecycle_cycle_ids == [f"{enqueued.run_id}:attempt:1"]
    assert provider.vector_delivery_source_ids == [source_id]


@pytest.mark.asyncio
async def test_source_sync_worker_keeps_success_when_vector_delivery_raises(
    db: Database,
    monkeypatch,
) -> None:
    import memforge.runtime as runtime

    source_id = "src-worker-vector-pending"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker vector pending",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )

    class PendingVectorRuntimeProvider:
        def __init__(self) -> None:
            self.memory_store = self

        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return self

        async def attempt_lifecycle_vector_delivery(self, *, source_id: str) -> None:
            raise RuntimeError(f"temporary vector failure for {source_id}")

        async def run_source_sync(self, **kwargs):
            return SyncState(
                source=str(kwargs["source"]["id"]),
                last_sync_at=datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc),
                last_sync_status="success",
            )

    logged_errors: list[tuple] = []
    monkeypatch.setattr(
        runtime.logger,
        "exception",
        lambda *args, **kwargs: logged_errors.append((args, kwargs)),
    )
    provider = PendingVectorRuntimeProvider()
    service = SyncService(db, AppConfig(), runtime_provider=provider)
    enqueued = await service.enqueue_source(source_id, trigger="manual")
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=provider,
        worker_id="worker-vector-pending",
    )

    await worker.run_once()

    completed = await db.get_source_sync_run(enqueued.run_id)
    assert completed is not None
    assert completed.status == "success"
    assert len(logged_errors) == 1
    assert logged_errors[0][0][1] == source_id


@pytest.mark.asyncio
async def test_source_sync_worker_gives_each_retry_attempt_a_new_lifecycle_cycle(
    db: Database,
):
    import memforge.runtime as runtime

    source_id = "src-worker-retry-cycle"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker retry cycle",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )

    class RetryRuntimeProvider:
        def __init__(self) -> None:
            self.lifecycle_cycle_ids: list[str] = []
            self.scope_transition_run_ids: list[str] = []

        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            self.lifecycle_cycle_ids.append(str(kwargs["lifecycle_cycle_id"]))
            self.scope_transition_run_ids.append(str(kwargs["scope_transition_run_id"]))
            if len(self.lifecycle_cycle_ids) == 1:
                raise RuntimeError("retry this attempt")
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            )

    provider = RetryRuntimeProvider()
    enqueued = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
    )
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(
            sync=SyncConfig(
                worker_retry_base_seconds=0,
                worker_retry_max_seconds=0,
            )
        ),
        runtime_provider=provider,
        worker_id="worker-a",
    )

    await worker.run_once()
    await asyncio.sleep(0.12)
    await worker.run_once()

    completed = await db.get_source_sync_run(enqueued.run_id)
    assert completed is not None
    assert completed.status == "success"
    assert provider.lifecycle_cycle_ids == [
        f"{enqueued.run_id}:attempt:1",
        f"{enqueued.run_id}:attempt:2",
    ]
    assert provider.scope_transition_run_ids == [
        enqueued.run_id,
        enqueued.run_id,
    ]


@pytest.mark.asyncio
async def test_source_sync_worker_rejects_legacy_unsupported_run_without_runtime(db: Database):
    import memforge.runtime as runtime

    source_id = "src-agent-session-worker-policy"
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Codex Session",
        config_json='{"documents_dir":"/missing/local-only-path"}',
        access_policy="private",
        owner_user_id="dev",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
    )

    class RejectRuntimeProvider:
        async def build_sync_runtime(self, *args, **kwargs):
            raise AssertionError("unsupported source reached runtime construction")

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(sync=SyncConfig(worker_max_attempts=1)),
        runtime_provider=RejectRuntimeProvider(),
        worker_id="worker-agent-session-policy",
    )

    await worker.run_once()

    failed = await db.get_source_sync_run(enqueued.run_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.next_attempt_at is None
    assert failed.error_message is not None
    assert "does not support ordinary sync" in failed.error_message


@pytest.mark.asyncio
async def test_source_sync_worker_does_not_reprocess_unchanged_complete_input_snapshot(db: Database):
    import memforge.runtime as runtime

    source_id = "src-snapshot-worker"
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Snapshot Worker",
        config_json='{"documents_dir":"/server/inbox"}',
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_legacy_snapshot_run_uses_snapshot_membership_without_watermark(
    db: Database,
):
    import memforge.runtime as runtime

    source_id = "src-legacy-snapshot"
    snapshot_id = "snapshot-before-watermarks"
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Legacy Snapshot",
        config_json='{"root":"/vault","vault_id":"vault-a"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://legacy-snapshot-doc",
        raw_sha256="sha-legacy-snapshot-doc",
        raw_content_type="application/json",
        metadata={
            "doc_id": "doc-legacy",
            "manifest_entry": {"doc_id": "doc-legacy", "version": "v1"},
        },
        sync_snapshot_id=snapshot_id,
    )
    run = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        input_snapshot_id=snapshot_id,
    )
    await db.db.execute(
        "UPDATE source_sync_runs SET input_generation_watermark = NULL WHERE run_id = ?",
        (run.run_id,),
    )
    await db.db.commit()

    class CapturingRuntimeProvider:
        source: dict | None = None

        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            self.source = kwargs["source"]
            return SyncState(source=source_id, last_sync_status="success")

    provider = CapturingRuntimeProvider()
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=provider,
        worker_id="worker-legacy-snapshot",
    )

    await worker.run_once()

    assert provider.source is not None
    assert provider.source["config"]["local_agent_package_manifest"] == [
        {
            "doc_id": "doc-legacy",
            "version": "v1",
            "package_uri": "object://legacy-snapshot-doc",
            "input_sha256": "sha-legacy-snapshot-doc",
        }
    ]


@pytest.mark.asyncio
async def test_legacy_incremental_local_run_without_watermark_fails_closed(
    db: Database,
):
    import memforge.runtime as runtime

    source_id = "src-legacy-incremental"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Legacy Incremental",
        config_json='{"conversation_ids":["chat-a"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://legacy-window",
        raw_sha256="sha-legacy-window",
        raw_content_type="application/json",
        metadata={
            "doc_id": "doc-window",
            "manifest_entry": {"doc_id": "doc-window", "version": "v1"},
        },
    )
    run = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
    )
    await db.db.execute(
        "UPDATE source_sync_runs SET input_generation_watermark = NULL WHERE run_id = ?",
        (run.run_id,),
    )
    await db.db.commit()

    class MustNotRunProvider:
        async def build_sync_runtime(self, *args, **kwargs):
            raise AssertionError("legacy unbounded run must fail before runtime construction")

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=MustNotRunProvider(),
        worker_id="worker-legacy-incremental",
    )

    await worker.run_once()
    failed = await db.get_source_sync_run(run.run_id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "local-agent sync run is missing its input generation boundary"


@pytest.mark.asyncio
async def test_source_sync_worker_consumes_only_inputs_at_run_watermark(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-watermark"
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Watermark Worker",
        config_json='{"root":"/vault","vault_id":"vault-a"}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    first = await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://first",
        raw_sha256="sha-first",
        raw_content_type="application/json",
        metadata={"manifest_entry": {"doc_id": "doc-first", "version": "v1"}},
    )
    run = await db.enqueue_source_sync_run(source_id=source_id, trigger="local_agent")
    await db.create_source_sync_input(
        source_id=source_id,
        raw_uri="object://second",
        raw_sha256="sha-second",
        raw_content_type="application/json",
        metadata={"manifest_entry": {"doc_id": "doc-second", "version": "v2"}},
    )

    class CapturingRuntimeProvider:
        source: dict | None = None

        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            self.source = kwargs["source"]
            return SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="success",
            )

    provider = CapturingRuntimeProvider()
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=provider,
        worker_id="worker-watermark",
    )

    await worker.run_once()

    assert run.input_generation_watermark == first.input_generation
    assert provider.source is not None
    assert provider.source["config"]["local_agent_package_manifest"] == [
        {
            "doc_id": "doc-first",
            "version": "v1",
            "package_uri": "object://first",
            "input_sha256": "sha-first",
        }
    ]


@pytest.mark.asyncio
async def test_source_sync_worker_rejects_stale_run_config_revision(db: Database, monkeypatch):
    import memforge.runtime as runtime

    source_id = "src-worker-stale-config"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["chat-a"]}',
        access_policy="workspace",
        owner_user_id="dev",
    )
    source = await db.get_source(source_id)
    assert source is not None
    run = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="local_agent",
        source_config_revision=local_agent_source_config_revision(source),
    )
    changed_source = {
        **source,
        "config": {"conversation_ids": ["chat-b"]},
    }

    async def get_changed_source(requested_source_id: str):
        assert requested_source_id == source_id
        return changed_source

    class MustNotRunProvider:
        async def build_sync_runtime(self, *args, **kwargs):
            raise AssertionError("stale run must fail before runtime construction")

    monkeypatch.setattr(db, "get_source", get_changed_source)
    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=MustNotRunProvider(),
        worker_id="worker-stale-config",
    )

    await worker.run_once()
    failed = await db.get_source_sync_run(run.run_id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_message == "source config revision changed before sync execution"


@pytest.mark.asyncio
async def test_source_sync_worker_marks_missing_source_terminal(db: Database, monkeypatch):
    import memforge.runtime as runtime

    await db.upsert_source(
        id="src-deleted-before-run",
        type="jira",
        name="Deleted Before Run",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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


@pytest.mark.asyncio
async def test_source_sync_worker_resumes_progress_from_an_expired_lease(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-recovered-progress"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Architecture Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-worker-recovered-progress",
    )
    enqueued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    expired_at = datetime(2026, 7, 13, 7, 0, tzinfo=timezone.utc)
    first_attempt = await db.lease_next_source_sync_run(
        worker_id="worker-a",
        lease_seconds=1,
        now=expired_at,
    )
    assert first_attempt is not None
    assert await db.report_source_sync_run_progress(
        enqueued.run_id,
        worker_id="worker-a",
        lease_attempt_count=first_attempt.lease_attempt_count,
        progress={
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 29, "total": 58, "unit": "file"},
            "counts": {"changed": 11, "memories_created": 101},
        },
        now=expired_at,
    )

    class RecoveredProgressRuntimeProvider:
        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            kwargs["progress_callback"](
                {
                    "phase": "processing",
                    "current": 2,
                    "total": 58,
                    "docs_updated": 1,
                    "memories_extracted": 3,
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
        runtime_provider=RecoveredProgressRuntimeProvider(),
        worker_id="worker-b",
        progress_flush_seconds=0.01,
    )

    await worker.run_once()
    completed = await db.get_source_sync_run(enqueued.run_id)

    assert completed is not None
    assert completed.status == "success"
    assert completed.recovery_count == 1
    assert completed.progress == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 29, "total": 58, "unit": "file"},
        "counts": {"changed": 12, "memories_created": 104},
    }


def test_reconciliation_without_measurable_work_is_indeterminate():
    from memforge.sync_progress import source_sync_progress_from_pipeline

    assert source_sync_progress_from_pipeline(
        {"phase": "detecting_deletions", "current": 0, "total": 0},
        source_type="confluence",
    ) == {"schema_version": 1, "phase": "reconciling"}


def test_recovered_source_sync_progress_preserves_run_level_work():
    from memforge.sync_progress import SourceSyncProgressAccumulator

    progress = SourceSyncProgressAccumulator(
        {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 29, "total": 58, "unit": "file"},
            "counts": {"changed": 11, "failed": 1, "memories_created": 101},
        }
    )

    assert progress.update(
        {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 2, "total": 58, "unit": "file"},
            "counts": {"changed": 1, "failed": 0, "memories_created": 3},
        }
    ) == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 29, "total": 58, "unit": "file"},
        "counts": {"changed": 12, "failed": 0, "memories_created": 104},
    }


def test_recovered_source_sync_progress_does_not_reuse_a_changed_workset():
    from memforge.sync_progress import SourceSyncProgressAccumulator

    progress = SourceSyncProgressAccumulator(
        {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 29, "total": 58, "unit": "file"},
        }
    )

    assert progress.update(
        {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 2, "total": 60, "unit": "file"},
        }
    )["progress"] == {"completed": 2, "total": 60, "unit": "file"}


def test_source_sync_progress_preserves_attempt_counts_between_phases():
    from memforge.sync_progress import SourceSyncProgressAccumulator

    progress = SourceSyncProgressAccumulator()
    progress.update(
        {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 29, "total": 58, "unit": "file"},
            "counts": {"changed": 11, "memories_created": 101},
        }
    )

    assert progress.update(
        {
            "schema_version": 1,
            "phase": "reconciling",
        }
    )["counts"] == {"changed": 11, "memories_created": 101}


@pytest.mark.asyncio
async def test_source_sync_worker_does_not_complete_after_losing_lease(db: Database, monkeypatch):
    import memforge.runtime as runtime

    source_id = "src-worker-lost-lease"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Lost Lease",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_rebaseline_admission_interrupts_an_inflight_durable_worker(
    db: Database,
) -> None:
    import memforge.runtime as runtime

    source_id = "src-worker-rebaseline-cancel"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Worker rebaseline cancel",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    enqueued = await db.enqueue_source_sync_run(
        source_id=source_id,
        trigger="manual",
        force_full_sync=True,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingRuntimeProvider:
        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=BlockingRuntimeProvider(),
        worker_id="worker-before-rebaseline",
        lease_seconds=1,
        heartbeat_seconds=0.01,
        progress_flush_seconds=0.01,
    )
    worker_task = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(started.wait(), timeout=1)

    job = await db.create_source_rebaseline_job(
        LifecycleBackfillJob(
            id="rebaseline-during-worker",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    await asyncio.wait_for(worker_task, timeout=1)

    terminal = await db.get_source_sync_run(enqueued.run_id)
    assert cancelled.is_set()
    assert job.status is LifecycleBackfillJobStatus.QUEUED
    assert terminal is not None
    assert terminal.status == "failed"
    assert terminal.error_message == ("cancelled_by_source_lifecycle_maintenance:rebaseline-during-worker")


@pytest.mark.asyncio
async def test_source_sync_worker_run_forever_polls_until_cancelled(db: Database):
    import memforge.runtime as runtime

    source_id = "src-worker-loop"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Worker Loop",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_idle_source_sync_worker_skips_relation_runtime_without_ready_work(
    db: Database,
) -> None:
    import memforge.runtime as runtime

    class BlockingRuntimeProvider:
        async def build_sync_runtime(self, db, config, **kwargs):
            del db, config, kwargs
            await asyncio.Event().wait()

    worker = runtime.SourceSyncWorker(
        db,
        AppConfig(),
        runtime_provider=BlockingRuntimeProvider(),
        worker_id="worker-empty-relation-queue",
    )

    result = await asyncio.wait_for(worker.run_once(), timeout=0.1)

    assert result is None


@pytest.mark.asyncio
async def test_source_sync_worker_interleaves_relation_work_during_source_backlog(
    db: Database,
    monkeypatch,
) -> None:
    import memforge.runtime as runtime

    worker = runtime.SourceSyncWorker(db, AppConfig(), worker_id="worker-fairness")
    source_runs = 0
    relation_slices = 0

    async def always_has_source_work():
        nonlocal source_runs
        source_runs += 1
        if source_runs == 3:
            raise asyncio.CancelledError
        return object()

    async def process_relation_slice():
        nonlocal relation_slices
        relation_slices += 1
        return True

    monkeypatch.setattr(worker, "run_once", always_has_source_work)
    monkeypatch.setattr(worker, "_process_relation_discovery_once", process_relation_slice)

    with pytest.raises(asyncio.CancelledError):
        await worker.run_forever(poll_seconds=0.01)

    assert source_runs == 3
    assert relation_slices == 2


@pytest.mark.asyncio
async def test_sync_service_passes_force_full_sync_to_source_task(db: Database, monkeypatch):
    source_id = "src-force-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Force Service",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.upsert_source(
        id="src-b",
        type="jira",
        name="Source B",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
            access_policy="workspace",
            owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.upsert_source(
        id="src-queued",
        type="jira",
        name="Queued Source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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


@pytest.mark.asyncio
async def test_cancel_source_terminates_durable_pending_run(db: Database):
    await db.upsert_source(
        id="src-durable-pending",
        type="jira",
        name="Durable Pending Source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    service = SyncService(db, AppConfig())
    enqueued = await service.enqueue_source(
        "src-durable-pending",
        trigger="manual",
    )

    await service.cancel_source("src-durable-pending")

    cancelled = await db.get_source_sync_run(enqueued.run_id)
    assert cancelled is not None
    assert cancelled.status == "failed"
    assert cancelled.error_message == "cancelled_by_source_change"
    replacement = await service.enqueue_source(
        "src-durable-pending",
        trigger="manual",
    )
    assert replacement.run_id != enqueued.run_id
    assert replacement.coalesced is False


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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_projection_repair_targets_only_requested_documents_without_semantic_work_or_cursor_advance(
    db: Database,
):
    source_id = "src-jira-projection-repair"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Repair",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    prior_sync_at = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=prior_sync_at,
            last_sync_status="success",
            docs_processed=0,
            docs_updated=0,
        )
    )

    class SemanticWorkMustNotRun:
        async def enrich_document(self, **kwargs):
            raise AssertionError("projection repair must not enrich")

        async def extract_memories(self, **kwargs):
            raise AssertionError("projection repair must not extract")

        async def process_enrichment(self, **kwargs):
            raise AssertionError("projection repair must not process enrichment")

        async def apply_projected_lifecycle(self, **kwargs):
            raise AssertionError("projection repair must not apply lifecycle")

    release = asyncio.Event()
    release.set()
    semantic_guard = SemanticWorkMustNotRun()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=semantic_guard,
        memory_extractor=semantic_guard,
        memory_engine=semantic_guard,
        memory_store=None,
        vector_store=FailingVectorStore(),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=2, release=release),
        source_name="Jira Repair",
        source_id=source_id,
        force_full_sync=True,
        reprocess_doc_ids=frozenset({"jira-1"}),
        execution_mode=SourceSyncMode.PROJECTION_REPAIR,
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert await db.get_document("jira-0") is None
    assert await db.get_document("jira-1") is not None
    projection_rows = await db.db.execute_fetchall(
        "SELECT id FROM source_units WHERE source_id = ?",
        (source_id,),
    )
    assert len(projection_rows) == 1
    persisted_state = await db.get_sync_state(source_id)
    assert persisted_state is not None
    assert persisted_state.last_sync_at == prior_sync_at
    history_rows = await db.db.execute_fetchall(
        "SELECT id FROM sync_history WHERE source = ?",
        (source_id,),
    )
    assert history_rows == []


@pytest.mark.asyncio
async def test_rebaseline_preflight_reads_full_provider_corpus_without_persisting(
    db: Database,
):
    source_id = "src-jira-rebaseline-preflight"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Preflight",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )

    class SemanticWorkMustNotRun:
        async def enrich_document(self, **kwargs):
            raise AssertionError("rebaseline preflight must not enrich")

        async def extract_memories(self, **kwargs):
            raise AssertionError("rebaseline preflight must not extract")

        async def process_enrichment(self, **kwargs):
            raise AssertionError("rebaseline preflight must not process enrichment")

        async def apply_projected_lifecycle(self, **kwargs):
            raise AssertionError("rebaseline preflight must not apply lifecycle")

    release = asyncio.Event()
    release.set()
    semantic_guard = SemanticWorkMustNotRun()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=semantic_guard,
        memory_extractor=semantic_guard,
        memory_engine=semantic_guard,
        memory_store=None,
        vector_store=FailingVectorStore(),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=2, release=release),
        source_name="Jira Preflight",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 2
    assert await db.count_documents(source=source_id) == 0
    assert await db.get_sync_state(source_id) is None
    assert (
        await db.db.execute_fetchall(
            "SELECT id FROM source_units WHERE source_id = ?",
            (source_id,),
        )
        == []
    )
    assert (
        await db.db.execute_fetchall(
            "SELECT id FROM sync_history WHERE source = ?",
            (source_id,),
        )
        == []
    )


@pytest.mark.asyncio
async def test_rebaseline_preflight_accepts_proven_authoritative_teams_package(
    db: Database,
) -> None:
    source_id = "src-teams-authoritative-preflight"
    conversation_id = "19:conversation-a@thread.v2"
    root_message_id = "message-a"
    window_id = build_teams_window_id(
        source_id=source_id,
        conversation_id=conversation_id,
        root_or_anchor_message_id=root_message_id,
        window_type="time_block",
    )
    doc_id = build_teams_doc_id(source_id=source_id, window_id=window_id)
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Teams Authoritative Preflight",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )

    class AuthoritativeTeamsPackageGene:
        discovery_complete = False

        @classmethod
        def metadata(cls):
            return GeneMetadata(
                name="teams",
                display_name="Teams",
                description="",
                default_sync_interval_minutes=60,
                auth_method="browser",
                data_shape="conversation",
            )

        async def authenticate(self) -> None:
            return None

        async def discover(self, since=None):
            assert since is None
            yield ContentItem(
                item_id=doc_id,
                title="Conversation A",
                source_url="https://teams.example.test/conversation-a",
                last_modified=datetime(2026, 7, 16, 9, 0, tzinfo=timezone.utc),
                content_type="application/json",
                space_or_project="Conversation A",
                version="revision-a",
                extra={
                    "conversation_id": conversation_id,
                    "root_message_id": root_message_id,
                    "window_id": window_id,
                    "window_type": "time_block",
                },
            )

        async def fetch(self, item):
            return RawContent(
                item=item,
                body=json.dumps(
                    {
                        "package_kind": "teams_window_document",
                        "doc_id": doc_id,
                        "conversation_id": conversation_id,
                        "root_message_id": root_message_id,
                        "window_id": window_id,
                        "window_type": "time_block",
                        "raw_payload": {
                            "conversation_id": conversation_id,
                            "window_id": window_id,
                            "messages": [
                                {
                                    "id": root_message_id,
                                    "content": "Current decision",
                                    "time": "2026-07-16T09:00:00+00:00",
                                }
                            ],
                        },
                    }
                ).encode(),
                content_type="application/json",
            )

        async def normalize(self, raw):
            return NormalizedContent(
                item=raw.item,
                markdown_body="# Conversation A\n\nCurrent decision",
            )

    class SemanticWorkMustNotRun:
        async def enrich_document(self, **kwargs):
            raise AssertionError("rebaseline preflight must not enrich")

        async def extract_memories(self, **kwargs):
            raise AssertionError("rebaseline preflight must not extract")

        async def process_enrichment(self, **kwargs):
            raise AssertionError("rebaseline preflight must not process enrichment")

        async def apply_projected_lifecycle(self, **kwargs):
            raise AssertionError("rebaseline preflight must not apply lifecycle")

    semantic_guard = SemanticWorkMustNotRun()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=semantic_guard,
        memory_extractor=semantic_guard,
        memory_engine=semantic_guard,
        memory_store=None,
        vector_store=FailingVectorStore(),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=AuthoritativeTeamsPackageGene(),
        source_name="Teams Authoritative Preflight",
        source_id=source_id,
        force_full_sync=True,
        authoritative_snapshot=True,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_failed == 0
    assert await db.count_documents(source=source_id) == 0
    assert await db.list_current_source_unit_observation_ids(source_id) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("discovery_complete", "expected_status"),
    [(True, "success"), (False, "failed")],
)
async def test_rebaseline_preflight_accepts_missing_units_only_with_complete_discovery(
    db: Database,
    discovery_complete: bool,
    expected_status: str,
) -> None:
    source_id = f"src-rebaseline-absence-{discovery_complete}"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Rebaseline Absence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )
    assert (
        await orchestrator.sync_gene(
            gene=BlockingFetchGene(item_count=2, release=release),
            source_name="Jira Rebaseline Absence",
            source_id=source_id,
        )
    ).last_sync_status == "success"
    current_units = await db.list_current_source_unit_observation_ids(source_id)
    assert len(current_units) == 2

    class ReplayGene(BlockingFetchGene):
        pass

    ReplayGene.discovery_complete = discovery_complete
    state = await orchestrator.sync_gene(
        gene=ReplayGene(item_count=1, release=release),
        source_name="Jira Rebaseline Absence",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == expected_status
    if not discovery_complete:
        assert "rebaseline replay closure is incomplete" in (state.error_message or "")
    assert await db.list_current_source_unit_observation_ids(source_id) == current_units


@pytest.mark.asyncio
async def test_rebaseline_preflight_fails_closed_on_empty_normalized_content(
    db: Database,
) -> None:
    source_id = "src-empty-rebaseline-preflight"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Empty Preflight",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        retry_sleep=_skip_retry_delay,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyNormalizedBlockingFetchGene(item_count=1, release=release),
        source_name="Jira Empty Preflight",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert await db.count_documents(source=source_id) == 0
    assert (
        await db.db.execute_fetchall(
            "SELECT id FROM source_units WHERE source_id = ?",
            (source_id,),
        )
        == []
    )


@pytest.mark.asyncio
async def test_rebaseline_preflight_rejects_truncated_jira_projection(
    db: Database,
) -> None:
    source_id = "src-jira-truncated-rebaseline-preflight"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Truncated Preflight",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()

    class JiraPayloadGene(BlockingFetchGene):
        def __init__(self, payload: dict[str, object]) -> None:
            super().__init__(item_count=1, release=release)
            self.payload = payload

        async def fetch(self, item):
            return RawContent(
                item=item,
                body=json.dumps(self.payload).encode("utf-8"),
                content_type="application/json",
            )

        async def normalize(self, raw):
            return NormalizedContent(
                item=raw.item,
                markdown_body="# PAY-0\n\nIssue and comment context.",
            )

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        retry_sleep=_skip_retry_delay,
    )
    full_payload = {
        "id": "100000",
        "key": "PAY-0",
        "fields": {
            "summary": "Payroll",
            "description": None,
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-14T10:00:00Z",
        },
        "_comments": [{"id": "501", "body": "Keep A7"}],
        "_comments_included": True,
        "_comments_total": 1,
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    assert (
        await orchestrator.sync_gene(
            gene=JiraPayloadGene(full_payload),
            source_name="Jira Truncated Preflight",
            source_id=source_id,
        )
    ).last_sync_status == "success"
    current_units = await db.list_current_source_unit_observation_ids(source_id)
    assert len(next(iter(current_units.values()))) == 2

    partial_payload = {
        "id": "100000",
        "key": "PAY-0",
        "fields": {
            "summary": "Payroll",
            "description": None,
            "status": None,
            "priority": None,
            "assignee": None,
            "labels": [],
            "resolution": None,
            "updated": "2026-07-14T10:00:00Z",
        },
        "_comments": [],
        "_comments_included": True,
        "_comments_total": 1,
        "_comments_truncated": {"returned": 0, "total": 1},
        "changelog": {"startAt": 0, "histories": [], "total": 0},
    }
    state = await orchestrator.sync_gene(
        gene=JiraPayloadGene(partial_payload),
        source_name="Jira Truncated Preflight",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "coverage=partial_projection" in state.failed_docs[0].error
    assert await db.list_current_source_unit_observation_ids(source_id) == current_units

    complete_removal_payload = {
        **full_payload,
        "_comments": [],
        "_comments_total": 0,
    }
    complete_state = await orchestrator.sync_gene(
        gene=JiraPayloadGene(complete_removal_payload),
        source_name="Jira Truncated Preflight",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert complete_state.last_sync_status == "success"
    assert await db.list_current_source_unit_observation_ids(source_id) == current_units


@pytest.mark.asyncio
async def test_rebaseline_preflight_accepts_provider_attested_empty_page(
    db: Database,
) -> None:
    source_id = "src-attested-empty-rebaseline-preflight"
    await db.upsert_source(
        id=source_id,
        type="github_pages",
        name="Attested Empty Preflight",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGitHubPagesBlockingFetchGene(item_count=1, release=release),
        source_name="Attested Empty Preflight",
        source_id=source_id,
        execution_mode=SourceSyncMode.REBASELINE_PREFLIGHT,
    )

    assert state.last_sync_status == "success"
    assert state.docs_failed == 0
    assert await db.list_current_source_unit_observation_ids(source_id) == {}


@pytest.mark.asyncio
async def test_normal_sync_reconciles_empty_revision_without_llm_extraction(
    db: Database,
) -> None:
    source_id = "src-empty-normal-sync"
    await db.upsert_source(
        id=source_id,
        type="github_pages",
        name="GitHub Pages Empty Normal",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    first_engine = RecordingMemoryEngine()
    first = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=first_engine,
        memory_store=None,
    )
    assert (
        await first.sync_gene(
            gene=GitHubPagesBlockingFetchGene(item_count=1, release=release),
            source_name="GitHub Pages Empty Normal",
            source_id=source_id,
        )
    ).last_sync_status == "success"

    class SemanticWorkMustNotRun:
        async def enrich_document(self, **kwargs):
            raise AssertionError("empty revision must not enrich")

        async def extract_memories(self, **kwargs):
            raise AssertionError("empty revision must not extract")

    empty_engine = RecordingMemoryEngine()
    second = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=SemanticWorkMustNotRun(),
        memory_extractor=SemanticWorkMustNotRun(),
        memory_engine=empty_engine,
        memory_store=None,
    )
    state = await second.sync_gene(
        gene=EmptyGitHubPagesBlockingFetchGene(item_count=1, release=release),
        source_name="GitHub Pages Empty Normal",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(empty_engine.projected_lifecycle_calls) == 1
    assert empty_engine.projected_lifecycle_calls[0]["raw_memories"] == []
    assert empty_engine.projected_lifecycle_calls[0]["document_content"] == ""
    current = await db.get_document("jira-0")
    assert current is not None
    assert current.content_hash == content_hash("")


@pytest.mark.asyncio
async def test_projection_repair_fails_when_requested_document_is_not_discovered(db: Database):
    source_id = "src-jira-projection-missing"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Missing",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Missing",
        source_id=source_id,
        force_full_sync=True,
        reprocess_doc_ids=frozenset({"jira-absent"}),
        execution_mode=SourceSyncMode.PROJECTION_REPAIR,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_processed == 0
    assert state.docs_failed == 1
    assert state.failed_docs[0].doc_id == "jira-absent"
    assert "not returned by provider discovery" in state.failed_docs[0].error
    assert await db.get_sync_state(source_id) is None


@pytest.mark.asyncio
async def test_stable_github_file_move_reuses_metadata_without_reextracting(db: Database):
    source_id = "src-github-move"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Payroll Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    extractor = RecordingMemoryExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
    )

    first = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-old-design",
            relative_path="old/design.md",
            file_lineage_id=None,
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )
    extraction_calls_after_first = len(extractor.full_calls) + len(extractor.change_calls) + len(extractor.unit_calls)
    moved = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-new-design",
            relative_path="new/design.md",
            previous_filename="old/design.md",
            previous_document_id="github-old-design",
            file_lineage_id=None,
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )
    ordinary_after_move = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-new-design",
            relative_path="new/design.md",
            file_lineage_id=None,
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )
    moved_again = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-final-design",
            relative_path="final/design.md",
            previous_filename="new/design.md",
            previous_document_id="github-new-design",
            file_lineage_id=None,
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )

    assert first.last_sync_status == "success"
    assert moved.last_sync_status == "success"
    assert ordinary_after_move.last_sync_status == "success"
    assert moved_again.last_sync_status == "success"
    assert extraction_calls_after_first > 0
    assert (
        len(extractor.full_calls) + len(extractor.change_calls) + len(extractor.unit_calls)
    ) == extraction_calls_after_first
    assert await db.get_document("github-old-design") is None
    assert await db.get_document("github-new-design") is None
    assert await db.get_document("github-final-design") is not None
    assert await db.get_metadata("github-final-design") is not None
    unit_rows = await db.db.execute_fetchall(
        "SELECT id FROM source_units WHERE source_id = ?",
        (source_id,),
    )
    assert len(unit_rows) == 1
    assert await db.list_source_unit_document_ids(str(unit_rows[0]["id"])) == (
        "github-final-design",
        "github-new-design",
        "github-old-design",
    )
    moved_unit_id = str(unit_rows[0]["id"])

    reused_old_path = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-old-design",
            relative_path="old/design.md",
            file_lineage_id=None,
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )

    assert reused_old_path.last_sync_status == "success"
    current_reused = await db.find_source_unit_by_document_id(
        source_id,
        "github-old-design",
        current_only=True,
    )
    assert current_reused is not None
    assert current_reused.id != moved_unit_id
    unit_rows = await db.db.execute_fetchall(
        "SELECT id FROM source_units WHERE source_id = ?",
        (source_id,),
    )
    assert len(unit_rows) == 2


@pytest.mark.asyncio
async def test_exact_version_github_file_move_a_to_b_to_a_keeps_one_lineage(db: Database):
    source_id = "src-github-move-reversion"
    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Payroll Repo",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    extractor = RecordingMemoryExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
    )

    for gene in (
        MovingGithubFileGene(
            item_id="github-a",
            relative_path="a/design.md",
            file_lineage_id=None,
            version="same-blob-sha",
        ),
        MovingGithubFileGene(
            item_id="github-b",
            relative_path="b/design.md",
            previous_filename="a/design.md",
            previous_document_id="github-a",
            file_lineage_id=None,
            version="same-blob-sha",
        ),
        MovingGithubFileGene(
            item_id="github-a",
            relative_path="a/design.md",
            previous_filename="b/design.md",
            previous_document_id="github-b",
            file_lineage_id=None,
            version="same-blob-sha",
        ),
    ):
        state = await orchestrator.sync_gene(
            gene=gene,
            source_name="Payroll Repo",
            source_id=source_id,
        )
        assert state.last_sync_status == "success"

    assert await db.get_document("github-a") is not None
    assert await db.get_document("github-b") is None
    unit_rows = await db.db.execute_fetchall(
        "SELECT id FROM source_units WHERE source_id = ?",
        (source_id,),
    )
    assert len(unit_rows) == 1
    assert await db.list_source_unit_document_ids(str(unit_rows[0]["id"])) == (
        "github-a",
        "github-b",
    )
    assert len(extractor.full_calls) + len(extractor.change_calls) + len(extractor.unit_calls) == 1


@pytest.mark.asyncio
async def test_scope_transition_reuses_historical_document_unit_identity(db: Database) -> None:
    source_id = "src-github-ref-roundtrip"

    def config(ref: str) -> dict[str, object]:
        return {
            "ref": ref,
            "include_paths": ["docs"],
            "include_extensions": ["md"],
        }

    async def set_scope(previous_ref: str, target_ref: str) -> None:
        target_config = config(target_ref)
        await db.upsert_source(
            id=source_id,
            type="github_repo",
            name="Payroll Repo",
            config_json=json.dumps(target_config),
            access_policy="workspace",
            owner_user_id="dev",
            projection_scope_transition=ProjectionScopeTransition(
                id=f"scope-{previous_ref}-{target_ref}",
                source_id=source_id,
                previous_scope=canonical_projection_scope("github_repo", config(previous_ref)),
                target_scope=canonical_projection_scope("github_repo", target_config),
            ),
        )

    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Payroll Repo",
        config_json=json.dumps(config("ref-a")),
        access_policy="workspace",
        owner_user_id="dev",
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    first = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-ref-a",
            relative_path="docs/design.md",
            file_lineage_id=None,
            version="blob-a",
        ),
        source_name="Payroll Repo",
        source_id=source_id,
    )
    memory = Memory(
        id="mem-ref-roundtrip",
        memory_type="fact",
        content="A7 remains enabled.",
        content_hash="hash-ref-roundtrip",
    )
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        "github-ref-a",
        "github_repo",
        "Keep A7.",
        source_updated_at=datetime.now(timezone.utc),
    )
    await set_scope("ref-a", "ref-b")
    second = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-ref-b",
            relative_path="docs/design.md",
            file_lineage_id=None,
            version="blob-b",
        ),
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )
    await set_scope("ref-b", "ref-a")
    third = await orchestrator.sync_gene(
        gene=MovingGithubFileGene(
            item_id="github-ref-a",
            relative_path="docs/design.md",
            file_lineage_id=None,
            version="blob-a",
        ),
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )

    assert first.last_sync_status == "success"
    assert second.last_sync_status == "success"
    assert third.last_sync_status == "success"
    unit_rows = await db.db.execute_fetchall(
        "SELECT id, provider_key FROM source_units WHERE source_id = ?",
        (source_id,),
    )
    assert len(unit_rows) == 1
    assert unit_rows[0]["provider_key"] == "acme/payroll:docs/design.md"
    assert await db.list_source_unit_document_ids(str(unit_rows[0]["id"])) == (
        "github-ref-a",
        "github-ref-b",
    )
    stored_memory = await db.get_memory(memory.id)
    assert stored_memory is not None
    assert stored_memory.status == "active"
    assert [source.doc_id for source in await db.get_memory_sources(memory.id)] == ["github-ref-a"]


@pytest.mark.asyncio
async def test_scope_reentry_reextracts_exact_revision_without_reusing_retired_memory(
    db: Database,
) -> None:
    source_id = "src-github-scope-reentry-replay"

    def config(include_paths: list[str]) -> dict[str, object]:
        return {
            "ref": "ref-a",
            "include_paths": include_paths,
            "include_extensions": ["md"],
        }

    async def set_scope(previous_paths: list[str], target_paths: list[str]) -> None:
        target_config = config(target_paths)
        previous_transitions = await db.list_projection_scope_transitions(
            source_id,
            limit=1,
        )
        await db.upsert_source(
            id=source_id,
            type="github_repo",
            name="Payroll Repo",
            config_json=json.dumps(target_config),
            access_policy="workspace",
            owner_user_id="dev",
            projection_scope_transition=ProjectionScopeTransition(
                id=projection_scope_transition_id(
                    source_id,
                    canonical_projection_scope(
                        "github_repo",
                        config(previous_paths),
                    ),
                    canonical_projection_scope(
                        "github_repo",
                        target_config,
                    ),
                    predecessor_transition_id=(previous_transitions[0].id if previous_transitions else None),
                ),
                source_id=source_id,
                previous_scope=canonical_projection_scope(
                    "github_repo",
                    config(previous_paths),
                ),
                target_scope=canonical_projection_scope(
                    "github_repo",
                    target_config,
                ),
            ),
        )

    class OneClaimExtractor(RecordingMemoryExtractor):
        def __init__(self) -> None:
            super().__init__()
            self.claim_calls: list[str] = []

        def _result(self, mode: str) -> MemoryExtractionResult:
            self.claim_calls.append(mode)
            return MemoryExtractionResult(
                memories=[
                    RawMemory(
                        content="A7 remains enabled.",
                        memory_type="fact",
                        evidence_quote="Keep A7.",
                        extraction_context="Keep A7.",
                    )
                ]
            )

        async def extract_memories(self, **kwargs):
            self.full_calls.append(kwargs)
            return self._result("full")

        async def extract_memory_changes(self, **kwargs):
            self.change_calls.append(kwargs)
            return self._result("changes")

        async def extract_unit_memories(self, context, **kwargs):
            self.unit_calls.append({"context": context, **kwargs})
            return self._result("unit")

        async def extract_projection_batch_memories(self, batch, **kwargs):
            del kwargs
            return self._result(f"projection:{batch.batch_id}")

    class ReplayMemoryStore:
        def __init__(self, database: Database) -> None:
            self.db = database

        def operation_context(self, **kwargs):
            del kwargs
            return None

        async def record_audit_event(self, *args, **kwargs):
            del args, kwargs

        async def find_access_compatible_equivalence_candidates(self, *args, **kwargs):
            del args, kwargs
            return ()

        async def find_access_compatible_exact_candidate(
            self,
            memory,
            *,
            excluded_memory_ids=frozenset(),
        ):
            return await self.db.find_active_exact_claim_candidate(
                memory.content_hash,
                visibility=memory.visibility,
                owner_user_id=memory.owner_user_id,
                repo_identifier=memory.repo_identifier,
                excluded_memory_ids=tuple(sorted(excluded_memory_ids)),
            )

        async def attempt_lifecycle_vector_delivery(self, lifecycle_plan_id: str):
            from memforge.memory.lifecycle_plan import (
                LifecycleVectorDeliveryResult,
                LifecycleVectorDeliveryState,
            )

            for task in await self.db.list_lifecycle_vector_tasks(lifecycle_plan_id=lifecycle_plan_id):
                await self.db.complete_lifecycle_vector_task(task.id)
            return LifecycleVectorDeliveryResult(state=LifecycleVectorDeliveryState.DELIVERED)

        async def delete_projected_document(self, doc_id: str, **kwargs):
            del kwargs
            await self.db.delete_projected_document(doc_id)

    await db.upsert_source(
        id=source_id,
        type="github_repo",
        name="Payroll Repo",
        config_json=json.dumps(config(["docs"])),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await db.enable_lifecycle_gate(source_id)
    adapters = build_sqlite_adapters(db, object())
    memory_store = ReplayMemoryStore(db)
    extractor = OneClaimExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=InstantEnricher(),
        memory_extractor=extractor,
        memory_engine=MemoryEngine(
            cross_document_candidates=CrossDocumentCandidateRetriever(
                relational=adapters.relational,
                keyword=adapters.keyword,
                vector=adapters.vector,
            ),
            db=db,
            memory_store=memory_store,
        ),
        memory_store=memory_store,
        max_concurrent=1,
    )
    gene = MovingGithubFileGene(
        item_id="github-ref-a",
        relative_path="docs/design.md",
        file_lineage_id=None,
        version="blob-a",
    )

    first = await orchestrator.sync_gene(
        gene=gene,
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )
    [memory] = await db.list_memories(source=source_id, status="active")

    await set_scope(["docs"], ["other"])
    removed = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )
    retired = await db.get_memory(memory.id)
    assert retired is not None and retired.status == "retired"

    await set_scope(["other"], ["docs"])
    reentered = await orchestrator.sync_gene(
        gene=gene,
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )

    original_after_reentry = await db.get_memory(memory.id)
    [current] = await db.list_memories(source=source_id, status="active")
    support = await db.get_active_memory_support_evidence(
        current.id,
        source_id=source_id,
    )

    await set_scope(["docs"], ["other"])
    repeated_removal = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Payroll Repo",
        source_id=source_id,
        authoritative_snapshot=True,
    )

    assert first.last_sync_status == "success"
    assert removed.last_sync_status == "success"
    assert reentered.last_sync_status == "success"
    assert repeated_removal.last_sync_status == "success"
    assert len(extractor.claim_calls) == 2
    assert original_after_reentry is not None
    assert original_after_reentry.status == "retired"
    assert current.id != memory.id
    assert len(support) == 1
    assert support[0].anchor.observation_revision_id
    repeatedly_retired = await db.get_memory(current.id)
    assert repeatedly_retired is not None
    assert repeatedly_retired.status == "retired"


@pytest.mark.asyncio
async def test_full_document_extraction_failure_is_audited(db: Database):
    source_id = "src-full-extraction-failure"
    await db.upsert_source(
        id=source_id,
        type="docs",
        name="Docs",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        retry_sleep=_skip_retry_delay,
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
    assert len(memory_engine.projected_lifecycle_calls) == 1
    assert memory_engine.projected_lifecycle_calls[0]["update_mode"] == "diff_guided"
    assert "PostgreSQL 15" in memory_engine.projected_lifecycle_calls[0]["changed_hunks"]
    assert memory_engine.projected_lifecycle_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.projected_lifecycle_calls[0]["update_plan_stats"]["data_shape"] == "document"
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
async def test_diff_guided_extraction_rejects_candidates_outside_current_change(
    db: Database,
) -> None:
    source_id = "src-diff-evidence-boundary"
    old_markdown = "\n".join(
        (
            "# Shared HANA Database Connections",
            "",
            "| Thread Group | Min | Max |",
            "| payrollTaskExecutor | 5 | 5 |",
            "",
            "![](../../../../../Desktop/old.png)",
        )
    )
    new_markdown = "\n".join(
        (
            "# Shared HANA Database Connections",
            "",
            "| Thread Group | Min | Max |",
            "| payrollTaskExecutor | 5 | 5 |",
            "",
            "Here is an example of running threads:",
            "![](assets/list-of-threads.png)",
        )
    )
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id=source_id,
        title="Shared HANA Database Connections",
        markdown=old_markdown,
    )
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Shared HANA Database Connections",
        markdown=old_markdown,
        version="1",
        normalized_content_uri=normalized_content_uri,
    )
    extractor = DiffBoundaryViolatingMemoryExtractor()
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    extraction_rows = await db.list_memory_audit_events(
        event_type="memory_change_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert len(memory_engine.projected_lifecycle_calls) == 1
    raw_memories = memory_engine.projected_lifecycle_calls[0]["raw_memories"]
    assert [memory.content for memory in raw_memories] == [
        "The document now uses the repository-owned thread-list asset."
    ]
    assert len(extraction_rows) == 1
    assert extraction_rows[0].payload["extracted_count"] == 1
    assert extraction_rows[0].payload["rejected_outside_changed_range_count"] == 1


@pytest.mark.asyncio
async def test_large_single_observation_update_keeps_diff_guided_authority(
    db: Database,
) -> None:
    source_id = "src-large-diff-evidence-boundary"
    stable_body = "\n".join(f"Stable context line {index}." for index in range(4_000))
    old_markdown = f"# Design Doc\n\n{stable_body}\n\nThe service uses PostgreSQL 14."
    new_markdown = f"# Design Doc\n\n{stable_body}\n\nThe service uses PostgreSQL 15."
    doc_store = StubDocumentStore()
    normalized_content_uri = doc_store.store_normalized(
        source_id=source_id,
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
    extractor = ProjectionBatchRecordingExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=RecordingMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert len(extractor.change_calls) == 1
    assert extractor.projection_calls == []


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
    assert len(memory_engine.projected_lifecycle_calls) == 1
    assert memory_engine.projected_lifecycle_calls[0]["update_mode"] == "diff_guided"
    assert memory_engine.projected_lifecycle_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.projected_lifecycle_calls[0]["update_plan_stats"]["data_shape"] == "ticket"
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
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_document_admission_capacity_does_not_throttle_unit_extraction(db: Database):
    markdown = "# Design Doc\n\nIntro.\n\n" + "\n\n".join(
        f"## Section {index}\n\n" + ("Durable design detail. " * 900) for index in range(8)
    )
    extractor = BlockingUnitMemoryExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, "src-large-doc-admission"),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=2,
        document_lifecycle_admission=DocumentLifecycleAdmission(max_active=1),
    )

    task = asyncio.create_task(
        orchestrator._extract_full_document_units(
            markdown_body=markdown,
            source_type="github_pages",
            doc_type="reference",
            entity_names=[],
            existing_memories=[],
            doc_id="doc-large",
            source_id="src-large-doc-admission",
            document_title="Design Doc",
            document_url="https://example.test/design",
        )
    )

    try:
        await asyncio.wait_for(extractor.started_two.wait(), timeout=2)
        assert extractor.max_active == 2
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
        retry_sleep=_skip_retry_delay,
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
    assert len(memory_engine.projected_lifecycle_calls) == 0
    assert len(audit_rows) == 3
    assert audit_rows[0].reason == "partial_unit_failure"
    assert audit_rows[0].payload["failed_unit_count"] == 1
    assert audit_rows[0].payload["extracted_count"] == 0


@pytest.mark.asyncio
async def test_lifecycle_failure_preserves_projection_delta_for_ordinary_retry(db: Database):
    source_id = "src-lifecycle-retry"
    old_markdown = "# Design Doc\n\nThe service uses PostgreSQL 14."
    new_markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Documents",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    doc_store = StubDocumentStore()

    first = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
    )
    first_state = await first.sync_gene(
        gene=UpdatingDocumentGene(old_markdown, version="1"),
        source_name="Documents",
        source_id=source_id,
    )
    source_unit = await db.find_source_unit_by_document_id(source_id, "doc-1")
    assert first_state.last_sync_status == "success"
    assert source_unit is not None
    prior_revision = await db.get_current_source_unit_revision(source_unit.id)
    prior_document = await db.get_document("doc-1")
    assert prior_revision is not None and prior_document is not None

    failing_engine = FailingProjectedMemoryEngine()
    failed = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=failing_engine,
        memory_store=None,
        max_concurrent=1,
        retry_sleep=_skip_retry_delay,
    )
    failed_state = await failed.sync_gene(
        gene=UpdatingDocumentGene(new_markdown, version="2"),
        source_name="Documents",
        source_id=source_id,
    )

    assert failed_state.last_sync_status == "failed"
    assert failing_engine.calls == 3
    assert (await db.get_current_source_unit_revision(source_unit.id)).id == prior_revision.id  # type: ignore[union-attr]
    assert (await db.get_document("doc-1")).content_hash == prior_document.content_hash  # type: ignore[union-attr]

    retry_engine = RecordingMemoryEngine()
    retry = GeneSyncOrchestrator(
        db=db,
        doc_store=doc_store,
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=RecordingMemoryExtractor(),
        memory_engine=retry_engine,
        memory_store=None,
        max_concurrent=1,
    )
    retry_state = await retry.sync_gene(
        gene=UpdatingDocumentGene(new_markdown, version="2"),
        source_name="Documents",
        source_id=source_id,
    )

    assert retry_state.last_sync_status == "success"
    assert len(retry_engine.projected_lifecycle_calls) == 1
    current_revision = await db.get_current_source_unit_revision(source_unit.id)
    assert current_revision is not None and current_revision.id != prior_revision.id


@pytest.mark.asyncio
async def test_item_processing_is_bounded_by_max_concurrent(db: Database):
    source_id = "src-bounded-sync"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
async def test_sync_cancels_and_drains_sibling_items_when_one_item_is_cancelled(
    db: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_id = "src-cancelled-item-siblings"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release_siblings = asyncio.Event()
    siblings_started = asyncio.Event()
    active_siblings = 0
    gene = BlockingFetchGene(item_count=3, release=asyncio.Event())
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=3,
    )

    async def process_item(**kwargs) -> dict:
        nonlocal active_siblings
        item = kwargs["item"]
        if item.item_id == "jira-0":
            await siblings_started.wait()
            raise asyncio.CancelledError

        active_siblings += 1
        if active_siblings == 2:
            siblings_started.set()
        try:
            await release_siblings.wait()
            return {"updated": False, "memories_extracted": 0}
        finally:
            active_siblings -= 1

    monkeypatch.setattr(orchestrator, "_process_item", process_item)

    try:
        with pytest.raises(asyncio.CancelledError):
            await orchestrator.sync_gene(
                gene=gene,
                source_name="Jira Board",
                source_id=source_id,
            )
        await asyncio.sleep(0)
        assert active_siblings == 0
    finally:
        release_siblings.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_running_progress_reports_extracted_memories(db: Database):
    source_id = "src-running-memory-progress"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
    assert [event["current"] for event in progress_events if event.get("phase") == "discovering"] == [0, 1]
    reconciliation = [event for event in progress_events if event.get("phase") == "detecting_deletions"]
    assert reconciliation == [
        {
            "phase": "detecting_deletions",
            "current": 0,
            "total": 0,
            "title": None,
        }
    ]


@pytest.mark.asyncio
async def test_document_vector_failure_happens_before_memory_mutations(db: Database, monkeypatch):
    source_id = "src-vector-before-memory"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()
    memory_engine = CountingMemoryEngine(inserted=3)
    vector_store = FailingVectorStore()
    retry_delays: list[float] = []

    async def record_retry_delay(delay: float) -> None:
        retry_delays.append(delay)

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
        retry_sleep=record_retry_delay,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert retry_delays == [2, 4]
    assert memory_engine.enrichment_calls == 0
    assert memory_engine.projected_lifecycle_calls == 0
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
        access_policy="workspace",
        owner_user_id="dev",
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
        access_policy="workspace",
        owner_user_id="dev",
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
        retry_sleep=_skip_retry_delay,
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
async def test_authoritative_empty_confluence_page_does_not_require_pdf(db: Database):
    source_id = "src-authoritative-empty-confluence"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
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
        gene=AuthoritativeEmptyRequiredPdfGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_failed == 0
    assert document is not None
    assert document.content_hash == content_hash("")
    assert document.pdf_content_uri is None


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
        retry_sleep=_skip_retry_delay,
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
async def test_unchanged_document_with_complete_artifacts_does_not_rewrite_or_export_pdf(
    db: Database,
):
    source_id = "src-unchanged-complete-artifacts"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
        normalized_content_uri="file:///tmp/Architecture/existing.md",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    await db.db.execute(
        """UPDATE documents
           SET raw_content_uri = ?, raw_content_type = ?, pdf_content_uri = ?
           WHERE doc_id = ?""",
        (
            "file:///tmp/Architecture/existing.raw",
            "application/json",
            "file:///tmp/Architecture/existing.pdf",
            "jira-0",
        ),
    )
    await db.db.commit()
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=NoArtifactRewriteDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UnexpectedPdfExportGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert document is not None
    assert document.raw_content_uri == "file:///tmp/Architecture/existing.raw"
    assert document.normalized_content_uri == "file:///tmp/Architecture/existing.md"
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
        retry_sleep=_skip_retry_delay,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert "embedding config is missing" in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_unchanged_document_survives_pending_lifecycle_vector_delivery(
    db: Database,
) -> None:
    source_id = "src-unchanged-lifecycle-vector"
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
    memory_store = FailingLifecycleOutboxMemoryStore()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert state.docs_failed == 0
    assert state.docs_updated == 0
    assert memory_store.drain_calls == 1
    assert await db.get_document("jira-0") is not None


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
        access_policy="workspace",
        owner_user_id="dev",
    )
    release = asyncio.Event()
    release.set()

    def fake_embed_texts(texts, *args, **kwargs):
        raise OSError("[Errno 111] Connection refused")

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

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
        retry_sleep=_skip_retry_delay,
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
