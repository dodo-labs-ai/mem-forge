"""A person stops a Source's sync: the run ends like one whose lease was lost, and is not retried."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import memforge.runtime as runtime
from memforge.config import AppConfig
from memforge.local_agent.runner import CloudJobLeaseLost, _CloudJobLeaseHeartbeat
from memforge.models import ContentItem, ScopeListingKind
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.source_activity import SOURCE_SYNC_CANCELLED_BY_USER, SourceSyncHeartbeat, SourceSyncRunActive
from memforge.source_projection import ProjectionScopeTransition
from memforge.storage.database import Database
from tests.test_document_absence import JIRA_CONFIG, SOURCE_ID, JiraProject, Workspace
from tests.test_source_authority_api import _app, _confluence_payload, _connect_database, _jira_payload
from tests.test_sync_bookkeeping import (
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    StubDocumentStore,
    _skip_retry_delay,
)

NOW = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)


async def _source(db: Database, source_id: str = "src-stop") -> str:
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira",
        config_json=json.dumps(JIRA_CONFIG),
        access_policy="workspace",
        owner_user_id="dev",
    )
    return source_id


async def _activity(db: Database, source_id: str) -> list[dict]:
    rows = await db.db.execute_fetchall(
        "SELECT id, kind FROM source_activity_leases WHERE source_id = ?",
        (source_id,),
    )
    return [dict(row) for row in rows]


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "stop.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_a_pending_run_fails_at_once_and_records_it_was_stopped(db: Database):
    source_id = await _source(db)
    queued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    stopped = await db.request_source_sync_stop(source_id=source_id, now=NOW)

    assert stopped is not None
    assert stopped.run_id == queued.run_id
    assert stopped.status == "failed"
    assert stopped.error_message == SOURCE_SYNC_CANCELLED_BY_USER
    assert stopped.stop_requested_at == NOW
    assert stopped.next_attempt_at is None
    [history] = await db.get_sync_history(source=source_id, limit=5)
    assert (history["run_id"], history["status"], history["error_message"]) == (
        queued.run_id,
        "failed",
        SOURCE_SYNC_CANCELLED_BY_USER,
    )
    assert await db.lease_next_source_sync_run(worker_id="worker-a", now=NOW) is None
    assert await db.request_source_sync_stop(source_id=source_id) is None


@pytest.mark.asyncio
async def test_a_running_run_records_the_first_request_and_drops_the_sync_waiting_behind_it(db: Database):
    source_id = await _source(db)
    queued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    leased = await db.lease_next_source_sync_run(worker_id="worker-a", lease_seconds=60, now=NOW)
    assert leased is not None
    coalesced = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual", force_full_sync=True)
    assert coalesced.run_id == queued.run_id
    assert (await db.get_source_sync_run(queued.run_id)).rerun_requested

    first = await db.request_source_sync_stop(source_id=source_id, now=NOW + timedelta(seconds=1))
    again = await db.request_source_sync_stop(source_id=source_id, now=NOW + timedelta(seconds=2))

    assert first is not None and again is not None
    assert first.status == again.status == "running"
    assert again.stop_requested_at == NOW + timedelta(seconds=1)
    assert not again.rerun_requested
    heartbeat = await db.heartbeat_source_sync_run(
        queued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        lease_seconds=60,
        now=NOW + timedelta(seconds=10),
    )
    assert heartbeat is SourceSyncHeartbeat.STOP_REQUESTED
    assert (await db.get_source_sync_run(queued.run_id)).lease_expires_at == NOW + timedelta(seconds=70)

    assert await db.fail_source_sync_run(
        queued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message=SOURCE_SYNC_CANCELLED_BY_USER,
        retryable=False,
        failed_at=NOW + timedelta(seconds=11),
    )
    assert await db.get_latest_source_sync_run(source_id=source_id) == await db.get_source_sync_run(queued.run_id)
    assert await _activity(db, source_id) == []


@pytest.mark.asyncio
async def test_a_stopped_run_leaves_its_scope_transition_open_for_the_next_run(db: Database):
    source_id = await _source(db)
    transition = ProjectionScopeTransition(
        id="pst-stop",
        source_id=source_id,
        previous_scope={"jql": "project = PAY"},
        target_scope={"jql": "project in (PAY, OPS)"},
        created_at=NOW.isoformat(),
    )
    await db.create_projection_scope_transition(transition)
    queued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    leased = await db.lease_next_source_sync_run(worker_id="worker-a", lease_seconds=60)
    await db.start_projection_scope_transition(transition.id, run_id=queued.run_id)
    await db.request_source_sync_stop(source_id=source_id)
    assert await db.fail_source_sync_run(
        queued.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        error_message=SOURCE_SYNC_CANCELLED_BY_USER,
        retryable=False,
    )

    open_transition = await db.get_open_projection_scope_transition(source_id)
    assert open_transition is not None and open_transition.id == transition.id
    next_run = await db.enqueue_source_sync_run(source_id=source_id, trigger="schedule")
    assert await db.lease_next_source_sync_run(worker_id="worker-b", lease_seconds=60) is not None
    restarted = await db.start_projection_scope_transition(transition.id, run_id=next_run.run_id)
    assert restarted.run_id == next_run.run_id


@pytest.mark.asyncio
async def test_a_source_change_names_the_run_it_waits_for(db: Database):
    source_id = await _source(db)
    queued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")

    with pytest.raises(SourceSyncRunActive) as refused:
        await db.upsert_source(
            id=source_id,
            type="jira",
            name="Renamed Jira",
            config_json=json.dumps(JIRA_CONFIG),
            access_policy="workspace",
            owner_user_id="dev",
        )

    assert refused.value.run_id == queued.run_id


class _OneIssueHangs(JiraProject):
    """Fetching issue ``hanging`` waits until the test lets it go; every other issue is fetched at once."""

    def __init__(self, *, hanging: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hanging = f"jira-{hanging}"
        self.reached = asyncio.Event()
        self.released = asyncio.Event()

    async def fetch(self, item: ContentItem):
        if item.item_id == self.hanging:
            self.reached.set()
            await self.released.wait()
        return await super().fetch(item)


class _WorkspaceRuntime:
    def __init__(self, workspace: Workspace, gene: JiraProject) -> None:
        self.workspace = workspace
        self.gene = gene
        self.calls = 0

    async def build_sync_runtime(self, db, config, *, extraction_pool=None, document_lifecycle_admission=None):
        del db, config, extraction_pool, document_lifecycle_admission
        return object()

    async def run_source_sync(self, **kwargs):
        self.calls += 1
        return await GeneSyncOrchestrator(
            db=self.workspace.db,
            doc_store=StubDocumentStore(),
            memory_extractor=ProjectionFragmentRecordingExtractor(),
            memory_engine=self.workspace.engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.workspace.db),
            max_concurrent=1,
            retry_sleep=_skip_retry_delay,
        ).sync_gene(
            gene=self.gene,
            source_name="Jira",
            source_id=SOURCE_ID,
            progress_callback=kwargs["progress_callback"],
            source_activity=kwargs["source_activity"],
            lifecycle_cycle_id=kwargs["lifecycle_cycle_id"],
            scope_transition_run_id=kwargs["scope_transition_run_id"],
            record_terminal_result=False,
        )


@pytest.fixture
async def workspace(tmp_path):
    database = Database(str(tmp_path / "stop-pipeline.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    await database.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(JIRA_CONFIG),
        access_policy="workspace",
        owner_user_id="dev",
    )
    try:
        yield Workspace(database)
    finally:
        NoopMemoryEngine.db = None
        await database.close()


@pytest.mark.asyncio
async def test_a_stopped_run_keeps_what_it_committed_removes_nothing_and_keeps_the_cursor(workspace: Workspace):
    project = _OneIssueHangs(hanging=3, existing={0, 2}, listing_kind=ScopeListingKind.EXISTENCE)
    first = await workspace.sync(project)
    assert first.last_sync_status == "success"
    workspace.engine.events.clear()
    cursor = (await workspace.db.get_sync_state(SOURCE_ID)).last_sync_at

    # jira-2 is deleted at the provider, so an absence step would remove it;
    # jira-1 is new and is committed before the run reaches jira-3.
    project.move_on(existing={0, 1, 3}, changed={1, 3})
    run = await workspace.db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")
    provider = _WorkspaceRuntime(workspace, project)
    worker = runtime.SourceSyncWorker(
        workspace.db,
        AppConfig(),
        runtime_provider=provider,
        worker_id="worker-a",
        lease_seconds=60,
        heartbeat_seconds=0.01,
    )
    running = asyncio.create_task(worker.run_once())
    await asyncio.wait_for(project.reached.wait(), timeout=10)
    requested = await workspace.db.request_source_sync_stop(source_id=SOURCE_ID)
    assert requested is not None and requested.status == "running"
    await asyncio.wait_for(running, timeout=10)

    stopped = await workspace.db.get_source_sync_run(run.run_id)
    assert stopped.status == "failed"
    assert stopped.error_message == SOURCE_SYNC_CANCELLED_BY_USER
    assert stopped.next_attempt_at is None
    assert ("commit", "jira-1") in workspace.engine.events
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert (await workspace.db.get_sync_state(SOURCE_ID)).last_sync_at == cursor
    assert await _activity(workspace.db, SOURCE_ID) == []
    assert await worker.run_once() is None
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_a_run_its_worker_was_stopping_when_it_died_is_failed_without_running(db: Database):
    source_id = await _source(db)
    queued = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
    assert await db.lease_next_source_sync_run(worker_id="dead-worker", lease_seconds=60, now=NOW) is not None
    await db.request_source_sync_stop(source_id=source_id, now=NOW + timedelta(seconds=1))

    class NeverRuns:
        async def build_sync_runtime(self, *args, **kwargs):
            raise AssertionError("a stopped run builds no runtime")

        async def run_source_sync(self, **kwargs):
            raise AssertionError("a stopped run does not run")

    worker = runtime.SourceSyncWorker(db, AppConfig(), runtime_provider=NeverRuns(), worker_id="worker-b")
    # The dead worker's lease expires; the next worker ends the run.
    await db.db.execute(
        "UPDATE source_sync_runs SET lease_expires_at = ? WHERE run_id = ?",
        ((NOW - timedelta(days=1)).isoformat(), queued.run_id),
    )
    await db.db.execute(
        "UPDATE source_activity_leases SET lease_until = ? WHERE id = ?",
        ((NOW - timedelta(days=1)).isoformat(), queued.run_id),
    )
    await db.db.commit()

    assert (await worker.run_once()).run_id == queued.run_id
    stopped = await db.get_source_sync_run(queued.run_id)
    assert (stopped.status, stopped.error_message) == ("failed", SOURCE_SYNC_CANCELLED_BY_USER)
    assert await _activity(db, source_id) == []


def test_the_stop_route_answers_what_it_stopped(tmp_path: Path):
    database = _connect_database(tmp_path)
    try:
        with TestClient(_app(tmp_path, database)) as client:
            source_id = client.post("/api/v1/sources", json=_confluence_payload()).json()["id"]
            url = f"/api/v1/sources/{source_id}/sync/stop"
            idle = client.post(url)
            pending = asyncio.run(database.enqueue_source_sync_run(source_id=source_id, workspace_id="default"))
            stopped = client.post(url)
            running = asyncio.run(database.enqueue_source_sync_run(source_id=source_id, workspace_id="default"))
            assert asyncio.run(database.lease_next_source_sync_run(worker_id="worker-a")) is not None
            stopping = client.post(url)
            listed = client.get("/api/v1/sources").json()["data"]
            stranger = client.post(url, headers={"x-test-user": "stranger"})

        assert idle.status_code == 200, idle.text
        assert idle.json() == {
            "source_id": source_id,
            "outcome": "not_running",
            "run_id": None,
            "local_agent_job_ids": [],
        }
        assert stopped.json()["outcome"] == "stopped"
        assert stopped.json()["run_id"] == pending.run_id
        assert stopping.json()["outcome"] == "stopping"
        assert stopping.json()["run_id"] == running.run_id
        [row] = [row for row in listed if row["id"] == source_id]
        assert row["sync"]["run_id"] == running.run_id
        assert row["sync"]["stop_requested_at"] is not None
        assert stranger.status_code == 403, stranger.text
    finally:
        asyncio.run(database.close())


def test_pausing_stops_the_running_sync_and_a_change_names_the_way_out(tmp_path: Path):
    database = _connect_database(tmp_path)
    try:
        with TestClient(_app(tmp_path, database)) as client:
            payload = _confluence_payload()
            source_id = client.post("/api/v1/sources", json=payload).json()["id"]
            run = asyncio.run(database.enqueue_source_sync_run(source_id=source_id, workspace_id="default"))
            assert asyncio.run(database.lease_next_source_sync_run(worker_id="worker-a")) is not None
            renamed = client.put(f"/api/v1/sources/{source_id}", json={"name": "Renamed Wiki"})
            paused = client.put(f"/api/v1/sources/{source_id}", json={"status": "paused"})

        assert renamed.status_code == 409, renamed.text
        detail = renamed.json()["detail"]
        assert detail["code"] == "source_sync_running"
        assert detail["run_id"] == run.run_id
        assert f"/api/v1/sources/{source_id}/sync/stop" in detail["message"]
        assert paused.status_code == 200, paused.text
        stopping = asyncio.run(database.get_source_sync_run(run.run_id))
        assert stopping.stop_requested_at is not None
        assert asyncio.run(database.get_source(source_id))["status"] == "paused"
    finally:
        asyncio.run(database.close())


def test_a_stopped_device_job_fences_the_daemon_that_is_collecting_it(tmp_path: Path):
    database = _connect_database(tmp_path)
    headers = {"x-test-user": "owner-a", "x-test-workspace-role": "member"}
    try:
        with TestClient(_app(tmp_path, database)) as client:
            source_id = client.post(
                "/api/v1/sources", headers=headers, json=_jira_payload(sync_mode="local_agent")
            ).json()["id"]
            queued = client.post(f"/api/v1/sources/{source_id}/sync", headers=headers)
            [job] = client.post(
                "/api/cloud/local-agent/jobs/lease", headers=headers, json={"limit": 1, "lease_seconds": 60}
            ).json()["jobs"]

            stopped = client.post(f"/api/v1/sources/{source_id}/sync/stop", headers=headers)

            def heartbeat(job_id, attempt_count, lease_seconds, progress=None):
                response = client.post(
                    f"/api/cloud/local-agent/jobs/{job_id}/heartbeat",
                    headers=headers,
                    json={"attempt_count": attempt_count, "lease_seconds": lease_seconds, "progress": progress},
                )
                body = response.json()
                return body if response.is_success else {"status_code": response.status_code, **body}

            collector = _CloudJobLeaseHeartbeat(
                heartbeat=heartbeat,
                job_id=job["job_id"],
                attempt_count=job["attempt_count"],
                lease_seconds=60,
                interval_seconds=20,
            )
            collector._send_heartbeat()
            with pytest.raises(CloudJobLeaseLost):
                collector.report_progress({"schema_version": 1, "phase": "discovering", "progress": {"completed": 1}})
            lease_context = {
                "local_agent_job_id": job["job_id"],
                "local_agent_attempt_count": job["attempt_count"],
            }
            processed = client.post(f"/api/v1/sources/{source_id}/process", headers=headers, json=lease_context)
            completed = client.post(
                f"/api/cloud/local-agent/jobs/{job['job_id']}/complete",
                headers=headers,
                json={"attempt_count": job["attempt_count"], "status": "succeeded", "result": {}},
            )
            leased_again = client.post(
                "/api/cloud/local-agent/jobs/lease", headers=headers, json={"limit": 1, "lease_seconds": 60}
            ).json()["jobs"]

        assert queued.status_code == 202, queued.text
        assert stopped.status_code == 200, stopped.text
        assert stopped.json() == {
            "source_id": source_id,
            "outcome": "stopped",
            "run_id": None,
            "local_agent_job_ids": [job["job_id"]],
        }
        assert processed.status_code == 409, processed.text
        assert processed.json()["detail"] == "local_agent_lease_not_current"
        assert completed.json()["status"] == "failed"
        assert leased_again == []
        stored = asyncio.run(database.get_local_agent_job(job["job_id"]))
        assert (stored["status"], stored["last_error"]) == ("failed", SOURCE_SYNC_CANCELLED_BY_USER)
    finally:
        asyncio.run(database.close())
