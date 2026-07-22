from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from click.testing import CliRunner
import pytest

import memforge.main as main
import memforge.local_agent.runner as local_agent_runner
from memforge.local_agent.runner import CloudJobLeaseLost, LocalAgentRunner, _CloudJobLeaseHeartbeat
from memforge.local_agent.source_contract import SourceSyncRunReceiptError
from memforge.local_agent.state import LocalAgentStateStore
from memforge.main import cli


def test_local_agent_state_records_result_and_daemon_heartbeat(tmp_path):
    store = LocalAgentStateStore(tmp_path / "state.json")
    store.record_result(
        "cloud-jobs:lease",
        {
            "task_id": "cloud-jobs:lease",
            "status": "success",
            "finished_at": "2026-07-10T00:00:00+00:00",
            "payload": {"leased_count": 1},
        },
    )
    store.record_daemon_heartbeat(
        pid=123,
        started_at="2026-07-10T00:00:00+00:00",
        command=["memforge", "adapter", "daemon", "run"],
        target={"api_url": "https://memforge.example.test"},
    )

    state = store.load()
    assert state["tasks"]["cloud-jobs:lease"]["run_count"] == 1
    assert state["tasks"]["cloud-jobs:lease"]["last_status"] == "success"
    assert state["daemon"]["pid"] == 123
    assert state["daemon"]["target"] == {"api_url": "https://memforge.example.test"}


def test_local_agent_state_quarantines_corrupt_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("not-json", encoding="utf-8")

    state = LocalAgentStateStore(path).load()

    assert state == {"version": 1, "tasks": {}}
    assert not path.exists()
    corrupt_files = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].read_text(encoding="utf-8") == "not-json"


def test_local_agent_state_quarantines_version_mismatch(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    state_path = tmp_path / "agent-state.json"
    state_path.write_text(json.dumps({"version": 999, "tasks": {"old": {}}}), encoding="utf-8")

    payload = LocalAgentStateStore(state_path).load()

    assert payload == {"version": 1, "tasks": {}}
    assert not state_path.exists()
    assert len(list(tmp_path.glob("agent-state.json.corrupt-*"))) == 1


def test_local_agent_state_compacts_large_payloads(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    store = LocalAgentStateStore(tmp_path / "agent-state.json")

    payload = store.record_result(
        "github:arch",
        {
            "status": "success",
            "started_at": "2026-07-07T01:00:00+00:00",
            "finished_at": "2026-07-07T01:00:02+00:00",
            "payload": {
                "profile": "arch",
                "counts": {"pushed": 1000, "failed": 1},
                "pushed": [{"relative_path": f"doc-{index}.md"} for index in range(1000)],
                "failed": [{"relative_path": "bad.md", "error": "push failed"}],
            },
        },
    )

    stored = payload["tasks"]["github:arch"]["last_result"]["payload"]
    assert "pushed" not in stored
    assert "failed" not in stored
    assert stored["pushed_count"] == 1000
    assert stored["failed_count"] == 1
    assert stored["first_failed"] == {"relative_path": "bad.md", "error": "push failed"}


def test_local_agent_state_preserves_sync_audit_summary(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    store = LocalAgentStateStore(tmp_path / "agent-state.json")

    payload = store.record_result(
        "cloud-job:laj-teams",
        {
            "task_id": "cloud-job:laj-teams",
            "kind": "cloud_job",
            "status": "success",
            "started_at": "2026-07-09T00:00:00+00:00",
            "finished_at": "2026-07-09T00:00:01+00:00",
            "error": None,
            "payload": {
                "source_id": "src-teams",
                "counts": {"selected": 1, "pushed": 1, "failed": 0, "skipped_existing": 0, "polls": 1},
                "pushed": [{"window_id": "teams-thread:v1:opaque", "document_hash": "hash"}],
                "failed": [],
                "skipped_existing": [],
                "sync_started": True,
                "source_sync_run_id": "run-teams-1",
                "audit_log_path": "/Users/example/.memforge/teams-sync-audit.jsonl",
            },
        },
    )

    stored = payload["tasks"]["cloud-job:laj-teams"]["last_result"]["payload"]
    assert stored["sync_started"] is True
    assert stored["source_sync_run_id"] == "run-teams-1"
    assert stored["audit_log_path"].endswith("teams-sync-audit.jsonl")
    assert stored["pushed_count"] == 1
    assert stored["failed_count"] == 0
    assert stored["skipped_existing_count"] == 0


def test_local_agent_state_records_running_without_incrementing_run_count(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    store = LocalAgentStateStore(tmp_path / "agent-state.json")
    store.record_result(
        "cloud-job:laj-teams",
        {
            "task_id": "cloud-job:laj-teams",
            "kind": "cloud_job",
            "status": "failed",
            "started_at": "2026-07-09T00:00:00+00:00",
            "finished_at": "2026-07-09T00:00:01+00:00",
            "error": "old failure",
        },
    )

    payload = store.record_running(
        "cloud-job:laj-teams",
        {
            "task_id": "cloud-job:laj-teams",
            "kind": "cloud_job",
            "status": "running",
            "started_at": "2026-07-09T01:00:00+00:00",
            "payload": {"source_id": "src-teams", "operation": "teams_sync"},
        },
    )

    task = payload["tasks"]["cloud-job:laj-teams"]
    assert task["run_count"] == 1
    assert task["last_status"] == "running"
    assert task["last_started_at"] == "2026-07-09T01:00:00+00:00"
    assert task["last_finished_at"] is None
    assert task["last_error"] is None
    assert task["last_result"]["payload"] == {
        "source_id": "src-teams",
        "operation": "teams_sync",
    }


def test_local_adapter_capability_commands_include_teams():
    runner = CliRunner()

    listed = runner.invoke(cli, ["adapter", "list"])
    status = runner.invoke(cli, ["adapter", "status"])

    assert listed.exit_code == 0
    assert status.exit_code == 0
    listed_payload = json.loads(listed.output)
    status_payload = json.loads(status.output)
    assert any(item["type"] == "teams" for item in listed_payload["data"])
    assert "teams.auth" in status_payload["capabilities"]
    assert "teams.browse" in status_payload["capabilities"]
    assert "teams.sync" in status_payload["capabilities"]


def test_teams_browse_job_reauths_when_no_local_session(monkeypatch):
    calls = {"browse": 0}
    auth_jobs: list[dict] = []

    async def fake_browse_teams_conversations(*, region: str):
        calls["browse"] += 1
        if calls["browse"] == 1:
            raise ValueError("No Teams session found.")
        return {"favorites": [], "teams": [], "group_chats": [], "individual_chats": []}

    monkeypatch.setattr(
        "memforge.local_agent.teams_browse.browse_teams_conversations",
        fake_browse_teams_conversations,
    )
    monkeypatch.setattr(main, "_current_teams_chat_token_hashes", lambda: {"old-token-hash"})

    def fake_auth(job):
        auth_jobs.append(job)
        return {"operation": "teams_auth", "authenticated": True, "region": "emea", "token_count": 1}

    monkeypatch.setattr(main, "_run_cloud_teams_auth_job", fake_auth)

    result = main._run_cloud_teams_browse_job(
        {
            "job_id": "laj-browse",
            "operation": "teams_browse",
            "payload": {"region": "emea", "wait_seconds": 1},
        }
    )

    assert result == {
        "operation": "teams_browse",
        "region": "emea",
        "favorites": [],
        "teams": [],
        "group_chats": [],
        "individual_chats": [],
    }
    assert calls["browse"] == 2
    assert auth_jobs[0]["operation"] == "teams_auth"
    assert auth_jobs[0]["payload"]["rejected_token_hashes"] == ["old-token-hash"]


def test_teams_sync_job_reauths_when_no_local_session(monkeypatch, tmp_path):
    calls = {"collect": 0}
    auth_jobs: list[dict] = []

    async def fake_collect(job, *, source_id: str, limit: int, report_progress=None):
        calls["collect"] += 1
        if calls["collect"] == 1:
            raise ValueError("No Teams session found.")
        return {"documents": [], "poll_audits": []}

    def fake_auth(job):
        auth_jobs.append(job)
        return {"operation": "teams_auth", "authenticated": True, "region": "emea", "token_count": 1}

    class FakeClient:
        def for_workspace(self, workspace_id: str):
            assert workspace_id == "workspace-a"
            return self

        def get_source_projection_inventory(self, source_id: str, **filters):
            del filters
            assert source_id == "src-teams"
            return {"units": []}

        def prepare_local_source_snapshot(self, **kwargs):
            assert kwargs["coverage"] == "bounded_delta"
            assert kwargs["items"] == []
            return {"required_doc_ids": [], "reused_count": 0}

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect)
    monkeypatch.setattr(main, "_run_cloud_teams_auth_job", fake_auth)
    monkeypatch.setattr(main, "_current_teams_chat_token_hashes", lambda: set())

    result = main._run_cloud_teams_sync_job(
        {
            "job_id": "laj-sync",
            "attempt_count": 1,
            "operation": "teams_sync",
            "source_id": "src-teams",
            "workspace_id": "workspace-a",
            "payload": {
                "source_id": "src-teams",
                "workspace_id": "workspace-a",
                "audit_log_path": str(tmp_path / "teams-audit.jsonl"),
                "ledger_state_path": str(tmp_path / "teams-ledger.json"),
                "wait_seconds": 1,
                "conversation_ids": ["19:conversation@thread.tacv2"],
            },
        },
        FakeClient(),
    )

    assert result["operation"] == "teams_sync"
    assert result["source_id"] == "src-teams"
    assert result["counts"] == {
        "selected": 0,
        "pushed": 0,
        "failed": 0,
        "skipped_existing": 0,
        "polls": 0,
    }
    assert result["sync_started"] is False
    assert calls["collect"] == 2
    assert auth_jobs[0]["operation"] == "teams_auth"


def test_local_agent_state_records_daemon_heartbeat(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    store = LocalAgentStateStore(tmp_path / "agent-state.json")

    payload = store.record_daemon_heartbeat(
        pid=12345,
        started_at="2026-07-09T00:00:00+00:00",
        command=["memforge", "adapter", "daemon", "run"],
        target={
            "api_url": "https://memforge.example.test",
            "api_token_configured": True,
        },
    )

    assert payload["daemon"]["pid"] == 12345
    assert payload["daemon"]["started_at"] == "2026-07-09T00:00:00+00:00"
    assert payload["daemon"]["command"] == ["memforge", "adapter", "daemon", "run"]
    assert payload["daemon"]["target"]["api_token_configured"] is True
    assert payload["daemon"]["updated_at"]
    assert LocalAgentStateStore(tmp_path / "agent-state.json").load()["daemon"]["pid"] == 12345


def test_local_agent_daemon_lock_prevents_duplicate_runners(tmp_path):
    lock_path = tmp_path / "daemon.lock"

    first = main._acquire_local_agent_daemon_lock(lock_path)
    second = main._acquire_local_agent_daemon_lock(lock_path)

    assert first is not None
    assert second is None
    first.close()
    assert not lock_path.exists()


def test_local_agent_leases_heartbeats_and_completes_cloud_job(tmp_path):
    completed: list[tuple[str, int, str, dict, str | None]] = []
    heartbeats: list[tuple[str, int, int]] = []
    handled: list[str] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: handled.append(job["job_id"]) or {"count": 1},
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-1",
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completed.append((job_id, attempt_count, status, result, error)) or {"ok": True}
        ),
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds: (
            heartbeats.append((job_id, attempt_count, lease_seconds)) or {"ok": True}
        ),
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert handled == ["laj-1"]
    assert heartbeats == [("laj-1", 1, 60)]
    assert completed == [("laj-1", 1, "succeeded", {"count": 1}, None)]
    assert report["counts"] == {"total": 2, "success": 2, "failed": 0}


def test_local_agent_leases_only_the_job_it_is_ready_to_execute(tmp_path):
    available = [
        {"job_id": "laj-long-1", "source_id": "src-1", "attempt_count": 1},
        {"job_id": "laj-long-2", "source_id": "src-2", "attempt_count": 1},
    ]
    lease_limits: list[int] = []
    handled: list[str] = []

    def lease_jobs(*, limit, wait_seconds, lease_seconds):
        lease_limits.append(limit)
        leased = available[:limit]
        del available[:limit]
        return {"jobs": leased}

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: handled.append(job["job_id"]) or {},
        cloud_jobs_provider=lease_jobs,
        cloud_job_completer=lambda *args: {"ok": True},
        cloud_job_heartbeat=lambda *args: {"ok": True},
    )

    first = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))
    second = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert lease_limits == [1, 1]
    assert handled == ["laj-long-1", "laj-long-2"]
    assert first["results"][0]["payload"] == {"leased_count": 1}
    assert second["results"][0]["payload"] == {"leased_count": 1}


def test_local_agent_rejected_initial_heartbeat_never_runs_handler(tmp_path):
    completed: list[tuple[str, int, str, dict, str | None]] = []
    handled: list[str] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: handled.append(job["job_id"]) or {"count": 1},
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-legacy",
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completed.append((job_id, attempt_count, status, result, error)) or {"ok": True}
        ),
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds: {
            "error": "MemForge API request failed",
            "status_code": 409,
            "detail": '{"detail":"local_agent_source_activity_epoch_required"}',
        },
    )

    report = runner.run_once(now=datetime(2026, 7, 16, tzinfo=timezone.utc))

    assert handled == []
    assert completed == []
    assert report["results"][-1]["status"] == "failed"
    assert report["results"][-1]["error_type"] == "CloudJobLeaseLost"
    assert "local_agent_source_activity_epoch_required" in report["results"][-1]["error"]


def test_local_agent_forwards_retryable_handler_failure_to_broker(tmp_path):
    completed: list[tuple[str, int, str, dict, str | None]] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {
            "error": "source processing failed to start",
            "retryable": True,
        },
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-teams-retry",
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completed.append((job_id, attempt_count, status, result, error)) or {"ok": True}
        ),
    )

    runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert completed == [
        (
            "laj-teams-retry",
            1,
            "failed",
            {"error": "source processing failed to start", "retryable": True},
            "source processing failed to start",
        )
    ]


def test_local_agent_treats_missing_source_run_receipt_as_retryable(tmp_path):
    completed: list[tuple[str, int, str, dict, str | None]] = []

    def missing_receipt(_job):
        raise SourceSyncRunReceiptError("successful source processing response omitted run_id")

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=missing_receipt,
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-missing-receipt",
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completed.append((job_id, attempt_count, status, result, error)) or {"ok": True}
        ),
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert completed[0][2:4] == ("failed", {"retryable": True})
    assert "omitted run_id" in str(completed[0][4])
    assert report["results"][-1]["status"] == "failed"


def test_local_agent_reports_handler_progress_through_heartbeat(tmp_path):
    heartbeats: list[dict | None] = []
    completions: list[dict] = []

    def handle(job, *, report_progress):
        report_progress(
            {
                "schema_version": 1,
                "phase": "uploading",
                "progress": {"completed": 7, "total": 16, "unit": "message"},
            }
        )
        return {"count": 1}

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=handle,
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-progress",
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds, progress=None: (
            heartbeats.append(progress) or {"ok": True}
        ),
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completions.append(result) or {"ok": True}
        ),
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert report["counts"]["failed"] == 0
    assert heartbeats == [
        None,
        {
            "schema_version": 1,
            "phase": "uploading",
            "progress": {"completed": 7, "total": 16, "unit": "message"},
        },
    ]
    assert completions == [
        {
            "count": 1,
            "progress": {
                "schema_version": 1,
                "phase": "uploading",
                "progress": {"completed": 7, "total": 16, "unit": "message"},
            },
        }
    ]


def test_local_agent_flushes_dirty_progress_before_lease_heartbeat():
    flushed = threading.Event()

    def heartbeat(job_id, attempt_count, lease_seconds, progress=None):
        if progress is not None:
            flushed.set()
        return {"ok": True}

    with _CloudJobLeaseHeartbeat(
        heartbeat=heartbeat,
        job_id="laj-progress-cadence",
        attempt_count=1,
        lease_seconds=60,
        interval_seconds=20,
        progress_flush_seconds=1,
    ) as lease:
        lease.report_progress(
            {
                "schema_version": 1,
                "phase": "discovering",
                "progress": {"completed": 3, "unit": "file"},
            }
        )
        assert flushed.wait(1.5)


def test_local_agent_heartbeat_rejection_fences_further_handler_work():
    responses = iter(
        [
            {"ok": True},
            {
                "error": "MemForge API request failed",
                "status_code": 404,
                "detail": '{"detail":"local_agent_job_not_found"}',
            },
        ]
    )
    lease = _CloudJobLeaseHeartbeat(
        heartbeat=lambda *args, **kwargs: next(responses),
        job_id="laj-expired",
        attempt_count=1,
        lease_seconds=60,
        interval_seconds=20,
    )

    with pytest.raises(CloudJobLeaseLost, match="local_agent_job_not_found"):
        with lease:
            lease._send_heartbeat()
            lease.report_progress(
                {
                    "schema_version": 1,
                    "phase": "uploading",
                    "progress": {"completed": 555, "total": 555, "unit": "file"},
                }
            )


def test_local_agent_lease_loss_does_not_attempt_stale_completion(tmp_path):
    completed: list[tuple] = []

    def lost_lease(_job, *, report_progress):
        raise CloudJobLeaseLost("local_agent_job_not_found")

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lost_lease,
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-expired",
                    "operation": "github_repo_sync",
                    "source_id": "src-github",
                    "attempt_count": 1,
                }
            ]
        },
        cloud_job_completer=lambda *args: completed.append(args) or {"ok": True},
        cloud_job_heartbeat=lambda *args: {"ok": True},
    )

    report = runner.run_once(now=datetime(2026, 7, 22, tzinfo=timezone.utc))

    assert completed == []
    assert report["results"][-1]["status"] == "failed"
    assert report["results"][-1]["error_type"] == "CloudJobLeaseLost"


def test_local_agent_fences_itself_after_last_successful_lease_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(local_agent_runner.time, "monotonic", lambda: clock[0])

    def slow_heartbeat(*args, **kwargs):
        clock[0] += 10.0
        return {"ok": True}

    lease = _CloudJobLeaseHeartbeat(
        heartbeat=slow_heartbeat,
        job_id="laj-network-partition",
        attempt_count=1,
        lease_seconds=60,
        interval_seconds=20,
    )

    with pytest.raises(CloudJobLeaseLost, match="lease expired"):
        with lease:
            clock[0] = 160.5
            lease.report_progress(
                {
                    "schema_version": 1,
                    "phase": "uploading",
                    "progress": {"completed": 1, "total": 555, "unit": "file"},
                }
            )


def test_local_agent_completion_failure_does_not_abort_following_job(tmp_path):
    completed: list[str] = []
    available = [
        {"job_id": "laj-1", "source_id": "src-1", "attempt_count": 1},
        {"job_id": "laj-2", "source_id": "src-2", "attempt_count": 1},
    ]

    def complete(job_id, attempt_count, status, result, error=None):
        if job_id == "laj-1":
            raise RuntimeError("completion unavailable")
        completed.append(job_id)
        return {"ok": True}

    def lease_jobs(*, limit, **kwargs):
        leased = available[:limit]
        del available[:limit]
        return {"jobs": leased}

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {"job": job["job_id"]},
        cloud_jobs_provider=lease_jobs,
        cloud_job_completer=complete,
    )

    first = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))
    second = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert completed == ["laj-2"]
    assert first["counts"] == {"total": 2, "success": 1, "failed": 1}
    assert first["results"][1]["error"] == "completion unavailable"
    assert second["counts"] == {"total": 2, "success": 2, "failed": 0}


def test_local_agent_rejects_broker_batch_that_could_expire_before_execution(
    tmp_path,
):
    handled: list[str] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: handled.append(job["job_id"]) or {},
        cloud_jobs_provider=lambda: {
            "jobs": [
                {"job_id": "laj-1", "source_id": "src-1", "attempt_count": 1},
                {"job_id": "laj-2", "source_id": "src-2", "attempt_count": 1},
            ]
        },
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert handled == []
    assert report["counts"] == {"total": 1, "success": 0, "failed": 1}
    assert report["results"][0]["error_type"] == "CloudJobLeaseError"
    assert "more than one leased job" in report["results"][0]["error"]


def test_local_agent_rejects_job_without_attempt_count(tmp_path):
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {"job": job["job_id"]},
        cloud_jobs_provider=lambda: {"jobs": [{"job_id": "laj-1", "source_id": "src-1"}]},
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert report["counts"] == {"total": 2, "success": 1, "failed": 1}
    assert report["results"][1]["error"] == "cloud job is missing attempt_count"


def test_local_agent_run_forever_records_lease_errors(tmp_path):
    store = LocalAgentStateStore(tmp_path / "state.json")
    runner = LocalAgentRunner(
        state_store=store,
        cloud_job_handler=lambda job: {},
        cloud_jobs_provider=lambda: (_ for _ in ()).throw(RuntimeError("cloud unavailable")),
    )
    sleeps: list[float] = []

    runner.run_forever(
        poll_interval_seconds=7,
        stop_after_iterations=2,
        sleep=sleeps.append,
    )

    assert sleeps == [7]
    assert store.load()["tasks"]["cloud-jobs:lease"]["last_error"] == "cloud unavailable"


def test_local_agent_records_running_before_job_handler(tmp_path):
    store = LocalAgentStateStore(tmp_path / "state.json")

    def handle(job):
        running = store.load()["tasks"]["cloud-job:laj-running"]
        assert running["last_status"] == "running"
        assert running["last_result"]["payload"]["operation"] == "teams_sync"
        return {"ok": True}

    runner = LocalAgentRunner(
        state_store=store,
        cloud_job_handler=handle,
        cloud_job_completer=lambda *args: {"ok": True},
    )

    result = runner._run_cloud_job(
        {
            "job_id": "laj-running",
            "attempt_count": 1,
            "operation": "teams_sync",
            "source_id": "src-teams",
        },
        datetime(2026, 7, 10, tzinfo=timezone.utc),
    )

    assert result["status"] == "success"


def test_adapter_daemon_once_exits_nonzero_when_job_fails(monkeypatch):
    class FakeRunner:
        def run_once(self) -> dict:
            return {
                "status": "ok",
                "counts": {"total": 1, "success": 0, "failed": 1},
                "results": [{"task_id": "cloud-job:laj-1", "status": "failed", "error": "push failed"}],
            }

    monkeypatch.setattr(main, "_build_local_agent_runner", lambda *args, **kwargs: FakeRunner())

    result = CliRunner().invoke(cli, ["adapter", "daemon", "once"])

    assert result.exit_code == 1
    assert json.loads(result.output)["counts"]["failed"] == 1


def test_adapter_daemon_status_uses_environment_target_provenance(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_LOCK", str(tmp_path / "daemon.lock"))
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    monkeypatch.setenv("MEMFORGE_API_URL", "https://environment.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "ws-environment")
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.setenv("SAP_TOKEN", "inactive-profile-token")
    (tmp_path / "cli.toml").write_text(
        'active = "sap"\n\n[targets.sap]\n'
        'api_url = "https://profile.hana.ondemand.com"\nworkspace_id = "ws-profile"\n'
        'token_env = "SAP_TOKEN"\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target"] == {
        "edition": "cloud",
        "api_url": "https://environment.hana.ondemand.com",
        "workspace_id": "ws-environment",
        "active_target": "",
        "token_env": "MEMFORGE_API_TOKEN",
        "api_token_configured": False,
    }
    assert payload["recommendations"] == ["Set MEMFORGE_API_TOKEN before starting the daemon."]


def test_adapter_daemon_status_allows_cloud_target_without_global_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_LOCK", str(tmp_path / "daemon.lock"))
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "daemon-token")
    (tmp_path / "cli.toml").write_text(
        'active = "cloud"\n\n[targets.cloud]\n'
        'api_url = "https://memforge-dev.cfapps.eu12.hana.ondemand.com"\n'
        'token_env = "MEMFORGE_API_TOKEN"\n',
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status"])

    assert result.exit_code == 0, result.output
    target = json.loads(result.output)["target"]
    assert target["edition"] == "cloud"
    assert target["workspace_id"] is None
    assert target["api_token_configured"] is True


def test_adapter_daemon_status_verbose_includes_raw_state(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": {
                    "cloud-jobs:lease": {
                        "run_count": 2,
                        "last_status": "failed",
                        "last_error": "cloud unavailable",
                        "updated_at": "2026-07-10T01:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_STATE", str(state_path))
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_LOCK", str(tmp_path / "daemon.lock"))
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status", "--verbose"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"]["tasks"]["cloud-jobs:lease"]["run_count"] == 2
    assert payload["summary"]["last_cloud_job_lease"]["error"] == "cloud unavailable"
