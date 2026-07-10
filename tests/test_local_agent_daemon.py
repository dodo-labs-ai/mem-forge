from __future__ import annotations

import json
from datetime import datetime, timezone

from click.testing import CliRunner

import memforge.main as main
from memforge.local_agent.runner import LocalAgentRunner
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
    assert list(tmp_path.glob("state.json.corrupt-*"))


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
        cloud_job_handler=lambda job: handled.append(job["job_id"])
        or {"count": 1},
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
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: completed.append(
            (job_id, attempt_count, status, result, error)
        )
        or {"ok": True},
        cloud_job_heartbeat=lambda job_id, attempt_count, lease_seconds: heartbeats.append(
            (job_id, attempt_count, lease_seconds)
        )
        or {"ok": True},
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert handled == ["laj-1"]
    assert heartbeats == [("laj-1", 1, 60)]
    assert completed == [("laj-1", 1, "succeeded", {"count": 1}, None)]
    assert report["counts"] == {"total": 2, "success": 2, "failed": 0}


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
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: completed.append(
            (job_id, attempt_count, status, result, error)
        )
        or {"ok": True},
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


def test_local_agent_completion_failure_does_not_abort_following_job(tmp_path):
    completed: list[str] = []

    def complete(job_id, attempt_count, status, result, error=None):
        if job_id == "laj-1":
            raise RuntimeError("completion unavailable")
        completed.append(job_id)
        return {"ok": True}

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {"job": job["job_id"]},
        cloud_jobs_provider=lambda: {
            "jobs": [
                {"job_id": "laj-1", "source_id": "src-1", "attempt_count": 1},
                {"job_id": "laj-2", "source_id": "src-2", "attempt_count": 1},
            ]
        },
        cloud_job_completer=complete,
    )

    report = runner.run_once(now=datetime(2026, 7, 10, tzinfo=timezone.utc))

    assert completed == ["laj-2"]
    assert report["counts"] == {"total": 3, "success": 2, "failed": 1}
    assert report["results"][1]["error"] == "completion unavailable"
    assert report["results"][2]["status"] == "success"


def test_local_agent_rejects_job_without_attempt_count(tmp_path):
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {"job": job["job_id"]},
        cloud_jobs_provider=lambda: {
            "jobs": [{"job_id": "laj-1", "source_id": "src-1"}]
        },
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
                "results": [
                    {"task_id": "cloud-job:laj-1", "status": "failed", "error": "push failed"}
                ],
            }

    monkeypatch.setattr(main, "_build_local_agent_runner", lambda *args, **kwargs: FakeRunner())

    result = CliRunner().invoke(cli, ["adapter", "daemon", "once"])

    assert result.exit_code == 1
    assert json.loads(result.output)["counts"]["failed"] == 1


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

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status", "--verbose"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"]["tasks"]["cloud-jobs:lease"]["run_count"] == 2
    assert payload["summary"]["last_cloud_job_lease"]["error"] == "cloud unavailable"
