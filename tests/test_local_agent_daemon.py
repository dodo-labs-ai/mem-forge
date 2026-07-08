from __future__ import annotations

from datetime import datetime, timezone
import json
import re

from click.testing import CliRunner

import memforge.main as main
from memforge.main import cli


def test_local_agent_state_records_task_result(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    state_path = tmp_path / "agent-state.json"
    store = LocalAgentStateStore(state_path)

    payload = store.record_result(
        "kb:notes",
        {
            "status": "success",
            "started_at": "2026-07-07T01:00:00+00:00",
            "finished_at": "2026-07-07T01:00:02+00:00",
            "counts": {"pushed": 2, "failed": 0},
        },
    )

    assert payload["tasks"]["kb:notes"]["run_count"] == 1
    assert payload["tasks"]["kb:notes"]["last_status"] == "success"
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert LocalAgentStateStore(state_path).load()["tasks"]["kb:notes"]["last_result"]["counts"]["pushed"] == 2


def test_local_agent_state_quarantines_corrupt_json(tmp_path):
    from memforge.local_agent.state import LocalAgentStateStore

    state_path = tmp_path / "agent-state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    payload = LocalAgentStateStore(state_path).load()

    assert payload == {"version": 1, "tasks": {}}
    assert not state_path.exists()
    corrupt_files = list(tmp_path.glob("agent-state.json.corrupt-*"))
    assert len(corrupt_files) == 1
    assert corrupt_files[0].read_text(encoding="utf-8") == "{not-json"


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


def test_local_agent_discovers_linked_profiles_and_jira_origins():
    from memforge.local_agent.tasks import discover_jira_auth_tasks, discover_profile_tasks

    adapter_config = {
        "kb": {
            "notes": {"root": "/repo", "vault_id": "notes", "source_id": "src-notes"},
            "draft": {"root": "/draft", "vault_id": "draft"},
        },
        "github": {
            "arch": {
                "repo_url": "https://github.example/org/repo",
                "source_id": "src-arch",
                "repo_path": "/clone",
                "daemon_interval_seconds": 0,
            }
        },
    }

    tasks = discover_profile_tasks(adapter_config, default_interval_seconds=900)
    jira_tasks = discover_jira_auth_tasks(
        {"origins": [{"origin": "https://jira.tools.sap", "configured": True, "status": "active"}]},
        default_interval_seconds=1800,
    )

    assert [(task.task_id, task.kind, task.profile_name, task.interval_seconds) for task in tasks] == [
        ("github:arch", "github_sync", "arch", 900),
        ("kb:notes", "kb_sync", "notes", 900),
    ]
    assert [(task.task_id, task.kind, task.origin) for task in jira_tasks] == [
        ("jira-auth:https://jira.tools.sap", "jira_auth", "https://jira.tools.sap")
    ]


def test_local_agent_once_isolates_profile_failures(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    adapter_config = {
        "kb": {"notes": {"root": "/repo", "vault_id": "notes", "source_id": "src-notes"}},
        "github": {"arch": {"repo_url": "https://github.example/org/repo", "source_id": "src-arch"}},
    }
    calls: list[tuple[str, str | None]] = []

    def run_kb(name: str) -> dict:
        calls.append(("kb", name))
        raise RuntimeError("folder unavailable")

    def run_github(name: str) -> dict:
        calls.append(("github", name))
        return {"counts": {"pushed": 1, "failed": 0}}

    runner = LocalAgentRunner(
        adapter_config=adapter_config,
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=run_kb,
            run_github_profile=run_github,
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged", "cookie_hash": last_hash},
        ),
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert calls == [("github", "arch"), ("kb", "notes")]
    assert report["counts"] == {"total": 2, "success": 1, "failed": 1, "skipped": 0}
    assert report["results"][0]["status"] == "success"
    assert report["results"][1]["status"] == "failed"
    assert "folder unavailable" in report["results"][1]["error"]


def test_local_agent_reloads_adapter_config_each_run(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    configs = [
        {"github": {}},
        {"github": {"arch": {"repo_url": "https://github.example/org/repo", "source_id": "src-arch"}}},
    ]
    last_config = configs[-1]
    calls: list[str] = []

    def adapter_config_provider() -> dict:
        return configs.pop(0) if configs else last_config

    runner = LocalAgentRunner(
        adapter_config={},
        adapter_config_provider=adapter_config_provider,
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: calls.append(name) or {"counts": {"pushed": 1}},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
    )

    first = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)
    second = runner.run_once(now=datetime(2026, 7, 7, 1, tzinfo=timezone.utc), include_jira=False)

    assert first["counts"]["total"] == 0
    assert second["counts"] == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert calls == ["arch"]


def test_local_agent_marks_error_payloads_as_failed(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    runner = LocalAgentRunner(
        adapter_config={"github": {"arch": {"repo_url": "https://github.example/org/repo", "source_id": "src-arch"}}},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {"error": "one or more documents failed to push", "failed": ["bad.md"]},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert report["counts"] == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"] == "one or more documents failed to push"


def test_local_agent_leases_runs_and_completes_cloud_jobs(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    completed: list[tuple[str, int, str, dict, str | None]] = []
    handled_jobs: list[dict] = []

    def run_cloud_job(job: dict) -> dict:
        handled_jobs.append(job)
        return {"count": 1, "items": [{"path": "README.md"}]}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
            run_cloud_job=run_cloud_job,
        ),
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "laj-1",
                    "operation": "github_repo_preview_tree",
                    "source_id": "src-gh",
                    "attempt_count": 1,
                    "payload": {"repo_url": "https://github.example/org/repo"},
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: completed.append(
            (job_id, attempt_count, status, result, error)
        )
        or {"ok": True},
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert handled_jobs == [
        {
            "job_id": "laj-1",
            "operation": "github_repo_preview_tree",
            "source_id": "src-gh",
            "attempt_count": 1,
            "payload": {"repo_url": "https://github.example/org/repo"},
        }
    ]
    assert completed == [
        (
            "laj-1",
            1,
            "succeeded",
            {"count": 1, "items": [{"path": "README.md"}]},
            None,
        )
    ]
    assert report["counts"] == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert report["results"][0]["task_id"] == "cloud-job:laj-1"


def test_local_agent_cloud_job_loop_uses_long_poll_without_profile_tasks(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    lease_requests: list[dict] = []
    profile_calls: list[str] = []

    def lease_cloud_jobs(*, wait_seconds: int = 0) -> dict:
        lease_requests.append({"wait_seconds": wait_seconds})
        return {"jobs": []}

    runner = LocalAgentRunner(
        adapter_config={
            "kb": {"notes": {"root": "/repo", "vault_id": "notes", "source_id": "src-notes"}},
        },
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: profile_calls.append(name) or {"counts": {"pushed": 1}},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        cloud_jobs_provider=lease_cloud_jobs,
    )

    report = runner.run_cloud_jobs_once(
        now=datetime(2026, 7, 7, tzinfo=timezone.utc),
        wait_seconds=25,
    )

    assert report["counts"] == {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    assert lease_requests == [{"wait_seconds": 25}]
    assert profile_calls == []


def test_local_agent_cloud_job_loop_sleeps_after_lease_failure(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    sleeps: list[float] = []

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        cloud_jobs_provider=lambda wait_seconds=0: (_ for _ in ()).throw(RuntimeError("cloud unavailable")),
    )

    runner.run_forever(
        include_jira=False,
        poll_interval_seconds=7,
        cloud_job_wait_seconds=25,
        stop_after_iterations=2,
        sleep=sleeps.append,
    )

    assert sleeps == [7]


def test_local_agent_cloud_job_loop_sleeps_after_scheduled_task_failure(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    sleeps: list[float] = []

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        cloud_jobs_provider=lambda wait_seconds=0: {"jobs": []},
    )

    def fail_scheduled_tasks(**kwargs) -> list[dict]:
        raise RuntimeError("scheduled task discovery failed")

    runner._run_scheduled_tasks = fail_scheduled_tasks  # type: ignore[method-assign]

    runner.run_forever(
        include_jira=False,
        poll_interval_seconds=7,
        cloud_job_wait_seconds=25,
        stop_after_iterations=2,
        sleep=sleeps.append,
    )

    assert sleeps == [7]


def test_local_agent_cloud_lease_failure_does_not_abort_profile_tasks(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    calls: list[str] = []

    runner = LocalAgentRunner(
        adapter_config={"github": {"arch": {"repo_url": "https://github.example/org/repo", "source_id": "src-arch"}}},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: calls.append(name) or {"counts": {"pushed": 1}},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        cloud_jobs_provider=lambda: (_ for _ in ()).throw(RuntimeError("cloud unavailable")),
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert calls == ["arch"]
    assert report["counts"] == {"total": 2, "success": 1, "failed": 1, "skipped": 0}
    assert report["results"][1]["task_id"] == "cloud-jobs:lease"
    assert report["results"][1]["kind"] == "cloud_job"
    assert report["results"][1]["status"] == "failed"
    assert report["results"][1]["error"] == "cloud unavailable"


def test_local_agent_cloud_lease_error_response_is_reported(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        cloud_jobs_provider=lambda: {
            "error": "MemForge API request failed",
            "status_code": 401,
            "detail": "invalid token",
        },
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert report["counts"] == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert report["results"][0]["task_id"] == "cloud-jobs:lease"
    assert report["results"][0]["error_type"] == "CloudJobLeaseError"
    assert report["results"][0]["error"] == "MemForge API request failed: status_code=401: invalid token"


def test_local_agent_cloud_completion_failure_does_not_abort_following_jobs(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    completed: list[str] = []

    def complete(job_id: str, attempt_count: int, status: str, result: dict, error: str | None = None) -> dict:
        if job_id == "laj-1":
            raise RuntimeError("completion unavailable")
        completed.append(f"{job_id}:{attempt_count}")
        return {"ok": True}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
            run_cloud_job=lambda job: {"job": job["job_id"]},
        ),
        cloud_jobs_provider=lambda: {
            "jobs": [
                {"job_id": "laj-1", "source_id": "src-1", "attempt_count": 1},
                {"job_id": "laj-2", "source_id": "src-2", "attempt_count": 2},
            ]
        },
        cloud_job_completer=complete,
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert completed == ["laj-2:2"]
    assert report["counts"] == {"total": 2, "success": 1, "failed": 1, "skipped": 0}
    assert report["results"][0]["task_id"] == "cloud-job:laj-1"
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"] == "completion unavailable"
    assert report["results"][1]["task_id"] == "cloud-job:laj-2"
    assert report["results"][1]["status"] == "success"


def test_local_agent_cloud_completion_error_response_does_not_abort_following_jobs(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    def complete(job_id: str, attempt_count: int, status: str, result: dict, error: str | None = None) -> dict:
        if job_id == "laj-1":
            return {
                "error": "MemForge API request failed",
                "status_code": 404,
                "detail": "stale lease",
            }
        return {"ok": True}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
            run_cloud_job=lambda job: {"job": job["job_id"]},
        ),
        cloud_jobs_provider=lambda: {
            "jobs": [
                {"job_id": "laj-1", "source_id": "src-1", "attempt_count": 1},
                {"job_id": "laj-2", "source_id": "src-2", "attempt_count": 2},
            ]
        },
        cloud_job_completer=complete,
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert report["counts"] == {"total": 2, "success": 1, "failed": 1, "skipped": 0}
    assert report["results"][0]["task_id"] == "cloud-job:laj-1"
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error_type"] == "CloudJobCompletionError"
    assert report["results"][0]["error"] == "MemForge API request failed: status_code=404: stale lease"
    assert report["results"][1]["task_id"] == "cloud-job:laj-2"
    assert report["results"][1]["status"] == "success"


def test_local_agent_cloud_job_without_attempt_count_is_rejected_locally(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
            run_cloud_job=lambda job: {"job": job["job_id"]},
        ),
        cloud_jobs_provider=lambda: {"jobs": [{"job_id": "laj-1", "source_id": "src-1"}]},
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: {"ok": True},
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=False)

    assert report["counts"] == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert report["results"][0]["task_id"] == "cloud-job:laj-1"
    assert report["results"][0]["error"] == "cloud job is missing attempt_count"


def test_local_agent_failed_tasks_retry_after_short_backoff(tmp_path):
    from datetime import timedelta

    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    state_store = LocalAgentStateStore(tmp_path / "state.json")
    state_store.record_result(
        "github:arch",
        {
            "task_id": "github:arch",
            "kind": "github_sync",
            "status": "failed",
            "started_at": "2026-07-07T00:00:00+00:00",
            "finished_at": "2026-07-07T00:00:01+00:00",
            "error": "temporary failure",
        },
    )
    calls: list[str] = []
    runner = LocalAgentRunner(
        adapter_config={"github": {"arch": {"repo_url": "https://github.example/org/repo", "source_id": "src-arch"}}},
        state_store=state_store,
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: calls.append(name) or {"counts": {"pushed": 1}},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        default_sync_interval_seconds=3600,
    )

    early = runner.run_once(
        now=datetime(2026, 7, 7, tzinfo=timezone.utc) + timedelta(seconds=60),
        include_jira=False,
        only_due=True,
    )
    retry = runner.run_once(
        now=datetime(2026, 7, 7, tzinfo=timezone.utc) + timedelta(seconds=301),
        include_jira=False,
        only_due=True,
    )

    assert early["counts"] == {"total": 1, "success": 0, "failed": 0, "skipped": 1}
    assert calls == ["arch"]
    assert retry["counts"] == {"total": 1, "success": 1, "failed": 0, "skipped": 0}


def test_local_agent_jira_reuses_cookie_hash_and_marks_transport_error_failed(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    state_store = LocalAgentStateStore(tmp_path / "state.json")
    state_store.record_result(
        "jira-auth:https://jira.tools.sap",
        {
            "status": "success",
            "started_at": "2026-07-07T00:00:00+00:00",
            "finished_at": "2026-07-07T00:00:01+00:00",
            "payload": {"cookie_hash": "hash-1"},
        },
    )
    seen_hashes: list[str | None] = []

    def run_jira(origin: str, last_hash: str | None = None) -> dict:
        seen_hashes.append(last_hash)
        return {"action": "transport_error", "cookie_hash": last_hash, "detail": "server unavailable"}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=state_store,
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=run_jira,
        ),
        jira_origins_provider=lambda: {
            "origins": [{"origin": "https://jira.tools.sap", "configured": True, "status": "active"}]
        },
    )

    report = runner.run_once(now=datetime(2026, 7, 7, 1, tzinfo=timezone.utc), include_jira=True)

    assert seen_hashes == ["hash-1"]
    assert report["counts"]["failed"] == 1
    assert report["results"][0]["status"] == "failed"
    assert report["results"][0]["error"] == "Jira browser-session refresh returned transport_error"


def test_local_agent_jira_origin_discovery_failure_does_not_block_profiles(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    calls: list[str] = []

    def raise_cloud_unavailable() -> dict:
        raise RuntimeError("cloud unavailable")

    runner = LocalAgentRunner(
        adapter_config={"kb": {"notes": {"root": "/repo", "vault_id": "notes", "source_id": "src-notes"}}},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: calls.append(name) or {"counts": {"pushed": 1}},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
        jira_origins_provider=raise_cloud_unavailable,
    )

    report = runner.run_once(now=datetime(2026, 7, 7, tzinfo=timezone.utc), include_jira=True)

    assert calls == ["notes"]
    assert report["counts"] == {"total": 2, "success": 1, "failed": 1, "skipped": 0}
    assert report["results"][0]["task_id"] == "jira-auth:discovery"
    assert report["results"][0]["status"] == "failed"
    assert report["results"][1]["task_id"] == "kb:notes"


def test_local_agent_jira_origin_discovery_requires_positive_signal():
    from memforge.local_agent.tasks import discover_jira_auth_tasks

    tasks = discover_jira_auth_tasks(
        {
            "origins": [
                {"origin": "https://old.example.test"},
                {"origin": "https://configured.example.test", "configured": True},
                {"origin": "https://active.example.test", "status": "active"},
            ]
        },
        default_interval_seconds=1800,
    )

    assert [task.origin for task in tasks] == [
        "https://configured.example.test",
        "https://active.example.test",
    ]


def test_adapter_daemon_run_accepts_interval_seconds_alias():
    result = CliRunner().invoke(cli, ["adapter", "daemon", "run", "--help"])

    assert result.exit_code == 0, result.output
    assert "--interval-seconds" in result.output


def test_adapter_daemon_run_defaults_to_fast_cloud_job_poll():
    result = CliRunner().invoke(cli, ["adapter", "daemon", "run", "--help"])

    assert result.exit_code == 0, result.output
    assert re.search(r"\[default:\s*10\]", result.output)


def test_local_agent_caches_jira_discovery_between_due_windows(tmp_path):
    from datetime import timedelta

    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    provider_calls = 0

    def provide_origins() -> dict:
        nonlocal provider_calls
        provider_calls += 1
        return {"origins": [{"origin": "https://jira.tools.sap", "configured": True, "status": "active"}]}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged", "cookie_hash": last_hash},
        ),
        jira_origins_provider=provide_origins,
        jira_interval_seconds=1800,
    )

    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    runner.run_once(now=now, include_jira=True, only_due=True)
    report = runner.run_once(now=now + timedelta(seconds=60), include_jira=True, only_due=True)

    assert provider_calls == 1
    assert report["counts"] == {"total": 1, "success": 0, "failed": 0, "skipped": 1}
    assert report["results"][0]["task_id"] == "jira-auth:https://jira.tools.sap"


def test_local_agent_retries_empty_jira_discovery_after_failure(tmp_path):
    from datetime import timedelta

    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    provider_calls = 0

    def provide_origins() -> dict:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise RuntimeError("cloud unavailable")
        return {"origins": [{"origin": "https://jira.tools.sap", "configured": True, "status": "active"}]}

    runner = LocalAgentRunner(
        adapter_config={},
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged", "cookie_hash": last_hash},
        ),
        jira_origins_provider=provide_origins,
        jira_interval_seconds=1800,
    )

    now = datetime(2026, 7, 7, tzinfo=timezone.utc)
    first = runner.run_once(now=now, include_jira=True, only_due=True)
    second = runner.run_once(now=now + timedelta(seconds=60), include_jira=True, only_due=True)

    assert provider_calls == 2
    assert first["results"][0]["task_id"] == "jira-auth:discovery"
    assert second["results"][0]["task_id"] == "jira-auth:https://jira.tools.sap"


def test_local_agent_run_forever_records_runner_errors(tmp_path):
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.local_agent.tasks import LocalAgentHandlers

    state_store = LocalAgentStateStore(tmp_path / "state.json")
    runner = LocalAgentRunner(
        adapter_config={},
        state_store=state_store,
        handlers=LocalAgentHandlers(
            run_kb_profile=lambda name: {},
            run_github_profile=lambda name: {},
            run_jira_auth=lambda origin, last_hash=None: {"action": "unchanged"},
        ),
    )

    def raise_once(**kwargs) -> dict:
        raise RuntimeError("unexpected loop failure")

    logs: list[str] = []
    runner.run_once = raise_once  # type: ignore[method-assign]

    runner.run_forever(stop_after_iterations=1, sleep=lambda seconds: None, log=logs.append)

    assert logs == ["Local agent daemon iteration failed: unexpected loop failure"]
    state = state_store.load()
    assert state["tasks"]["runner:error"]["last_status"] == "failed"
    assert state["tasks"]["runner:error"]["last_error"] == "unexpected loop failure"


def test_adapter_daemon_once_exits_nonzero_when_task_fails(monkeypatch):
    class FakeRunner:
        def run_once(self, *, include_jira: bool = True) -> dict:
            return {
                "status": "ok",
                "counts": {"total": 1, "success": 0, "failed": 1, "skipped": 0},
                "results": [{"task_id": "github:arch", "status": "failed", "error": "push failed"}],
            }

    monkeypatch.setattr(main, "_build_local_agent_runner", lambda *args, **kwargs: FakeRunner())

    result = CliRunner().invoke(cli, ["adapter", "daemon", "once", "--no-include-jira"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "one or more local agent tasks failed"
    assert payload["counts"]["failed"] == 1


def test_jira_auth_once_preserves_previous_hash_when_no_new_hash(monkeypatch):
    async def fake_watch_tick(**kwargs):
        assert kwargs["last_hash"] == "hash-1"
        return "expired", None

    monkeypatch.setattr(main, "run_watch_tick", fake_watch_tick)

    payload = main._run_jira_auth_once(
        client=object(),
        origin="https://jira.tools.sap",
        browser=None,
        last_hash="hash-1",
    )

    assert payload == {"action": "expired", "cookie_hash": "hash-1", "ok": False}


def test_adapter_daemon_status_summarizes_state_by_default(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": {
                    "github:arch": {
                        "run_count": 2,
                        "last_status": "success",
                        "last_finished_at": "2026-07-07T01:00:00+00:00",
                    },
                    "cloud-jobs:lease": {
                        "run_count": 3,
                        "last_status": "failed",
                        "last_error": "missing MEMFORGE_WORKSPACE_ID",
                        "updated_at": "2026-07-07T01:02:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_STATE", str(state_path))
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    (tmp_path / "cli.toml").write_text(
        'active = "dev"\n\n[targets.dev]\napi_url = "https://memforge.example.test"\ntoken_env = "MEMFORGE_API_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "_read_adapter_config", lambda: {"kb": {}, "github": {}})

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "stopped"
    assert "state" not in payload
    assert payload["target"]["api_url"] == "https://memforge.example.test"
    assert payload["target"]["api_token_configured"] is False
    assert payload["target"]["workspace_id_configured"] is False
    assert payload["recommendations"] == [
        "Set MEMFORGE_API_TOKEN before starting the daemon.",
        "Set MEMFORGE_WORKSPACE_ID for hosted multi-workspace MemForge targets.",
    ]
    assert payload["summary"]["total_recorded_tasks"] == 2
    assert payload["summary"]["last_cloud_job_lease"]["status"] == "failed"
    assert payload["summary"]["last_cloud_job_lease"]["error"] == "missing MEMFORGE_WORKSPACE_ID"
    assert payload["recent_tasks"] == [
        {
            "task_id": "cloud-jobs:lease",
            "status": "failed",
            "last_finished_at": None,
            "updated_at": "2026-07-07T01:02:00+00:00",
            "run_count": 3,
            "error": "missing MEMFORGE_WORKSPACE_ID",
        },
        {
            "task_id": "github:arch",
            "status": "success",
            "last_finished_at": "2026-07-07T01:00:00+00:00",
            "updated_at": None,
            "run_count": 2,
            "error": None,
        },
    ]


def test_adapter_daemon_status_verbose_includes_raw_state(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "tasks": {
                    "github:arch": {
                        "run_count": 2,
                        "last_status": "success",
                        "last_finished_at": "2026-07-07T01:00:00+00:00",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_LOCAL_AGENT_STATE", str(state_path))
    monkeypatch.setattr(main, "_read_adapter_config", lambda: {"kb": {}, "github": {}})

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status", "--verbose"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"]["tasks"]["github:arch"]["run_count"] == 2


def test_recent_local_agent_tasks_sorts_mixed_timestamp_offsets():
    tasks = {
        "earlier-offset": {
            "run_count": 1,
            "last_status": "success",
            "updated_at": "2026-07-07T02:00:00+02:00",
        },
        "later-z": {
            "run_count": 1,
            "last_status": "failed",
            "updated_at": "2026-07-07T00:30:00Z",
        },
    }

    recent = main._recent_local_agent_tasks(tasks, limit=2)

    assert [task["task_id"] for task in recent] == ["later-z", "earlier-offset"]
