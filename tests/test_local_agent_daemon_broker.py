"""The daemon executes server-owned jobs and owns no recurring schedule."""

from __future__ import annotations

import json

from click.testing import CliRunner

import memforge.main as main
from memforge.local_agent.runner import LocalAgentRunner
from memforge.local_agent.state import LocalAgentStateStore
from memforge.main import cli


def test_local_agent_runner_exposes_only_server_job_execution(tmp_path):
    lease_calls: list[int] = []
    handled: list[str] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: handled.append(job["job_id"]) or {"ok": True},
        cloud_jobs_provider=lambda wait_seconds=0: lease_calls.append(wait_seconds)
        or {
            "jobs": [
                {
                    "job_id": "laj-1",
                    "attempt_count": 1,
                    "operation": "teams_sync",
                    "source_id": "src-teams",
                }
            ]
        },
        cloud_job_completer=lambda *args: {},
    )

    report = runner.run_once(wait_seconds=7)

    assert lease_calls == [7]
    assert handled == ["laj-1"]
    assert report["counts"] == {"total": 2, "success": 2, "failed": 0}
    assert not hasattr(runner, "discover_tasks")
    assert not hasattr(runner, "_run_scheduled_tasks")


def test_daemon_commands_do_not_offer_local_schedule_controls():
    runner = CliRunner()

    run_help = runner.invoke(cli, ["adapter", "daemon", "run", "--help"])
    once_help = runner.invoke(cli, ["adapter", "daemon", "once", "--help"])
    status_help = runner.invoke(cli, ["adapter", "daemon", "status", "--help"])

    assert run_help.exit_code == once_help.exit_code == status_help.exit_code == 0
    combined = "\n".join((run_help.output, once_help.output, status_help.output))
    assert "include-jira" not in combined
    assert "default-sync-interval" not in combined
    assert "jira-interval" not in combined


def test_daemon_status_reports_job_loop_not_profile_tasks(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"version": 1, "tasks": {}}), encoding="utf-8")
    monkeypatch.setattr(main, "_local_agent_state_path", lambda: state_path)
    monkeypatch.setattr(main, "_local_agent_lock_path", lambda: tmp_path / "daemon.lock")
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
    for name in ("MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID"):
        monkeypatch.delenv(name, raising=False)

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "configured_tasks" not in payload
    assert payload["summary"]["last_cloud_job_lease"] is None


def test_local_agent_runner_reports_malformed_lease_response(tmp_path):
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: {"ok": True},
        cloud_jobs_provider=lambda **kwargs: {"jobs": "not-a-list"},
    )

    report = runner.run_once()

    assert report["counts"] == {"total": 1, "success": 0, "failed": 1}
    result = report["results"][0]
    assert result["task_id"] == "cloud-jobs:lease"
    assert result["error_type"] == "CloudJobLeaseError"
    assert "malformed" in result["error"]


def test_local_agent_runner_retries_unclassified_handler_exceptions(tmp_path):
    completions: list[tuple] = []

    def fail_handler(_job):
        raise ConnectionError("temporary upload failure")

    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=fail_handler,
        cloud_jobs_provider=lambda **_kwargs: {
            "jobs": [
                {
                    "job_id": "laj-transient",
                    "attempt_count": 1,
                    "operation": "local_markdown_sync",
                    "source_id": "src-local",
                }
            ]
        },
        cloud_job_completer=lambda *args: completions.append(args) or {},
    )

    runner.run_once()

    assert completions
    assert completions[0][2] == "failed"
    assert completions[0][3]["retryable"] is True
