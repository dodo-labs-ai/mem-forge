from __future__ import annotations

from datetime import datetime, timezone
import json

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


def test_adapter_daemon_status_reads_state(monkeypatch, tmp_path):
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

    result = CliRunner().invoke(cli, ["adapter", "daemon", "status"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "stopped"
    assert payload["state"]["tasks"]["github:arch"]["run_count"] == 2
