from __future__ import annotations

import ast
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_memforge_plugin_config(monkeypatch, tmp_path):
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing-codex-config.toml"))
    try:
        from memforge import plugin_config

        monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    except Exception:
        pass


def _normalized_plugin_config_ast(source: str) -> str:
    module = ast.parse(source)
    expected_import_switches = [
        ast.parse(
            """
if __package__:
    from .api_target import MemForgeTarget, build_target
else:
    from memforge.api_target import MemForgeTarget, build_target
"""
        ).body[0],
        ast.parse(
            """
if __package__:
    from .memforge_api_target import MemForgeTarget, build_target
else:
    from memforge_api_target import MemForgeTarget, build_target
"""
        ).body[0],
    ]
    expected_import_switch_asts = {ast.dump(node, include_attributes=False) for node in expected_import_switches}
    normalized_body: list[ast.stmt] = []
    for node in module.body:
        contains_target_import = any(
            isinstance(child, ast.ImportFrom)
            and child.module
            in {
                "api_target",
                "memforge.api_target",
                "memforge_api_target",
            }
            for child in ast.walk(node)
        )
        if contains_target_import:
            assert ast.dump(node, include_attributes=False) in expected_import_switch_asts, (
                "target import compatibility block must be exact"
            )
            continue
        normalized_body.append(node)
    module.body = normalized_body
    return ast.dump(module, include_attributes=False)


def _init_git_repo_with_origin(path: Path, origin_url: str) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(
        ["git", "remote", "add", "origin", origin_url],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_hook_adapter_returns_additional_context_for_prompt_hook(monkeypatch, capsys):
    from memforge import hook_adapter

    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {
            "should_inject": True,
            "context_markdown": "## MemForge Memory Context\n- Use MemoryStore.",
            "memories": [{"id": "mem-1"}],
            "recent_changes": [],
            "warnings": [],
        }

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "sess-1",
                "cwd": "/tmp/mem-forge",
                "prompt": "What memory lifecycle decisions matter?",
            }
        ),
    )

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    assert requests[0][0] == "/hooks/context"
    assert requests[0][1]["client"] == "codex"
    assert requests[0][1]["hook"] == "UserPromptSubmit"
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "MemoryStore" in output["hookSpecificOutput"]["additionalContext"]


def test_hook_adapter_context_sends_canonical_remote_repo_for_claude(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    _init_git_repo_with_origin(tmp_path, "git@github.tools.sap:HCM/memforge-cloud.git")
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"should_inject": False}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "sess-claude-repo",
                "cwd": str(tmp_path),
                "prompt": "What do we know about this repo?",
            }
        ),
    )

    exit_code = hook_adapter.main(["context", "--client", "claude-code"])

    assert exit_code == 0
    assert requests[0][0] == "/hooks/context"
    assert requests[0][1]["client"] == "claude-code"
    assert requests[0][1]["repo"] == "github.tools.sap/hcm/memforge-cloud"
    assert capsys.readouterr().out == ""


def test_hook_adapter_injects_session_start_memforge_usage_guidance_without_api(monkeypatch, capsys):
    from memforge import hook_adapter

    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        raise AssertionError("SessionStart guidance should not call the API")

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "SessionStart",
                "session_id": "sess-start",
                "cwd": "/tmp/mem-forge",
            }
        ),
    )

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    assert requests == []
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    context = output["hookSpecificOutput"]["additionalContext"]
    assert "MemForge Usage Guidance" in context
    assert "MCP search" in context
    assert "list_sources" in context
    assert "source_ids" in context
    assert "source_types" not in context
    assert "get_memory" in context
    assert "get_resource" in context
    assert "confirmed content must be the durable memory only" in context
    assert "why-the-tool-was-called out of content" in context
    assert "Relevant Memories" not in context


def test_hook_adapter_submits_lifecycle_receipt_when_transcript_is_missing(monkeypatch, capsys):
    from memforge import hook_adapter

    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"receipt_id": "agent-hook-receipt"}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "Stop",
                "session_id": "sess-2",
                "cwd": "/tmp/mem-forge",
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert requests[0][0] == "/hooks/receipts"
    payload = requests[0][1]
    assert payload["client"] == "codex"
    assert payload["hook"] == "Stop"
    assert payload["metadata"]["has_transcript_path"] is False
    assert "transcript_path" not in payload["metadata"]
    assert "document_markdown" not in payload
    assert "process_now" not in payload
    assert capsys.readouterr().out == ""


def test_hook_adapter_precompact_posts_window_when_transcript_exists(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"user","message":"please remember this design"}\n{"type":"assistant","message":"Done."}\n',
        encoding="utf-8",
    )
    queue_db = tmp_path / "queue.sqlite"
    requests: list[tuple[str, dict]] = []
    spawned_workers: list[float] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"receipt_id": "receipt-precompact"}

    def fake_spawn_worker(*, timeout: float):
        spawned_workers.append(timeout)

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(hook_adapter, "_spawn_agent_window_worker", fake_spawn_worker)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "PreCompact",
                "session_id": "sess-precompact",
                "cwd": str(tmp_path),
                "transcript_path": str(transcript),
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert [request[0] for request in requests] == ["/hooks/receipts"]
    assert spawned_workers == [180.0]
    with sqlite3.connect(queue_db) as connection:
        rows = connection.execute(
            "SELECT capture_pending, captured_through, pending_trigger, transcript_path, session_id FROM session_cursor"
        ).fetchall()
    assert len(rows) == 1
    capture_pending, captured_through, pending_trigger, tpath, sess = rows[0]
    assert capture_pending == 1
    assert captured_through == 0
    assert pending_trigger == "REQUIRED_CAPTURE"
    assert tpath == str(transcript)
    assert sess == "sess-precompact"
    assert requests[0][1]["hook"] == "PreCompact"
    assert capsys.readouterr().out == ""


def test_hook_adapter_reprocesses_appended_transcript_window(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"first edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    requests: list[tuple[str, dict]] = []
    spawned_workers: list[float] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    def fake_spawn_worker(*, timeout: float):
        spawned_workers.append(timeout)

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(hook_adapter, "_spawn_agent_window_worker", fake_spawn_worker)

    payload = {
        "hook_event_name": "PreCompact",
        "session_id": "sess-precompact",
        "cwd": str(tmp_path),
        "transcript_path": str(transcript),
    }
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin(payload))
    assert hook_adapter.main(["submit-session"]) == 0

    transcript.write_text(
        transcript.read_text(encoding="utf-8")
        + '{"type":"tool","name":"exec_command","input":"pytest tests/test_hook_adapter.py -q"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin(payload))
    assert hook_adapter.main(["submit-session"]) == 0

    assert [request[0] for request in requests] == [
        "/hooks/receipts",
        "/hooks/receipts",
    ]
    assert spawned_workers == [180.0, 180.0]
    with sqlite3.connect(queue_db) as connection:
        rows = connection.execute(
            "SELECT capture_pending, captured_through, transcript_path, session_id FROM session_cursor"
        ).fetchall()
    assert len(rows) == 1
    capture_pending, captured_through, tpath, sess = rows[0]
    assert capture_pending == 1
    assert captured_through == 0
    assert tpath == str(transcript)
    assert sess == "sess-precompact"
    assert capsys.readouterr().out == ""


def test_hook_adapter_posts_receipt_when_window_queue_fails(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    def fail_request(*args, **kwargs):
        raise OSError("queue unavailable")

    monkeypatch.setattr(hook_adapter, "request_session_capture", fail_request)
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "PreCompact",
                "session_id": "sess-queue-fail",
                "cwd": str(tmp_path),
                "transcript_path": str(transcript),
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert [request[0] for request in requests] == ["/hooks/receipts"]
    assert requests[0][1]["metadata"]["has_transcript_path"] is True
    assert capsys.readouterr().out == ""


def test_hook_adapter_trivial_stop_does_not_post_window(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"assistant","message":"ok"}\n', encoding="utf-8")
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(tmp_path / "queue.sqlite"))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "Stop",
                "session_id": "sess-trivial",
                "cwd": str(tmp_path),
                "transcript_path": str(transcript),
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert [request[0] for request in requests] == ["/hooks/receipts"]
    assert requests[0][1]["hook"] == "Stop"
    assert capsys.readouterr().out == ""


def test_hook_adapter_stop_with_edit_and_test_signal_enqueues_window_worker(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"apply_patch","input":"update src/memforge/hook_adapter.py"}\n'
        '{"type":"tool","name":"exec_command","input":"pytest tests/test_hook_adapter.py -q"}\n'
        '{"type":"assistant","message":"Implemented and tests pass."}\n',
        encoding="utf-8",
    )
    requests: list[tuple[str, dict]] = []
    spawned_workers: list[float] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"receipt_id": "receipt-stop"}

    def fake_spawn_worker(*, timeout: float):
        spawned_workers.append(timeout)

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(tmp_path / "queue.sqlite"))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(hook_adapter, "_spawn_agent_window_worker", fake_spawn_worker, raising=False)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "Stop",
                "session_id": "sess-stop",
                "cwd": str(tmp_path),
                "transcript_path": str(transcript),
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert [request[0] for request in requests] == ["/hooks/receipts"]
    assert spawned_workers == [180.0]
    with sqlite3.connect(tmp_path / "queue.sqlite") as connection:
        rows = connection.execute("SELECT capture_pending, pending_trigger, session_id FROM session_cursor").fetchall()
    assert len(rows) == 1
    capture_pending, pending_trigger, sess = rows[0]
    assert capture_pending == 1
    assert pending_trigger == "GATED_CAPTURE"
    assert sess == "sess-stop"
    assert capsys.readouterr().out == ""


def test_gated_capture_scans_tail_without_materializing_full_delta(monkeypatch, tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps({"type": "assistant", "message": f"line {index}"}) for index in range(20))
        + "\n"
        + json.dumps({"type": "tool", "name": "exec_command", "input": "pytest tests"})
        + "\n",
        encoding="utf-8",
    )

    def fail_full_slice(*args, **kwargs):
        raise AssertionError("gate must not materialize the whole transcript tail")

    monkeypatch.setattr(hook_adapter, "_get_captured_through", lambda client, session_id: 0)
    monkeypatch.setattr(hook_adapter, "_transcript_line_slice", fail_full_slice)

    assert (
        hook_adapter._should_request_capture(
            "GATED_CAPTURE",
            str(transcript),
            client="codex",
            payload={"session_id": "sess-stream-gate"},
        )
        is True
    )


def test_hook_adapter_context_wakes_pending_queue_without_changing_output(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"apply_patch","input":"edit"}\n',
        encoding="utf-8",
    )
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="REQUIRED_CAPTURE",
        queue_db_path=queue_db,
    )
    requests: list[tuple[str, dict]] = []
    spawned_workers: list[float] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {
            "should_inject": True,
            "context_markdown": "## MemForge Memory Context\n- Keep queue output quiet.",
        }

    def fake_spawn_worker(*, timeout: float):
        spawned_workers.append(timeout)

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(hook_adapter, "_spawn_agent_window_worker", fake_spawn_worker, raising=False)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "UserPromptSubmit",
                "session_id": "sess-context",
                "cwd": str(tmp_path),
                "prompt": "continue",
            }
        ),
    )

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    assert [request[0] for request in requests] == ["/hooks/context"]
    assert spawned_workers == [180.0]
    output = json.loads(capsys.readouterr().out)
    assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "Keep queue output quiet." in output["hookSpecificOutput"]["additionalContext"]


def test_hook_adapter_worker_run_once_drains_pending_queue(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"apply_patch","input":"edit"}\n',
        encoding="utf-8",
    )
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )
    requests: list[tuple[str, dict, float]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload, timeout))
        return {"window_id": "queued-window"}

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    exit_code = hook_adapter.main(
        [
            "worker-run-once",
            "--timeout",
            "77",
        ]
    )

    assert exit_code == 0
    assert len(requests) == 1
    path, payload, timeout = requests[0]
    assert path == "/agent-sessions/windows"
    assert payload["trigger"] == "GATED_CAPTURE"
    assert payload["events"][0]["name"] == "apply_patch"
    assert payload["process_now"] is False
    assert payload["history_window"]["start"] == "0"
    assert payload["history_window"]["end"] == "1"
    assert timeout == 77.0
    with sqlite3.connect(queue_db) as connection:
        rows = connection.execute("SELECT capture_pending, captured_through FROM session_cursor").fetchall()
    assert rows == [(0, 1)]
    assert capsys.readouterr().out == ""


def test_worker_normalizes_legacy_capture_trigger_names(monkeypatch, tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="legacy-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )
    with sqlite3.connect(queue_db) as connection:
        connection.execute("UPDATE session_cursor SET pending_trigger = 'BOUNDARY' WHERE session_id = 'legacy-session'")
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(timeout=5, queue_db_path=queue_db)

    assert submitted == 1
    assert requests[0][1]["trigger"] == "REQUIRED_CAPTURE"


def test_hook_adapter_worker_leaves_source_sync_to_service(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"apply_patch","input":"edit"}\n',
        encoding="utf-8",
    )
    queue_db = tmp_path / "queue.sqlite"
    for index in range(2):
        hook_adapter.request_session_capture(
            client="codex",
            session_id=f"queued-session-{index}",
            transcript_path=str(transcript),
            workspace=str(tmp_path),
            trigger="GATED_CAPTURE",
            queue_db_path=queue_db,
        )
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(
        timeout=77,
    )

    assert submitted == 2
    assert [request[0] for request in requests] == [
        "/agent-sessions/windows",
        "/agent-sessions/windows",
    ]
    assert requests[0][1]["process_now"] is False
    assert requests[1][1]["process_now"] is False
    assert capsys.readouterr().out == ""


def test_hook_adapter_worker_splits_large_window_without_advancing_past_upload(monkeypatch, tmp_path):
    from memforge import hook_adapter

    lines = [json.dumps({"type": "tool", "name": f"tool-{index}", "input": "x" * 20}) for index in range(3)]
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="REQUIRED_CAPTURE",
        queue_db_path=queue_db,
    )
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "MAX_TRANSCRIPT_CHARS", len("\n".join(lines[:2])))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(
        timeout=77,
    )

    assert submitted == 1
    payload = requests[0][1]
    assert payload["history_window"]["start"] == "0"
    uploaded_end = int(payload["history_window"]["end"])
    assert 0 < uploaded_end < 3
    assert payload["history_window"]["truncated"] is True
    assert "tool-0" in payload["transcript_markdown"]
    assert "tool-2" not in payload["transcript_markdown"]
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute("SELECT captured_through, capture_pending FROM session_cursor").fetchone()
    assert row == (uploaded_end, 1)


def test_hook_adapter_worker_records_window_submission_error(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"apply_patch","input":"edit"}\n',
        encoding="utf-8",
    )
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    def fail_post_json(path: str, payload: dict, *, timeout: float):
        raise TimeoutError("timed out after 77 seconds")

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fail_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(
        timeout=77,
    )

    assert submitted == 0
    with sqlite3.connect(queue_db) as connection:
        rows = connection.execute(
            "SELECT capture_pending, captured_through, last_error, last_attempt_at, lease_until FROM session_cursor"
        ).fetchall()
    assert len(rows) == 1
    capture_pending, captured_through, last_error, last_attempt_at, lease_until = rows[0]
    assert capture_pending == 1
    assert captured_through == 0
    assert "TimeoutError: timed out after 77 seconds" in last_error
    assert last_attempt_at
    assert lease_until is None
    assert capsys.readouterr().out == ""


def test_hook_adapter_worker_keeps_pending_when_transcript_disappears(monkeypatch, tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )
    transcript.unlink()
    requests: list[tuple[str, dict]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, payload))
        return {"ok": True}

    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(
        timeout=77,
    )

    assert submitted == 0
    assert requests == []
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute(
            "SELECT capture_pending, captured_through, last_error, lease_until FROM session_cursor"
        ).fetchone()
    assert row[0] == 1
    assert row[1] == 0
    assert "transcript unavailable" in row[2]
    assert row[3] is None


def test_request_session_capture_creates_session_cursor_schema(tmp_path):
    from memforge import hook_adapter

    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="queued-session",
        transcript_path=str(tmp_path / "missing.jsonl"),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    with sqlite3.connect(queue_db) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(session_cursor)")}
        row = connection.execute(
            "SELECT capture_pending, captured_through, pending_trigger, request_seq, lease_token FROM session_cursor"
        ).fetchone()
    assert {
        "captured_through",
        "capture_pending",
        "pending_trigger",
        "lease_until",
        "lease_token",
        "request_seq",
        "last_error",
    } <= columns
    assert row == (1, 0, "GATED_CAPTURE", 1, None)


def test_codex_and_claude_plugins_include_hooks_and_adapter_wrappers():
    root = Path(__file__).resolve().parents[1]

    codex_root = root / "integrations" / "codex" / "memforge-memory"
    claude_root = root / "integrations" / "claude-code" / "memforge-memory"

    assert (codex_root / ".codex-plugin" / "plugin.json").exists()
    assert (codex_root / "hooks" / "hooks.json").exists()
    assert (codex_root / "scripts" / "memforge_hook.py").exists()
    assert (codex_root / "scripts" / "memforge_hook_adapter.py").exists()
    assert (codex_root / "scripts" / "memforge_mcp.py").exists()
    assert (codex_root / "scripts" / "memforge_repo_identity.py").exists()
    assert (codex_root / ".mcp.json").exists()

    assert (claude_root / ".claude-plugin" / "plugin.json").exists()
    assert (claude_root / "hooks" / "hooks.json").exists()
    assert (claude_root / "scripts" / "memforge_hook.py").exists()
    assert (claude_root / "scripts" / "memforge_hook_adapter.py").exists()
    assert (claude_root / "scripts" / "memforge_mcp.py").exists()
    assert (claude_root / "scripts" / "memforge_repo_identity.py").exists()
    assert (claude_root / ".mcp.json").exists()

    codex_manifest = json.loads((codex_root / ".codex-plugin" / "plugin.json").read_text())
    assert codex_manifest["hooks"] == "./hooks/hooks.json"
    claude_manifest = json.loads((claude_root / ".claude-plugin" / "plugin.json").read_text())
    assert codex_manifest["version"] == claude_manifest["version"]

    codex_hooks = json.loads((codex_root / "hooks" / "hooks.json").read_text())
    claude_hooks = json.loads((claude_root / "hooks" / "hooks.json").read_text())

    for hooks in (codex_hooks, claude_hooks):
        assert "SessionStart" in hooks["hooks"]
        assert "PreCompact" in hooks["hooks"]
        assert "Stop" in hooks["hooks"]
        commands = json.dumps(hooks)
        hook_commands = [
            hook["command"]
            for event_entries in hooks["hooks"].values()
            for event_entry in event_entries
            for hook in event_entry["hooks"]
        ]
        assert "memforge_hook.py" in commands
        assert "submit-session" in commands
        assert any("${PLUGIN_ROOT:-}" in command or "${PLUGIN_ROOT:-}}" in command for command in hook_commands)
        assert all("plugins/cache/memforge/memory" not in command for command in hook_commands)
        assert all("version=" not in command for command in hook_commands)
        assert "plugins/cache/memforge/memory/*" not in commands
        assert " -nt " not in commands
    codex_mcp = json.loads((codex_root / ".mcp.json").read_text())
    claude_mcp = json.loads((claude_root / ".mcp.json").read_text())

    codex_memforge = codex_mcp["mcpServers"]["memforge"]
    assert codex_memforge["command"] == "python3"
    assert codex_memforge["args"] == ["scripts/memforge_mcp.py"]
    assert codex_memforge["cwd"] == "."
    assert codex_memforge["env_vars"] == ["CODEX_WORKSPACE_ROOT"]

    claude_memforge = claude_mcp["mcpServers"]["memforge"]
    assert claude_memforge["command"] == "sh"
    assert claude_memforge["args"] == [
        "-lc",
        (
            "root=$CLAUDE_PLUGIN_ROOT; "
            'if [ -z "$root" ]; then root=$PLUGIN_ROOT; fi; '
            'if [ -z "$root" ]; then root=.; fi; '
            'cd "$root" && exec python3 scripts/memforge_mcp.py'
        ),
    ]
    assert "cwd" not in claude_memforge
    assert "SubagentStop" in claude_hooks["hooks"]


def test_packaged_plugin_version_0_1_28_is_consistent():
    root = Path(__file__).resolve().parents[1]
    version = "0.1.28"
    canonical_mcp = (root / "src" / "memforge" / "plugin_mcp_proxy.py").read_text()
    canonical_hook = (root / "src" / "memforge" / "hook_adapter.py").read_text()

    assert f'SERVER_VERSION = "{version}"' in canonical_mcp
    assert f'PLUGIN_VERSION = "{version}"' in canonical_hook

    for client, manifest_dir in (
        ("codex", ".codex-plugin"),
        ("claude-code", ".claude-plugin"),
    ):
        plugin_root = root / "integrations" / client / "memforge-memory"
        manifest = json.loads((plugin_root / manifest_dir / "plugin.json").read_text())
        assert manifest["version"] == version
        assert f"version is `{version}`" in (plugin_root / "README.md").read_text()
        assert f'SERVER_VERSION = "{version}"' in (plugin_root / "scripts" / "memforge_mcp.py").read_text()
        assert f'PLUGIN_VERSION = "{version}"' in (
            plugin_root / "scripts" / "memforge_hook_adapter.py"
        ).read_text()


@pytest.mark.parametrize(
    ("plugin_path", "home_cache", "env_root"),
    [
        (
            "integrations/codex/memforge-memory/hooks/hooks.json",
            ".codex/plugins/cache/memforge/memory",
            "CODEX_PLUGIN_ROOT",
        ),
        (
            "integrations/claude-code/memforge-memory/hooks/hooks.json",
            ".claude/plugins/cache/memforge/memory",
            "CLAUDE_PLUGIN_ROOT",
        ),
    ],
)
def test_plugin_hook_commands_skip_when_registered_root_is_stale(tmp_path, plugin_path, home_cache, env_root):
    root = Path(__file__).resolve().parents[1]
    hooks = json.loads((root / plugin_path).read_text())
    command = hooks["hooks"]["Stop"][0]["hooks"][0]["command"]
    marker = tmp_path / "argv.txt"
    cache_script = tmp_path / home_cache / "0.1.20" / "scripts" / "memforge_hook.py"
    cache_script.parent.mkdir(parents=True)
    cache_script.write_text(
        "import os, pathlib, sys\npathlib.Path(os.environ['MEMFORGE_TEST_MARKER']).write_text(' '.join(sys.argv[1:]))\n"
    )
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env[env_root] = str(tmp_path / "deleted-plugin-root")
    env["PLUGIN_ROOT"] = str(tmp_path / "also-missing")
    env["MEMFORGE_TEST_MARKER"] = str(marker)

    result = subprocess.run(
        command,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "MemForge hook script not found" in result.stderr
    assert not marker.exists()


def test_plugin_wrapper_runs_from_repo_checkout_without_pythonpath():
    root = Path(__file__).resolve().parents[1]
    wrapper = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_hook.py"

    result = subprocess.run(
        [sys.executable, str(wrapper), "context"],
        input="{}",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output["continue"] is True
    assert "MemForge hook skipped" in output["systemMessage"]


def test_plugin_wrapper_runs_from_copied_package_without_pythonpath(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    for client in ("codex", "claude-code"):
        source_plugin = root / "integrations" / client / "memforge-memory"
        copied_plugin = tmp_path / client / "memforge-memory"
        shutil.copytree(source_plugin, copied_plugin)
        wrapper = copied_plugin / "scripts" / "memforge_hook.py"

        result = subprocess.run(
            [sys.executable, str(wrapper), "context"],
            input="{}",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=False,
        )

        assert result.returncode == 0
        assert result.stderr == ""
        output = json.loads(result.stdout)
        assert output["continue"] is True
        assert "MemForge hook skipped" in output["systemMessage"]


def test_plugin_adapters_match_canonical_adapter():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "hook_adapter.py").read_text()

    for adapter in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_hook_adapter.py",
    ):
        assert adapter.read_text() == canonical


def test_packaged_repo_identity_matches_canonical_helper():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "repo_identity.py").read_text()

    for helper in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_repo_identity.py",
    ):
        assert helper.read_text() == canonical


def test_packaged_plugin_config_matches_canonical_target_and_helpers():
    root = Path(__file__).resolve().parents[1]
    canonical = (root / "src" / "memforge" / "plugin_config.py").read_text()

    for helper in (
        root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
        root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_plugin_config.py",
    ):
        assert _normalized_plugin_config_ast(helper.read_text()) == _normalized_plugin_config_ast(canonical)


def test_mcp_and_hook_share_cloud_resource_url(monkeypatch):
    from memforge import hook_adapter, plugin_mcp_proxy

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")

    assert plugin_mcp_proxy._resource_url("/sources") == (
        "https://cloud.example.hana.ondemand.com/api/workspaces/mount_tai/api/sources"
    )
    assert hook_adapter._resource_url("/hooks/receipts") == (
        "https://cloud.example.hana.ondemand.com/api/workspaces/mount_tai/api/hooks/receipts"
    )


def test_invalid_hook_target_fails_before_urlopen(monkeypatch):
    from memforge import hook_adapter

    monkeypatch.setenv("MEMFORGE_API_URL", "https://cloud.example.hana.ondemand.com")
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setattr(
        hook_adapter.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("network"),
    )

    assert hook_adapter.main(["submit-session"]) == 1


def test_plugin_config_parity_normalizer_rejects_mixed_target_import_block():
    source = """
if __package__:
    from .api_target import MemForgeTarget, build_target
    unrelated_statement = True
else:
    from memforge.api_target import MemForgeTarget, build_target
"""

    with pytest.raises(AssertionError, match="target import compatibility block must be exact"):
        _normalized_plugin_config_ast(source)


def test_plugin_mcp_launchers_match_each_other():
    root = Path(__file__).resolve().parents[1]
    canonical = root / "src" / "memforge" / "plugin_mcp_proxy.py"
    codex_launcher = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    claude_launcher = root / "integrations" / "claude-code" / "memforge-memory" / "scripts" / "memforge_mcp.py"

    assert codex_launcher.read_text() == canonical.read_text()
    assert codex_launcher.read_text() == claude_launcher.read_text()


def test_mcp_proxy_starts_without_memforge_executable():
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    manifest = json.loads(
        (root / "integrations" / "codex" / "memforge-memory" / ".codex-plugin" / "plugin.json").read_text()
    )
    request = _mcp_initialize_request()
    frame = b"Content-Length: " + str(len(request)).encode() + b"\r\n\r\n" + request

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=frame,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    _, payload = result.stdout.split(b"\r\n\r\n", 1)
    response = json.loads(payload)
    assert response["result"]["serverInfo"]["name"] == "memforge"
    assert response["result"]["serverInfo"]["version"] == manifest["version"]
    assert response["result"]["capabilities"]["tools"]["listChanged"] is False


def test_mcp_proxy_supports_json_line_stdio():
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / "codex" / "memforge-memory" / "scripts" / "memforge_mcp.py"
    request = _mcp_initialize_request() + b"\n"

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=request,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    response = json.loads(result.stdout)
    assert response["result"]["serverInfo"]["name"] == "memforge"
    assert response["result"]["capabilities"]["tools"]["listChanged"] is False


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_mcp_proxy_launchers_request_roots_when_client_supports_roots(client):
    root = Path(__file__).resolve().parents[1]
    proxy = root / "integrations" / client / "memforge-memory" / "scripts" / "memforge_mcp.py"
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    ).encode()
    initialized = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode()
    frame = (
        b"Content-Length: "
        + str(len(initialize)).encode()
        + b"\r\n\r\n"
        + initialize
        + b"Content-Length: "
        + str(len(initialized)).encode()
        + b"\r\n\r\n"
        + initialized
    )

    result = subprocess.run(
        [sys.executable, str(proxy)],
        input=frame,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert result.stderr == b""
    messages = _mcp_content_length_messages(result.stdout)
    assert messages[0]["result"]["serverInfo"]["name"] == "memforge"
    assert messages[1]["method"] == "roots/list"


def test_mcp_proxy_retries_roots_request_after_list_changed_while_previous_request_is_pending():
    proxy = _load_plugin_mcp_proxy()
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    first_roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert first_roots_request["method"] == "roots/list"

    retry_roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/roots/list_changed"})

    assert retry_roots_request["method"] == "roots/list"
    assert retry_roots_request["id"] == first_roots_request["id"]


def _mcp_initialize_request() -> bytes:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "test"}},
        }
    ).encode()


def _mcp_content_length_messages(stream: bytes) -> list[dict]:
    messages = []
    remaining = stream
    while remaining:
        header, payload_start = remaining.split(b"\r\n\r\n", 1)
        length = int(header.removeprefix(b"Content-Length: ").strip())
        payload = payload_start[:length]
        messages.append(json.loads(payload))
        remaining = payload_start[length:]
    return messages


def test_mcp_proxy_forwards_search_to_service_with_token(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["authorization"] = request.get_header("Authorization")
            captured["content_type"] = request.get_header("Content-type")
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "artifact cache"})

    assert result == {"results": []}
    assert captured["url"] == "https://self.example/api/memories/search"
    assert captured["authorization"] == "Bearer token-123"
    assert captured["content_type"] == "application/json"
    assert json.loads(captured["body"].decode()) == {
        "query": "artifact cache",
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_forwards_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "deployment owner",
            "entities": ["payroll control center", "deployment owner"],
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode()) == {
        "query": "deployment owner",
        "entities": ["payroll control center", "deployment owner"],
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_trims_and_dedupes_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "deployment owner",
            "entities": [" Payroll Control Center ", "payroll control center", "Deployment Owner"],
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode())["entities"] == [
        "Payroll Control Center",
        "Deployment Owner",
    ]


def test_mcp_proxy_omits_empty_explicit_search_entities(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "deployment owner", "entities": []})

    assert result == {"results": []}
    assert json.loads(captured["body"].decode()) == {
        "query": "deployment owner",
        "include_private": True,
        "include_superseded": False,
    }


@pytest.mark.parametrize(
    "entities",
    [
        "payroll control center",
        ["payroll control center", 3],
        ["payroll control center", ""],
        ["payroll control center", "   "],
        ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"],
        ["x" * 129],
    ],
)
def test_mcp_proxy_rejects_invalid_explicit_search_entities(monkeypatch, entities):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "deployment owner", "entities": entities})

    assert result == {"error": "entities must be an array of 1-8 strings, each 1-128 characters after trimming"}


def test_mcp_proxy_forwards_list_sources_to_searchable_sources(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return (
                b'{"data":[{"source_id":"src-mounttai","name":"MountTai Defects",'
                b'"type":"jira","status":"active","doc_count":68,"memory_count":199,'
                b'"last_synced_at":"2026-06-27T00:00:00Z"}]}'
            )

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool("list_sources", {})

    assert result["data"][0]["source_id"] == "src-mounttai"
    assert (
        captured["url"] == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/sources/searchable"
    )
    assert captured["authorization"] == "Bearer token-123"


def test_mcp_proxy_adds_client_root_git_remote_as_ranking_hint_only(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    _provide_mcp_roots(proxy, repo_root)
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool("search", {"query": "scheduler fix"})

    body = json.loads(captured["body"].decode())
    assert body["active_repo_identifier"] == "github.com/dodoman-sun/memforge-cloud"
    assert "source_filter" not in body


def test_mcp_proxy_rejects_current_repo_filter_when_repo_context_is_unavailable(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setenv("MEMFORGE_ACTIVE_REPO_IDENTIFIER", "github.tools.sap/hcm/memforge-cloud")

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
            "workspace roots. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_even_when_client_roots_are_available(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )
    assert roots_request["method"] == "roots/list"

    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "result": {"roots": [{"uri": repo_root.as_uri(), "name": "memforge-cloud"}]},
            }
        )
        is None
    )
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
            "workspace roots. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_when_cwd_is_git_repo_but_roots_are_missing(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    monkeypatch.chdir(repo_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
            "workspace roots. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_uses_codex_workspace_root_when_roots_are_not_advertised(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "memforge-cloud"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    plugin_root = tmp_path / "plugin-cache"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", str(repo_root))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
        },
    )

    body = json.loads(captured["body"].decode())
    assert body["active_repo_identifier"] == "github.com/dodoman-sun/memforge-cloud"
    assert "source_filter" not in body


def test_mcp_proxy_rejects_current_repo_filter_when_root_has_no_git_remote(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    repo_root = tmp_path / "local-only"
    repo_root.mkdir()
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _provide_mcp_roots(proxy, repo_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
            "workspace roots. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_current_repo_filter_when_roots_have_multiple_git_remotes(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    first_root = tmp_path / "memforge-cloud"
    second_root = tmp_path / "mem-forge"
    first_root.mkdir()
    second_root.mkdir()
    _init_git_repo_with_origin(first_root, "https://github.com/dodoman-sun/memforge-cloud.git")
    _init_git_repo_with_origin(second_root, "https://github.com/shno-labs/mem-forge.git")
    _provide_mcp_roots(proxy, first_root, second_root)

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("current_repo_only should fail before posting")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "current_repo_only": True,
            },
        },
    )

    assert result == {
        "error": (
            "current_repo_only is disabled for MCP search because repo-scoped search depends on reliable "
            "workspace roots. Omit the filter to search all visible memories."
        )
    }


def test_mcp_proxy_rejects_unadvertised_search_source_ids(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {"query": "scheduler fix", "sources": ["src-hidden"]},
    )

    assert result == {"error": ("Unsupported search parameter(s): sources. Omit unknown filters instead of guessing.")}


def test_mcp_proxy_rejects_explicit_repo_hint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "active_repo_identifier": "github.tools.sap/hcm/other",
        },
    )

    assert result == {
        "error": ("Unsupported search parameter(s): active_repo_identifier. Omit unknown filters instead of guessing.")
    }


def test_mcp_proxy_search_schema_exposes_validated_facets_not_recent_changes():
    proxy = _load_plugin_mcp_proxy()

    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    assert "search" in tools
    assert "list_sources" in tools
    assert "list_recent_changes" not in tools
    assert "submit_agent_session_document" not in tools
    assert "suggest_memory_replacement" not in tools
    assert "create_memory" in tools
    assert "retire_memory" in tools
    assert "replace_memory" in tools
    assert "list_memory_reviews" in tools
    assert "get_memory_review" in tools
    assert "resolve_memory_review" in tools

    search_schema = tools["search"]["inputSchema"]
    properties = search_schema["properties"]
    assert "source_types" not in properties["source_filter"]["properties"]
    assert properties["source_filter"]["properties"]["source_ids"]["items"]["type"] == "string"
    assert "list_sources" in properties["source_filter"]["properties"]["source_ids"]["description"]
    assert properties["source_filter"]["properties"]["clients"]["items"]["enum"] == [
        "claude-code",
        "codex",
    ]
    assert "current_repo_only" not in properties["source_filter"]["properties"]
    assert "repo_identifiers" not in properties["source_filter"]["properties"]
    assert "source_instance_ids" not in properties["source_filter"]["properties"]
    assert "sources" not in properties
    assert set(properties) == {"query", "source_filter", "time_range", "top_k", "offset", "entities"}
    assert properties["entities"]["type"] == "array"
    assert properties["entities"]["maxItems"] == 8
    assert properties["entities"]["items"]["type"] == "string"
    assert properties["entities"]["items"]["minLength"] == 1
    assert properties["entities"]["items"]["maxLength"] == 128
    assert "agent-selected entity hints" in properties["entities"]["description"]
    assert "keeping query unchanged" in properties["entities"]["description"]
    assert "not filters or authority" in properties["entities"]["description"]
    assert "backlog" not in tools["search"]["description"].lower()
    assert "not a hard cap" in properties["top_k"]["description"]
    assert "up to 50" in properties["top_k"]["description"]
    assert "complete list" in properties["top_k"]["description"]
    assert "enumeration" in properties["top_k"]["description"]
    assert "backlog" not in properties["top_k"]["description"].lower()
    assert properties["top_k"]["minimum"] == 1
    assert properties["top_k"]["maximum"] == 50
    assert properties["offset"]["default"] == 0
    assert properties["offset"]["minimum"] == 0
    assert "next page" in properties["offset"]["description"]
    assert "required" not in search_schema or "query" not in search_schema.get("required", [])
    time_range_schema = properties["time_range"]
    assert "Omit time_range" in time_range_schema["description"]
    assert "start_date and end_date are individually optional" in time_range_schema["description"]
    assert time_range_schema["anyOf"] == [{"required": ["start_date"]}, {"required": ["end_date"]}]
    assert set(time_range_schema["properties"]) == {"date_type", "start_date", "end_date"}
    assert "include_private" not in properties
    assert "include_superseded" not in properties
    assert "status" not in properties
    assert "memory_types" not in properties
    assert "active_repo_identifier" not in properties
    assert "search -> get_memory -> get_resource" in tools["get_resource"]["description"]

    create_schema = tools["create_memory"]["inputSchema"]
    assert create_schema["required"] == ["content", "provenance"]
    assert create_schema["properties"]["memory_type"]["enum"] == ["fact", "decision", "convention", "procedure"]
    assert "provenance" in create_schema["properties"]
    assert "reason" not in create_schema["properties"]
    assert "client" not in create_schema["properties"]
    assert "repo_identifier" not in create_schema["properties"]
    assert "readable preview" in tools["create_memory"]["description"]
    assert "request_user_input" in tools["create_memory"]["description"]
    assert "durable memory content" in tools["create_memory"]["description"]
    assert "provenance" in tools["create_memory"]["description"]
    assert "Do not put confirmation details" in create_schema["properties"]["content"]["description"]

    retire_schema = tools["retire_memory"]["inputSchema"]
    assert retire_schema["required"] == ["memory_id", "reason", "expected_content_hash"]
    assert "status" not in retire_schema["properties"]

    replace_schema = tools["replace_memory"]["inputSchema"]
    assert replace_schema["required"] == [
        "memory_id",
        "replacement_content",
        "provenance",
        "reason",
        "expected_content_hash",
    ]
    assert "provenance" in replace_schema["properties"]
    assert "Do not put confirmation details" in replace_schema["properties"]["replacement_content"]["description"]
    assert "provenance" in tools["replace_memory"]["description"]
    assert (
        "old claim, new claim, provenance/evidence, scope, and replacement reason"
        in tools["replace_memory"]["description"]
    )
    assert replace_schema["properties"]["replacement_kind"]["enum"] == ["revision", "supersession"]
    assert "status" not in replace_schema["properties"]

    resolve_schema = tools["resolve_memory_review"]["inputSchema"]
    assert resolve_schema["properties"]["decision"]["enum"] == ["approve", "reject", "refresh"]
    assert "required when decision is reject" in resolve_schema["properties"]["note"]["description"]


def test_mcp_proxy_source_selection_descriptions_guide_scoped_and_global_search():
    proxy = _load_plugin_mcp_proxy()
    tools = {tool["name"]: tool for tool in proxy.TOOLS}

    search_description = tools["search"]["description"]
    list_sources_description = tools["list_sources"]["description"]

    assert "call list_sources first" in search_description
    assert "exact source_ids" in search_description
    assert "omit source_filter" in search_description
    assert "time_range only when explicitly requested" in search_description
    assert "deterministic source/time listings" in search_description
    assert "total_candidates and offset" in search_description
    assert "Ranked queries are not exhaustive" in search_description
    assert "Call get_memory for provenance" in search_description
    assert len(search_description) <= 500

    assert "Use before source-specific search" in list_sources_description
    assert "exact source_ids" in list_sources_description
    assert "skip for broad or cross-source requests" in list_sources_description
    assert "Returns source_id, name, type, status, counts, and last_synced_at" in list_sources_description
    assert len(list_sources_description) <= 260


def test_mcp_proxy_forwards_search_to_hosted_workspace(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "artifact cache"})

    assert result == {"results": []}
    assert captured["url"] == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/memories/search"
    assert captured["authorization"] == "Bearer token-123"
    assert json.loads(captured["body"].decode()) == {
        "query": "artifact cache",
        "include_private": True,
        "include_superseded": False,
    }


def test_mcp_proxy_forwards_retire_memory_to_lifecycle_endpoint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"retired","memory_id":"mem-123"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "retire_memory",
        {
            "memory_id": "mem-123",
            "reason": "User confirmed this memory is obsolete.",
            "expected_content_hash": "hash-old",
        },
    )

    assert result == {"status": "retired", "memory_id": "mem-123"}
    assert captured["method"] == "POST"
    assert (
        captured["url"]
        == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/memories/mem-123/retire"
    )
    assert json.loads(captured["body"].decode()) == {
        "reason": "User confirmed this memory is obsolete.",
        "expected_content_hash": "hash-old",
    }


def test_mcp_proxy_forwards_create_memory_with_plugin_client_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setenv("MEMFORGE_MCP_CLIENT", "claude-code")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: "github.com/shno-labs/mem-forge")

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
            "confidence": 0.9,
        },
    )

    assert result == {"status": "inserted", "memory_id": "mem-new"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/memories/create"
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "confidence": 0.9,
        "client": "claude-code",
        "repo_identifier": "github.com/shno-labs/mem-forge",
    }


def test_mcp_proxy_create_memory_uses_codex_workspace_root_when_roots_are_not_advertised(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    repo_root = tmp_path / "mem-forge"
    repo_root.mkdir()
    _init_git_repo_with_origin(repo_root, "https://github.com/shno-labs/mem-forge.git")
    plugin_root = tmp_path / "plugin-cache"
    plugin_root.mkdir()
    monkeypatch.chdir(plugin_root)

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"inserted","memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("CODEX_WORKSPACE_ROOT", str(repo_root))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
        },
    )

    assert result == {"status": "inserted", "memory_id": "mem-new"}
    assert json.loads(captured["body"].decode()) == {
        "content": "Use readable confirmation previews before memory mutations.",
        "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
        "memory_type": "convention",
        "client": "codex",
        "repo_identifier": "github.com/shno-labs/mem-forge",
    }


def test_mcp_proxy_rejects_create_memory_when_repo_roots_are_missing(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("create_memory should fail before posting without repo roots")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
        },
    )

    assert result == {
        "error": (
            "create_memory requires exactly one git remote from MCP workspace roots; "
            "the MCP client did not advertise roots support. Refusing to create an unscoped memory."
        )
    }


def test_mcp_proxy_reports_roots_error_response_for_create_memory(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert roots_request["method"] == "roots/list"
    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "error": {"code": -32601, "message": "roots/list rejected by host"},
            }
        )
        is None
    )

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("create_memory should fail before posting without repo roots")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "provenance": "User confirmed this convention after reviewing the MemForge MCP UX.",
            "memory_type": "convention",
        },
    )

    assert result == {
        "error": (
            "create_memory requires exactly one git remote from MCP workspace roots; "
            "the MCP client returned an error for roots/list: roots/list rejected by host. "
            "Refusing to create an unscoped memory."
        )
    }


def test_mcp_proxy_rejects_create_memory_without_provenance(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("create_memory should fail before posting without provenance")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: "github.com/shno-labs/mem-forge")

    result = proxy._call_tool(
        "create_memory",
        {
            "content": "Use readable confirmation previews before memory mutations.",
            "memory_type": "convention",
        },
    )

    assert result == {"error": "provenance is required"}


def test_mcp_proxy_forwards_replace_memory_to_lifecycle_endpoint(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"status":"superseded","memory_id":"mem-old","replacement_memory_id":"mem-new"}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "replace_memory",
        {
            "memory_id": "mem-old",
            "replacement_content": "Use the new deployment route.",
            "provenance": "User corrected this while reviewing the deployment guide.",
            "reason": "User corrected the stale route.",
            "expected_content_hash": "hash-old",
            "replacement_kind": "revision",
        },
    )

    assert result["replacement_memory_id"] == "mem-new"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://self.example/api/memories/mem-old/replace"
    assert json.loads(captured["body"].decode()) == {
        "replacement_content": "Use the new deployment route.",
        "provenance": "User corrected this while reviewing the deployment guide.",
        "reason": "User corrected the stale route.",
        "expected_content_hash": "hash-old",
        "replacement_kind": "revision",
    }


def test_mcp_proxy_rejects_replace_memory_without_provenance(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FailOpener:
        def open(self, *_args, **_kwargs):
            raise AssertionError("replace_memory should fail before posting without provenance")

    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FailOpener())

    result = proxy._call_tool(
        "replace_memory",
        {
            "memory_id": "mem-old",
            "replacement_content": "Use the new deployment route.",
            "reason": "User corrected the stale route.",
            "expected_content_hash": "hash-old",
            "replacement_kind": "revision",
        },
    )

    assert result == {"error": "provenance is required"}


def test_mcp_proxy_forwards_memory_review_tools(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    calls = []

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __init__(self, body: bytes = b'{"ok":true}'):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return self.body

    class FakeOpener:
        def open(self, request, timeout):
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "body": request.data,
                }
            )
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    assert proxy._call_tool("list_memory_reviews", {"status": "open", "limit": 5}) == {"ok": True}
    assert proxy._call_tool("get_memory_review", {"review_id": "rev-1"}) == {"ok": True}
    assert proxy._call_tool(
        "resolve_memory_review",
        {"review_id": "rev-1", "decision": "reject", "note": "Not durable enough."},
    ) == {"ok": True}

    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://self.example/api/memory-reviews?status=open&limit=5&offset=0"
    assert calls[1]["method"] == "GET"
    assert calls[1]["url"] == "https://self.example/api/memory-reviews/rev-1"
    assert calls[2]["method"] == "POST"
    assert calls[2]["url"] == "https://self.example/api/memory-reviews/rev-1/reject"
    assert json.loads(calls[2]["body"].decode()) == {"note": "Not durable enough."}


def test_mcp_proxy_requires_note_before_rejecting_memory_review(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeOpener:
        def open(self, request, timeout):
            raise AssertionError("reject without note should fail before HTTP")

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    assert proxy._call_tool(
        "resolve_memory_review",
        {"review_id": "rev-1", "decision": "reject"},
    ) == {"error": "note is required when decision is reject"}


def test_mcp_proxy_rejects_invalid_memory_review_pagination():
    proxy = _load_plugin_mcp_proxy()

    assert proxy._call_tool("list_memory_reviews", {"limit": "ten"}) == {"error": "limit must be an integer"}
    assert proxy._call_tool("list_memory_reviews", {"offset": "soon"}) == {"error": "offset must be an integer"}


def test_mcp_proxy_forwards_queryless_source_id_time_range(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-mounttai"]},
            "time_range": {
                "date_type": "source_updated_at",
                "start_date": "2026-06-20",
                "end_date": "2026-06-26",
            },
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode()) == {
        "include_private": True,
        "include_superseded": False,
        "source_filter": {"source_ids": ["src-mounttai"]},
        "time_range": {
            "date_type": "source_updated_at",
            "start_date": "2026-06-20",
            "end_date": "2026-06-26",
        },
    }


def test_mcp_proxy_forwards_search_offset_for_deterministic_listing(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[],"total_candidates":52,"candidate_count_kind":"exact","limit":10,"offset":10}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-backlog"]},
            "time_range": {"date_type": "source_updated_at", "start_date": "2026-06-29"},
            "top_k": 10,
            "offset": 10,
        },
    )

    assert result["total_candidates"] == 52
    assert result["has_more"] is True
    assert json.loads(captured["body"].decode()) == {
        "include_private": True,
        "include_superseded": False,
        "source_filter": {"source_ids": ["src-backlog"]},
        "time_range": {"date_type": "source_updated_at", "start_date": "2026-06-29"},
        "top_k": 10,
        "offset": 10,
    }


def test_mcp_proxy_does_not_guess_has_more_for_windowed_legacy_response(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[],"total_candidates":61,"candidate_count_kind":"windowed","limit":10,"offset":0}'

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "create blocker hint"})

    assert result == {
        "results": [],
        "total_candidates": 61,
        "candidate_count_kind": "windowed",
        "limit": 10,
        "offset": 0,
    }


def test_mcp_proxy_compacts_search_response_for_agent_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "query_analysis": {
                        "detected_entities": ["periodcutoffblockerhint"],
                        "entity_linking": [{"entity_id": 95, "matched_alias": "blocker hint"}],
                        "strategies_used": ["vector", "bm25_metadata_tokens", "graph"],
                    },
                    "results": [
                        {
                            "memory_id": "mem-1",
                            "memory_type": "fact",
                            "summary": "Create blocker hint task details",
                            "confidence": 0.9,
                            "relevance_score": 0.99,
                            "freshness": "current",
                            "status": "active",
                            "follow_up": {"suggested_tool": "get_memory"},
                            "retrieval_evidence": {"metadata_lexical": {"matched_text": ["large debug text"]}},
                            "repo_identifier": "repo",
                        }
                    ],
                    "total_candidates": 61,
                    "candidate_count_kind": "windowed",
                    "ranking_window_size": 50,
                    "limit": 10,
                    "offset": 0,
                    "has_more": True,
                    "retrieval_time_ms": 123,
                }
            ).encode()

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "create blocker hint"})

    assert result == {
        "results": [
            {
                "memory_id": "mem-1",
                "memory_type": "fact",
                "summary": "Create blocker hint task details",
                "confidence": 0.9,
                "relevance_score": 0.99,
                "freshness": "current",
                "status": "active",
                "follow_up": {"suggested_tool": "get_memory"},
            }
        ],
        "total_candidates": 61,
        "candidate_count_kind": "windowed",
        "ranking_window_size": 50,
        "limit": 10,
        "offset": 0,
        "has_more": True,
    }


def test_mcp_proxy_compacts_get_memory_response_for_agent_context(monkeypatch):
    proxy = _load_plugin_mcp_proxy()

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return json.dumps(
                {
                    "id": "mem-1",
                    "memory_type": "procedure",
                    "content": "Persist blocker hints.",
                    "content_hash": "sha256-secret",
                    "visibility": "workspace",
                    "owner_user_id": "user-1",
                    "project_key": "PAY",
                    "confidence": 0.93,
                    "corroboration_count": 1,
                    "contradiction_count": 0,
                    "status": "active",
                    "retirement_reason": None,
                    "replacement_reason": None,
                    "extraction_context": "large admin context",
                    "entity_refs": ["periodcutoffblockerhint"],
                    "sources": [
                        {
                            "doc_id": "jira-SFPAY-179397",
                            "source_type": "jira",
                            "support_kind": "extracted",
                            "doc_title": "SFPAY-179397: Create Blocker Hint",
                            "source_url": "https://jira.example/browse/SFPAY-179397",
                            "content_url": "/api/documents/jira-SFPAY-179397/content",
                            "pdf_url": None,
                            "source_updated_at": "2026-07-02T03:50:32+00:00",
                            "excerpt": "large excerpt",
                            "added_at": "2026-07-02T04:00:00+00:00",
                        }
                    ],
                }
            ).encode()

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool("get_memory", {"memory_id": "mem-1"})

    assert result == {
        "id": "mem-1",
        "memory_type": "procedure",
        "content": "Persist blocker hints.",
        "content_hash": "sha256-secret",
        "confidence": 0.93,
        "status": "active",
        "entity_refs": ["periodcutoffblockerhint"],
        "sources": [
            {
                "doc_id": "jira-SFPAY-179397",
                "source_type": "jira",
                "support_kind": "extracted",
                "doc_title": "SFPAY-179397: Create Blocker Hint",
                "source_url": "https://jira.example/browse/SFPAY-179397",
                "content_url": "/api/documents/jira-SFPAY-179397/content",
                "pdf_url": None,
                "source_updated_at": "2026-07-02T03:50:32+00:00",
            }
        ],
    }


@pytest.mark.parametrize("offset", [-1, "10", True])
def test_mcp_proxy_rejects_invalid_search_offset(monkeypatch, offset):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": ["src-backlog"]},
            "offset": offset,
        },
    )

    assert result == {"error": "offset must be a non-negative integer"}


@pytest.mark.parametrize("top_k", [0, 51, "10", True])
def test_mcp_proxy_rejects_invalid_search_top_k(monkeypatch, top_k):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "payroll",
            "top_k": top_k,
        },
    )

    assert result == {"error": "top_k must be an integer from 1 to 50"}


def test_mcp_proxy_rejects_queryless_without_deterministic_filter(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {})

    assert result == {"error": "search.query may be omitted only when source_filter or time_range is provided"}


def test_mcp_proxy_forwards_explicit_optional_date_bounds(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "jira memories",
            "time_range": {
                "date_type": "source_updated_at",
                "start_date": "2026-06-19",
            },
        },
    )

    assert result == {"results": []}
    assert json.loads(captured["body"].decode())["time_range"] == {
        "date_type": "source_updated_at",
        "start_date": "2026-06-19",
    }


@pytest.mark.parametrize(
    "time_range",
    [
        {},
        {"start_date": "2026-06-20T00:00:00Z"},
        {"after": "2026-06-20"},
        {"date_type": "created_at", "start_date": "2026-06-20"},
        {"start_date": "2026-06-21", "end_date": "2026-06-20"},
    ],
)
def test_mcp_proxy_rejects_invalid_time_range_shapes(monkeypatch, time_range):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool("search", {"query": "jira memories", "time_range": time_range})

    assert "error" in result
    assert "time_range" in result["error"]


def test_mcp_proxy_rejects_empty_source_ids(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "source_filter": {"source_ids": []},
            "time_range": {"start_date": "2026-06-20"},
        },
    )

    assert result == {"error": "source_filter.source_ids must be a non-empty array of source IDs from list_sources"}


def test_mcp_proxy_rejects_removed_backend_search_knobs(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "private memories",
            "include_private": True,
            "status": "active",
            "memory_types": ["decision"],
        },
    )

    assert result == {
        "error": (
            "Unsupported search parameter(s): include_private, memory_types, status. "
            "Omit unknown filters instead of guessing."
        )
    }


def test_mcp_proxy_rejects_unadvertised_source_filter_facets(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setattr(proxy, "_active_repo_identifier", lambda: None)

    result = proxy._call_tool(
        "search",
        {
            "query": "scheduler fix",
            "source_filter": {
                "source_types": ["agent_session"],
                "repo_identifiers": ["github.tools.sap/hcm/other"],
            },
        },
    )

    assert result == {
        "error": (
            "Unsupported source_filter parameter(s): repo_identifiers, source_types. "
            "Omit repo-scoped facets until MCP roots are available."
        )
    }


def test_mcp_proxy_fetches_resource_through_hosted_workspace(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    captured = {}

    class FakeResponse:
        headers = {"content-type": "text/markdown"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b"# Source"

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._call_tool(
        "get_resource",
        {"url": "/api/documents/doc-1/content", "mode": "text"},
    )

    assert result["text"] == "# Source"
    assert result["url"] == "/api/documents/doc-1/content"
    assert (
        captured["url"]
        == "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/documents/doc-1/content"
    )
    assert captured["authorization"] == "Bearer token-123"


def test_mcp_proxy_downloads_resource_to_local_cache(monkeypatch, tmp_path):
    proxy = _load_plugin_mcp_proxy()
    captured = {}
    body = b"%PDF-1.4\n%local-cache\n"

    class FakeResponse:
        headers = {
            "content-type": "application/pdf",
            "content-disposition": 'attachment; filename="source.pdf"',
            "content-length": str(len(body)),
        }

        def __init__(self):
            self._offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size=-1):
            if size is None or size < 0:
                size = len(body) - self._offset
            chunk = body[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(proxy, "build_opener", lambda *_handlers: FakeOpener())

    result = proxy._handle_get_resource(
        {
            "url": "/api/documents/doc-123/pdf",
            "mode": "file",
        }
    )

    local_path = Path(result["local_path"])
    assert captured["url"] == "http://memforge.test/api/documents/doc-123/pdf"
    assert result["mode"] == "file"
    assert result["content_type"] == "application/pdf"
    assert result["size_bytes"] == len(body)
    assert local_path.parent == tmp_path
    assert local_path.read_bytes() == body


def test_mcp_proxy_rejects_foreign_and_ambiguous_resource_urls(monkeypatch):
    proxy = _load_plugin_mcp_proxy()
    monkeypatch.setenv("MEMFORGE_API_URL", "https://self.example")

    foreign = proxy._handle_get_resource(
        {
            "url": "https://evil.example/api/documents/doc-123/pdf",
            "mode": "file",
        }
    )
    encoded_slash = proxy._handle_get_resource(
        {
            "url": "/api/documents/doc%2F123/pdf",
            "mode": "file",
        }
    )

    assert foreign["error"] == "unsupported resource URL"
    assert encoded_slash["error"] == "unsupported resource URL"


def _load_plugin_mcp_proxy():
    root = Path(__file__).resolve().parents[1]
    proxy_path = root / "src" / "memforge" / "plugin_mcp_proxy.py"
    spec = importlib.util.spec_from_file_location("memforge_mcp_test", proxy_path)
    assert spec is not None
    assert spec.loader is not None
    proxy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proxy)
    return proxy


def _provide_mcp_roots(proxy, *roots: Path) -> None:
    initialize = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test"},
            },
        }
    )
    assert initialize["result"]["serverInfo"]["name"] == "memforge"
    roots_request = proxy._handle_rpc_message(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
    )
    assert roots_request["method"] == "roots/list"
    assert (
        proxy._handle_rpc_message(
            {
                "jsonrpc": "2.0",
                "id": roots_request["id"],
                "result": {"roots": [{"uri": root.as_uri(), "name": root.name} for root in roots]},
            }
        )
        is None
    )


def test_malformed_timeout_env_fails_open(monkeypatch, capsys):
    from memforge import hook_adapter

    monkeypatch.setenv("MEMFORGE_HOOK_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({}))

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["continue"] is True
    assert "MemForge hook skipped" in output["systemMessage"]


def test_http_error_body_is_not_echoed_into_hook_output(monkeypatch, capsys):
    from memforge import hook_adapter

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        raise RuntimeError("/hooks/context returned HTTP 500: secret stack trace")

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(hook_adapter.sys, "stdin", _Stdin({"hook_event_name": "UserPromptSubmit"}))

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["continue"] is True
    assert "secret stack trace" not in output["systemMessage"]
    assert "MemForge hook skipped" in output["systemMessage"]


def test_post_json_targets_zero_configuration_local_oss(monkeypatch):
    from memforge import hook_adapter

    urls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout: float):
        urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setattr(hook_adapter.urllib.request, "urlopen", fake_urlopen)

    hook_adapter._post_json("/hooks/context", {}, timeout=1)
    hook_adapter._post_json("/hooks/context", {}, timeout=1)

    assert urls == [
        "http://127.0.0.1:8765/api/hooks/context",
        "http://127.0.0.1:8765/api/hooks/context",
    ]


def test_post_json_targets_hosted_workspace_when_configured(monkeypatch):
    from memforge import hook_adapter

    urls: list[str] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout: float):
        urls.append(request.full_url)
        return FakeResponse()

    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setenv("MEMFORGE_API_URL", "https://memforge.example.hana.ondemand.com")
    monkeypatch.setattr(hook_adapter.urllib.request, "urlopen", fake_urlopen)

    hook_adapter._post_json("/hooks/context", {}, timeout=1)
    hook_adapter._post_json("/agent-sessions/windows", {}, timeout=1)

    assert urls == [
        "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/hooks/context",
        "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/agent-sessions/windows",
    ]


def test_post_json_includes_configured_bearer_token(monkeypatch):
    from memforge import hook_adapter

    auth_headers: list[str | None] = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout: float):
        auth_headers.append(request.get_header("Authorization"))
        return FakeResponse()

    monkeypatch.setenv("MEMFORGE_API_TOKEN", "secret-token")
    monkeypatch.setattr(hook_adapter.urllib.request, "urlopen", fake_urlopen)

    hook_adapter._post_json("/hooks/context", {}, timeout=1)

    assert auth_headers == ["Bearer secret-token"]


def test_hook_adapter_uses_codex_plugin_config_when_hook_env_is_absent(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter
    from memforge import plugin_config

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
[memforge]
MEMFORGE_API_URL = "https://memforge.example.hana.ondemand.com"
MEMFORGE_API_TOKEN = "config-token"
MEMFORGE_WORKSPACE_ID = "mount_tai"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(codex_home / "config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)

    requests: list[tuple[str, float]] = []

    def fake_post_json(path: str, payload: dict, *, timeout: float):
        requests.append((path, timeout))
        return {}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "Stop",
                "session_id": "sess-config-fallback",
                "cwd": str(tmp_path),
            }
        ),
    )

    exit_code = hook_adapter.main(["submit-session"])

    assert exit_code == 0
    assert requests == [("/hooks/receipts", 5.0)]
    assert capsys.readouterr().out == ""


def test_post_json_uses_codex_plugin_config_for_workspace_and_token(monkeypatch, tmp_path):
    from memforge import hook_adapter
    from memforge import plugin_config

    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text(
        """
[memforge]
MEMFORGE_API_URL = "https://memforge.example.hana.ondemand.com"
MEMFORGE_API_TOKEN = "config-token"
MEMFORGE_WORKSPACE_ID = "mount_tai"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(codex_home / "config.toml"))
    monkeypatch.setattr(plugin_config, "_CONFIG_CACHE", None)
    observed: dict[str, str | None] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout: float):
        observed["url"] = request.full_url
        observed["authorization"] = request.get_header("Authorization")
        return FakeResponse()

    monkeypatch.setattr(hook_adapter.urllib.request, "urlopen", fake_urlopen)

    hook_adapter._post_json("/hooks/receipts", {}, timeout=1)

    assert observed == {
        "url": "https://memforge.example.hana.ondemand.com/api/workspaces/mount_tai/api/hooks/receipts",
        "authorization": "Bearer config-token",
    }


def test_session_start_guidance_explains_agentic_memforge_usage(monkeypatch, capsys):
    from memforge import hook_adapter

    monkeypatch.setattr(hook_adapter, "_drain_pending_agent_windows_if_present", lambda **_: None)
    monkeypatch.setattr(
        hook_adapter.sys,
        "stdin",
        _Stdin(
            {
                "hook_event_name": "SessionStart",
                "session_id": "sess-guidance",
            }
        ),
    )

    exit_code = hook_adapter.main(["context"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    guidance = output["hookSpecificOutput"]["additionalContext"]
    assert "Use it agentically" in guidance
    assert "get_memory" in guidance
    assert "get_resource" in guidance
    assert "Treat memory as context, not current truth" in guidance


def test_session_window_payload_redacts_before_network_and_versions_contract(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "tool",
                "name": "exec_command",
                "input": "Authorization: Bearer raw-secret-token",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": "api_key: raw-api-secret",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-redact",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=2,
        trigger="GATED_CAPTURE",
    )

    serialized = json.dumps(payload)
    assert "raw-secret-token" not in serialized
    assert "raw-api-secret" not in serialized
    assert "[REDACTED]" in serialized
    assert payload["schema_version"] == "agent-session-window/v1"
    assert payload["plugin_version"] == hook_adapter.PLUGIN_VERSION
    assert payload["receipt"]["metadata"]["uploaded_to_line"] == 2
    assert payload["receipt"]["metadata"]["observed_to_line"] == 2


def test_session_window_payload_sends_canonical_remote_repo_for_codex(tmp_path):
    from memforge import hook_adapter

    _init_git_repo_with_origin(tmp_path, "https://github.com/shno-labs/mem-forge.git")
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-codex-repo",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=1,
        trigger="REQUIRED_CAPTURE",
    )

    assert payload["repo"] == "github.com/shno-labs/mem-forge"


def test_bounded_transcript_slice_never_advances_past_lines_read(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        '{"type":"tool","name":"one","input":"edit"}\n{"type":"tool","name":"two","input":"test"}\n',
        encoding="utf-8",
    )

    lines, effective_end, truncated = hook_adapter._bounded_transcript_line_slice(str(transcript), 0, 3)

    assert len(lines) == 2
    assert effective_end == 2
    assert truncated is False


def test_extract_transcript_events_understands_codex_payload_shape():
    from memforge import hook_adapter

    text = "\n".join(
        [
            json.dumps(
                {
                    "timestamp": "2026-05-30T12:00:00Z",
                    "type": "response_item",
                    "payload": {
                        "type": "function_call",
                        "name": "apply_patch",
                        "arguments": '{"cmd":"edit docs"}',
                    },
                }
            ),
            json.dumps(
                {
                    "timestamp": "2026-05-30T12:00:01Z",
                    "type": "event_msg",
                    "payload": {
                        "type": "agent_message",
                        "message": "Implemented the boundary fix.",
                    },
                }
            ),
        ]
    )

    events = hook_adapter._extract_transcript_events(text)

    assert events == [
        {
            "kind": "tool_call",
            "actor": "assistant",
            "source_type": "response_item",
            "native_type": "function_call",
            "name": "apply_patch",
            "timestamp": "2026-05-30T12:00:00Z",
            "text": '{"cmd":"edit docs"}',
        },
        {
            "kind": "assistant_message",
            "actor": "assistant",
            "source_type": "event_msg",
            "native_type": "agent_message",
            "timestamp": "2026-05-30T12:00:01Z",
            "text": "Implemented the boundary fix.",
        },
    ]


def test_extract_transcript_events_understands_claude_nested_content_shape():
    from memforge import hook_adapter

    text = json.dumps(
        {
            "type": "assistant",
            "timestamp": "2026-05-30T12:00:02Z",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "pytest tests/test_hook_adapter.py -q"},
                    },
                    {"type": "text", "text": "I will run the focused test."},
                ],
            },
        }
    )

    events = hook_adapter._extract_transcript_events(text)

    assert events == [
        {
            "kind": "tool_call",
            "actor": "assistant",
            "source_type": "assistant",
            "native_type": "tool_use",
            "name": "Bash",
            "timestamp": "2026-05-30T12:00:02Z",
            "text": '{"command":"pytest tests/test_hook_adapter.py -q"}',
        }
    ]


def test_extract_transcript_events_keeps_recent_within_cap():
    from memforge import hook_adapter

    text = "\n".join(
        json.dumps({"type": "tool", "name": f"tool-{i:03d}", "input": "x"}) for i in range(hook_adapter.MAX_EVENTS + 20)
    )
    events = hook_adapter._extract_transcript_events(text)

    assert len(events) == hook_adapter.MAX_EVENTS
    # The most recent events are kept, not the earliest, so the structured events
    # line up with the tail the transcript text is taken from.
    assert events[0]["name"] == "tool-020"
    assert events[-1]["name"] == f"tool-{hook_adapter.MAX_EVENTS + 19:03d}"


def test_session_window_payload_events_and_text_track_same_tail(tmp_path):
    from memforge import hook_adapter

    n = hook_adapter.MAX_EVENTS + 20
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        "\n".join(json.dumps({"type": "tool", "name": f"tool-{i:03d}", "input": "x"}) for i in range(n)) + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-tail",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=n,
        trigger="GATED_CAPTURE",
    )

    events = payload["events"]
    assert len(events) == hook_adapter.MAX_EVENTS
    assert events[0]["name"] == "tool-000"
    assert events[-1]["name"] == f"tool-{hook_adapter.MAX_EVENTS - 1:03d}"
    # The window is now a bounded prefix chunk over canonical evidence, so the
    # bookmark stops at the last uploaded event instead of claiming the full tail.
    assert f"tool-{hook_adapter.MAX_EVENTS - 1:03d}" in payload["transcript_markdown"]
    assert f"tool-{hook_adapter.MAX_EVENTS:03d}" not in payload["transcript_markdown"]
    assert payload["history_window"]["start"] == "0"
    assert payload["history_window"]["end"] == str(hook_adapter.MAX_EVENTS)
    assert payload["history_window"]["truncated"] is True


def test_session_window_payload_filters_codex_bootstrap_before_budgeting(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "codex-long-bootstrap.jsonl"
    lines = [
        json.dumps({"type": "session_meta", "payload": {"cwd": str(tmp_path)}}),
        json.dumps({"type": "turn_context", "payload": {"huge": "startup"}}),
        json.dumps({"type": "event_msg", "payload": {"type": "task_started", "message": "boot"}}),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "private developer instruction"}],
                },
            }
        ),
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /tmp\n<INSTRUCTIONS>\nnoise\n</INSTRUCTIONS>",
                        }
                    ],
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-05-30T12:00:10Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "apply_patch",
                    "arguments": '{"file":"src/memforge/hook_adapter.py"}',
                },
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-05-30T12:00:11Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "Updated canonical evidence extraction.",
                },
            }
        ),
    ]
    transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-bootstrap",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=len(lines),
        trigger="REQUIRED_CAPTURE",
    )

    assert payload["history_window"]["start"] == "0"
    assert payload["history_window"]["end"] == str(len(lines))
    assert payload["events"] == [
        {
            "kind": "tool_call",
            "actor": "assistant",
            "source_type": "response_item",
            "native_type": "function_call",
            "name": "apply_patch",
            "timestamp": "2026-05-30T12:00:10Z",
            "text": '{"file":"src/memforge/hook_adapter.py"}',
        },
        {
            "kind": "tool_result",
            "actor": "tool",
            "source_type": "response_item",
            "native_type": "function_call_output",
            "timestamp": "2026-05-30T12:00:11Z",
            "text": "Updated canonical evidence extraction.",
        },
    ]
    assert "session_meta" not in payload["transcript_markdown"]
    assert "task_started" not in payload["transcript_markdown"]
    assert "private developer instruction" not in payload["transcript_markdown"]
    assert "AGENTS.md" not in payload["transcript_markdown"]
    assert "apply_patch" in payload["transcript_markdown"]
    assert payload["source_updated_at"] == "2026-05-30T12:00:10+00:00"
    assert payload["receipt"]["metadata"]["omissions"]["metadata_or_context"] == 5


def test_session_window_payload_uses_earliest_offset_aware_event_timestamp(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "codex-out-of-order.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-30T12:00:30Z",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "later event"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-30T12:00:10+00:00",
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "earlier event"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-out-of-order",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=2,
        trigger="REQUIRED_CAPTURE",
    )

    assert payload["source_updated_at"] == "2026-05-30T12:00:10+00:00"


def test_session_window_payload_rejects_naive_source_updated_at(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "codex-naive-timestamp.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-30T12:00:10",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "naive timestamp"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="timezone offset"):
        hook_adapter._session_window_payload(
            client="codex",
            session_id="sess-naive-timestamp",
            transcript_path=str(transcript),
            workspace=str(tmp_path),
            from_line=0,
            to_line=1,
            trigger="REQUIRED_CAPTURE",
        )


def test_session_window_payload_rejects_invalid_source_updated_at(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "codex-invalid-timestamp.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "timestamp": "not-a-timestamp",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "invalid timestamp"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid event timestamp"):
        hook_adapter._session_window_payload(
            client="codex",
            session_id="sess-invalid-timestamp",
            transcript_path=str(transcript),
            workspace=str(tmp_path),
            from_line=0,
            to_line=1,
            trigger="REQUIRED_CAPTURE",
        )


def test_session_window_payload_omits_source_updated_at_without_event_timestamp(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "codex-no-timestamp.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "pytest passed",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-no-timestamp",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=1,
        trigger="REQUIRED_CAPTURE",
    )

    assert "source_updated_at" not in payload


def test_session_window_payload_middle_truncates_oversized_evidence_line(monkeypatch, tmp_path):
    from memforge import hook_adapter

    monkeypatch.setattr(hook_adapter, "MAX_TRANSCRIPT_CHARS", 220)
    transcript = tmp_path / "codex-huge-line.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "command started\n" + ("x" * 1000) + "\npytest passed",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="codex",
        session_id="sess-huge",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=1,
        trigger="REQUIRED_CAPTURE",
    )

    assert payload["history_window"]["end"] == "1"
    assert payload["history_window"]["truncated"] is True
    assert payload["events"][0]["kind"] == "tool_result"
    assert "command started" in payload["events"][0]["text"]
    assert "pytest passed" in payload["events"][0]["text"]
    assert "truncated" in payload["events"][0]["text"]


def test_session_window_payload_redacts_claude_nested_json_secret_before_network(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "claude-json-secret.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {
                                "command": "pytest tests/test_hook_adapter.py -q",
                                "api_key": "claude-json-api-key-value",
                            },
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = hook_adapter._session_window_payload(
        client="claude-code",
        session_id="sess-claude-json-secret",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        from_line=0,
        to_line=1,
        trigger="REQUIRED_CAPTURE",
    )

    serialized = json.dumps(payload)
    assert "claude-json-api-key-value" not in serialized
    assert "api_key" in serialized
    assert "[REDACTED]" in serialized


def test_worker_single_flight_skips_when_locked(monkeypatch, tmp_path):
    from memforge import hook_adapter

    if hook_adapter.fcntl is None:
        pytest.skip("file locking unavailable on this platform")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="sess-lock",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    posts: list[str] = []

    def fake_post_json(path, payload, *, timeout):
        posts.append(path)
        return {"ok": True}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    held = hook_adapter._acquire_worker_lock(queue_db)
    assert held is not None and held is not hook_adapter._NULL_WORKER_LOCK
    try:
        skipped = hook_adapter.run_agent_window_worker_once(timeout=5, queue_db_path=queue_db)
    finally:
        hook_adapter._release_worker_lock(held)

    assert skipped == 0
    assert posts == []  # nothing processed while another worker holds the lock

    processed = hook_adapter.run_agent_window_worker_once(timeout=5, queue_db_path=queue_db)
    assert processed == 1
    assert posts and posts[0] == "/agent-sessions/windows"


def test_drain_does_not_spawn_when_no_pending(monkeypatch, tmp_path):
    from memforge import hook_adapter

    queue_db = tmp_path / "queue.sqlite"
    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(queue_db))
    spawned: list = []
    monkeypatch.setattr(
        hook_adapter,
        "_spawn_agent_window_worker",
        lambda *, timeout: spawned.append(timeout),
    )

    # No queue file yet: nothing pending, so no worker is spawned.
    hook_adapter._drain_pending_agent_windows_if_present()
    assert spawned == []

    # A session_cursor with capture_pending=0 also has nothing pending.
    with sqlite3.connect(queue_db) as connection:
        hook_adapter._ensure_session_cursor(connection)
        connection.execute(
            "INSERT INTO session_cursor (client, session_id, captured_through, capture_pending, created_at, updated_at) "
            "VALUES ('codex', 's', 3, 0, 'now', 'now')"
        )
    hook_adapter._drain_pending_agent_windows_if_present()
    assert spawned == []


def test_worker_keeps_pending_when_capture_requested_during_upload(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="sess",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    def fake_post_json(path, payload, *, timeout):
        if path == "/agent-sessions/windows":
            # A hook fires during the in-flight upload, requesting another capture.
            hook_adapter.request_session_capture(
                client="codex",
                session_id="sess",
                transcript_path=str(transcript),
                workspace=str(tmp_path),
                trigger="GATED_CAPTURE",
                queue_db_path=queue_db,
            )
        return {"ok": True}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(timeout=5, queue_db_path=queue_db)

    assert submitted == 1
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute("SELECT capture_pending, captured_through FROM session_cursor").fetchone()
    # Bookmark advanced to the uploaded count, but the mid-upload request is preserved.
    assert row == (1, 1)
    assert capsys.readouterr().out == ""


def test_worker_keeps_pending_when_request_timestamp_collides(monkeypatch, tmp_path, capsys):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n', encoding="utf-8")
    queue_db = tmp_path / "queue.sqlite"
    constant_now = "2026-05-30T00:00:00+00:00"
    monkeypatch.setattr(hook_adapter, "_now_iso", lambda: constant_now)
    hook_adapter.request_session_capture(
        client="codex",
        session_id="sess",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    def fake_post_json(path, payload, *, timeout):
        if path == "/agent-sessions/windows":
            transcript.write_text(
                '{"type":"tool","name":"apply_patch","input":"edit"}\n'
                '{"type":"tool","name":"pytest","input":"tests"}\n',
                encoding="utf-8",
            )
            hook_adapter.request_session_capture(
                client="codex",
                session_id="sess",
                transcript_path=str(transcript),
                workspace=str(tmp_path),
                trigger="GATED_CAPTURE",
                queue_db_path=queue_db,
            )
        return {"ok": True}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    submitted = hook_adapter.run_agent_window_worker_once(timeout=5, queue_db_path=queue_db)

    assert submitted == 1
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute("SELECT capture_pending, captured_through, request_seq FROM session_cursor").fetchone()
    assert row == (1, 1, 2)
    assert capsys.readouterr().out == ""


def test_stale_worker_cannot_rewind_bookmark_after_lease_reclaim(monkeypatch, tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_lines(transcript, 10)
    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="sess",
        transcript_path=str(transcript),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )
    calls = {"window_uploads": 0}

    def fake_post_json(path, payload, *, timeout):
        if path == "/agent-sessions/windows":
            calls["window_uploads"] += 1
            if calls["window_uploads"] == 1:
                _write_transcript_lines(transcript, 20)
                with sqlite3.connect(queue_db) as connection:
                    connection.execute(
                        "UPDATE session_cursor SET lease_until = ? WHERE client = ? AND session_id = ?",
                        ("1970-01-01T00:00:00+00:00", "codex", "sess"),
                    )
                hook_adapter._process_session_captures(queue_db, timeout=5, max_sessions=5)
        return {"ok": True}

    monkeypatch.setattr(hook_adapter, "_post_json", fake_post_json)

    hook_adapter._process_session_captures(queue_db, timeout=5, max_sessions=5)

    with sqlite3.connect(queue_db) as connection:
        row = connection.execute("SELECT captured_through, capture_pending FROM session_cursor").fetchone()
    assert row == (20, 0)


def test_recover_does_not_overwrite_concurrent_required_capture_request(monkeypatch, tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_lines(transcript, 6)
    queue_db = tmp_path / "queue.sqlite"
    _seed_session_cursor(queue_db, transcript_path=str(transcript), captured_through=2, capture_pending=0)
    requested = {"done": False}

    def racing_line_count(path):
        if not requested["done"]:
            requested["done"] = True
            hook_adapter.request_session_capture(
                client="codex",
                session_id="sess",
                transcript_path=str(transcript),
                workspace=str(tmp_path),
                trigger="REQUIRED_CAPTURE",
                queue_db_path=queue_db,
            )
        return 6

    monkeypatch.setattr(hook_adapter, "_transcript_line_count", racing_line_count)

    rearmed = hook_adapter._recover_incomplete_sessions(client="codex", queue_db_path=queue_db)

    assert rearmed == 0
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute(
            "SELECT capture_pending, pending_trigger, captured_through, request_seq FROM session_cursor"
        ).fetchone()
    assert row == (1, "REQUIRED_CAPTURE", 2, 1)


class _Stdin:
    def __init__(self, payload: dict) -> None:
        self._text = json.dumps(payload)

    def read(self) -> str:
        return self._text


def _seed_session_cursor(queue_db, **fields):
    from memforge import hook_adapter

    columns = (
        "client",
        "session_id",
        "transcript_path",
        "captured_through",
        "capture_pending",
        "pending_trigger",
        "created_at",
        "updated_at",
    )
    values = {
        "client": "codex",
        "session_id": "sess",
        "transcript_path": None,
        "captured_through": 0,
        "capture_pending": 0,
        "pending_trigger": None,
        "created_at": "now",
        "updated_at": "now",
    }
    values.update(fields)
    with sqlite3.connect(queue_db) as connection:
        hook_adapter._ensure_session_cursor(connection)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO session_cursor ({', '.join(columns)}) VALUES ({placeholders})",
            tuple(values[name] for name in columns),
        )


def _write_transcript_lines(path, count):
    path.write_text("\n".join(f'{{"i":{index}}}' for index in range(count)) + "\n", encoding="utf-8")


def test__is_recover_event_only_session_start():
    from memforge import hook_adapter

    assert hook_adapter._is_recover_event("SessionStart") is True
    for event in ("Stop", "SubagentStop", "PreCompact", "UserPromptSubmit", "UnknownHook"):
        assert hook_adapter._is_recover_event(event) is False


def test_recover_rearms_session_with_uncaptured_tail(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_lines(transcript, 6)
    queue_db = tmp_path / "queue.sqlite"
    _seed_session_cursor(queue_db, transcript_path=str(transcript), captured_through=2, capture_pending=0)

    rearmed = hook_adapter._recover_incomplete_sessions(client="codex", queue_db_path=queue_db)

    assert rearmed == 1
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute(
            "SELECT capture_pending, pending_trigger, captured_through FROM session_cursor"
        ).fetchone()
    assert row == (1, "RECOVER", 2)  # re-armed for the tail, bookmark left where it was


def test_recover_skips_caught_up_session(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_lines(transcript, 6)
    queue_db = tmp_path / "queue.sqlite"
    _seed_session_cursor(queue_db, transcript_path=str(transcript), captured_through=6, capture_pending=0)

    rearmed = hook_adapter._recover_incomplete_sessions(client="codex", queue_db_path=queue_db)

    assert rearmed == 0
    with sqlite3.connect(queue_db) as connection:
        assert connection.execute("SELECT capture_pending FROM session_cursor").fetchone() == (0,)


def test_recover_skips_already_pending_session(tmp_path):
    from memforge import hook_adapter

    transcript = tmp_path / "transcript.jsonl"
    _write_transcript_lines(transcript, 6)
    queue_db = tmp_path / "queue.sqlite"
    _seed_session_cursor(
        queue_db,
        transcript_path=str(transcript),
        captured_through=2,
        capture_pending=1,
        pending_trigger="REQUIRED_CAPTURE",
    )

    rearmed = hook_adapter._recover_incomplete_sessions(client="codex", queue_db_path=queue_db)

    assert rearmed == 0
    with sqlite3.connect(queue_db) as connection:
        row = connection.execute("SELECT capture_pending, pending_trigger FROM session_cursor").fetchone()
    assert row == (1, "REQUIRED_CAPTURE")  # a pending row is left untouched, its required capture preserved


def test_recover_skips_missing_transcript(tmp_path):
    from memforge import hook_adapter

    queue_db = tmp_path / "queue.sqlite"
    _seed_session_cursor(queue_db, transcript_path=str(tmp_path / "gone.jsonl"), captured_through=2, capture_pending=0)

    rearmed = hook_adapter._recover_incomplete_sessions(client="codex", queue_db_path=queue_db)

    assert rearmed == 0
    with sqlite3.connect(queue_db) as connection:
        assert connection.execute("SELECT capture_pending FROM session_cursor").fetchone() == (0,)


def test_queue_uses_wal_mode(tmp_path):
    from memforge import hook_adapter

    queue_db = tmp_path / "queue.sqlite"
    hook_adapter.request_session_capture(
        client="codex",
        session_id="sess",
        transcript_path=str(tmp_path / "t.jsonl"),
        workspace=str(tmp_path),
        trigger="GATED_CAPTURE",
        queue_db_path=queue_db,
    )

    with sqlite3.connect(queue_db) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
