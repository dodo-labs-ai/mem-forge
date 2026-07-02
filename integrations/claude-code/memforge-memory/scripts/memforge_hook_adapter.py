"""Command hook adapter for Codex and Claude Code plugins.

The adapter is intentionally thin: it translates provider hook JSON into
MemForge Admin API calls and translates responses back into hook output.
It does not perform memory extraction and it does not store transcripts.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from .plugin_config import configured_api_token, configured_api_url, configured_workspace_id
except ImportError:  # pragma: no cover - copied plugin package or direct file load
    try:
        from memforge_plugin_config import configured_api_token, configured_api_url, configured_workspace_id
    except ImportError:
        import importlib.util

        _config_path = Path(__file__).with_name("memforge_plugin_config.py")
        if not _config_path.exists():
            _config_path = Path(__file__).with_name("plugin_config.py")
        _config_spec = importlib.util.spec_from_file_location("memforge_plugin_config", _config_path)
        if _config_spec is None or _config_spec.loader is None:
            raise
        _config_module = importlib.util.module_from_spec(_config_spec)
        _config_spec.loader.exec_module(_config_module)
        configured_api_token = _config_module.configured_api_token
        configured_api_url = _config_module.configured_api_url
        configured_workspace_id = _config_module.configured_workspace_id

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

DEFAULT_API_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_AGENT_WORKER_TIMEOUT_SECONDS = 180.0
DEFAULT_AGENT_QUEUE_DB = Path.home() / ".memforge-agent" / "queue.sqlite"
MAX_TRANSCRIPT_CHARS = 60000
MAX_EVENTS = 40
WORKER_LEASE_BUFFER_SECONDS = 60.0
QUEUE_BUSY_TIMEOUT_MS = 5000  # how long a queue connection waits on a busy lock
WINDOW_SCHEMA_VERSION = "agent-session-window/v1"
PLUGIN_VERSION = "0.1.21-rc.5"
SESSION_START_USAGE_GUIDANCE = (
    "## MemForge Usage Guidance\n\n"
    "MemForge is long-term memory for prior decisions, conventions, debugging "
    "history, source context, and user preferences. Use it agentically:\n\n"
    "- Before non-trivial repo or workspace work, use the MemForge MCP search "
    "tool with a natural language query. Omit filters when unsure so you do not "
    "accidentally hide relevant memories.\n"
    "- Use source filters only when the user or task gives a clear facet. Use "
    "`current_repo_only` for current-repo agent-session history and `clients` "
    "only when the user names Codex or Claude Code. When the user names a "
    "configured knowledge source, call `list_sources` first and pass exact "
    "`source_ids` from that result; do not guess source ids from source names "
    "or source types.\n"
    "- If a memory affects an answer, review, or code change, call `get_memory` "
    "for full content and provenance before relying on it.\n"
    "- Search results do not include source links or artifact URLs. Use "
    "`search -> get_memory -> get_resource` when the user needs source "
    "evidence, quotes, or exact document context.\n"
    "- Treat memory as context, not current truth. Verify current files, tests, "
    "runtime state, or external systems when facts may have changed.\n"
    "- Proactively detect memory creation, corrections, or retirements (`remember this`, "
    "`not X, actually Y`, `don't use this anymore`). For create, search first "
    "to avoid duplicates. For replace/retire, locate the memory with "
    "`search`/`get_memory`. Show a readable preview: new claim or old/new claim, "
    "provenance/evidence, scope, and type/tags.\n"
    "- For `create_memory`, confirmed content must be the durable memory only. Keep "
    "confirmation details, test/deploy notes, and why-the-tool-was-called out "
    "of content; put source evidence/details in `provenance`.\n"
    "- Never mutate memory silently. Before `create_memory`, `replace_memory`, "
    "`retire_memory`, or `resolve_memory_review`, confirm via `request_user_input` if available; "
    "otherwise ask a concise text question. Do not show raw tool arguments unless needed.\n"
    "- If no relevant memory is found, continue normally and say so only when it "
    "matters to the user.\n"
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer", re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+")),
    ("json", re.compile(r"(?i)([\"']?(?:api[_-]?key|token|password|secret)[\"']?\s*:\s*[\"'])[^\"']+")),
    ("generic", re.compile(r"(?i)\b(api[_-]?key|token|password|secret)\b\s*[:=]\s*[^,\s\"'}]+")),
)
STOP_SIGNAL_TERMS = (
    "apply_patch",
    "write_file",
    "edit_file",
    '"write"',
    "file edit",
    "pytest",
    "npm test",
    "pnpm test",
    "yarn test",
    "mvn test",
    "go test",
    "cargo test",
    "uv run",
    "git diff",
    "git status",
    "git commit",
    "verification",
    "verified",
    "remember",
    "capture",
    "submit",
    "sync",
    "implemented",
    "fixed",
    "tests pass",
    "tests passed",
    "complete",
    "completed",
)
NOISE_EVENT_TYPES = {
    "session_meta",
    "turn_context",
    "compacted",
    "context_compaction",
    "compaction",
    "compaction_trigger",
    "task_started",
    "reasoning",
    "thinking",
    "system",
    "queue-operation",
    "last-prompt",
    "attachment",
}
CODEX_TOOL_CALL_TYPES = {
    "function_call",
    "tool_search_call",
    "custom_tool_call",
    "local_shell_call",
    "web_search_call",
}
CODEX_TOOL_RESULT_TYPES = {
    "function_call_output",
    "tool_search_output",
    "custom_tool_call_output",
}

# Capture policies. Each client hook maps onto one of these.
REQUIRED_CAPTURE_TRIGGER = "REQUIRED_CAPTURE"  # context about to be lost: always capture
GATED_CAPTURE_TRIGGER = "GATED_CAPTURE"  # turn ended: capture only on durable signal
RECOVER_TRIGGER = "RECOVER"  # session resumed: re-arm an uncaptured tail
LEGACY_REQUIRED_CAPTURE_TRIGGER = "BOUNDARY"
LEGACY_GATED_CAPTURE_TRIGGER = "FLUSH"
_CAPTURE_TRIGGER_ALIASES = {
    LEGACY_REQUIRED_CAPTURE_TRIGGER: REQUIRED_CAPTURE_TRIGGER,
    LEGACY_GATED_CAPTURE_TRIGGER: GATED_CAPTURE_TRIGGER,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MemForge agent hook adapter")
    parser.add_argument("mode", choices=("context", "submit-session", "worker-run-once"))
    parser.add_argument("--client", default=os.getenv("MEMFORGE_HOOK_CLIENT", "codex"))
    parser.add_argument("--api-url", default=configured_api_url(DEFAULT_API_URL))
    parser.add_argument(
        "--timeout", type=float, default=_env_float("MEMFORGE_HOOK_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "worker-run-once":
            run_agent_window_worker_once(api_url=args.api_url, timeout=args.timeout)
            return 0
        payload = _read_hook_payload()
        if args.mode == "context":
            return _run_context(payload, client=args.client, api_url=args.api_url, timeout=args.timeout)
        return _run_submit_session(payload, client=args.client, api_url=args.api_url, timeout=args.timeout)
    except Exception as exc:  # Hooks must not crash the coding session.
        print(
            json.dumps(
                {
                    "continue": True,
                    "systemMessage": f"MemForge hook skipped: {_safe_exception_message(exc)}",
                }
            ),
        )
        return 0


def _run_context(payload: dict[str, Any], *, client: str, api_url: str, timeout: float) -> int:
    event_name = _event_name(payload)
    if event_name == "SessionStart":
        _emit_additional_context(event_name, SESSION_START_USAGE_GUIDANCE)
        if _is_recover_event(event_name):
            try:
                _recover_incomplete_sessions(
                    client=client,
                    session_id=str(payload.get("session_id") or "") or None,
                )
            except Exception:
                pass
        _drain_pending_agent_windows_if_present(api_url=api_url, timeout=timeout)
        return 0

    request = {
        "client": client,
        "hook": event_name,
        "workspace": _workspace(payload),
        "repo": _repo_name(payload),
        "branch": _git_value(payload, ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "prompt": payload.get("prompt"),
        "touched_files": _touched_files(payload),
        "max_memories": _env_int("MEMFORGE_HOOK_MAX_MEMORIES", 5),
    }
    response = _post_json("/api/hooks/context", request, api_url=api_url, timeout=timeout)
    if not response.get("should_inject"):
        context = None
    else:
        context = response.get("context_markdown")
    if context:
        _emit_additional_context(event_name, context)

    if _is_recover_event(event_name):
        try:
            _recover_incomplete_sessions(
                client=client,
                session_id=str(payload.get("session_id") or "") or None,
            )
        except Exception:
            pass
    _drain_pending_agent_windows_if_present(api_url=api_url, timeout=timeout)
    return 0


def _emit_additional_context(event_name: str, context: str) -> None:
    print(
        json.dumps(
            {
                "continue": True,
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": context,
                },
            }
        )
    )


def _run_submit_session(payload: dict[str, Any], *, client: str, api_url: str, timeout: float) -> int:
    event_name = _event_name(payload)
    transcript_path = _transcript_path(payload)
    trigger = _capture_trigger(event_name)
    if trigger and transcript_path:
        try:
            if _should_request_capture(trigger, transcript_path, client=client, payload=payload):
                request_session_capture(
                    client=client,
                    session_id=str(payload.get("session_id") or "unknown-session"),
                    transcript_path=transcript_path,
                    workspace=_workspace(payload),
                    trigger=trigger,
                )
                _spawn_agent_window_worker(api_url=api_url, timeout=_agent_worker_timeout())
        except Exception:
            pass

    workspace = _workspace(payload)
    request = {
        "client": client,
        "session_id": str(payload.get("session_id") or "unknown-session"),
        "hook": event_name,
        "workspace": workspace,
        "repo": _repo_name(payload),
        "branch": _git_value(payload, ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_sha": _git_value(payload, ["git", "rev-parse", "HEAD"]),
        "metadata": _session_metadata(payload),
    }
    _post_json("/api/hooks/receipts", request, api_url=api_url, timeout=timeout)
    return 0


# ---------------------------------------------------------------------------
# Capture gate: which hooks request a window, measured over the new delta
# ---------------------------------------------------------------------------


def _capture_trigger(event_name: str) -> str | None:
    """Map a client hook event onto a capture policy."""
    if event_name == "PreCompact":
        return REQUIRED_CAPTURE_TRIGGER
    if event_name in ("Stop", "SubagentStop"):
        return GATED_CAPTURE_TRIGGER
    return None


def _normalize_capture_trigger(trigger: str | None) -> str | None:
    if trigger is None:
        return None
    return _CAPTURE_TRIGGER_ALIASES.get(trigger, trigger)


def _should_request_capture(
    trigger: str,
    transcript_path: str,
    *,
    client: str,
    payload: dict[str, Any],
) -> bool:
    """Decide whether to request a capture, measured over the delta since the bookmark.

    REQUIRED_CAPTURE always captures new content. GATED_CAPTURE captures only
    when the delta since the last captured window carries durable signal, so
    trivial turns fold forward into the next window instead.
    """
    trigger = _normalize_capture_trigger(trigger)
    if not Path(transcript_path).exists():
        return False
    session_id = str(payload.get("session_id") or "unknown-session")
    captured_through = _get_captured_through(client, session_id)
    count = _transcript_line_count(transcript_path)
    if count <= captured_through:
        return False
    if trigger == REQUIRED_CAPTURE_TRIGGER:
        return True
    if trigger != GATED_CAPTURE_TRIGGER:
        return False
    return _has_stop_window_signal_in_transcript(transcript_path, captured_through, count)


def request_session_capture(
    *,
    client: str,
    session_id: str,
    transcript_path: str,
    workspace: str,
    trigger: str,
    queue_db_path: str | Path | None = None,
) -> None:
    """Mark a session as needing a capture. The range is materialized at upload.

    Setting the flag is idempotent, so any number of hooks firing before the
    worker runs collapse into a single pending capture. A REQUIRED_CAPTURE
    request wins over a GATED_CAPTURE request for the same session.
    """
    trigger = _normalize_capture_trigger(trigger) or trigger
    db_path = _agent_queue_db_path(queue_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    now = _now_iso()
    with sqlite3.connect(db_path) as connection:
        _ensure_session_cursor(connection)
        connection.execute(
            """
            INSERT INTO session_cursor (
                client, session_id, transcript_path, workspace,
                captured_through, capture_pending, pending_trigger, request_seq, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, 0, 1, ?, 1, ?, ?)
            ON CONFLICT(client, session_id) DO UPDATE SET
                transcript_path = excluded.transcript_path,
                workspace = excluded.workspace,
                capture_pending = 1,
                request_seq = session_cursor.request_seq + 1,
                pending_trigger = CASE
                    WHEN session_cursor.pending_trigger IN ('REQUIRED_CAPTURE', 'BOUNDARY')
                         OR excluded.pending_trigger IN ('REQUIRED_CAPTURE', 'BOUNDARY')
                    THEN 'REQUIRED_CAPTURE'
                    ELSE excluded.pending_trigger
                END,
                updated_at = excluded.updated_at
            """,
            (client, session_id, transcript_path, workspace, trigger, now, now),
        )


def _get_captured_through(client: str, session_id: str, queue_db_path: str | Path | None = None) -> int:
    db_path = _agent_queue_db_path(queue_db_path)
    if not db_path.exists():
        return 0
    try:
        with sqlite3.connect(db_path) as connection:
            _ensure_session_cursor(connection)
            row = connection.execute(
                "SELECT captured_through FROM session_cursor WHERE client = ? AND session_id = ?",
                (client, session_id),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    except sqlite3.Error:
        return 0


# ---------------------------------------------------------------------------
# Self-heal: re-arm sessions that grew but never requested a final capture
# ---------------------------------------------------------------------------


def _is_recover_event(event_name: str) -> bool:
    """A RECOVER event (session start or resume) re-arms any uncaptured tail."""
    return event_name == "SessionStart"


def _incomplete_sessions(
    connection: sqlite3.Connection,
    *,
    client: str,
    session_id: str | None = None,
    max_sessions: int = 10,
) -> list[tuple]:
    """Return idle sessions whose tail may be uncaptured (client-side completeness).

    Only rows with capture_pending = 0 are returned: a row that is already
    pending or leased is being handled by the worker and must not be disturbed.
    The caller compares the transcript line count to captured_through to decide
    which of these idle sessions still has an uncaptured tail.
    """
    query = (
        "SELECT session_id, transcript_path, captured_through FROM session_cursor "
        "WHERE client = ? AND capture_pending = 0"
    )
    params: list[Any] = [client]
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY updated_at LIMIT ?"
    params.append(max_sessions)
    return connection.execute(query, params).fetchall()


def _recover_incomplete_sessions(
    *,
    client: str,
    session_id: str | None = None,
    queue_db_path: str | Path | None = None,
    max_sessions: int = 10,
) -> int:
    """Re-arm idle sessions whose transcript grew past the bookmark.

    On RECOVER, an idle session whose transcript has more lines than its
    bookmark is marked capture_pending so the normal worker drains the
    uncaptured tail. The update is conditional on the row still being idle and
    still having the observed bookmark, so concurrent capture requests win. It
    never raises: a hook must not crash the session.
    """
    db_path = _agent_queue_db_path(queue_db_path)
    if not db_path.exists():
        return 0
    rearmed = 0
    try:
        with sqlite3.connect(db_path) as connection:
            connection.isolation_level = None
            _ensure_session_cursor(connection)
            now = _now_iso()
            for row_session_id, transcript_path, captured_through in _incomplete_sessions(
                connection,
                client=client,
                session_id=session_id,
                max_sessions=max_sessions,
            ):
                if not transcript_path or not Path(transcript_path).exists():
                    continue
                if _transcript_line_count(transcript_path) <= int(captured_through or 0):
                    continue
                cursor = connection.execute(
                    "UPDATE session_cursor SET capture_pending = 1, pending_trigger = ?, "
                    "request_seq = request_seq + 1, updated_at = ? "
                    "WHERE client = ? AND session_id = ? AND capture_pending = 0 "
                    "AND captured_through = ?",
                    (RECOVER_TRIGGER, now, client, row_session_id, int(captured_through or 0)),
                )
                if cursor.rowcount:
                    rearmed += 1
    except sqlite3.Error:
        return rearmed
    return rearmed


# ---------------------------------------------------------------------------
# Worker: single-flight, lease-claimed, materializes the window at upload time
# ---------------------------------------------------------------------------


_NULL_WORKER_LOCK = object()


def _acquire_worker_lock(db_path: Path):
    """Take a non-blocking single-flight lock for the queue.

    Returns a held lock handle on success, a null sentinel when file locking is
    unavailable (proceed without a lock), or None when another worker on this
    host already holds the lock (the caller should exit without processing).
    """
    if fcntl is None:
        return _NULL_WORKER_LOCK
    lock_path = db_path.parent / "worker.lock"
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "w")
    except OSError:
        return _NULL_WORKER_LOCK
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def _release_worker_lock(lock) -> None:
    if lock is None or lock is _NULL_WORKER_LOCK:
        return
    try:
        fcntl.flock(lock, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock.close()
    except Exception:
        pass


def _new_lease_token() -> str:
    return uuid.uuid4().hex


def run_agent_window_worker_once(
    *,
    api_url: str,
    timeout: float,
    queue_db_path: str | Path | None = None,
    max_sessions: int = 5,
) -> int:
    db_path = _agent_queue_db_path(queue_db_path)
    if not db_path.exists():
        return 0
    lock = _acquire_worker_lock(db_path)
    if lock is None:
        # Another worker on this host already holds the queue lock; skip rather
        # than double-process the same pending sessions.
        return 0
    try:
        return _process_session_captures(db_path, api_url=api_url, timeout=timeout, max_sessions=max_sessions)
    finally:
        _release_worker_lock(lock)


def _claim_pending_sessions(
    connection: sqlite3.Connection,
    *,
    timeout: float,
    max_sessions: int,
) -> tuple[list[tuple], str]:
    """Atomically claim pending sessions by leasing them, so two workers never collide."""
    now = _now_iso()
    lease_until = _iso_after(timeout + WORKER_LEASE_BUFFER_SECONDS)
    connection.execute("BEGIN IMMEDIATE")
    try:
        rows = connection.execute(
            """
            SELECT client, session_id, transcript_path, workspace, captured_through, pending_trigger, request_seq
            FROM session_cursor
            WHERE capture_pending = 1 AND (lease_until IS NULL OR lease_until < ?)
            ORDER BY updated_at
            LIMIT ?
            """,
            (now, max_sessions),
        ).fetchall()
        claimed = []
        for row in rows:
            lease_token = _new_lease_token()
            connection.execute(
                "UPDATE session_cursor SET lease_until = ?, lease_token = ?, updated_at = ? "
                "WHERE client = ? AND session_id = ?",
                (lease_until, lease_token, now, row[0], row[1]),
            )
            claimed.append((*row, lease_token))
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    return claimed, now


def _process_session_captures(db_path: Path, *, api_url: str, timeout: float, max_sessions: int) -> int:
    connection = sqlite3.connect(db_path)
    connection.isolation_level = None  # manage the lease claim transaction explicitly
    try:
        _ensure_session_cursor(connection)
        claimed, _claim_now = _claim_pending_sessions(connection, timeout=timeout, max_sessions=max_sessions)
        submitted = 0
        for (
            client,
            session_id,
            transcript_path,
            workspace,
            captured_through,
            trigger,
            request_seq,
            lease_token,
        ) in claimed:
            now = _now_iso()
            captured_through = int(captured_through or 0)
            request_seq = int(request_seq or 0)
            try:
                if not transcript_path or not Path(transcript_path).exists():
                    raise FileNotFoundError("transcript unavailable")
                count = _transcript_line_count(transcript_path)
                # Shrink/rotation guard: if the transcript has fewer lines than the
                # bookmark, the file rotated; re-capture from the start.
                start = 0 if count < captured_through else captured_through
                if count <= start:
                    cursor = connection.execute(
                        "UPDATE session_cursor SET captured_through = "
                        "CASE WHEN ? < ? THEN ? ELSE max(captured_through, ?) END, "
                        "capture_pending = CASE WHEN request_seq = ? THEN 0 ELSE 1 END, "
                        "pending_trigger = CASE WHEN request_seq = ? THEN NULL ELSE pending_trigger END, "
                        "lease_until = NULL, lease_token = NULL, updated_at = ? "
                        "WHERE client = ? AND session_id = ? AND lease_token = ?",
                        (
                            count,
                            captured_through,
                            count,
                            count,
                            request_seq,
                            request_seq,
                            now,
                            client,
                            session_id,
                            lease_token,
                        ),
                    )
                    continue
                trigger = _normalize_capture_trigger(trigger) or GATED_CAPTURE_TRIGGER
                window_payload = _session_window_payload(
                    client=client,
                    session_id=session_id,
                    transcript_path=transcript_path,
                    workspace=workspace,
                    from_line=start,
                    to_line=count,
                    trigger=trigger,
                )
                _post_json("/api/agent-sessions/windows", window_payload, api_url=api_url, timeout=timeout)
            except Exception as exc:
                # Keep capture_pending set so the next pass retries; release the lease.
                connection.execute(
                    "UPDATE session_cursor SET lease_until = NULL, lease_token = NULL, last_error = ?, "
                    "last_attempt_at = ?, updated_at = ? "
                    "WHERE client = ? AND session_id = ? AND lease_token = ?",
                    (_queue_error_message(exc), now, now, client, session_id, lease_token),
                )
                continue
            # Confirmed upload: only the worker holding this lease token may
            # finish the row. If a hook requested another capture mid-upload,
            # request_seq changed and capture_pending stays set for the tail.
            uploaded_through = int(window_payload["history_window"]["end"])
            cursor = connection.execute(
                "UPDATE session_cursor SET captured_through = "
                "CASE WHEN ? < ? THEN ? ELSE max(captured_through, ?) END, "
                "capture_pending = CASE WHEN ? < ? THEN 1 WHEN request_seq = ? THEN 0 ELSE 1 END, "
                "pending_trigger = CASE WHEN ? < ? THEN pending_trigger WHEN request_seq = ? THEN NULL ELSE pending_trigger END, "
                "lease_until = NULL, lease_token = NULL, last_error = NULL, "
                "last_attempt_at = ?, updated_at = ? "
                "WHERE client = ? AND session_id = ? AND lease_token = ?",
                (
                    count,
                    captured_through,
                    count,
                    uploaded_through,
                    uploaded_through,
                    count,
                    request_seq,
                    uploaded_through,
                    count,
                    request_seq,
                    now,
                    now,
                    client,
                    session_id,
                    lease_token,
                ),
            )
            if cursor.rowcount:
                submitted += 1
        return submitted
    finally:
        connection.close()


def _read_hook_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("hook input must be a JSON object")
    return data


def _has_pending_session_captures(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as connection:
            _ensure_session_cursor(connection)
            row = connection.execute("SELECT 1 FROM session_cursor WHERE capture_pending = 1 LIMIT 1").fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def _drain_pending_agent_windows_if_present(*, api_url: str, timeout: float) -> None:
    try:
        if _has_pending_session_captures(_agent_queue_db_path()):
            _spawn_agent_window_worker(api_url=api_url, timeout=_agent_worker_timeout())
    except Exception:
        return


def _spawn_agent_window_worker(*, api_url: str, timeout: float) -> None:
    command = _agent_window_worker_command(api_url=api_url, timeout=timeout)
    subprocess.Popen(  # noqa: S603 - plugin-local worker command is constructed without shell.
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )


def _agent_window_worker_command(*, api_url: str, timeout: float) -> list[str]:
    script_path = Path(sys.argv[0]).expanduser()
    if script_path.exists() and script_path.name.startswith("memforge_hook"):
        command = [sys.executable, str(script_path)]
    else:
        command = [sys.executable, "-m", "memforge.hook_adapter"]
    return [
        *command,
        "worker-run-once",
        "--api-url",
        api_url,
        "--timeout",
        str(timeout),
    ]


def _agent_worker_timeout() -> float:
    return _env_float("MEMFORGE_AGENT_WORKER_TIMEOUT_SECONDS", DEFAULT_AGENT_WORKER_TIMEOUT_SECONDS)


def _post_json(path: str, payload: dict[str, Any], *, api_url: str, timeout: float) -> dict[str, Any]:
    url = _admin_api_request_url(path, api_url=api_url)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_token = configured_api_token()
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    if not response_body:
        return {}
    data = json.loads(response_body)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} returned non-object JSON")
    return data


def _admin_api_base_url(api_url: str) -> str:
    base_url = api_url.rstrip("/")
    if base_url.endswith("/api"):
        return base_url[:-4]
    return base_url


def _workspace_id() -> str:
    return configured_workspace_id()


def _admin_api_request_url(path: str, *, api_url: str) -> str:
    base_url = _admin_api_base_url(api_url)
    workspace_id = _workspace_id()
    if not workspace_id or not path.startswith("/api"):
        return f"{base_url}{path}"
    quoted_workspace = urllib.parse.quote(workspace_id, safe="")
    if path == "/api":
        return f"{base_url}/api/workspaces/{quoted_workspace}/api"
    if path.startswith("/api/"):
        return f"{base_url}/api/workspaces/{quoted_workspace}/api/{path[len('/api/') :]}"
    return f"{base_url}{path}"


def _event_name(payload: dict[str, Any]) -> str:
    return str(payload.get("hook_event_name") or payload.get("hookEventName") or payload.get("event") or "UnknownHook")


def _workspace(payload: dict[str, Any]) -> str:
    return str(payload.get("cwd") or payload.get("workspace") or os.getcwd())


def _repo_name(payload: dict[str, Any]) -> str | None:
    remote = _git_value(payload, ["git", "remote", "get-url", "origin"])
    normalized_remote = _normalize_repo_identifier(remote)
    if normalized_remote:
        return normalized_remote
    root = _git_value(payload, ["git", "rev-parse", "--show-toplevel"])
    if root:
        return Path(root).name
    workspace = _workspace(payload)
    return Path(workspace).name if workspace else None


def _normalize_repo_identifier(repo: str | None) -> str | None:
    """Normalize a VCS remote or repo slug to the stable session repo key."""
    if repo is None:
        return None
    value = repo.strip()
    if not value:
        return None

    ssh_match = re.match(r"^[^/@]+@([^:/]+):(.+)$", value)
    if ssh_match:
        host, path = ssh_match.groups()
        value = f"{host}/{path}"
    else:
        value = re.sub(r"^[a-z][a-z0-9+.-]*://", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^[^@/]+@", "", value)

    value = value.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    value = re.sub(r"/+", "/", value)
    return value.lower() or None


def _git_value(payload: dict[str, Any], command: list[str]) -> str | None:
    cwd = _workspace(payload)
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def _touched_files(payload: dict[str, Any]) -> list[str]:
    files: list[str] = []
    for key in ("file_path", "path"):
        value = payload.get(key)
        if isinstance(value, str):
            files.append(value)
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict):
        for key in ("file_path", "path"):
            value = tool_input.get(key)
            if isinstance(value, str):
                files.append(value)
    return files


def _transcript_path(payload: dict[str, Any]) -> str | None:
    value = payload.get("transcript_path") or payload.get("transcriptPath")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _has_stop_window_signal(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in STOP_SIGNAL_TERMS):
        return True
    tool_mentions = lowered.count('"type":"tool"') + lowered.count('"type": "tool"') + lowered.count('"tool"')
    return tool_mentions >= 2


def _has_stop_window_signal_in_transcript(transcript_path: str, start: int, end: int) -> bool:
    tool_mentions = 0
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= end:
                    break
                if index < start:
                    continue
                lowered = line.lower()
                if any(term in lowered for term in STOP_SIGNAL_TERMS):
                    return True
                tool_mentions += (
                    lowered.count('"type":"tool"') + lowered.count('"type": "tool"') + lowered.count('"tool"')
                )
                if tool_mentions >= 2:
                    return True
    except OSError:
        return False
    return False


def _session_window_payload(
    *,
    client: str,
    session_id: str,
    transcript_path: str,
    workspace: str,
    from_line: int,
    to_line: int,
    trigger: str,
) -> dict[str, Any]:
    """Build the window upload for the live [from_line, to_line) transcript delta."""
    events, effective_to_line, truncated, omissions = _bounded_transcript_evidence_window(
        transcript_path,
        from_line,
        to_line,
    )
    events = _redact_agent_payload(events)
    window_text = _render_canonical_events_markdown(events)
    payload_like = {
        "cwd": workspace,
        "session_id": session_id,
        "_client": client,
        "transcript_path": transcript_path,
        "hook_event_name": trigger,
    }
    payload = {
        "schema_version": WINDOW_SCHEMA_VERSION,
        "plugin_version": os.getenv("MEMFORGE_PLUGIN_VERSION", PLUGIN_VERSION),
        "client": client,
        "session_id": session_id,
        "trigger": trigger,
        "workspace": workspace,
        "repo": _repo_name(payload_like),
        "branch": _git_value(payload_like, ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_sha": _git_value(payload_like, ["git", "rev-parse", "HEAD"]),
        "history_window": {
            "kind": "transcript_window",
            "transcript_path": transcript_path,
            "start": str(from_line),
            "end": str(effective_to_line),
            "line_count": effective_to_line - from_line,
            "truncated": truncated,
        },
        "events": events,
        # Keep the legacy field name for the v1 API, but send compact canonical
        # evidence rather than raw transcript JSONL.
        "transcript_markdown": window_text,
        "receipt": {
            "client": client,
            "session_id": session_id,
            "hook": trigger,
            "workspace": workspace,
            "metadata": {
                "trigger": trigger,
                "from_line": from_line,
                "uploaded_to_line": effective_to_line,
                "observed_to_line": to_line,
                "omissions": dict(omissions),
            },
        },
        "retention": "none",
        "process_now": False,
    }
    source_updated_at = _first_reliable_event_timestamp(events)
    if source_updated_at is not None:
        payload["source_updated_at"] = source_updated_at
    return payload


def _first_reliable_event_timestamp(events: list[dict[str, Any]]) -> str | None:
    """Return the earliest explicit offset-aware timestamp in the submitted evidence window.

    This records when the source window started, not when an individual claim was
    first stated. If the transcript omits reliable absolute time, callers leave
    source_updated_at unset instead of falling back to upload time.
    """
    parsed_times: list[datetime] = []
    for event in events:
        value = event.get("timestamp")
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(f"invalid event timestamp {value!r}") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"event timestamp must include timezone offset: {value!r}")
        parsed_times.append(parsed.astimezone(timezone.utc))
    if not parsed_times:
        return None
    return min(parsed_times).isoformat()


def _bounded_transcript_line_slice(transcript_path: str, start: int, end: int) -> tuple[list[str], int, bool]:
    """Return complete transcript lines that fit in the upload budget.

    The bookmark unit is a transcript line. If a window is too large, upload a
    prefix that fits and leave the rest pending for another pass instead of
    uploading only the tail and marking the whole range captured.
    """
    lines = _transcript_line_slice(transcript_path, start, end)
    selected: list[str] = []
    used_chars = 0
    for line in lines:
        separator_chars = 1 if selected else 0
        next_chars = separator_chars + len(line)
        if selected and used_chars + next_chars > MAX_TRANSCRIPT_CHARS:
            return selected, start + len(selected), True
        if not selected and next_chars > MAX_TRANSCRIPT_CHARS:
            return [_compact_oversized_transcript_line(line)], start + 1, True
        selected.append(line)
        used_chars += next_chars
    return selected, start + len(selected), False


def _compact_oversized_transcript_line(line: str) -> str:
    event = _parse_transcript_event(line)
    if event:
        event["truncated"] = True
        compact = json.dumps(event, sort_keys=True, separators=(",", ":"))
        if len(compact) <= MAX_TRANSCRIPT_CHARS:
            return compact
    return line[:MAX_TRANSCRIPT_CHARS]


def _bounded_transcript_evidence_window(
    transcript_path: str,
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], int, bool, Counter[str]]:
    """Return canonical evidence from complete transcript units in [start, end).

    The cursor still advances by native transcript lines, but upload budgeting is
    applied after client-format parsing and memory-relevance filtering. Lines
    that only contain bootstrap/context metadata are represented in the omission
    counters instead of being sent as raw LLM input.
    """
    selected: list[dict[str, Any]] = []
    omissions: Counter[str] = Counter()
    effective_to = start
    used_chars = 0
    if end <= start:
        return selected, effective_to, False, omissions

    try:
        handle = open(transcript_path, encoding="utf-8", errors="replace")
    except OSError:
        return selected, effective_to, False, omissions

    with handle:
        for index, line in enumerate(handle):
            if index >= end:
                break
            if index < start:
                continue

            events = _parse_transcript_events_from_line(line.rstrip("\n"))
            if not events:
                omissions["metadata_or_context"] += 1
                effective_to = index + 1
                continue

            line_events: list[dict[str, Any]] = []
            line_chars = 0
            for event in events:
                event = _fit_event_to_budget(event, MAX_TRANSCRIPT_CHARS)
                event_chars = _event_budget_chars(event)
                line_events.append(event)
                line_chars += event_chars

            if selected and (
                len(selected) + len(line_events) > MAX_EVENTS or used_chars + line_chars > MAX_TRANSCRIPT_CHARS
            ):
                return selected, index, True, omissions

            if not selected and (len(line_events) > MAX_EVENTS or line_chars > MAX_TRANSCRIPT_CHARS):
                selected = _fit_events_to_empty_window(line_events)
                omissions["oversized_event"] += max(0, len(line_events) - len(selected))
                return selected, index + 1, True, omissions

            selected.extend(line_events)
            used_chars += line_chars
            effective_to = index + 1

    return selected, effective_to, False, omissions


def _fit_events_to_empty_window(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_chars = 0
    for event in events:
        if len(selected) >= MAX_EVENTS:
            break
        remaining = MAX_TRANSCRIPT_CHARS - used_chars
        if remaining <= 0:
            break
        event = _fit_event_to_budget(event, remaining)
        event_chars = _event_budget_chars(event)
        if selected and used_chars + event_chars > MAX_TRANSCRIPT_CHARS:
            break
        selected.append(event)
        used_chars += event_chars
    return selected


def _fit_event_to_budget(event: dict[str, Any], budget_chars: int) -> dict[str, Any]:
    if _event_budget_chars(event) <= budget_chars:
        return event
    text = str(event.get("text") or "")
    if not text:
        return event
    without_text = {key: value for key, value in event.items() if key != "text"}
    overhead = _event_budget_chars(without_text) + 24
    text_budget = min(len(text), max(80, budget_chars - overhead))
    fitted = dict(event)
    fitted["text"] = _truncate_middle_text(text, text_budget)
    fitted["truncation"] = {
        "strategy": "middle",
        "original_chars": len(text),
    }
    return fitted


def _event_budget_chars(event: dict[str, Any]) -> int:
    return len(json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)) + 1


def _truncate_middle_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return f"...{len(text)} chars truncated..."
    if len(text) <= max_chars:
        return text
    marker = "...truncated..."
    if max_chars <= len(marker):
        return marker
    left = (max_chars - len(marker)) // 2
    right = max_chars - len(marker) - left
    return text[:left] + marker + text[-right:]


def _redact_agent_payload(value: Any) -> Any:
    if isinstance(value, str):
        redacted = value
        for kind, pattern in _SECRET_PATTERNS:
            if kind == "bearer":
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            elif kind == "json":
                redacted = pattern.sub(r"\1[REDACTED]", redacted)
            else:
                redacted = pattern.sub(lambda match: f"{match.group(1)}: [REDACTED]", redacted)
        return redacted
    if isinstance(value, list):
        return [_redact_agent_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_agent_payload(item) for key, item in value.items()}
    return value


def _extract_transcript_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        events.extend(_parse_transcript_events_from_line(line))
    # Keep the most recent events so the structured events describe the same tail
    # of the window as the transcript text.
    return events[-MAX_EVENTS:]


def _parse_transcript_event(line: str) -> dict[str, Any] | None:
    events = _parse_transcript_events_from_line(line)
    return events[0] if events else None


def _parse_transcript_events_from_line(line: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    payload = data.get("payload")
    if not isinstance(payload, dict):
        payload = {}

    source_type = _first_string(data, "type")
    if _looks_like_claude_record(data):
        return _parse_claude_content_events(data)
    if source_type in ("response_item", "event_msg"):
        return _parse_codex_events(data, payload)
    if source_type in NOISE_EVENT_TYPES:
        return []

    native_type = source_type or _first_string(payload, "type", "item_type", "role")
    if not native_type:
        return []
    kind, actor = _infer_generic_kind(native_type, data, payload)
    if not kind:
        return []
    text = _first_preview_value(
        data,
        payload,
        keys=("input", "message", "content", "arguments", "output", "summary", "stdout", "stderr", "text"),
    )
    return [
        _canonical_event(
            kind=kind,
            actor=actor,
            source_type=source_type,
            native_type=native_type,
            name=_first_string(data, "name", "tool_name") or _first_string(payload, "name", "tool_name"),
            text=text,
            timestamp=_first_string(data, "timestamp") or _first_string(payload, "timestamp"),
        )
    ]


def _parse_codex_events(data: dict[str, Any], payload: dict[str, Any]) -> list[dict[str, Any]]:
    source_type = str(data.get("type") or "")
    native_type = _first_string(payload, "type", "item_type", "role")
    if not native_type or native_type in NOISE_EVENT_TYPES:
        return []
    timestamp = _first_string(data, "timestamp") or _first_string(payload, "timestamp")

    if source_type == "response_item" and native_type == "message":
        role = _first_string(payload, "role") or ""
        if role == "developer":
            return []
        text = _message_content_text(payload.get("content") or payload.get("message"))
        if role == "user" and _is_memory_excluded_contextual_user_fragment(text):
            return []
        if role == "user":
            kind, actor = "user_message", "user"
        elif role == "assistant":
            kind, actor = "assistant_message", "assistant"
        else:
            return []
        return [
            _canonical_event(
                kind=kind,
                actor=actor,
                source_type=source_type,
                native_type=native_type,
                text=text,
                timestamp=timestamp,
            )
        ]

    if source_type == "event_msg" and native_type == "agent_message":
        text = _first_preview_value(payload, keys=("message", "content", "summary", "text"))
        return [
            _canonical_event(
                kind="assistant_message",
                actor="assistant",
                source_type=source_type,
                native_type=native_type,
                text=text,
                timestamp=timestamp,
            )
        ]

    if native_type in CODEX_TOOL_CALL_TYPES:
        text = _first_preview_value(payload, keys=("arguments", "input", "command", "content", "summary"))
        return [
            _canonical_event(
                kind="tool_call",
                actor="assistant",
                source_type=source_type,
                native_type=native_type,
                name=_first_string(payload, "name", "tool_name"),
                text=text,
                timestamp=timestamp,
            )
        ]

    if native_type in CODEX_TOOL_RESULT_TYPES:
        text = _first_preview_value(payload, keys=("output", "content", "summary", "stdout", "stderr"))
        return [
            _canonical_event(
                kind="tool_result",
                actor="tool",
                source_type=source_type,
                native_type=native_type,
                name=_first_string(payload, "name", "tool_name"),
                text=text,
                timestamp=timestamp,
            )
        ]

    if source_type == "event_msg" and native_type.startswith("exec_command"):
        text = _first_preview_value(payload, keys=("command", "summary", "stdout", "stderr", "aggregated_output"))
        return [
            _canonical_event(
                kind="tool_result",
                actor="tool",
                source_type=source_type,
                native_type=native_type,
                name=_first_string(payload, "name", "tool_name") or "exec_command",
                text=text,
                timestamp=timestamp,
            )
        ]
    return []


def _looks_like_claude_record(data: dict[str, Any]) -> bool:
    if isinstance(data.get("message"), dict):
        return True
    return str(data.get("type") or "") in {"assistant", "user"}


def _parse_claude_content_events(data: dict[str, Any]) -> list[dict[str, Any]]:
    message = data.get("message")
    source_type = _first_string(data, "type")
    timestamp = _first_string(data, "timestamp")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = data.get("content") or data.get("text")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    tool_events: list[dict[str, Any]] = []
    text_events: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = _first_string(part, "type")
        if part_type in NOISE_EVENT_TYPES:
            continue
        if part_type == "tool_use":
            tool_events.append(
                _canonical_event(
                    kind="tool_call",
                    actor="assistant",
                    source_type=source_type,
                    native_type=part_type,
                    name=_first_string(part, "name"),
                    text=_first_preview_value(part, keys=("input", "content", "text")),
                    timestamp=timestamp,
                )
            )
        elif part_type == "tool_result":
            tool_events.append(
                _canonical_event(
                    kind="tool_result",
                    actor="tool",
                    source_type=source_type,
                    native_type=part_type,
                    name=_first_string(part, "name"),
                    text=_first_preview_value(part, keys=("content", "text", "output")),
                    timestamp=timestamp,
                )
            )
        elif part_type == "text":
            role = source_type or _first_string(message if isinstance(message, dict) else {}, "role") or ""
            if role == "user":
                kind, actor = "user_message", "user"
            elif role == "assistant":
                kind, actor = "assistant_message", "assistant"
            else:
                continue
            text = _first_preview_value(part, keys=("text", "content"))
            if role == "user" and _is_memory_excluded_contextual_user_fragment(text or ""):
                continue
            text_events.append(
                _canonical_event(
                    kind=kind,
                    actor=actor,
                    source_type=source_type,
                    native_type=part_type,
                    text=text,
                    timestamp=timestamp,
                )
            )
    return tool_events or text_events


def _canonical_event(
    *,
    kind: str,
    actor: str,
    source_type: str | None,
    native_type: str | None,
    text: str | None = None,
    name: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "kind": kind,
        "actor": actor,
    }
    if source_type:
        event["source_type"] = source_type
    if native_type:
        event["native_type"] = native_type
    if name:
        event["name"] = name
    if timestamp:
        event["timestamp"] = timestamp
    if text:
        event["text"] = text
    return event


def _infer_generic_kind(
    native_type: str,
    data: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[str | None, str]:
    role = _first_string(data, "role") or _first_string(payload, "role")
    if native_type in ("tool", "tool_call", "tool_use", "function_call"):
        return "tool_call", role or "assistant"
    if native_type in ("tool_result", "tool_output", "tool_use_result", "function_call_output"):
        return "tool_result", "tool"
    if native_type == "user" or role == "user":
        return "user_message", "user"
    if native_type == "assistant" or role == "assistant":
        return "assistant_message", "assistant"
    return None, role or "system"


def _message_content_text(value: Any) -> str | None:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _message_content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("text", "content", "message", "input", "output"):
            text = _message_content_text(value.get(key))
            if text:
                return text
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _is_memory_excluded_contextual_user_fragment(text: str | None) -> bool:
    if not text:
        return False
    return _matches_marked_fragment(
        text, "# AGENTS.md instructions for ", "</INSTRUCTIONS>"
    ) or _matches_marked_fragment(text, "<skill>", "</skill>")


def _matches_marked_fragment(text: str, start_marker: str, end_marker: str) -> bool:
    trimmed = text.strip()
    return trimmed.lower().startswith(start_marker.lower()) and trimmed.lower().endswith(end_marker.lower())


def _render_canonical_events_markdown(events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        kind = str(event.get("kind") or "event")
        actor = str(event.get("actor") or "")
        name = event.get("name")
        label = kind if not name else f"{kind}:{name}"
        suffix = f" ({actor})" if actor else ""
        text = str(event.get("text") or "").strip()
        lines.append(f"{index}. {label}{suffix}\n{text}".strip())
    return "\n\n".join(lines)


def _first_string(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_preview_value(*mappings: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for mapping in mappings:
        for key in keys:
            value = mapping.get(key)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return str(value)
    return None


def _transcript_line_count(transcript_path: str) -> int:
    """Return the number of events (JSONL lines) in the transcript: the bookmark unit."""
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _transcript_line_slice(transcript_path: str, start: int, end: int) -> list[str]:
    """Return transcript lines in the half-open range [start, end)."""
    lines: list[str] = []
    if end <= start:
        return lines
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= end:
                    break
                if index >= start:
                    lines.append(line.rstrip("\n"))
    except OSError:
        return []
    return lines


def _agent_queue_db_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return Path(os.getenv("MEMFORGE_AGENT_QUEUE_DB") or DEFAULT_AGENT_QUEUE_DB).expanduser()


def _ensure_session_cursor(connection: sqlite3.Connection) -> None:
    # WAL gives concurrent readers a writer without blocking, and the busy
    # timeout absorbs the brief cross-process contention of overlapping hooks
    # and the worker. Every queue connection runs this, so both apply in one
    # place; journal_mode is a persistent property, so re-setting it is a no-op.
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(f"PRAGMA busy_timeout={QUEUE_BUSY_TIMEOUT_MS}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_cursor (
            client TEXT NOT NULL,
            session_id TEXT NOT NULL,
            transcript_path TEXT,
            workspace TEXT,
            captured_through INTEGER NOT NULL DEFAULT 0,
            capture_pending INTEGER NOT NULL DEFAULT 0,
            pending_trigger TEXT,
            lease_until TEXT,
            lease_token TEXT,
            request_seq INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            last_attempt_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client, session_id)
        )
        """
    )
    _ensure_column(connection, "session_cursor", "lease_token", "TEXT")
    _ensure_column(connection, "session_cursor", "request_seq", "INTEGER NOT NULL DEFAULT 0")


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _queue_error_message(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".strip()
    return message.replace("\x00", "")[:1000]


def _session_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "hook_event_name": _event_name(payload),
        "permission_mode": payload.get("permission_mode"),
        "has_transcript_path": bool(_transcript_path(payload)),
        "turn_id": payload.get("turn_id"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _safe_exception_message(exc: Exception) -> str:
    text = str(exc).replace("`", "'")
    if " returned HTTP " in text:
        return text.split(":", 1)[0]
    return text.splitlines()[0][:300]


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_after(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
