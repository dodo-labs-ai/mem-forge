"""Queue/OS boundary regressions for the event-driven capture owner."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import sqlite3
import subprocess
import sys
import threading

import pytest

from memforge import hook_adapter as h


@pytest.fixture
def queue(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMFORGE_AGENT_QUEUE_DB", str(tmp_path / "queue.sqlite"))
    monkeypatch.setenv("MEMFORGE_EDITION", "cloud")
    monkeypatch.delenv("MEMFORGE_WORKSPACE_ID", raising=False)
    monkeypatch.setenv("MEMFORGE_CODEX_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(h, "resolve_workspace_binding", lambda **kw: type("Binding", (), {"workspace_id": "pinned"})())
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"type":"tool","name":"apply_patch","input":"edit"}\n')

    def request(session="current", trigger="REQUIRED_CAPTURE"):
        h.request_session_capture(
            client="codex",
            session_id=session,
            transcript_path=str(transcript),
            workspace=str(tmp_path),
            trigger=trigger,
        )

    request()
    return tmp_path / "queue.sqlite", transcript, request


def read(db, columns="capture_pending, captured_through, wake_requested_at", session="current"):
    with sqlite3.connect(db) as c:
        return c.execute(f"SELECT {columns} FROM session_cursor WHERE session_id = ?", (session,)).fetchone()


def drain(db):
    owner = h._acquire_worker_lock(db)
    assert owner is not None
    fd = os.dup(owner.fileno())
    owner.close()
    return h.run_agent_window_worker(timeout=1, owner_fd=fd)


@pytest.mark.parametrize("event,pending", [("Stop", 1), ("PreCompact", 1), ("SessionStart", 1), ("SessionStart", 0)])
def test_hooks_wake_current_before_six_historical_rows(queue, monkeypatch, event, pending):
    db, transcript, request = queue
    for i in range(6):
        request(f"old-{i}")
    with sqlite3.connect(db) as c:
        c.execute("UPDATE session_cursor SET wake_requested_at=NULL")
        c.execute('UPDATE session_cursor SET capture_pending=? WHERE session_id="current"', (pending,))
    monkeypatch.setattr(h, "_spawn_agent_window_worker", lambda **kw: None)
    monkeypatch.setattr(h, "_post_json", lambda *a, **kw: {})
    payload = dict(
        hook_event_name=event, session_id="current", transcript_path=str(transcript), cwd=str(transcript.parent)
    )
    if event == "SessionStart":
        h._run_context(payload, client="codex", timeout=1)
    else:
        h._run_submit_session(payload, client="codex", timeout=1)
    with sqlite3.connect(db) as c:
        rows, _ = h._claim_pending_sessions(c, timeout=1, max_sessions=5)
    assert rows[0][1] == "current"
    assert len(rows) == 5
    assert read(db, "wake_requested_at, lease_token")[0] is None


def test_hundred_concurrent_hooks_coalesce_without_spawning_waiters(queue, monkeypatch):
    db, _, request = queue
    held = h._acquire_worker_lock(db)
    spawned = []
    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **kw: spawned.append(a))
    first_wake = read(db)[2]

    def hook(i):
        request("current" if i % 2 else f"session-{i % 10}")
        h._spawn_agent_window_worker(timeout=1)

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(hook, range(100)))
        assert spawned == []
        assert read(db)[2] == first_wake
        with sqlite3.connect(db) as c:
            assert c.execute("SELECT sum(request_seq), count(*) FROM session_cursor").fetchone() == (101, 6)
    finally:
        h._release_worker_lock(held)


def test_spawn_transfers_real_lock_and_closes_unrelated_fds(queue, monkeypatch):
    db, _, _ = queue
    real_popen = subprocess.Popen
    child = []
    extra_read, extra_write = os.pipe()

    def spawn(command, **kwargs):
        fd = kwargs["pass_fds"][0]
        assert command[-4:] == ["--timeout", "1", "--owner-fd", str(fd)]
        script = """
import os, sys
fd, extra = map(int, sys.argv[1:])
try:
    os.fstat(extra)
except OSError:
    print('closed', flush=True)
else:
    raise AssertionError('unrelated fd inherited')
sys.stdin.readline()
os.close(fd)
"""
        kwargs.update(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        child.append(real_popen([sys.executable, "-c", script, str(fd), str(extra_write)], **kwargs))

    os.set_inheritable(extra_write, True)
    monkeypatch.setattr(h.subprocess, "Popen", spawn)
    try:
        h._spawn_agent_window_worker(timeout=1)
        assert child[0].stdout.readline().strip() == "closed"
        assert h._acquire_worker_lock(db) is None
        out, err = child[0].communicate("\n", timeout=5)
        assert child[0].returncode == 0, err
        lock = h._acquire_worker_lock(db)
        assert lock is not None
        h._release_worker_lock(lock)
    finally:
        os.close(extra_read)
        os.close(extra_write)
        for proc in child:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def test_spawn_failure_preserves_wake_and_releases_owner(queue, monkeypatch):
    db, _, _ = queue

    def fail(*a, **kw):
        raise OSError("spawn failed")

    monkeypatch.setattr(h.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="spawn failed"):
        h._spawn_agent_window_worker(timeout=1)
    assert read(db)[:2] == (1, 0)
    assert read(db)[2]
    owner = h._acquire_worker_lock(db)
    assert owner is not None
    h._release_worker_lock(owner)


def test_non_posix_fails_closed(queue, monkeypatch):
    db, _, _ = queue
    monkeypatch.setattr(h, "fcntl", None)
    with pytest.raises(RuntimeError, match="POSIX"):
        h.run_agent_window_worker_once(timeout=1)
    assert read(db)[:2] == (1, 0)


def test_prefixes_and_mid_upload_request_are_drained(queue, monkeypatch):
    db, transcript, request = queue
    line = transcript.read_text()
    transcript.write_text(line * 3)
    monkeypatch.setattr(h, "MAX_EVENTS", 1)
    posted = []

    def post(path, payload, **kw):
        assert kw["workspace_id"] == "pinned"
        posted.append(payload["history_window"])
        if len(posted) == 1:
            transcript.write_text(line * 4)
            request()
        return {}

    monkeypatch.setattr(h, "_post_json", post)
    assert drain(db) == 4
    assert [(x["start"], x["end"]) for x in posted] == [("0", "1"), ("1", "2"), ("2", "3"), ("3", "4")]
    assert read(db) == (0, 4, None)


def test_historical_budget_is_per_activation(queue, monkeypatch):
    db, _, request = queue
    for i in range(9):
        request(f"old-{i}")
    with sqlite3.connect(db) as c:
        c.execute('UPDATE session_cursor SET wake_requested_at=NULL WHERE session_id != "current"')
    posted = []
    monkeypatch.setattr(h, "_post_json", lambda path, payload, **kw: posted.append(payload["session_id"]) or {})
    assert drain(db) == 6
    assert posted[0] == "current"
    with sqlite3.connect(db) as c:
        assert c.execute("SELECT count(*) FROM session_cursor WHERE capture_pending=1").fetchone()[0] == 4


def test_failure_completion_cooldown_and_no_retry_in_same_activation(queue, monkeypatch):
    db, _, request = queue
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    clock = [now]
    monkeypatch.setattr(h, "_now_iso", lambda: clock[0].isoformat())
    monkeypatch.setattr(h, "_iso_after", lambda seconds: (clock[0] + timedelta(seconds=seconds)).isoformat())
    posted = []

    def post(path, payload, **kw):
        posted.append(payload["session_id"])
        clock[0] += timedelta(seconds=90)
        if payload["session_id"] == "current":
            request()
            request("new")
            raise OSError("slow failure")
        return {}

    monkeypatch.setattr(h, "_post_json", post)
    assert drain(db) == 1
    assert posted == ["current", "new"]
    assert read(db, "last_attempt_at")[0] == (now + timedelta(seconds=90)).isoformat()
    # Even after 90 more seconds in the same activation, the failure was not retried.
    clock[0] = now + timedelta(seconds=149)
    assert drain(db) == 0
    monkeypatch.setattr(h, "_post_json", lambda *a, **kw: {})
    clock[0] += timedelta(seconds=1)
    assert drain(db) == 1
    assert read(db)[:2] == (0, 1)


@pytest.mark.parametrize("request_first", [True, False])
def test_exit_handoff_serializes_request_and_unlock(queue, monkeypatch, request_first):
    db, _, request = queue
    with sqlite3.connect(db) as c:
        c.execute("UPDATE session_cursor SET capture_pending=0, wake_requested_at=NULL")
    owner = h._acquire_worker_lock(db)
    activation = h._CaptureDrain()
    spawned = []
    monkeypatch.setattr(h.subprocess, "Popen", lambda *a, **kw: spawned.append(a))
    if request_first:
        request()
        h._spawn_agent_window_worker(timeout=1)
        assert spawned == []
        assert h._finish_capture_drain(db, owner, activation) is False
        h._release_worker_lock(owner)
    else:
        at_unlock = threading.Event()
        producer_started = threading.Event()
        real_release = h._release_worker_lock

        def release(lock):
            at_unlock.set()
            assert producer_started.wait(5)
            real_release(lock)

        monkeypatch.setattr(h, "_release_worker_lock", release)

        def producer():
            assert at_unlock.wait(5)
            producer_started.set()
            request()
            h._spawn_agent_window_worker(timeout=1)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(producer)
            assert h._finish_capture_drain(db, owner, activation) is True
            future.result(timeout=5)
        assert len(spawned) == 1
    assert read(db)[:2] == (1, 0)
    assert read(db)[2]


@pytest.mark.parametrize("claimed", [False, True])
def test_killed_owner_preserves_request_until_later_event(queue, monkeypatch, claimed):
    db, _, _ = queue
    script = """
import os, sys
from memforge import hook_adapter as h
lock = h._acquire_worker_lock(h._agent_queue_db_path())
import sqlite3
if sys.argv[1] == "claimed":
    with sqlite3.connect(h._agent_queue_db_path()) as c:
        h._claim_pending_sessions(c, timeout=180, max_sessions=5)
print("ready", flush=True)
sys.stdin.readline()
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, "claimed" if claimed else "reserved"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.kill()
        proc.wait(timeout=5)
        assert read(db)[:2] == (1, 0)
        monkeypatch.setattr(h, "_post_json", lambda *a, **kw: {})
        if claimed:
            assert drain(db) == 0
            with sqlite3.connect(db) as c:
                c.execute("UPDATE session_cursor SET lease_until='2000-01-01T00:00:00+00:00'")
        assert drain(db) == 1
        assert read(db)[:2] == (0, 1)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.mark.parametrize("maintenance", [False, True])
def test_recover_during_upload_preserves_and_drains_new_tail(queue, monkeypatch, maintenance):
    db, transcript, _ = queue
    line = transcript.read_text()
    posted = []

    def post(path, payload, **kw):
        posted.append(payload["history_window"]["end"])
        if len(posted) == 1:
            transcript.write_text(line * 2)
            h.schedule_capture(dict(session_id="current"), client="codex", policy="RECOVER")
        return {}

    monkeypatch.setattr(h, "_post_json", post)
    assert (h.run_agent_window_worker_once(timeout=1) if maintenance else drain(db)) == 2
    assert posted == ["1", "2"]
    assert read(db) == (0, 2, None)


def test_failure_exclusion_does_not_grow_sql_expression(queue):
    db, _, _ = queue
    activation = h._CaptureDrain(failed={("codex", str(i)) for i in range(2000)})
    with sqlite3.connect(db) as c:
        assert len(h._pending_capture_rows(c, max_sessions=5, drain=activation)) == 1
        activation.failed.add(("codex", "current"))
        assert h._pending_capture_rows(c, max_sessions=5, drain=activation) == []


def test_session_start_queue_failure_keeps_one_json_response(queue, monkeypatch, capsys):
    import io
    import json

    def fail(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(h, "schedule_capture", fail)
    monkeypatch.setattr(h.sys, "stdin", io.StringIO('{"hook_event_name":"SessionStart"}'))
    assert h.main(["context"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "database is locked" in output.err


def test_worker_rejects_unowned_descriptor_while_another_owner_is_active(queue, monkeypatch):
    db, _, _ = queue
    owner = h._acquire_worker_lock(db)
    fd = os.open(db.parent / "worker.lock", os.O_WRONLY)
    posts = []
    monkeypatch.setattr(h, "_post_json", lambda *a, **kw: posts.append(a))
    try:
        with pytest.raises(BlockingIOError):
            h.run_agent_window_worker(timeout=1, owner_fd=fd)
        assert posts == []
        assert read(db)[:2] == (1, 0)
    finally:
        h._release_worker_lock(owner)


def test_recovery_requires_positive_round_capacity(queue):
    with pytest.raises(ValueError, match="positive"):
        h.run_agent_window_worker_once(timeout=1, max_sessions=0)


@pytest.mark.parametrize("success_before_recover", [False, True])
def test_recover_and_upload_completion_have_no_idle_promotion_gap(queue, monkeypatch, success_before_recover):
    db, transcript, _ = queue
    line = transcript.read_text()
    uploading = threading.Event()
    allow_success = threading.Event()
    worker_finished = threading.Event()
    posted = []

    def post(path, payload, **kw):
        posted.append(payload["history_window"]["end"])
        if len(posted) == 1:
            uploading.set()
            assert allow_success.wait(5)
        return {}

    monkeypatch.setattr(h, "_post_json", post)

    def worker():
        try:
            return drain(db)
        finally:
            worker_finished.set()

    real_recover = h._recover_incomplete_sessions

    def recover(**kw):
        result = real_recover(**kw)
        allow_success.set()
        assert worker_finished.wait(5)
        return result

    monkeypatch.setattr(h, "_recover_incomplete_sessions", recover)
    monkeypatch.setattr(h, "_spawn_agent_window_worker", lambda **kw: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(worker)
        assert uploading.wait(5)
        transcript.write_text(line * 2)
        if success_before_recover:
            allow_success.set()
            assert worker_finished.wait(5)
        h.schedule_capture(dict(session_id="current"), client="codex", policy="RECOVER")
        future.result(timeout=5)
    if success_before_recover:
        assert read(db)[0] == 1
        assert read(db)[2]
        drain(db)
    assert posted == ["1", "2"]
    assert read(db) == (0, 2, None)
