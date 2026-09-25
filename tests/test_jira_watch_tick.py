from __future__ import annotations

from memforge.auth.jira_auth import JiraAuthSessionMissingError


class _Client:
    def __init__(self, upload_result=None):
        self.uploaded = []
        self.expired = []
        self._upload_result = upload_result or {"status": "active"}

    def upload_jira_session(self, *, base_url, cookie_header, browser=None, confirm_principal_change=False):
        self.uploaded.append(cookie_header)
        return self._upload_result

    def mark_jira_session_expired(self, *, base_url, error):
        self.expired.append(error)
        return {"ok": True}


async def _capture_good(base_url, *, browser=None):
    from memforge.auth.jira_capture import JiraCaptureResult

    return JiraCaptureResult(
        origin=base_url, cookie_header="SESSION=good", browser="Chrome", principal={"accountId": "u1"}
    )


async def _capture_dead(base_url, *, browser=None):
    raise JiraAuthSessionMissingError("dead")


async def test_tick_uploads_changed_cookie():
    from memforge.main import run_watch_tick

    client = _Client()
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash=None,
        capture=_capture_good,
        log=lambda m: None,
    )
    assert action == "uploaded"
    assert client.uploaded == ["SESSION=good"]
    assert new_hash is not None


async def test_tick_skips_unchanged_cookie():
    from memforge.main import run_watch_tick, _cookie_hash

    client = _Client()
    same = _cookie_hash("SESSION=good")
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash=same,
        capture=_capture_good,
        log=lambda m: None,
    )
    assert action == "unchanged"
    assert client.uploaded == []
    assert new_hash == same


async def test_tick_marks_expired_when_session_dead():
    from memforge.main import run_watch_tick

    client = _Client()
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash="abc",
        capture=_capture_dead,
        log=lambda m: None,
    )
    assert action == "expired"
    assert client.expired and new_hash is None


def _principal_changed():
    return {
        "error": "MemForge API request failed",
        "status_code": 409,
        "code": "jira_principal_changed",
        "detail": {
            "code": "jira_principal_changed",
            "origin": "https://jira.example.test",
            "old_principal_id": "old-user",
            "new_principal_id": "new-user",
        },
    }


async def test_tick_flags_principal_conflict():
    from memforge.main import run_watch_tick

    client = _Client(upload_result=_principal_changed())
    action, _ = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash=None,
        capture=_capture_good,
        log=lambda m: None,
    )
    assert action == "principal_conflict"


async def test_tick_reports_transport_error_on_upload_failure():
    from memforge.main import run_watch_tick

    client = _Client(upload_result={"error": "MemForge API unavailable", "detail": "connection refused"})
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash="abc",
        capture=_capture_good,
        log=lambda m: None,
    )
    assert action == "transport_error"
    # On a failed upload the old hash is retained so the next tick retries.
    assert new_hash == "abc"


async def test_tick_does_not_report_workspace_selection_conflict_as_principal_change():
    from memforge.main import run_watch_tick

    messages = []
    client = _Client(
        upload_result={
            "error": "MemForge API request failed",
            "status_code": 409,
            "code": "workspace_selection_required",
            "detail": "Select a workspace for this request.",
            "workspace_ids": ["ws-a", "ws-b"],
        }
    )
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": client},
        last_hash="abc",
        capture=_capture_good,
        log=messages.append,
    )

    assert action == "transport_error"
    assert new_hash == "abc"
    assert not any("different Jira user" in message for message in messages)
    assert any("workspace_selection_required" in message for message in messages)


async def test_tick_uploads_to_every_workspace_and_reports_each_result():
    from memforge.main import run_watch_tick

    messages = []
    accepted = _Client()
    changed = _Client(upload_result=_principal_changed())
    action, new_hash = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": accepted, "ws-b": changed},
        last_hash=None,
        capture=_capture_good,
        log=messages.append,
    )

    assert action == "principal_conflict"
    assert new_hash is None
    assert accepted.uploaded == ["SESSION=good"]
    assert changed.uploaded == ["SESSION=good"]
    assert any(message.startswith("[ws-a] Refreshed") for message in messages)
    assert any(message.startswith("[ws-b] A different Jira user") for message in messages)


async def test_tick_marks_every_workspace_expired():
    from memforge.main import run_watch_tick

    first, second = _Client(), _Client()
    action, _ = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        clients={"ws-a": first, "ws-b": second},
        last_hash="abc",
        capture=_capture_dead,
        log=lambda m: None,
    )

    assert action == "expired"
    assert first.expired and second.expired
