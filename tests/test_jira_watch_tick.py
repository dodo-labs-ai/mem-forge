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
        client=client,
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
        client=client,
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
        client=client,
        last_hash="abc",
        capture=_capture_dead,
        log=lambda m: None,
    )
    assert action == "expired"
    assert client.expired and new_hash is None


async def test_tick_flags_principal_conflict():
    from memforge.main import run_watch_tick

    client = _Client(upload_result={"error": "MemForge API request failed", "status_code": 409, "detail": "{}"})
    action, _ = await run_watch_tick(
        base_url="https://jira.example.test",
        browser=None,
        client=client,
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
        client=client,
        last_hash="abc",
        capture=_capture_good,
        log=lambda m: None,
    )
    assert action == "transport_error"
    # On a failed upload the old hash is retained so the next tick retries.
    assert new_hash == "abc"
