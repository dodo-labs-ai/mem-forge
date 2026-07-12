from __future__ import annotations

import pytest

from memforge.auth.jira_auth import JiraAuthSessionMissingError


class FakeBrowserSession:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.stored = []

    def capture(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0)

    def store(self, **kwargs):
        self.stored.append(kwargs)


async def test_capture_prefers_silent_profile_before_system_browser_cookie():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    session = FakeBrowserSession([JiraBrowserCapture.captured("SESSION=silent")])

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser_session=session,
        extractor=lambda *_args: (_ for _ in ()).throw(AssertionError("system browser should not be read")),
        validator=lambda origin, cookie, tls_config=None: {"accountId": "user-1"},
    )

    assert result.cookie_header == "SESSION=silent"
    assert session.calls[0]["interactive"] is False


async def test_capture_runs_sync_browser_boundary_outside_the_event_loop():
    import asyncio

    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    class LoopCheckingBrowserSession(FakeBrowserSession):
        def capture(self, **kwargs):
            with pytest.raises(RuntimeError, match="no running event loop"):
                asyncio.get_running_loop()
            return super().capture(**kwargs)

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser_session=LoopCheckingBrowserSession([JiraBrowserCapture.captured("SESSION=silent")]),
        extractor=lambda *_args: (_ for _ in ()).throw(AssertionError("system browser should not be read")),
        validator=lambda origin, cookie, tls_config=None: {"accountId": "user-1"},
    )

    assert result.cookie_header == "SESSION=silent"


async def test_capture_uses_interactive_profile_only_after_silent_paths_fail():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    session = FakeBrowserSession([
        JiraBrowserCapture.captured("SESSION=expired"),
        JiraBrowserCapture.captured("SESSION=interactive"),
    ])

    async def validator(_origin, cookie, _tls_config=None):
        if cookie == "SESSION=expired":
            raise JiraAuthSessionMissingError("expired")
        return {"accountId": "user-1"}

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        interactive=True,
        browser_session=session,
        extractor=lambda *_args: (_ for _ in ()).throw(JiraAuthSessionMissingError("no system cookie")),
        validator=validator,
    )

    assert result.cookie_header == "SESSION=interactive"
    assert [call["interactive"] for call in session.calls] == [False, True]
    assert session.calls[1]["rejected_cookie_hashes"]


async def test_capture_rejects_invalid_system_cookie_before_interactive_profile():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    session = FakeBrowserSession([
        JiraBrowserCapture.interaction_required(),
        JiraBrowserCapture.captured("SESSION=interactive"),
    ])

    async def validator(_origin, cookie, _tls_config=None):
        if cookie == "SESSION=system-expired":
            raise JiraAuthSessionMissingError("expired")
        return {"accountId": "user-1"}

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        interactive=True,
        browser_session=session,
        extractor=lambda *_args: ("SESSION=system-expired", "Chrome"),
        validator=validator,
    )

    assert result.cookie_header == "SESSION=interactive"
    assert session.calls[1]["rejected_cookie_hashes"]


async def test_capture_and_prevalidate_returns_cookie_and_principal():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    session = FakeBrowserSession([JiraBrowserCapture.interaction_required()])
    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser=None,
        extractor=lambda origin, browser: ("SESSION=good", "Chrome"),
        validator=lambda origin, cookie, tls_config=None: {"accountId": "user-1", "displayName": "Ann"},
        browser_session=session,
    )
    assert result.cookie_header == "SESSION=good"
    assert result.browser == "Chrome"
    assert result.principal["accountId"] == "user-1"
    assert result.cookie_header == session.stored[0]["cookie_header"]


async def test_capture_and_prevalidate_awaits_async_validator_principal():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    async def async_validator(origin, cookie, tls_config=None):
        return {"accountId": "user-2", "displayName": "Bo"}

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser="Edge",
        extractor=lambda origin, browser: ("SESSION=ok", "Edge"),
        validator=async_validator,
        browser_session=FakeBrowserSession([JiraBrowserCapture.interaction_required()]),
    )
    assert result.principal["accountId"] == "user-2"
    assert result.browser == "Edge"


async def test_capture_keeps_valid_system_session_when_profile_store_fails():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    class StoreFailingBrowserSession(FakeBrowserSession):
        def store(self, **kwargs):
            raise RuntimeError("profile busy")

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        extractor=lambda origin, browser: ("SESSION=valid", "Chrome"),
        validator=lambda origin, cookie, tls_config=None: {"accountId": "user-1"},
        browser_session=StoreFailingBrowserSession([JiraBrowserCapture.interaction_required()]),
    )

    assert result.cookie_header == "SESSION=valid"


async def test_capture_and_prevalidate_raises_when_session_dead():
    from memforge.auth import jira_capture
    from memforge.auth.jira_browser_session import JiraBrowserCapture

    async def dead_validator(origin, cookie, tls_config=None):
        raise JiraAuthSessionMissingError("not accepted")

    with pytest.raises(JiraAuthSessionMissingError):
        await jira_capture.capture_and_prevalidate(
            "https://jira.example.test",
            browser=None,
            extractor=lambda origin, browser: ("SESSION=dead", "Chrome"),
            validator=dead_validator,
            browser_session=FakeBrowserSession([JiraBrowserCapture.interaction_required()]),
        )
