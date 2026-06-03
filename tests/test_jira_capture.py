from __future__ import annotations

import pytest

from memforge.auth.jira_auth import JiraAuthSessionMissingError


async def test_capture_and_prevalidate_returns_cookie_and_principal():
    from memforge.auth import jira_capture

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser=None,
        extractor=lambda origin, browser: ("SESSION=good", "Chrome"),
        validator=lambda origin, cookie, tls_config=None: {"accountId": "user-1", "displayName": "Ann"},
    )
    assert result.cookie_header == "SESSION=good"
    assert result.browser == "Chrome"
    assert result.principal["accountId"] == "user-1"


async def test_capture_and_prevalidate_awaits_async_validator_principal():
    from memforge.auth import jira_capture

    async def async_validator(origin, cookie, tls_config=None):
        return {"accountId": "user-2", "displayName": "Bo"}

    result = await jira_capture.capture_and_prevalidate(
        "https://jira.example.test",
        browser="Edge",
        extractor=lambda origin, browser: ("SESSION=ok", "Edge"),
        validator=async_validator,
    )
    assert result.principal["accountId"] == "user-2"
    assert result.browser == "Edge"


async def test_capture_and_prevalidate_raises_when_session_dead():
    from memforge.auth import jira_capture

    async def dead_validator(origin, cookie, tls_config=None):
        raise JiraAuthSessionMissingError("not accepted")

    with pytest.raises(JiraAuthSessionMissingError):
        await jira_capture.capture_and_prevalidate(
            "https://jira.example.test",
            browser=None,
            extractor=lambda origin, browser: ("SESSION=dead", "Chrome"),
            validator=dead_validator,
        )
