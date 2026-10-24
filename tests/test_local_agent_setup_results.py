"""Each local setup job's handler reports a result its operation's model describes.

The handlers run with fakes for the folder picker, the file system, Git, Jira
capture and the Teams clients. Every result, success or failure, must
validate against the operation's model in ``SETUP_RESULT_MODELS``, carry
every field that model declares and be exactly what it dumps, so the
published types match what the daemon sends.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import memforge.auth.teams_auth as teams_auth
import memforge.local_agent.teams_browse as teams_browse
from memforge import main
from memforge.auth import jira_capture
from memforge.auth.jira_auth import JIRA_PRINCIPAL_CHANGED_CODE
from memforge.genes.teams_client import AuthenticationError
from memforge.local_agent.setup_results import SETUP_RESULT_MODELS, JiraAuthResult, LocalMarkdownPickRootResult
from memforge.server.admin_api import LOCAL_AGENT_SETUP_OPERATION_SOURCE_TYPES
from tests.test_cli_agent_tools import FakeToolClient, _cloud_test_client, _fake_github_remote_run
from tests.test_source_authority_api import _app, _connect_database

BROWSE_DATA = {
    "favorites": [{"id": "19:fav@thread.tacv2", "topic": "Engineering / General"}],
    "teams": [
        {
            "id": "team-1",
            "displayName": "Engineering",
            "channels": [{"id": "19:channel@thread.tacv2", "displayName": "Architecture"}],
        }
    ],
    "group_chats": [{"id": "19:group@thread.v2", "topic": "Planning", "lastActivity": "2026-10-20T09:00:00+00:00"}],
    "individual_chats": [{"id": "19:dm@thread.v2", "topic": "Ada", "lastActivity": None}],
}


def _run(operation: str, payload: dict | None = None, **job) -> dict:
    return main._run_cloud_local_agent_job(
        {"job_id": f"laj-{operation}", "operation": operation, "payload": payload or {}, **job},
        _cloud_test_client(),
        browser="chrome",
    )


def _assert_described(operation: str, result: dict) -> None:
    model = SETUP_RESULT_MODELS[operation]
    assert result.keys() == model.model_fields.keys()
    assert model.model_validate(result).model_dump() == result


def test_every_setup_operation_has_a_result_model():
    assert set(SETUP_RESULT_MODELS) == set(LOCAL_AGENT_SETUP_OPERATION_SOURCE_TYPES)


def test_the_live_document_publishes_every_setup_result_model(tmp_path):
    database = _connect_database(tmp_path)
    try:
        with TestClient(_app(tmp_path, database)) as client:
            schemas = client.get("/openapi.json").json()["components"]["schemas"]
    finally:
        asyncio.run(database.close())

    for model in SETUP_RESULT_MODELS.values():
        assert schemas[model.__name__]["properties"].keys() == model.model_fields.keys()
    assert schemas["TeamsBrowseResult"]["properties"]["teams"]["anyOf"][0]["items"] == {
        "$ref": "#/components/schemas/TeamsTeamOption"
    }


def test_github_repo_preview_tree_result(monkeypatch):
    monkeypatch.setattr(main.subprocess, "run", _fake_github_remote_run)

    result = _run(
        "github_repo_preview_tree",
        {"repo_url": "https://github.internal.example.com/acme-gamma/architecture", "ref": "main", "limit": 2},
        source_id="src-gh",
    )

    _assert_described("github_repo_preview_tree", result)
    assert result["counts"] == {"included": 3, "ignored": 2}
    assert len(result["items"]) == 2


def test_local_markdown_preview_tree_result(tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision\n\nUse the daemon.", encoding="utf-8")

    result = _run("local_markdown_preview_tree", {"root": str(root), "vault_id": "engineering", "limit": 20})

    _assert_described("local_markdown_preview_tree", result)
    assert result["items"][0]["relative_path"] == "Decision.md"


def _picker(outcome: Callable[[], str]):
    def pick_folder(*, title: str | None = None, initial_directory: str | None = None) -> str:
        return outcome()

    return pick_folder


def _raise(exc: Exception) -> Callable[[], str]:
    def outcome() -> str:
        raise exc

    return outcome


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (lambda: "/Users/me/notes", {"root": "/Users/me/notes"}),
        (_raise(main.FolderPickerCancelled("closed")), {"cancelled": True}),
        (_raise(main.FolderPickerUnavailable("type the folder path instead")), {"error": "type the folder path instead"}),
    ],
    ids=["chosen", "cancelled", "unavailable"],
)
def test_local_markdown_pick_root_result(monkeypatch, outcome, expected):
    monkeypatch.setattr(main, "pick_folder", _picker(outcome))

    result = _run("local_markdown_pick_root", {"title": "Choose folder to sync"})

    _assert_described("local_markdown_pick_root", result)
    assert result == LocalMarkdownPickRootResult(operation="local_markdown_pick_root", **expected).model_dump()


def _capture(*, fail: Exception | None = None):
    async def capture_and_prevalidate(base_url, *, browser=None, tls_config=None, interactive=False):
        if fail is not None:
            raise fail
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="SESSION=interactive",
            browser="MemForge Chrome profile",
            principal={"accountId": "jira-user"},
        )

    return capture_and_prevalidate


PRINCIPAL_CHANGED = {
    "code": JIRA_PRINCIPAL_CHANGED_CODE,
    "detail": {"origin": "https://jira.example.test", "old_principal_id": "ada", "new_principal_id": "bob"},
}


@pytest.mark.parametrize(
    ("upload", "fail", "expected"),
    [
        ({"status": "active"}, None, {"authenticated": True, "origin": "https://jira.example.test"}),
        (PRINCIPAL_CHANGED, None, {"error": "principal_changed", "new_principal_id": "bob"}),
        ({"error": "storage unavailable"}, None, {"error_type": "JiraSessionUploadError"}),
        ({}, RuntimeError("browser closed"), {"error": "browser closed", "error_type": "RuntimeError"}),
    ],
    ids=["renewed", "principal-changed", "upload-failed", "capture-failed"],
)
def test_jira_auth_result(monkeypatch, upload, fail, expected):
    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", _capture(fail=fail))
    FakeToolClient.reset(upload)

    result = _run(
        "jira_auth",
        {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie"},
        source_id="src-jira",
        workspace_id="ws-1",
    )

    _assert_described("jira_auth", result)
    assert expected.items() <= result.items()


def test_jira_auth_without_a_source_is_a_described_failure():
    result = _run("jira_auth", {"base_url": "https://jira.example.test"})

    _assert_described("jira_auth", result)
    assert result == JiraAuthResult(operation="jira_auth", error="source_id is required", retryable=False).model_dump()


class _NoTeamsSession:
    @staticmethod
    def load_tokens():
        return None

    def authenticate(self, **_kwargs):
        raise RuntimeError("Teams sign-in timed out.")


class _SignedInTeams:
    @staticmethod
    def load_tokens():
        return None

    def authenticate(self, *, region: str = "emea", **_kwargs):
        return {"region": region, "tokens": {"https://ic3.teams.office.com": {"expiresAt": 4_102_444_800}}}


@pytest.mark.parametrize(
    ("status", "authenticated"),
    [
        ({"authenticated": True, "expires_in_minutes": 42, "error": None}, True),
        (None, False),
    ],
    ids=["signed-in", "no-session"],
)
def test_teams_auth_check_result(monkeypatch, status, authenticated):
    if status is None:
        monkeypatch.setattr(teams_browse, "TeamsAuthenticator", _NoTeamsSession)
    else:
        monkeypatch.setattr(teams_browse, "teams_auth_status", lambda: status)

    result = _run("teams_auth_check", {"region": "apac"})

    _assert_described("teams_auth_check", result)
    assert result["authenticated"] is authenticated


@pytest.mark.parametrize(("authenticator", "authenticated"), [(_SignedInTeams, True), (_NoTeamsSession, False)])
def test_teams_auth_result(monkeypatch, authenticator, authenticated):
    monkeypatch.setattr(teams_auth, "TeamsAuthenticator", authenticator)

    result = _run("teams_auth", {"region": "amer", "wait_seconds": 1})

    _assert_described("teams_auth", result)
    assert result["authenticated"] is authenticated


def _browse(fail: Exception | None = None):
    async def browse_teams_conversations(*, region: str = "emea"):
        if fail is not None:
            raise fail
        return BROWSE_DATA

    return browse_teams_conversations


@pytest.mark.parametrize(
    ("fail", "expected"),
    [
        (None, {"teams": BROWSE_DATA["teams"]}),
        (AuthenticationError("Teams session expired."), {"error": "Teams sign-in timed out."}),
        (ValueError("Teams is unreachable"), {"error": "Teams is unreachable", "error_type": "ValueError"}),
    ],
    ids=["listed", "renewal-failed", "unreachable"],
)
def test_teams_browse_result(monkeypatch, fail, expected):
    monkeypatch.setattr(teams_browse, "browse_teams_conversations", _browse(fail))
    monkeypatch.setattr(teams_auth, "TeamsAuthenticator", _NoTeamsSession)

    result = _run("teams_browse", {"region": "emea", "wait_seconds": 1})

    _assert_described("teams_browse", result)
    assert result["operation"] == "teams_browse"
    assert expected.items() <= result.items()
