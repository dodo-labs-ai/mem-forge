"""Tests for Jira origin history: delete_auth_session, jira-list, jira-forget."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from click.testing import CliRunner

from memforge.main import cli
from memforge.storage.database import Database

_SECRET_SENTINEL = "ENC-SECRET-DO-NOT-LEAK"


def _seed_db(db_path: Path, *, with_session: bool = False, with_source: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))
    asyncio.run(db.connect())

    async def _seed():
        if with_session:
            await db.upsert_auth_session(
                provider="jira",
                origin="https://jira.tools.sap",
                secret_encrypted=_SECRET_SENTINEL,
                principal_id="JIRAUSER1",
                principal_name="Sun, Andrew",
                principal_email="andrew@example.com",
                browser="Chrome",
                status="active",
                captured_at="2026-06-02T00:00:00+00:00",
                validated_at="2026-06-02T00:00:00+00:00",
                last_error=None,
            )
        if with_source:
            await db.upsert_source(
                id="src-jira1",
                type="jira",
                name="SFPAY Board",
                config_json=json.dumps({"base_url": "https://jira.other.example", "auth_mode": "browser_cookie"}),
            )

    asyncio.run(_seed())
    asyncio.run(db.close())


def test_delete_auth_session_removes_row(tmp_path):
    db_path = tmp_path / "db" / "memforge.db"
    _seed_db(db_path, with_session=True)
    db = Database(str(db_path))
    asyncio.run(db.connect())
    try:
        before = asyncio.run(db.list_auth_sessions("jira"))
        removed = asyncio.run(db.delete_auth_session("jira", "https://jira.tools.sap"))
        after = asyncio.run(db.list_auth_sessions("jira"))
        removed_again = asyncio.run(db.delete_auth_session("jira", "https://jira.tools.sap"))
    finally:
        asyncio.run(db.close())
    assert len(before) == 1
    assert removed is True
    assert after == []
    assert removed_again is False


def test_jira_list_merges_sessions_and_sources(monkeypatch, tmp_path):
    base = tmp_path / "mem"
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(base))
    _seed_db(base / "db" / "memforge.db", with_session=True, with_source=True)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "list"])

    assert result.exit_code == 0, result.output
    # Parse stdout only: gene-registration logging lands on stderr, and the Node
    # CLI likewise reads stdout. So the payload is always clean JSON here.
    payload = json.loads(result.stdout)
    origins = {o["origin"]: o for o in payload["origins"]}
    assert origins["https://jira.tools.sap"]["status"] == "active"
    assert origins["https://jira.tools.sap"]["principal_name"] == "Sun, Andrew"
    assert origins["https://jira.other.example"]["configured"] is True
    # The encrypted cookie must never be emitted.
    assert _SECRET_SENTINEL not in result.stdout
    assert "secret_encrypted" not in result.stdout


def test_jira_forget_deletes_session(monkeypatch, tmp_path):
    base = tmp_path / "mem"
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(base))
    _seed_db(base / "db" / "memforge.db", with_session=True)

    forget = CliRunner().invoke(cli, ["adapter", "auth", "jira", "forget", "--base-url", "https://jira.tools.sap"])
    listing = CliRunner().invoke(cli, ["adapter", "auth", "jira", "list"])

    assert forget.exit_code == 0, forget.output
    assert json.loads(forget.stdout)["forgotten"] is True
    assert json.loads(listing.stdout)["origins"] == []
