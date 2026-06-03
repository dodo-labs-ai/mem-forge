"""Tests for Jira origin history at the database layer (delete_auth_session)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from memforge.storage.database import Database

_SECRET_SENTINEL = "ENC-SECRET-DO-NOT-LEAK"


def _seed_db(db_path: Path, *, with_session: bool = False) -> None:
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
