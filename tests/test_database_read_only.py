from __future__ import annotations

import sqlite3

import aiosqlite
import pytest

from memforge.storage.database import Database


@pytest.mark.asyncio
async def test_non_migrating_connect_opens_existing_database_read_only(tmp_path) -> None:
    path = tmp_path / "workspace.sqlite"
    writer = Database(str(path))
    await writer.connect()
    await writer.upsert_source(
        id="src-read-only",
        type="local_markdown",
        name="Read only source",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    await writer.close()
    before = path.read_bytes()

    reader = Database(str(path))
    await reader.connect(run_migrations=False)
    try:
        assert [source["id"] for source in await reader.list_sources()] == [
            "src-read-only"
        ]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            await reader.db.execute("DELETE FROM sources")
    finally:
        await reader.close()

    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_non_migrating_connect_does_not_create_missing_database(tmp_path) -> None:
    path = tmp_path / "missing" / "workspace.sqlite"
    database = Database(str(path))

    with pytest.raises(sqlite3.OperationalError, match="unable to open database"):
        await database.connect(run_migrations=False)

    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.asyncio
async def test_failed_connect_drains_aiosqlite_worker_before_raising(
    tmp_path,
    monkeypatch,
) -> None:
    class Worker:
        joined = False

        def is_alive(self) -> bool:
            return True

        def join(self) -> None:
            self.joined = True

    class FailedConnectionAttempt:
        def __init__(self) -> None:
            self._thread = Worker()

        def __await__(self):
            async def fail():
                raise sqlite3.OperationalError("unable to open database file")

            return fail().__await__()

    attempt = FailedConnectionAttempt()
    monkeypatch.setattr(aiosqlite, "connect", lambda *args, **kwargs: attempt)
    database = Database(str(tmp_path / "missing.sqlite"))

    with pytest.raises(sqlite3.OperationalError, match="unable to open database"):
        await database.connect(run_migrations=False)

    assert attempt._thread.joined
