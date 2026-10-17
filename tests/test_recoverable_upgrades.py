"""Upgrading a local installation: atomic migrations, recovery points, downgrade refusal and restore."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner
from rich.console import Console

from memforge.main import cli
from memforge.storage import database as database_module
from memforge.storage.database import (
    Database,
    MigrationFailed,
    SchemaNewerThanRelease,
    latest_schema_version,
)
from memforge.storage.recovery_points import (
    PRE_RESTORE,
    PRE_UPGRADE,
    RETAINED_RECOVERY_POINTS,
    DataRootBusy,
    RecoveryPointError,
    RecoveryPoints,
)

RELEASED_MIGRATIONS = list(database_module.MIGRATIONS)
LATEST = latest_schema_version()
PREVIOUS = LATEST - 1


@pytest.fixture
def base_dir(tmp_path) -> Path:
    return tmp_path / "mem"


@pytest.fixture
def db_path(base_dir) -> Path:
    return base_dir / "db" / "memforge.db"


@pytest.fixture
def chroma_path(base_dir) -> Path:
    return base_dir / "vectors" / "chroma"


@pytest.fixture
def points(base_dir, db_path, chroma_path) -> RecoveryPoints:
    return RecoveryPoints(base_dir / "backups", db_path=db_path, chroma_path=chroma_path)


async def _add_source(path: Path, source_id: str) -> None:
    database = Database(str(path))
    await database.connect()
    try:
        await database.upsert_source(
            id=source_id,
            type="local_markdown",
            name=source_id,
            config_json="{}",
            access_policy="workspace",
            owner_user_id="owner-1",
        )
    finally:
        await database.close()


def _forget_migrations_after(path: Path, version: int) -> None:
    """Make the database look as a release whose newest migration is ``version`` left it."""
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DELETE FROM schema_migrations WHERE version > ?", (version,))
        connection.commit()


async def _previous_release_installation(db_path: Path, chroma_path: Path, *, vectors: str = "v1") -> None:
    await _add_source(db_path, "src-before-upgrade")
    _forget_migrations_after(db_path, PREVIOUS)
    chroma_path.mkdir(parents=True)
    (chroma_path / "chroma.sqlite3").write_text(vectors)


def _schema(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])


def _source_ids(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {row[0] for row in connection.execute("SELECT id FROM sources")}


def _table_exists(path: Path, table: str) -> bool:
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def _with_probe_migration(monkeypatch, statements: list[str]) -> int:
    """Make the release one migration newer, applying ``statements``; return its version."""
    version = LATEST + 1
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        [*RELEASED_MIGRATIONS, (version, "Rebuild the probe table", statements)],
    )
    return version


# ---------------------------------------------------------------------------
# One transaction per migration
# ---------------------------------------------------------------------------


async def test_failed_migration_rolls_back_whole_and_the_next_start_applies_it(db_path, monkeypatch) -> None:
    await _add_source(db_path, "src-kept")
    _with_probe_migration(
        monkeypatch,
        [
            "CREATE TABLE probe_new (id TEXT PRIMARY KEY)",
            "INSERT INTO probe_new (id) VALUES ('row-1')",
            "INSERT INTO probe_new (id) VALUES ('row-1')",
        ],
    )

    database = Database(str(db_path))
    with pytest.raises(MigrationFailed) as failed:
        await database.connect()
    await database.close()

    assert f"Migration {LATEST + 1} (Rebuild the probe table) failed and was rolled back" in str(failed.value)
    assert f"the database stays at schema {LATEST}" in str(failed.value)
    assert "UNIQUE constraint failed" in str(failed.value)
    assert not _table_exists(db_path, "probe_new")
    assert _schema(db_path) == LATEST
    assert _source_ids(db_path) == {"src-kept"}

    probe = _with_probe_migration(
        monkeypatch,
        ["CREATE TABLE probe_new (id TEXT PRIMARY KEY)", "INSERT INTO probe_new (id) VALUES ('row-1')"],
    )
    database = Database(str(db_path))
    await database.connect()
    await database.close()

    assert _schema(db_path) == probe
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("SELECT id FROM probe_new").fetchall() == [("row-1",)]


async def test_failed_data_step_rolls_back_the_statements_before_it(db_path, monkeypatch) -> None:
    await _add_source(db_path, "src-kept")
    probe = _with_probe_migration(monkeypatch, ["ALTER TABLE sources ADD COLUMN probe_marker TEXT"])
    apply_migration = Database._apply_migration_unlocked

    async def apply_then_fail(self, version, statements):
        await apply_migration(self, version, statements)
        if version == probe:
            raise RuntimeError("data step failed")

    monkeypatch.setattr(Database, "_apply_migration_unlocked", apply_then_fail)

    database = Database(str(db_path))
    with pytest.raises(MigrationFailed, match="RuntimeError: data step failed"):
        await database.connect()
    await database.close()

    with closing(sqlite3.connect(db_path)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(sources)")}
    assert "probe_marker" not in columns
    assert _schema(db_path) == LATEST


async def test_process_stopped_mid_migration_leaves_the_previous_schema_and_restarts_cleanly(
    db_path,
    monkeypatch,
) -> None:
    await _add_source(db_path, "src-kept")
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import sys

        from memforge.storage import database
        from memforge.storage.database import Database

        probe = database.latest_schema_version() + 1
        database.MIGRATIONS = [
            *database.MIGRATIONS,
            (probe, "Rebuild the probe table", [
                "CREATE TABLE probe_new (id TEXT PRIMARY KEY)",
                "INSERT INTO probe_new (id) VALUES ('row-1')",
            ]),
        ]
        apply_migration = Database._apply_migration_unlocked

        async def apply_then_stop(self, version, statements):
            await apply_migration(self, version, statements)
            if version == probe:
                os._exit(3)

        Database._apply_migration_unlocked = apply_then_stop

        async def main():
            await Database(sys.argv[1]).connect()

        asyncio.run(main())
        """
    )
    stopped = subprocess.run(
        [sys.executable, "-c", script, str(db_path)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert stopped.returncode == 3, stopped.stderr

    assert not _table_exists(db_path, "probe_new")
    assert _schema(db_path) == LATEST

    probe = _with_probe_migration(
        monkeypatch,
        ["CREATE TABLE probe_new (id TEXT PRIMARY KEY)", "INSERT INTO probe_new (id) VALUES ('row-1')"],
    )
    database = Database(str(db_path))
    await database.connect()
    await database.close()

    assert _schema(db_path) == probe
    assert _source_ids(db_path) == {"src-kept"}


async def test_table_rebuild_runs_without_foreign_keys_and_restores_enforcement(db_path, monkeypatch) -> None:
    await _add_source(db_path, "src-kept")
    probe = _with_probe_migration(
        monkeypatch,
        ["CREATE TABLE probe_foreign_keys AS SELECT foreign_keys FROM pragma_foreign_keys"],
    )
    monkeypatch.setattr(database_module, "_MIGRATIONS_WITHOUT_FOREIGN_KEYS", frozenset({probe}))

    database = Database(str(db_path))
    await database.connect()
    try:
        async with database.db.execute("SELECT foreign_keys FROM probe_foreign_keys") as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [(0,)]
        async with database.db.execute("PRAGMA foreign_keys") as cursor:
            assert (await cursor.fetchone())[0] == 1
    finally:
        await database.close()


# ---------------------------------------------------------------------------
# Recovery points before an upgrade
# ---------------------------------------------------------------------------


async def test_upgrade_takes_a_recovery_point_of_database_and_vectors_first(
    db_path,
    chroma_path,
    points,
) -> None:
    await _previous_release_installation(db_path, chroma_path)

    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()

    [point] = points.list()
    assert point.reason == PRE_UPGRADE
    assert point.schema_version == PREVIOUS
    assert point.id.startswith(f"{PRE_UPGRADE}-")
    assert f"-schema-{PREVIOUS}-memforge-" in point.id
    assert _schema(point.database_path) == PREVIOUS
    assert _source_ids(point.database_path) == {"src-before-upgrade"}
    assert point.includes_vectors
    assert (point.vectors_path / "chroma.sqlite3").read_text() == "v1"
    assert _schema(db_path) == LATEST

    reopened = Database(str(db_path), recovery_points=points)
    await reopened.connect()
    await reopened.close()

    assert [existing.id for existing in points.list()] == [point.id]


async def test_fresh_installation_takes_no_recovery_point(db_path, points) -> None:
    db_path.parent.mkdir(parents=True)
    db_path.touch()

    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()

    assert _schema(db_path) == LATEST
    assert points.list() == []
    assert not points.directory.exists()


async def test_failed_upgrade_names_the_recovery_point_taken_before_it(
    db_path,
    chroma_path,
    points,
    monkeypatch,
) -> None:
    await _previous_release_installation(db_path, chroma_path)
    _with_probe_migration(monkeypatch, ["SELECT * FROM no_such_table"])

    database = Database(str(db_path), recovery_points=points)
    with pytest.raises(MigrationFailed) as failed:
        await database.connect()
    await database.close()

    [point] = points.list()
    assert f"kept as recovery point {point.id}" in str(failed.value)
    # Each migration that succeeded before the failing one is committed.
    assert _schema(db_path) == LATEST


async def test_upgrade_is_refused_while_another_process_has_the_data_open(db_path, chroma_path, points) -> None:
    await _add_source(db_path, "src-open")
    chroma_path.mkdir(parents=True)
    running = Database(str(db_path), recovery_points=points)
    await running.connect()
    try:
        _forget_migrations_after(db_path, PREVIOUS)
        upgrading = Database(str(db_path), recovery_points=points)
        with pytest.raises(DataRootBusy, match="Another MemForge process has the data"):
            await upgrading.connect()
        await upgrading.close()
    finally:
        await running.close()

    assert points.list() == []
    assert _schema(db_path) == PREVIOUS


def _set_schema(db_path: Path, version: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
        connection.execute("DELETE FROM schema_migrations")
        connection.execute("INSERT INTO schema_migrations VALUES (?)", (version,))
        connection.commit()


def test_pre_restore_points_keep_the_newest_three(db_path, points) -> None:
    _set_schema(db_path, 7)
    pre_upgrade = points.create(PRE_UPGRADE)
    taken = [points.create(PRE_RESTORE) for _ in range(RETAINED_RECOVERY_POINTS + 2)]

    removed = points.prune(PRE_RESTORE)

    assert removed == [point.id for point in reversed(taken[:2])]
    assert [point.id for point in points.list()] == [point.id for point in reversed(taken[2:])] + [pre_upgrade.id]


def test_pre_upgrade_points_keep_the_newest_of_each_recent_schema(db_path, points) -> None:
    taken = {}
    for version in (5, 6, 7):
        _set_schema(db_path, version)
        taken[version] = points.create(PRE_UPGRADE)
    _set_schema(db_path, 8)
    repeated = [points.create(PRE_UPGRADE) for _ in range(RETAINED_RECOVERY_POINTS + 1)]

    removed = points.prune(PRE_UPGRADE)

    assert set(removed) == {taken[5].id, *(point.id for point in repeated[:-1])}
    assert [point.id for point in points.list()] == [repeated[-1].id, taken[7].id, taken[6].id]
    assert points.newest_readable(7) == taken[7]


def test_a_new_recovery_point_removes_what_an_interrupted_one_left(db_path, points) -> None:
    _set_schema(db_path, 7)
    leftover = points.directory / ".partial-interrupted"
    (leftover / "chroma").mkdir(parents=True)
    (leftover / "memforge.db").write_bytes(b"partial copy")

    point = points.create(PRE_UPGRADE)

    assert not leftover.exists()
    assert [entry.name for entry in points.directory.iterdir()] == [point.id]


async def test_retried_upgrade_keeps_the_point_taken_before_it_began(
    db_path,
    chroma_path,
    points,
    monkeypatch,
) -> None:
    await _previous_release_installation(db_path, chroma_path)
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        [
            *RELEASED_MIGRATIONS,
            (LATEST + 1, "Add the probe table", ["CREATE TABLE probe_added (id TEXT)"]),
            (LATEST + 2, "Fill the probe table", ["INSERT INTO no_such_table VALUES (1)"]),
        ],
    )

    failures = []
    for _ in range(RETAINED_RECOVERY_POINTS + 1):
        database = Database(str(db_path), recovery_points=points)
        with pytest.raises(MigrationFailed) as failed:
            await database.connect()
        await database.close()
        failures.append(str(failed.value))

    [point] = points.list()
    assert point.schema_version == PREVIOUS
    assert _schema(db_path) == LATEST + 1
    assert all(f"kept as recovery point {point.id}" in failure for failure in failures)
    assert points.newest_readable(PREVIOUS) == point


async def test_retries_by_another_release_keep_a_point_the_older_release_reads(
    db_path,
    chroma_path,
    points,
    monkeypatch,
) -> None:
    await _previous_release_installation(db_path, chroma_path)
    monkeypatch.setattr(
        database_module,
        "MIGRATIONS",
        [
            *RELEASED_MIGRATIONS,
            (LATEST + 1, "Add the probe table", ["CREATE TABLE probe_added (id TEXT)"]),
            (LATEST + 2, "Fill the probe table", ["INSERT INTO no_such_table VALUES (1)"]),
        ],
    )
    database = Database(str(db_path), recovery_points=points)
    with pytest.raises(MigrationFailed):
        await database.connect()
    await database.close()
    [before_upgrade] = points.list()

    monkeypatch.setattr("memforge.storage.recovery_points.installed_release", lambda: "9.9.9")
    monkeypatch.setattr(database_module, "installed_release", lambda: "9.9.9")
    for _ in range(RETAINED_RECOVERY_POINTS + 1):
        database = Database(str(db_path), recovery_points=points)
        with pytest.raises(MigrationFailed):
            await database.connect()
        await database.close()

    assert [point.schema_version for point in points.list()] == [LATEST + 1, PREVIOUS]
    assert points.newest_readable(PREVIOUS) == before_upgrade


# ---------------------------------------------------------------------------
# Downgrade refusal
# ---------------------------------------------------------------------------


async def _upgraded_by_a_newer_release(db_path: Path, chroma_path: Path, points: RecoveryPoints, monkeypatch):
    await _previous_release_installation(db_path, chroma_path)
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    with monkeypatch.context() as newer_release:
        _with_probe_migration(newer_release, ["CREATE TABLE probe_newer (id TEXT)"])
        newer = Database(str(db_path), recovery_points=points)
        await newer.connect()
        await newer.close()
    return points.list()


async def test_older_release_refuses_a_newer_schema_and_names_the_point_to_restore(
    db_path,
    chroma_path,
    points,
    monkeypatch,
) -> None:
    taken = await _upgraded_by_a_newer_release(db_path, chroma_path, points, monkeypatch)
    readable = next(point for point in taken if point.schema_version <= LATEST)
    before = db_path.read_bytes()

    database = Database(str(db_path), recovery_points=points)
    with pytest.raises(SchemaNewerThanRelease) as refused:
        await database.connect()
    await database.close()

    message = str(refused.value)
    assert f"has schema {LATEST + 1}" in message
    assert f"reads schema {LATEST} and older, and has not changed it" in message
    assert f"memforge backup restore {readable.id}" in message
    assert db_path.read_bytes() == before
    assert points.list() == taken

    reader = Database(str(db_path))
    with pytest.raises(SchemaNewerThanRelease):
        await reader.connect(run_migrations=False)
    await reader.close()


@pytest.fixture
def installation_env(base_dir, monkeypatch) -> None:
    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(base_dir))
    for name in ("MEMFORGE_STORAGE_DB_PATH", "MEMFORGE_STORAGE_CHROMA_PATH", "MEMFORGE_STORAGE_DOCS_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("memforge.main.console", Console(width=400))


def test_cli_refuses_a_newer_schema_with_the_restore_command(
    db_path,
    chroma_path,
    points,
    monkeypatch,
    installation_env,
) -> None:
    asyncio.run(_upgraded_by_a_newer_release(db_path, chroma_path, points, monkeypatch))

    result = CliRunner().invoke(cli, ["memories", "stats"])

    assert result.exit_code == 1
    assert "memforge backup restore pre-upgrade-" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


async def test_restore_puts_the_pre_upgrade_data_back_and_keeps_what_it_replaced(
    db_path,
    chroma_path,
    points,
) -> None:
    await _previous_release_installation(db_path, chroma_path, vectors="v1")
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    [point] = points.list()
    await _add_source(db_path, "src-after-upgrade")
    (chroma_path / "chroma.sqlite3").write_text("v2")

    result = points.restore(point.id, readable_schema_version=LATEST)

    assert result.restored == point
    assert _source_ids(db_path) == {"src-before-upgrade"}
    assert _schema(db_path) == PREVIOUS
    assert (chroma_path / "chroma.sqlite3").read_text() == "v1"
    assert result.replaced is not None and result.replaced.reason == PRE_RESTORE
    assert _source_ids(result.replaced.database_path) == {"src-before-upgrade", "src-after-upgrade"}
    assert (result.replaced.vectors_path / "chroma.sqlite3").read_text() == "v2"

    undone = points.restore(result.replaced.id, readable_schema_version=LATEST)
    assert _source_ids(db_path) == {"src-before-upgrade", "src-after-upgrade"}
    assert (chroma_path / "chroma.sqlite3").read_text() == "v2"
    assert undone.replaced is not None

    reopened = Database(str(db_path), recovery_points=points)
    await reopened.connect()
    await reopened.close()
    assert _schema(db_path) == LATEST


async def test_restore_is_refused_while_memforge_has_the_data_open(db_path, chroma_path, points) -> None:
    await _previous_release_installation(db_path, chroma_path)
    running = Database(str(db_path), recovery_points=points)
    await running.connect()
    try:
        [point] = points.list()
        with pytest.raises(DataRootBusy, match="Stop the MemForge server"):
            points.restore(point.id, readable_schema_version=LATEST)
    finally:
        await running.close()

    assert [existing.reason for existing in points.list()] == [PRE_UPGRADE]


async def test_restore_refuses_a_damaged_or_too_new_point(db_path, chroma_path, points) -> None:
    await _previous_release_installation(db_path, chroma_path)
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    [point] = points.list()

    with pytest.raises(RecoveryPointError, match="this release reads schema"):
        points.restore(point.id, readable_schema_version=PREVIOUS - 1)
    with pytest.raises(RecoveryPointError, match="There is no recovery point"):
        points.restore("pre-upgrade-missing", readable_schema_version=LATEST)
    with point.database_path.open("ab") as damaged:
        damaged.write(b"\0")
    with pytest.raises(RecoveryPointError, match="does not match the digest"):
        points.restore(point.id, readable_schema_version=LATEST)

    assert _schema(db_path) == LATEST


async def test_cli_lists_and_restores_a_recovery_point(
    db_path,
    chroma_path,
    points,
    installation_env,
) -> None:
    await _previous_release_installation(db_path, chroma_path)
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    [point] = points.list()

    listed = CliRunner().invoke(cli, ["backup", "list"])
    assert listed.exit_code == 0, listed.output
    assert point.id in listed.output
    assert f"This release reads schema {LATEST} and older" in listed.output

    declined = CliRunner().invoke(cli, ["backup", "restore", point.id], input="n\n")
    assert declined.exit_code == 1
    assert _schema(db_path) == LATEST

    restored = CliRunner().invoke(cli, ["backup", "restore", point.id, "--yes"])
    assert restored.exit_code == 0, restored.output
    assert f"Restored recovery point {point.id}" in restored.output
    assert "kept as recovery point pre-restore-" in restored.output
    assert f"upgrades it to schema {LATEST} when it next starts" in restored.output
    assert _schema(db_path) == PREVIOUS

    missing = CliRunner().invoke(cli, ["backup", "restore", "pre-upgrade-missing", "--yes"])
    assert missing.exit_code == 1
    assert "There is no recovery point" in missing.output


async def test_restore_over_a_damaged_database_is_refused_with_guidance(db_path, chroma_path, points) -> None:
    await _previous_release_installation(db_path, chroma_path)
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    [point] = points.list()
    with db_path.open("r+b") as live:
        live.seek(4096)
        live.write(os.urandom(4096 * 4))

    with pytest.raises(RecoveryPointError, match="has not changed anything") as refused:
        points.restore(point.id, readable_schema_version=LATEST)
    assert f"move {db_path.name} and any {db_path.name}-wal" in str(refused.value)
    assert [existing.id for existing in points.list()] == [point.id]
    assert not any(entry.name.startswith(".partial-") for entry in points.directory.iterdir())

    damaged = db_path.with_name("damaged")
    damaged.mkdir()
    db_path.rename(damaged / db_path.name)
    result = points.restore(point.id, readable_schema_version=LATEST)

    assert result.replaced is None
    assert _source_ids(db_path) == {"src-before-upgrade"}


async def test_interrupted_restore_is_refused_until_it_is_run_again(
    db_path,
    chroma_path,
    points,
    monkeypatch,
) -> None:
    await _previous_release_installation(db_path, chroma_path, vectors="v1")
    database = Database(str(db_path), recovery_points=points)
    await database.connect()
    await database.close()
    [point] = points.list()
    await _add_source(db_path, "src-after-upgrade")
    (chroma_path / "chroma.sqlite3").write_text("v2")

    def stop_before_the_database(source, target):
        raise KeyboardInterrupt

    with monkeypatch.context() as interrupted:
        interrupted.setattr("memforge.storage.recovery_points._copy_database", stop_before_the_database)
        with pytest.raises(KeyboardInterrupt):
            points.restore(point.id, readable_schema_version=LATEST)
    [replaced] = [existing for existing in points.list() if existing.reason == PRE_RESTORE]
    assert (chroma_path / "chroma.sqlite3").read_text() == "v1"
    assert "src-after-upgrade" in _source_ids(db_path)

    opening = Database(str(db_path), recovery_points=points)
    with pytest.raises(RecoveryPointError, match=f"memforge backup restore {point.id}` again"):
        await opening.connect()
    await opening.close()

    result = points.restore(point.id, readable_schema_version=LATEST)

    assert result.replaced == replaced
    assert [existing.id for existing in points.list() if existing.reason == PRE_RESTORE] == [replaced.id]
    assert _source_ids(db_path) == {"src-before-upgrade"}
    reopened = Database(str(db_path), recovery_points=points)
    await reopened.connect()
    await reopened.close()
    assert _schema(db_path) == LATEST
