"""Recovery points: copies of a local installation's data that a restore puts back.

MemForge takes a recovery point before it applies pending schema migrations to
an existing database, so an upgrade that goes wrong, or a release that turns
out to be unwanted, is undone by restoring the data as it was before the
upgrade. A restore takes one of the data it replaces, so a restore is undone
the same way.

A recovery point is one directory under ``<base_dir>/backups``:

- ``memforge.db``, a consistent copy of the SQLite database made with the
  SQLite Online Backup API, verified with ``PRAGMA quick_check``;
- ``chroma/``, a copy of the Chroma vector store, present when the
  installation has one. It is copied while no MemForge process has the data
  open, so the vectors match the relational state of the copy beside them;
- ``manifest.json``, which records why and when the point was taken, the
  release that took it, the schema version the database copy holds and the
  SHA-256 digest of that copy.

A recovery point is written under a hidden name and renamed into place once
complete, so a listing never shows a partial one, and the next point taken
removes what an interrupted one left behind.

Of the points taken before an upgrade, the newest of each of the
``RETAINED_RECOVERY_POINTS`` most recent schema versions is kept, so the
release each of those schemas came from still has a point it can restore.
Of the points taken before a restore, the newest ``RETAINED_RECOVERY_POINTS``
are kept.

A restore marks itself in progress until both the vector store and the
database are back in place; while the mark is there, MemForge refuses to open
the data and asks for the restore to be run again.

Source documents under ``documents/`` are not part of a recovery point: no
migration changes them, and a restore leaves them as they are.

The data-root lock keeps an upgrade and a restore from running while another
MemForge process has the data open. Every process that opens the database for
writing holds the lock shared; applying migrations and restoring need it
exclusively, and are refused rather than kept waiting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

if TYPE_CHECKING:
    from memforge.config import AppConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BACKUPS_DIRECTORY",
    "DataRootBusy",
    "DataRootLock",
    "PRE_RESTORE",
    "PRE_UPGRADE",
    "RETAINED_RECOVERY_POINTS",
    "RecoveryPoint",
    "RecoveryPointError",
    "RecoveryPoints",
    "RestoreResult",
    "UpgradeError",
    "installed_release",
]

BACKUPS_DIRECTORY = "backups"
PRE_UPGRADE = "pre-upgrade"
PRE_RESTORE = "pre-restore"
RETAINED_RECOVERY_POINTS = 3

_DISTRIBUTION = "memforge-ai"
_MANIFEST_FILE = "manifest.json"
_DATABASE_FILE = "memforge.db"
_VECTORS_DIRECTORY = "chroma"
_MANIFEST_FORMAT = 1
_PARTIAL_PREFIX = ".partial-"
_RESTORE_IN_PROGRESS_FILE = ".restore-in-progress.json"
_REASONS = (PRE_UPGRADE, PRE_RESTORE)


class UpgradeError(RuntimeError):
    """MemForge cannot open its local data with this release."""


class DataRootBusy(UpgradeError):
    """Another MemForge process has the data open."""


class RecoveryPointError(UpgradeError):
    """A recovery point is missing, damaged, or cannot be restored by this release."""


def installed_release() -> str:
    """Return the installed MemForge release, or ``unknown`` for an uninstalled checkout."""
    try:
        return metadata.version(_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class RecoveryPoint:
    """One complete recovery point, as its manifest describes it."""

    id: str
    path: Path
    reason: str
    created_at: datetime
    schema_version: int
    taken_by_release: str
    database_sha256: str
    includes_vectors: bool

    @property
    def database_path(self) -> Path:
        return self.path / _DATABASE_FILE

    @property
    def vectors_path(self) -> Path | None:
        return self.path / _VECTORS_DIRECTORY if self.includes_vectors else None

    def size_bytes(self) -> int:
        return sum(entry.stat().st_size for entry in self.path.rglob("*") if entry.is_file())


@dataclass(frozen=True)
class RestoreResult:
    restored: RecoveryPoint
    replaced: RecoveryPoint | None


class DataRootLock:
    """An advisory lock on one data root, shared by readers of the data and held alone to change it."""

    def __init__(self, path: Path, data_root: Path) -> None:
        self._path = path
        self._data_root = data_root
        self._handle: int | None = None

    def acquire(self, *, exclusive: bool) -> None:
        """Hold the lock shared or alone, without waiting; raise ``DataRootBusy`` when another process prevents it."""
        if fcntl is None:
            return
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self._handle, mode | fcntl.LOCK_NB)
        except BlockingIOError:
            self.release()
            if exclusive:
                raise DataRootBusy(
                    f"Another MemForge process has the data in {self._data_root} open. "
                    "Stop the MemForge server and every other memforge command using this data, then try again."
                ) from None
            raise DataRootBusy(
                f"The data in {self._data_root} is being upgraded or restored by another MemForge process. "
                "Wait for it to finish, then try again."
            ) from None

    def release(self) -> None:
        if self._handle is None:
            return
        os.close(self._handle)
        self._handle = None


class RecoveryPoints:
    """The recovery points of one data root, and the restore that puts one back."""

    def __init__(self, directory: Path, *, db_path: Path, chroma_path: Path) -> None:
        self.directory = directory
        self._db_path = db_path
        self._chroma_path = chroma_path

    @classmethod
    def for_config(cls, config: "AppConfig") -> "RecoveryPoints":
        return cls(
            Path(config.base_dir).expanduser() / BACKUPS_DIRECTORY,
            db_path=Path(config.storage.db_path).expanduser(),
            chroma_path=Path(config.storage.chroma_path).expanduser(),
        )

    def lock(self) -> DataRootLock:
        return DataRootLock(self._db_path.with_name(f"{self._db_path.name}.lock"), self._db_path.parent)

    # ------------------------------------------------------------------
    # Taking and listing
    # ------------------------------------------------------------------

    def create(self, reason: str) -> RecoveryPoint:
        """Take a recovery point of the database and vector store as they are now.

        The caller holds the data-root lock exclusively, so nothing writes the
        data while it is copied, and no other point is being taken: what an
        interrupted one left in the staging area is removed first.
        """
        if reason not in _REASONS:
            raise ValueError(f"unknown recovery point reason: {reason}")
        self.directory.mkdir(parents=True, exist_ok=True)
        for leftover in self.directory.glob(f"{_PARTIAL_PREFIX}*"):
            shutil.rmtree(leftover, ignore_errors=True)
        created_at = datetime.now(timezone.utc)
        staging = self.directory / f"{_PARTIAL_PREFIX}{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            database_copy = staging / _DATABASE_FILE
            _snapshot_database(self._db_path, database_copy)
            schema_version = _schema_version(database_copy)
            includes_vectors = self._chroma_path.is_dir()
            if includes_vectors:
                shutil.copytree(self._chroma_path, staging / _VECTORS_DIRECTORY)
            release = installed_release()
            manifest = {
                "format": _MANIFEST_FORMAT,
                "reason": reason,
                "created_at": created_at.isoformat(),
                "schema_version": schema_version,
                "taken_by_release": release,
                "database": {"file": _DATABASE_FILE, "sha256": _sha256(database_copy)},
                "vectors": {"directory": _VECTORS_DIRECTORY} if includes_vectors else None,
            }
            (staging / _MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            final = self._available_path(reason, created_at, schema_version, release)
            staging.rename(final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        point = _read_point(final)
        logger.info("Took recovery point %s in %s", point.id, self.directory)
        return point

    def list(self) -> list[RecoveryPoint]:
        """Return the complete recovery points, newest first."""
        if not self.directory.is_dir():
            return []
        points = []
        for entry in self.directory.iterdir():
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            try:
                points.append(_read_point(entry))
            except RecoveryPointError as exc:
                logger.warning("Skipping %s: %s", entry, exc)
        return sorted(points, key=lambda point: point.created_at, reverse=True)

    def get(self, point_id: str) -> RecoveryPoint:
        for point in self.list():
            if point.id == point_id:
                return point
        raise RecoveryPointError(
            f"There is no recovery point {point_id!r} in {self.directory}. Run `memforge backup list` to see them."
        )

    def newest_readable(self, schema_version: int) -> RecoveryPoint | None:
        """Return the newest recovery point holding ``schema_version`` or an older schema."""
        return next((point for point in self.list() if point.schema_version <= schema_version), None)

    def upgrade_in_progress(self, schema_version: int, release: str) -> RecoveryPoint | None:
        """Return the point taken before an upgrade ``release`` began and has not finished.

        That is the newest recovery point when ``release`` took it before an
        upgrade and the database, at ``schema_version``, has gained
        migrations since: an earlier start of ``release`` applied some of its
        migrations and stopped at a later one. The point still holds the data
        as it was before that upgrade.
        """
        points = self.list()
        if not points:
            return None
        newest = points[0]
        if (
            newest.reason == PRE_UPGRADE
            and newest.taken_by_release == release
            and newest.schema_version < schema_version
        ):
            return newest
        return None

    def prune(self, reason: str, *, keep: frozenset[str] = frozenset()) -> list[str]:
        """Delete the points of ``reason`` past what is retained; return their ids.

        Of ``pre-upgrade`` points, the newest of each of the
        ``RETAINED_RECOVERY_POINTS`` most recent schema versions is retained;
        of ``pre-restore`` points, the newest ``RETAINED_RECOVERY_POINTS``.
        The points named in ``keep`` are kept as well.
        """
        same_reason = [point for point in self.list() if point.reason == reason]
        retained: set[str] = set()
        if reason == PRE_UPGRADE:
            schemas: set[int] = set()
            for point in same_reason:
                if point.schema_version not in schemas and len(schemas) < RETAINED_RECOVERY_POINTS:
                    schemas.add(point.schema_version)
                    retained.add(point.id)
        else:
            retained = {point.id for point in same_reason[:RETAINED_RECOVERY_POINTS]}
        removed = []
        for point in same_reason:
            if point.id in retained or point.id in keep:
                continue
            shutil.rmtree(point.path)
            removed.append(point.id)
            logger.info("Deleted recovery point %s", point.id)
        return removed

    # ------------------------------------------------------------------
    # Restoring
    # ------------------------------------------------------------------

    def restore(self, point_id: str, *, readable_schema_version: int) -> RestoreResult:
        """Put the database and vector store of a recovery point back in place.

        ``readable_schema_version`` is the newest schema the running release
        reads; a point holding a newer one is refused. The data being replaced
        is first taken as a ``pre-restore`` recovery point; when a restore that
        was interrupted is run again, the point that restore took stands for
        it. The restore holds the data-root lock exclusively, so it is refused
        while the server or another memforge command has the data open.
        """
        point = self.get(point_id)
        if point.schema_version > readable_schema_version:
            raise RecoveryPointError(
                f"Recovery point {point.id} holds schema {point.schema_version}; this release reads schema "
                f"{readable_schema_version} and older. Restore it with MemForge {point.taken_by_release} or newer."
            )
        if _sha256(point.database_path) != point.database_sha256:
            raise RecoveryPointError(
                f"The database copy in recovery point {point.id} does not match the digest in its manifest; "
                "it is damaged and is not restored."
            )
        lock = self.lock()
        lock.acquire(exclusive=True)
        try:
            interrupted = self._read_restore_in_progress()
            if interrupted is not None:
                replaced = next((kept for kept in self.list() if kept.id == interrupted["replaced"]), None)
            else:
                replaced = self._keep_data_before_restore(point)
            self._write_restore_in_progress(point, replaced)
            try:
                _replace_directory(self._chroma_path, point.vectors_path)
                _copy_database(point.database_path, self._db_path)
            except Exception as exc:
                kept = "" if replaced is None else f" The data it was replacing is kept as recovery point {replaced.id}."
                raise RecoveryPointError(
                    f"The restore of {point.id} stopped part-way ({exc}); run it again.{kept}"
                ) from exc
            self._restore_in_progress_path.unlink()
            self.prune(PRE_RESTORE, keep=frozenset({point.id}))
        finally:
            lock.release()
        logger.info("Restored recovery point %s", point.id)
        return RestoreResult(restored=point, replaced=replaced)

    def refuse_interrupted_restore(self) -> None:
        """Raise ``RecoveryPointError`` while a restore that stopped part-way has not been run again."""
        interrupted = self._read_restore_in_progress()
        if interrupted is None:
            return
        point_id = interrupted.get("restoring") or "ID"
        raise RecoveryPointError(
            f"The restore of recovery point {point_id} stopped before it finished, so the database and the "
            f"vector index may not match each other. Run `memforge backup restore {point_id}` again, then "
            "start MemForge."
        )

    def _keep_data_before_restore(self, point: RecoveryPoint) -> RecoveryPoint | None:
        if not self._db_path.exists():
            return None
        try:
            return self.create(PRE_RESTORE)
        except (sqlite3.Error, OSError, RecoveryPointError) as exc:
            database = self._db_path
            raise RecoveryPointError(
                f"The restore of {point.id} has not changed anything: the current database {database} could "
                f"not be kept as a recovery point first ({exc}). If it is damaged, move {database.name} and any "
                f"{database.name}-wal and {database.name}-shm files out of {database.parent}, then run the "
                "restore again."
            ) from exc

    @property
    def _restore_in_progress_path(self) -> Path:
        return self.directory / _RESTORE_IN_PROGRESS_FILE

    def _read_restore_in_progress(self) -> dict[str, str | None] | None:
        """Return the mark of a restore in progress, naming its point and the point it kept, or ``None``."""
        try:
            marker = json.loads(self._restore_in_progress_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except ValueError:
            marker = None
        if not isinstance(marker, dict):
            return {"restoring": None, "replaced": None}
        return {key: None if marker.get(key) is None else str(marker[key]) for key in ("restoring", "replaced")}

    def _write_restore_in_progress(self, point: RecoveryPoint, replaced: RecoveryPoint | None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = {"restoring": point.id, "replaced": None if replaced is None else replaced.id}
        staging = self._restore_in_progress_path.with_suffix(".tmp")
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(marker) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(self._restore_in_progress_path)

    def _available_path(self, reason: str, created_at: datetime, schema_version: int, release: str) -> Path:
        safe_release = re.sub(r"[^A-Za-z0-9._-]", "_", release)
        stem = f"{reason}-{created_at.strftime('%Y%m%dT%H%M%SZ')}-schema-{schema_version}-memforge-{safe_release}"
        candidate = self.directory / stem
        counter = 2
        while candidate.exists():
            candidate = self.directory / f"{stem}-{counter}"
            counter += 1
        return candidate


def _read_point(path: Path) -> RecoveryPoint:
    try:
        manifest = json.loads((path / _MANIFEST_FILE).read_text(encoding="utf-8"))
        return RecoveryPoint(
            id=path.name,
            path=path,
            reason=str(manifest["reason"]),
            created_at=datetime.fromisoformat(str(manifest["created_at"])),
            schema_version=int(manifest["schema_version"]),
            taken_by_release=str(manifest["taken_by_release"]),
            database_sha256=str(manifest["database"]["sha256"]),
            includes_vectors=manifest.get("vectors") is not None,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RecoveryPointError(f"{path.name} has no readable manifest ({exc})") from exc


def _snapshot_database(source: Path, target: Path) -> None:
    """Copy the database at ``source`` to ``target`` with the Online Backup API and verify the copy."""
    with closing(sqlite3.connect(source)) as live, closing(sqlite3.connect(target)) as copy:
        live.backup(copy)
        # A self-contained file: the copy keeps no write-ahead log beside it.
        copy.execute("PRAGMA journal_mode = DELETE")
        (check,) = copy.execute("PRAGMA quick_check").fetchone()
    if check != "ok":
        raise RecoveryPointError(f"The copy of {source} failed its integrity check: {check}")


def _copy_database(source: Path, target: Path) -> None:
    """Replace the database at ``target`` with ``source`` through the Online Backup API."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as copy,
        closing(sqlite3.connect(target)) as live,
    ):
        copy.backup(live)


def _replace_directory(target: Path, source: Path | None) -> None:
    """Make ``target`` a copy of ``source``, or remove it when ``source`` is ``None``."""
    staging = target.with_name(f".{target.name}.restoring")
    replaced = target.with_name(f".{target.name}.replaced")
    for leftover in (staging, replaced):
        shutil.rmtree(leftover, ignore_errors=True)
    if source is not None:
        shutil.copytree(source, staging)
    if target.exists():
        target.rename(replaced)
    if source is not None:
        staging.rename(target)
    shutil.rmtree(replaced, ignore_errors=True)


def _schema_version(database: Path) -> int:
    with closing(sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        try:
            (version,) = connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        except sqlite3.OperationalError:
            return 0
    return int(version or 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
