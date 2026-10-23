"""Device snapshot confirmations, excluded-scope listings and the sync history confirmation count."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import aiosqlite
import pytest
import pytest_asyncio

from memforge.local_agent.snapshot_manifest import (
    ExcludedScopeListing,
    SnapshotConfirmation,
    SnapshotConfirmationAnswer,
)
from memforge.memory.evidence import EVIDENCE_UNIT_SUPPORT_SCOPE
from memforge.models import SyncState
from memforge.source_activity import SourceActivityConflict
from memforge.storage.database import MIGRATIONS, SCHEMA, Database

SOURCE = "src-jira"
WORKSPACE = "ws"
DIGEST = "a" * 64
ABSENT = SnapshotConfirmationAnswer.ABSENT
EXISTS = SnapshotConfirmationAnswer.EXISTS
HIDDEN = SnapshotConfirmationAnswer.HIDDEN
FAILED = SnapshotConfirmationAnswer.FAILED
ISSUED = (
    SnapshotConfirmation("jira-PAY-1", "PAY-1"),
    SnapshotConfirmation("jira-PAY-2", "PAY-2"),
    SnapshotConfirmation("jira-PAY-3", "PAY-3"),
)
LISTINGS = (
    ExcludedScopeListing("scope-old", frozenset({"jira-PAY-1", "jira-PAY-9"})),
    ExcludedScopeListing("scope-empty"),
    ExcludedScopeListing("scope-rejected", rejection="JQL rejected: unknown field"),
)


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "confirmations.db"))
    await database.connect()
    await database.upsert_source(
        id=SOURCE,
        type="jira",
        name="Jira",
        config_json=json.dumps({"base_url": "https://jira.example"}),
        access_policy="workspace",
        owner_user_id="dev",
    )
    try:
        yield database
    finally:
        await database.close()


async def _attach(db: Database, snapshot_id: str = "snap-1", **kwargs) -> None:
    arguments = {
        "source_id": SOURCE,
        "workspace_id": WORKSPACE,
        "snapshot_id": snapshot_id,
        "memberships": [],
        "coverage": "complete_snapshot",
        "manifest_count": 0,
        "manifest_sha256": DIGEST,
        "manifest_items": [],
        "local_agent_job_id": "job-1",
        "local_agent_attempt_count": 1,
        "source_config_revision": "rev-1",
        **kwargs,
    }
    await db.attach_source_sync_inputs_to_snapshot(**arguments)


async def _answer(db: Database, snapshot_id: str = "snap-1", **kwargs) -> None:
    arguments = {
        "source_id": SOURCE,
        "workspace_id": WORKSPACE,
        "snapshot_id": snapshot_id,
        "manifest_sha256": DIGEST,
        "local_agent_job_id": "job-1",
        "local_agent_attempt_count": 1,
        **kwargs,
    }
    await db.write_source_sync_snapshot_confirmations(**arguments)


def _answers(*answers: SnapshotConfirmationAnswer) -> tuple[SnapshotConfirmation, ...]:
    return tuple(
        SnapshotConfirmation(item.doc_id, item.provider_key, answer) for item, answer in zip(ISSUED, answers)
    )


async def _status(db: Database, snapshot_id: str = "snap-1") -> tuple[bool, bool]:
    status = await db.get_source_sync_snapshot_manifest_status(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id=snapshot_id
    )
    assert status is not None
    return status["awaiting_confirmations"], status["ready"]


async def _confirmations(db: Database, snapshot_id: str = "snap-1") -> tuple[SnapshotConfirmation, ...]:
    return await db.list_source_sync_snapshot_confirmations(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id=snapshot_id
    )


async def test_issued_confirmations_hold_the_manifest_until_answered(db: Database) -> None:
    """Answering three items in one batch also shows the batch's rowcount totals its rows."""
    await _attach(db, confirm=ISSUED, excluded_listings=LISTINGS)
    issued = await _confirmations(db)
    awaiting = await _status(db)

    await _answer(db, answers=_answers(ABSENT, EXISTS, HIDDEN))
    await _answer(db, answers=_answers(ABSENT, EXISTS, HIDDEN))

    assert issued == ISSUED
    assert awaiting == (True, False)
    assert await _status(db) == (False, True)
    assert await _confirmations(db) == _answers(ABSENT, EXISTS, HIDDEN)
    assert await db.list_source_sync_snapshot_excluded_listings(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-1"
    ) == tuple(sorted(LISTINGS, key=lambda listing: listing.scope_key))


async def test_a_manifest_issued_no_confirmations_awaits_nothing(db: Database) -> None:
    await _attach(db)
    await _answer(db)

    assert await _status(db) == (False, True)
    assert await _confirmations(db) == ()
    assert await db.list_source_sync_snapshot_excluded_listings(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-1"
    ) == ()


async def test_a_failed_confirmation_is_stored_on_every_issued_item(db: Database) -> None:
    await _attach(db, confirm=ISSUED)

    await _answer(db, failure_cause="HTTP 401 from Jira")
    await _answer(db, failure_cause="HTTP 401 from Jira")

    assert await _confirmations(db) == tuple(
        SnapshotConfirmation(item.doc_id, item.provider_key, FAILED, "HTTP 401 from Jira") for item in ISSUED
    )
    assert await _status(db) == (False, True)


@pytest.mark.parametrize(
    ("submission", "refusal"),
    [
        ({"answers": _answers(ABSENT, ABSENT, EXISTS)}, "already answered differently"),
        ({"failure_cause": "timeout"}, "already answered differently"),
        ({"manifest_sha256": "b" * 64, "answers": _answers(ABSENT, EXISTS, HIDDEN)}, "another manifest digest"),
        ({"local_agent_attempt_count": 2, "answers": _answers(ABSENT, EXISTS, HIDDEN)}, "another collection attempt"),
        ({"local_agent_job_id": "job-2", "answers": _answers(ABSENT, EXISTS, HIDDEN)}, "another collection attempt"),
    ],
)
async def test_a_differing_resubmission_is_refused(db: Database, submission, refusal) -> None:
    await _attach(db, confirm=ISSUED)
    await _answer(db, answers=_answers(ABSENT, EXISTS, HIDDEN))

    with pytest.raises(SourceActivityConflict, match=refusal):
        await _answer(db, **submission)

    assert await _confirmations(db) == _answers(ABSENT, EXISTS, HIDDEN)


@pytest.mark.parametrize(
    ("answers", "refusal"),
    [
        (
            (*_answers(ABSENT, EXISTS), SnapshotConfirmation("jira-PAY-4", "PAY-4", ABSENT)),
            "was not issued: jira-PAY-4",
        ),
        (
            (*_answers(ABSENT, EXISTS), SnapshotConfirmation("jira-PAY-3", "PAY-33", ABSENT)),
            "was not issued: jira-PAY-3",
        ),
        (_answers(ABSENT, EXISTS), "must answer every issued item"),
        ((*_answers(ABSENT, EXISTS), SnapshotConfirmation("jira-PAY-3", "PAY-3", FAILED, "x")), "absent, exists or hidden"),
        ((*_answers(ABSENT, EXISTS, HIDDEN), _answers(EXISTS)[0]), "answer an item twice"),
    ],
)
async def test_an_unissued_or_incomplete_answer_is_refused(db: Database, answers, refusal) -> None:
    await _attach(db, confirm=ISSUED)

    with pytest.raises(ValueError, match=refusal):
        await _answer(db, answers=answers)

    assert await _confirmations(db) == ISSUED


async def test_answers_need_the_manifest(db: Database) -> None:
    with pytest.raises(ValueError, match="manifest not found"):
        await _answer(db, answers=())


async def test_a_manifest_replay_keeps_its_confirm_set_and_listings(db: Database) -> None:
    await _attach(db, confirm=ISSUED, excluded_listings=LISTINGS)

    await _attach(db, confirm=tuple(reversed(ISSUED)), excluded_listings=tuple(reversed(LISTINGS)))
    with pytest.raises(SourceActivityConflict, match="manifest changed"):
        await _attach(db, confirm=ISSUED[:2], excluded_listings=LISTINGS)
    with pytest.raises(SourceActivityConflict, match="manifest changed"):
        await _attach(db, confirm=ISSUED, excluded_listings=LISTINGS[:2])


async def test_confirmations_name_only_unlisted_items(db: Database) -> None:
    with pytest.raises(ValueError, match="does not list"):
        await _attach(
            db,
            manifest_count=1,
            manifest_items=[("jira-PAY-1", "rev-1", "upsert")],
            confirm=ISSUED,
        )
    with pytest.raises(ValueError, match="issued unanswered"):
        await _attach(db, confirm=_answers(ABSENT))

    assert await db.get_source_sync_snapshot_manifest_status(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-1"
    ) is None


@pytest.mark.parametrize(
    ("coverage", "change_kind", "refusal"),
    [
        ("bounded_delta", "upsert", "coverage is invalid"),
        ("partial", "upsert", "coverage is invalid"),
        ("complete_snapshot", "tombstone", "change_kind is invalid"),
    ],
)
async def test_a_manifest_is_a_complete_snapshot_of_upserts(db: Database, coverage, change_kind, refusal) -> None:
    with pytest.raises(ValueError, match=refusal):
        await _attach(
            db,
            coverage=coverage,
            manifest_count=1,
            manifest_items=[("jira-PAY-1", "rev-1", change_kind)],
        )

    assert await db.get_source_sync_snapshot_manifest_status(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-1"
    ) is None


async def test_a_stored_legacy_manifest_still_reads_and_reports_its_status(db: Database) -> None:
    # Manifests written before only complete snapshots of upserts were accepted.
    await db.db.execute(
        """INSERT INTO source_sync_snapshot_manifests (
               workspace_id, source_id, snapshot_id, coverage, item_count, manifest_sha256,
               local_agent_job_id, local_agent_attempt_count, source_config_revision, created_at
           ) VALUES (?, ?, 'snap-legacy', 'bounded_delta', 2, ?, 'job-legacy', 1, 'rev-1', '2026-06-01T00:00:00+00:00')""",
        (WORKSPACE, SOURCE, DIGEST),
    )
    await db.db.executemany(
        """INSERT INTO source_sync_snapshot_manifest_items (
               workspace_id, source_id, snapshot_id, doc_id, revision, change_kind
           ) VALUES (?, ?, 'snap-legacy', ?, ?, ?)""",
        [(WORKSPACE, SOURCE, "doc-kept", "rev-1", "upsert"), (WORKSPACE, SOURCE, "doc-gone", "rev-2", "tombstone")],
    )
    await db.db.commit()

    status = await db.get_source_sync_snapshot_manifest_status(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-legacy"
    )
    confirmations = await _confirmations(db, "snap-legacy")
    attested = await db.find_source_sync_input_attestations(
        source_id=SOURCE, workspace_id=WORKSPACE, manifest_items=[("doc-kept", "rev-1", "upsert")]
    )

    assert status is not None
    assert (status["coverage"], status["item_count"], status["materialized_count"]) == ("bounded_delta", 2, 0)
    assert (status["awaiting_confirmations"], status["ready"]) == (False, False)
    assert confirmations == ()
    assert attested == []
    # A replay of the legacy manifest is refused as a write, and leaves it stored.
    with pytest.raises(ValueError, match="coverage is invalid"):
        await _attach(
            db,
            "snap-legacy",
            coverage="bounded_delta",
            manifest_count=2,
            manifest_items=[("doc-kept", "rev-1", "upsert"), ("doc-gone", "rev-2", "tombstone")],
            local_agent_job_id="job-legacy",
        )
    assert await db.get_source_sync_snapshot_manifest_status(
        source_id=SOURCE, workspace_id=WORKSPACE, snapshot_id="snap-legacy"
    ) == status


async def test_the_confirmation_count_round_trips_through_sync_history(db: Database) -> None:
    started_at = datetime(2026, 10, 23, 8, 0, tzinfo=timezone.utc)
    for run_id, count, offset in (("run-counted", 4, 0), ("run-unconfirmed", None, 1)):
        await db.record_source_sync_result(
            SyncState(
                source=SOURCE,
                last_sync_at=started_at,
                last_sync_status="success",
                unlisted_present_count=count,
            ),
            started_at=started_at + timedelta(minutes=offset),
            finished_at=started_at + timedelta(minutes=offset, seconds=5),
            run_id=run_id,
        )

    history = await db.get_sync_history(source=SOURCE)

    assert [(row["run_id"], row["unlisted_present_count"]) for row in history] == [
        ("run-unconfirmed", None),
        ("run-counted", 4),
    ]


async def _database_at_schema_126(path: str) -> None:
    """A database at schema 126, before device confirmations and the confirmation count."""

    connection = await aiosqlite.connect(path)
    await connection.executescript(SCHEMA)
    await connection.executescript(
        """
        DROP TABLE source_sync_snapshot_confirmations;
        DROP TABLE source_sync_snapshot_excluded_listings;
        ALTER TABLE sync_history DROP COLUMN unlisted_present_count;
        """
    )
    for version, description, _statements in MIGRATIONS:
        if version > 126:
            break
        await connection.execute(
            "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
            (version, description, "2026-10-01T00:00:00+00:00"),
        )
    await connection.execute(
        "INSERT INTO system_contract_markers (marker_key, marker_value, updated_at) VALUES (?, ?, ?)",
        ("support_scope_version", EVIDENCE_UNIT_SUPPORT_SCOPE, "2026-10-01T00:00:00+00:00"),
    )
    await connection.execute(
        """INSERT INTO sync_history (source, status, started_at, finished_at, run_id)
           VALUES ('src-old', 'success', '2026-10-01T00:00:00+00:00', '2026-10-01T00:01:00+00:00', 'run-old')"""
    )
    await connection.commit()
    await connection.close()


async def test_migration_adds_the_confirmation_tables_and_count(tmp_path) -> None:
    path = str(tmp_path / "schema-126.db")
    await _database_at_schema_126(path)

    database = Database(path)
    await database.connect()
    try:
        tables = {
            str(row[0])
            for row in await database.db.execute_fetchall("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        history = await database.get_sync_history(source="src-old")
    finally:
        await database.close()

    assert {"source_sync_snapshot_confirmations", "source_sync_snapshot_excluded_listings"} <= tables
    assert [(row["run_id"], row["unlisted_present_count"]) for row in history] == [("run-old", None)]


async def test_failure_and_answers_together_are_refused(db: Database) -> None:
    await _attach(db, confirm=ISSUED)

    with pytest.raises(ValueError, match="carries its cause and no answers"):
        await _answer(db, answers=_answers(ABSENT, EXISTS, HIDDEN), failure_cause="timeout")


async def test_a_manifest_replay_after_answers_is_accepted(db: Database) -> None:
    await _attach(db, confirm=ISSUED, excluded_listings=LISTINGS)
    await _answer(db, answers=_answers(ABSENT, EXISTS, HIDDEN))

    await _attach(db, confirm=ISSUED, excluded_listings=LISTINGS)

    assert await _confirmations(db) == _answers(ABSENT, EXISTS, HIDDEN)


async def test_answers_to_one_snapshot_leave_another_untouched(db: Database) -> None:
    await _attach(db, "snap-a", confirm=ISSUED)
    await _attach(db, "snap-b", confirm=ISSUED)

    await _answer(db, "snap-b", answers=_answers(ABSENT, EXISTS, HIDDEN))

    assert await _confirmations(db, "snap-a") == ISSUED
    assert await _status(db, "snap-a") == (True, False)


async def test_answers_stored_meanwhile_by_another_writer_refuse_the_submission(db: Database, monkeypatch) -> None:
    await _attach(db, confirm=ISSUED)
    await _answer(db, answers=_answers(ABSENT, ABSENT, ABSENT))

    async def read_before_the_other_writer_committed(*_args):
        return ISSUED

    with monkeypatch.context() as patch:
        patch.setattr(db, "_list_source_sync_snapshot_confirmations_unlocked", read_before_the_other_writer_committed)
        with pytest.raises(SourceActivityConflict, match="answered by another submission"):
            await _answer(db, answers=_answers(EXISTS, EXISTS, EXISTS))

    assert await _confirmations(db) == _answers(ABSENT, ABSENT, ABSENT)
