"""A Source Unit derivation or lifecycle failure is retryable only when its cause proves it transient."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import httpx
import pytest

from memforge.llm.embedding import EmbeddingDeadlineExceeded, EmbeddingProviderError
from memforge.llm.structured import StructuredLlmError, failure_retryable
from memforge.memory.candidate_admission import CandidateAdmissionError
from memforge.memory.engine import SourceUnitLifecycleExecutionError
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.models import RawMemory
from memforge.pipeline.projection_images import ProjectionImageLoadError
from memforge.pipeline.sync import MAX_RETRIES
from memforge.source_activity import SourceActivityConflict
from memforge.source_item_failures import FailedItemRetry, RunItemFailure, SourceItemFailure
from memforge.storage.database import Database
from tests.coordination_fixture import AUDIT, DOC_ID, ONE, RETENTION, TWO, ScriptedClient, coordination_engine, seeded_page
from tests.test_projected_lifecycle_integration import db as db, _selected
from tests.test_stored_input_currency import JiraProvider, SOURCE_ID, doc_id, synced


def _lock_error(path) -> sqlite3.OperationalError:
    """The error SQLite raises for a write while another connection holds the write lock."""
    holder = sqlite3.connect(path)
    holder.execute("CREATE TABLE probe (id INTEGER)")
    holder.execute("BEGIN IMMEDIATE")
    waiter = sqlite3.connect(path, timeout=0)
    try:
        waiter.execute("INSERT INTO probe (id) VALUES (1)")
    except sqlite3.OperationalError as error:
        return error
    finally:
        waiter.close()
        holder.rollback()
        holder.close()
    raise AssertionError("a write under another connection's write lock succeeded")


def _driver_error(path, statement: str) -> sqlite3.Error:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS probe (id INTEGER PRIMARY KEY)")
        connection.execute(statement)
    except sqlite3.Error as error:
        return error
    finally:
        connection.close()
    raise AssertionError(f"{statement} succeeded")


def _http_answer(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://jira.example.test/rest/api/2/issue/PAY-1")
    return httpx.HTTPStatusError(
        f"status {status_code}", request=request, response=httpx.Response(status_code, request=request),
    )


class _ProviderAnswer(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider answered {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (KeyError("obs-carried-member"), False),
        (ValueError("invalid lifecycle plan"), False),
        (RuntimeError("lifecycle apply failed"), False),
        (StructuredLlmError("unavailable", terminal_category="provider_error"), True),
        (StructuredLlmError("rate limited", terminal_category="provider_error", status_code=429), True),
        (StructuredLlmError("overloaded", terminal_category="provider_error", status_code=503), True),
        (StructuredLlmError("unauthorized", terminal_category="provider_error", status_code=401), False),
        (StructuredLlmError("forbidden", terminal_category="provider_error", status_code=403), False),
        (StructuredLlmError("no such model", terminal_category="provider_error", status_code=404), False),
        (StructuredLlmError("too long", terminal_category="provider_error", status_code=400), False),
        (CandidateAdmissionError(
            "admission_failed", "failed",
            cause=StructuredLlmError("unauthorized", terminal_category="provider_error", status_code=401),
        ), False),
        (MemoryPairClassificationError("failed", terminal_category="provider_error", status_code=401), False),
        (MemoryPairClassificationError("failed", terminal_category="provider_error", status_code=503), True),
        (StructuredLlmError("timed out", terminal_category="deadline_exceeded"), True),
        (StructuredLlmError("rejected", terminal_category="request_error"), False),
        (StructuredLlmError("unreadable", terminal_category="invalid_response"), False),
        (_ProviderAnswer(503), True),
        (_ProviderAnswer(429), True),
        (_ProviderAnswer(400), False),
        (TimeoutError(), True),
        (ConnectionResetError(), True),
        (httpx.ConnectError("connection refused"), True),
        (httpx.ReadTimeout("read timed out"), True),
        (httpx.RemoteProtocolError("server disconnected"), True),
        (_http_answer(503), True),
        (_http_answer(429), True),
        (_http_answer(404), False),
        (_http_answer(401), False),
        (EmbeddingDeadlineExceeded("deadline"), True),
        (EmbeddingProviderError("unreachable", unreachable=True), True),
        (EmbeddingProviderError("unavailable", status_code=502), True),
        (EmbeddingProviderError("rejected", status_code=400), False),
        (SourceActivityConflict("source activity fence is not current: sync-run"), True),
        (ProjectionImageLoadError(error_code="artifact_unavailable"), True),
        (ProjectionImageLoadError(error_code="artifact_integrity_failed"), False),
        (SourceUnitLifecycleExecutionError("wrapped", None, retryable=False), False),
        (SourceUnitLifecycleExecutionError("wrapped", None, retryable=True), True),
    ],
)
def test_only_a_cause_that_proves_itself_transient_is_retryable(error: BaseException, retryable: bool) -> None:
    assert failure_retryable(error) is retryable


async def test_a_storage_error_is_retryable_only_when_its_store_classifies_it_transient(tmp_path) -> None:
    database = Database(str(tmp_path / "classify.db"))
    locked = _lock_error(tmp_path / "locked.db")

    assert failure_retryable(locked) is False
    assert failure_retryable(locked, store=database) is True
    assert failure_retryable(KeyError("obs"), store=database) is False
    # A store that names no transient errors classifies none.
    assert failure_retryable(locked, store=object()) is False


def test_sqlite_classifies_only_a_busy_or_locked_database_as_transient(tmp_path) -> None:
    database = Database(str(tmp_path / "classify.db"))
    locked = _lock_error(tmp_path / "locked.db")
    missing_table = _driver_error(tmp_path / "missing.db", "SELECT * FROM missing")
    duplicate_key = _driver_error(tmp_path / "duplicate.db", "INSERT INTO probe (id) VALUES (1), (1)")

    assert locked.sqlite_errorcode & 0xFF == sqlite3.SQLITE_BUSY
    assert database.is_transient_error(locked) is True
    # An error constructed without a result code is classified by its message.
    assert database.is_transient_error(sqlite3.OperationalError("database table is locked")) is True
    assert isinstance(missing_table, sqlite3.OperationalError)
    assert database.is_transient_error(missing_table) is False
    assert isinstance(duplicate_key, sqlite3.IntegrityError)
    assert database.is_transient_error(duplicate_key) is False
    assert database.is_transient_error(TimeoutError()) is False


def _failure(stage: str, error_class: str) -> RunItemFailure:
    return RunItemFailure(
        document_id="jira-PAY-1", stage=stage, error_class=error_class, message="failed", retryable=True,
    )


def test_the_ledger_counts_a_failure_only_while_it_repeats_in_one_release() -> None:
    at = datetime(2026, 10, 17, tzinfo=timezone.utc)

    def record(previous, stage="plan_construction", error_class="builtins.KeyError", release="1.0.0"):
        return SourceItemFailure.recorded(
            previous, _failure(stage, error_class), source_id=SOURCE_ID, release=release, failed_at=at,
        )

    first = record(None)
    second = record(first)
    third = record(second)

    assert [(failure.failures_in_release, failure.budget_exhausted) for failure in (first, second, third)] == [
        (1, False), (2, False), (3, True),
    ]
    assert record(second, stage="lifecycle_commit").failures_in_release == 1
    assert record(second, error_class="builtins.ValueError").failures_in_release == 1
    assert record(third, release="1.0.1").failures_in_release == 1


async def test_a_database_before_the_ledger_gains_it_and_the_release_of_each_derivation(tmp_path) -> None:
    path = tmp_path / "upgrade.db"
    legacy = Database(str(path))
    await legacy.connect()
    try:
        await legacy.db.execute("DROP TABLE source_item_failures")
        await legacy.db.execute("ALTER TABLE source_derivation_attempts DROP COLUMN release")
        await legacy.db.execute("DELETE FROM schema_migrations WHERE version = 116")
        await legacy.db.commit()
    finally:
        await legacy.close()

    upgraded = Database(str(path))
    await upgraded.connect()
    try:
        columns = {
            str(row[1]) for row in await upgraded.db.execute_fetchall("PRAGMA table_info(source_derivation_attempts)")
        }
        assert "release" in columns
        assert await upgraded.list_source_item_failures() == ()
    finally:
        await upgraded.close()


@pytest.mark.parametrize(
    ("cause", "retryable"),
    [(lambda path: KeyError("obs-carried-member"), False), (_lock_error, True)],
    ids=["undeclared", "locked_storage"],
)
async def test_the_lifecycle_engine_carries_the_classification_and_stage_of_a_plan_construction_failure(
    db: Database, monkeypatch, tmp_path, cause, retryable: bool,
) -> None:
    page, _memory = await seeded_page(db, TWO, RETENTION)
    error = cause(tmp_path / "locked.db")

    def failing_validation(*args, **kwargs):
        raise error

    monkeypatch.setattr("memforge.memory.engine.validate_destructive_operations", failing_validation)
    projection = page.next(TWO, RETENTION, AUDIT)
    with pytest.raises(SourceUnitLifecycleExecutionError) as raised:
        await coordination_engine(db, ScriptedClient()).prepare_and_commit_projected_lifecycle(
            projection=projection, doc_id=DOC_ID,
            raw_memories=_selected(projection, [RawMemory(content=AUDIT, memory_type="fact")]),
            doc_type="design-doc", project_key="ENG", repo_identifier=None,
            document_content=projection.observation_revisions[-1].content, update_mode="full_document",
            changed_hunks=None, update_plan_stats=None, source_updated_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-plan-construction:lease-1",
        )

    assert raised.value.__cause__ is error
    assert (raised.value.retryable, raised.value.stage) == (retryable, "plan_construction")


@pytest.mark.parametrize(
    ("cause", "retryable"),
    [(lambda path: KeyError("incumbent"), False), (_lock_error, True)],
    ids=["undeclared", "locked_storage"],
)
async def test_a_reconciliation_failure_is_classified_with_the_store_of_its_work(
    db: Database, tmp_path, cause, retryable: bool,
) -> None:
    page, _memory = await seeded_page(db, TWO, RETENTION)
    error = cause(tmp_path / "locked.db")

    class FailingRelation(ScriptedClient):
        async def assess_claim_revisions(self, prompt, **kwargs):
            raise error

    projection = page.next(TWO, RETENTION, AUDIT, ONE)
    with pytest.raises(SourceUnitLifecycleExecutionError) as raised:
        await coordination_engine(db, FailingRelation()).prepare_and_commit_projected_lifecycle(
            projection=projection, doc_id=DOC_ID,
            raw_memories=_selected(projection, [RawMemory(content=claim, memory_type="fact") for claim in (AUDIT, ONE)]),
            doc_type="design-doc", project_key="ENG", repo_identifier=None,
            document_content=projection.observation_revisions[-1].content, update_mode="full_document",
            changed_hunks=None, update_plan_stats=None, source_updated_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            lifecycle_execution_owner_id="sync-reconciliation:lease-1",
        )

    assert raised.value.retryable is retryable


class _UnreachableProvider(JiraProvider):
    """A provider whose issue PAY-1 cannot be fetched while ``failure`` is set."""

    def __init__(self, failure: BaseException) -> None:
        super().__init__()
        self.failure: BaseException | None = failure
        self.unreachable: set[str] = set()
        self.fetches: list[str] = []

    async def discover(self, since=None):
        async for item in super().discover(since):
            if since is None:
                yield item

    async def fetch(self, item):
        issue_key = item.extra["issue_key"]
        self.fetches.append(issue_key)
        if issue_key in self.unreachable and self.failure is not None:
            raise self.failure
        return await super().fetch(item)


@pytest.mark.parametrize(
    "failure",
    [httpx.ConnectError("connection refused"), _http_answer(503), _http_answer(429)],
    ids=["connect_error", "status_503", "status_429"],
)
async def test_a_provider_network_failure_while_fetching_a_document_stays_retryable(
    db: Database, failure: BaseException,
) -> None:
    from tests.test_sync_bookkeeping import NoopMemoryEngine

    NoopMemoryEngine.db = db
    try:
        harness = await synced(db)
        harness.provider = provider = _UnreachableProvider(failure)
        provider.histories["PAY-1"].append({**provider.histories["PAY-1"][0], "id": "9003"})
        provider.unreachable.add("PAY-1")
        await db.reset_source_sync_cursor(SOURCE_ID)

        state = await harness.sync(retry=FailedItemRetry.RETRYABLE)
        [row] = await db.list_source_item_failures(source_id=SOURCE_ID)

        # The provider recovers; the next scheduled run, whose discovery does
        # not return PAY-1, fetches it again from the ledger.
        provider.failure = None
        fetches = provider.fetches.count("PAY-1")
        recovered = await harness.sync(retry=FailedItemRetry.RETRYABLE)
    finally:
        NoopMemoryEngine.db = None

    assert [failed.doc_id for failed in state.failed_docs] == [doc_id("PAY-1")]
    assert fetches == MAX_RETRIES
    assert provider.fetches.count("PAY-1") > fetches
    assert state.failure_retryable is True
    assert (row.document_id, row.stage, row.retryable) == (doc_id("PAY-1"), "fetch", True)
    assert (recovered.last_sync_status, recovered.docs_failed) == ("success", 0)
    assert await db.list_source_item_failures(source_id=SOURCE_ID) == ()
