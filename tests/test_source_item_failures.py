"""The failure ledger owns the retry of each failed Document, once per release."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from memforge.config import AppConfig, SyncConfig
from memforge.pipeline.collection_family import SnapshotBinding
from memforge.models import ScopeListingKind, SyncState
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.release import current_release
from memforge.runtime import SourceSyncWorker, SyncService, failed_item_health, failed_item_retry_for_run
from memforge.source_activity import SourceActivityConflict, SourceActivityKind, SourceActivityLease
from memforge.source_derivation import DERIVATION_RELEASE_SUPERSEDED
from memforge.source_item_failures import FailedItemRetry, RunItemFailure, SourceItemFailure, with_inherited_rename
from memforge.storage.database import Database
from tests.test_stored_input_currency import (
    ISSUES,
    SOURCE_ID,
    SPRINT_HISTORY,
    SYNCED_AT,
    Harness,
    JiraProvider,
    _FailingLifecycleEngine,
    _ledger,
    authored,
    db as db,
    doc_id,
    synced,
)
from tests.test_sync_bookkeeping import ProjectionFragmentRecordingExtractor

SCHEDULED = FailedItemRetry.RETRYABLE


class BreakableProvider(JiraProvider):
    """A Jira provider whose issues can fail in a chosen step, and whose incremental discovery returns changed issues.

    ``changed`` holds the issues an incremental discovery returns; a full
    discovery returns every issue. ``broken`` maps an issue to the error its
    fetch or normalization raises. ``modified`` holds each changed issue's
    change time, which its discovered item carries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.changed: set[str] = set()
        self.broken: dict[str, tuple[str, type[Exception]]] = {}
        self.fetches: list[str] = []
        self.modified: dict[str, datetime] = {}

    def _item(self, issue_key: str):
        item = super()._item(issue_key)
        item.last_modified = self.modified.get(issue_key, item.last_modified)
        return item

    async def discover(self, since=None):
        async for item in super().discover(since):
            if since is None or item.extra["issue_key"] in self.changed:
                yield item

    def _break(self, step: str, issue_key: str) -> None:
        if self.broken.get(issue_key, (None,))[0] == step:
            raise self.broken[issue_key][1](f"{step} of {issue_key} failed")

    async def fetch(self, item):
        issue_key = item.extra["issue_key"]
        self.fetches.append(issue_key)
        self._break("fetch", issue_key)
        return await super().fetch(item)

    async def normalize(self, raw):
        self._break("normalize", raw.item.extra["issue_key"])
        return await super().normalize(raw)


class RaisingExtractor(ProjectionFragmentRecordingExtractor):
    async def extract_projection_fragment_memories(self, catalog, **kwargs):
        raise ValueError("extraction raised")


def _change(harness, issue_key: str) -> str:
    """A new changelog entry on the issue, which an incremental discovery returns; returns the entry's id."""
    histories = harness.provider.histories[issue_key]
    entry_id = str(9100 + len(histories))
    histories.append(authored({**SPRINT_HISTORY, "id": entry_id}, "Ann"))
    harness.provider.changed.add(issue_key)
    harness.provider.modified[issue_key] = SYNCED_AT + timedelta(minutes=len(histories))
    return entry_id


async def breakable(db: Database):
    harness = await synced(db)
    harness.provider = BreakableProvider()
    return harness


async def _cursor(db: Database) -> datetime:
    state = await db.get_sync_state(SOURCE_ID)
    assert state is not None and state.last_sync_at is not None
    return state.last_sync_at


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["fetch", "normalize", "projection", "derivation", "lifecycle"])
async def test_a_failure_in_any_stage_is_recorded_and_the_cursor_advances(db, monkeypatch, stage):
    harness = await breakable(db)
    before = await _cursor(db)
    _change(harness, "PAY-1")
    _change(harness, "PAY-2")
    extractor = None
    if stage in {"fetch", "normalize"}:
        harness.provider.broken["PAY-1"] = (stage, ValueError)
    elif stage == "projection":
        project = GeneSyncOrchestrator._project_fetched_item

        async def failing_projection(self, **kwargs):
            if kwargs["item"].item_id == doc_id("PAY-1"):
                raise ValueError("projection failed")
            return await project(self, **kwargs)

        monkeypatch.setattr(GeneSyncOrchestrator, "_project_fetched_item", failing_projection)
    elif stage == "derivation":
        extractor = RaisingExtractor()
    else:
        harness.engine = _FailingLifecycleEngine(db, "PAY-1", lambda: KeyError("obs"), wrapped=False)

    state = await harness.sync(extractor, retry=SCHEDULED)

    failed = {failure.doc_id for failure in state.failed_docs}
    assert doc_id("PAY-1") in failed
    assert state.last_sync_status == "partial"
    recorded = (await _ledger(db))["PAY-1"]
    assert (recorded.stage, recorded.release, recorded.failures_in_release) == (stage, current_release(), 1)
    assert recorded.retryable is False
    assert recorded.item is not None and recorded.item.item_id == doc_id("PAY-1")
    # The ledger retries the Document, so it does not hold back the Source's cursor.
    assert state.last_sync_at > before


@pytest.mark.asyncio
async def test_a_failure_of_the_run_itself_keeps_the_cursor(db, monkeypatch):
    harness = await breakable(db)
    before = await _cursor(db)
    _change(harness, "PAY-1")

    async def failing_count(*args, **kwargs):
        raise RuntimeError("document count unavailable")

    monkeypatch.setattr(db, "count_documents", failing_count)
    state = await harness.sync(retry=SCHEDULED)

    assert state.last_sync_status == "failed"
    assert state.last_sync_at == before


@pytest.mark.asyncio
async def test_a_scheduled_run_retries_a_retryable_failure_up_to_the_budget_and_never_a_deterministic_one(db):
    harness = await breakable(db)
    provider = harness.provider
    _change(harness, "PAY-1")
    provider.broken["PAY-1"] = ("fetch", TimeoutError)

    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    provider.changed.clear()
    for failures in (2, 3):
        fetches = provider.fetches.count("PAY-1")
        # Discovery does not return the issue; the ledger retries it.
        state = await harness.sync(retry=SCHEDULED)
        assert provider.fetches.count("PAY-1") > fetches
        assert [failure.doc_id for failure in state.failed_docs] == [doc_id("PAY-1")]
        recorded = (await _ledger(db))["PAY-1"]
        assert (recorded.retryable, recorded.failures_in_release) == (True, failures)

    fetches = provider.fetches.count("PAY-1")
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "success"
    assert provider.fetches.count("PAY-1") == fetches

    # A deterministic failure is left alone while discovery returns its Document unchanged.
    _change(harness, "PAY-2")
    provider.broken["PAY-2"] = ("fetch", ValueError)
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    assert (await _ledger(db))["PAY-2"].retryable is False
    fetches = provider.fetches.count("PAY-2")
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "success"
    assert provider.fetches.count("PAY-2") == fetches

    # Once the provider changes it, a scheduled run processes it again.
    _change(harness, "PAY-2")
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    assert provider.fetches.count("PAY-2") > fetches
    assert (await _ledger(db))["PAY-2"].item.last_modified == provider.modified["PAY-2"]

    # A requested sync retries every failed Document.
    provider.broken.clear()
    provider.changed.clear()
    state = await harness.sync()
    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert {"PAY-1", "PAY-2"} <= {issue for issue in provider.fetches[-2:]}
    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_new_release_retries_each_failure_of_the_earlier_release_once(db, monkeypatch):
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-1")
    harness = await breakable(db)
    _change(harness, "PAY-1")
    harness.engine = engine = _FailingLifecycleEngine(db, "PAY-1", lambda: KeyError("obs"))
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    # Recovery ends the staged derivation; afterwards scheduled runs of this release leave it alone.
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    failed_calls = engine.failed_calls
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "success"
    assert engine.failed_calls == failed_calls
    service = SyncService(db, AppConfig(sync=SyncConfig(max_extraction_workers=1)))
    assert await service.enqueue_release_recovery() == 0

    # The fix ships in the next release; startup queues one reprocess of the failed Document.
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-2")
    engine.failure = None
    assert await service.enqueue_release_recovery() == 1
    run = await db.get_latest_source_sync_run(source_id=SOURCE_ID)
    assert (run.trigger, run.reprocess_document_ids) == ("reprocess", (doc_id("PAY-1"),))
    health = await failed_item_health(db)
    assert (health.status, health.payload["failed_items"], health.payload["retrying_after_upgrade"]) == (
        "warning", 1, 1,
    )

    state = await harness.reprocess(*[document_id.removeprefix("jira-") for document_id in run.reprocess_document_ids])

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert engine.projected_lifecycle_calls[-1]["doc_id"] == doc_id("PAY-1")
    assert await _ledger(db) == {}
    assert (await failed_item_health(db)).status == "ok"


async def _first_sync_failing_to_fetch(db: Database, monkeypatch) -> Harness:
    """A new Source whose first sync could not fetch PAY-1, so PAY-1 never committed a Unit."""
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-1")
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace", owner_user_id="dev",
    )
    harness = Harness(db)
    harness.provider = BreakableProvider()
    harness.provider.broken["PAY-1"] = ("fetch", ValueError)
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    assert await db.find_source_unit_by_document_id(SOURCE_ID, doc_id("PAY-1"), current_only=True) is None
    harness.provider.broken.clear()
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-2")
    return harness


@pytest.mark.asyncio
async def test_a_document_that_never_committed_is_reprocessed_from_the_item_its_failure_recorded(db, monkeypatch):
    harness = await _first_sync_failing_to_fetch(db, monkeypatch)
    service = SyncService(db, AppConfig(sync=SyncConfig(max_extraction_workers=1)))

    assert await service.enqueue_release_recovery() == 1
    run = await db.get_latest_source_sync_run(source_id=SOURCE_ID)
    assert (run.trigger, run.reprocess_document_ids) == ("reprocess", (doc_id("PAY-1"),))
    state = await harness.reprocess("PAY-1")

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert "PAY-1" in harness.provider.rediscovered
    assert await db.find_source_unit_by_document_id(SOURCE_ID, doc_id("PAY-1"), current_only=True) is not None
    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_source_that_never_completed_a_sync_syncs_to_retry_its_failed_documents(db, monkeypatch):
    harness = await _first_sync_failing_to_fetch(db, monkeypatch)
    # A Source whose earlier runs left no cursor.
    await db.reset_source_sync_cursor(SOURCE_ID)
    service = SyncService(db, AppConfig(sync=SyncConfig(max_extraction_workers=1)))
    with pytest.raises(Exception, match="has not completed a sync"):
        await service.enqueue_reprocess(SOURCE_ID, (doc_id("PAY-1"),))

    assert await service.enqueue_release_recovery() == 1
    run = await db.get_latest_source_sync_run(source_id=SOURCE_ID)
    assert (run.trigger, run.reprocess_document_ids) == ("release_recovery", ())
    assert failed_item_retry_for_run(run) is SCHEDULED
    state = await harness.sync(retry=SCHEDULED)

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert await db.find_source_unit_by_document_id(SOURCE_ID, doc_id("PAY-1"), current_only=True) is not None
    assert await _ledger(db) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("new_release", [False, True], ids=["same_release", "new_release"])
async def test_a_derivation_of_an_earlier_release_is_neither_resumed_nor_replayed(db, monkeypatch, new_release):
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-1")
    harness = await breakable(db)
    _change(harness, "PAY-1")
    harness.engine = _FailingLifecycleEngine(db, "PAY-1", lambda: sqlite3.OperationalError("database is locked"))
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    [staged] = await db.list_source_derivation_attempts(source_id=SOURCE_ID, statuses=("completed",))
    assert staged.release == current_release()
    harness.provider.changed.clear()
    harness.engine.failure = None
    harness.extractor.fragment_calls.clear()
    if new_release:
        monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-2")

    state = await harness.sync(retry=SCHEDULED)

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    attempts = {attempt.id: attempt for attempt in await db.list_source_derivation_attempts(source_id=SOURCE_ID)}
    applied = [
        attempt for attempt in attempts.values()
        if attempt.status == "applied" and attempt.context.document.doc_id == doc_id("PAY-1")
        and attempt.created_at >= staged.created_at
    ]
    if new_release:
        assert (attempts[staged.id].status, attempts[staged.id].terminal_reason_code) == (
            "superseded", DERIVATION_RELEASE_SUPERSEDED,
        )
        # The Document is derived again: its model work runs anew under the new release.
        assert harness.extractor.fragment_calls
        assert [attempt.release for attempt in applied if attempt.id != staged.id] == [current_release()]
    else:
        assert attempts[staged.id].status == "applied"
        assert harness.extractor.fragment_calls == []
    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_staged_derivation_of_an_earlier_release_enters_the_ledger_when_a_reprocess_supersedes_it(
    db, monkeypatch,
):
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-1")
    harness = await breakable(db)
    _change(harness, "PAY-1")
    harness.engine = _FailingLifecycleEngine(db, "PAY-1", lambda: sqlite3.OperationalError("database is locked"))
    await harness.sync(retry=SCHEDULED)
    await db.delete_source_item_failures(SOURCE_ID, [doc_id("PAY-1")])
    harness.engine.failure = None
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-2")

    # A reprocess of another Document supersedes the staged derivation without deriving PAY-1.
    assert (await harness.reprocess("PAY-2")).last_sync_status == "success"

    recorded = (await _ledger(db))["PAY-1"]
    assert (recorded.error_code, recorded.retryable) == (DERIVATION_RELEASE_SUPERSEDED, True)
    assert recorded.recorded_by_another_release(current_release())
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "success"
    assert await _ledger(db) == {}


class _CapturingRuntimeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def build_sync_runtime(self, db, config, **kwargs):
        return object()

    async def run_source_sync(self, **kwargs):
        self.calls.append(kwargs)
        return SyncState(source=SOURCE_ID, last_sync_at=datetime.now(timezone.utc), last_sync_status="success")


async def _jira_source(db: Database) -> None:
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace", owner_user_id="dev",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "retry_failed_items", "retry"),
    [
        ("manual", False, FailedItemRetry.ALL),
        ("force", False, FailedItemRetry.ALL),
        ("local_agent", True, FailedItemRetry.ALL),
        ("local_agent", False, FailedItemRetry.RETRYABLE),
        ("schedule", False, FailedItemRetry.RETRYABLE),
    ],
)
async def test_the_worker_retries_every_failed_document_only_for_a_run_a_person_requested(
    db, trigger, retry_failed_items, retry,
):
    await _jira_source(db)
    config = AppConfig(sync=SyncConfig(max_extraction_workers=1))
    provider = _CapturingRuntimeProvider()
    await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger=trigger, retry_failed_items=retry_failed_items)

    await SourceSyncWorker(db, config, runtime_provider=provider, worker_id="worker").run_once()

    [call] = provider.calls
    assert call["failed_item_retry"] is retry


@pytest.mark.asyncio
async def test_a_requested_sync_keeps_its_intent_when_it_coalesces_into_a_pending_run(db):
    await _jira_source(db)
    scheduled = await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="schedule")
    assert scheduled.retry_failed_items is False

    coalesced = await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")

    assert (coalesced.run_id, coalesced.trigger, coalesced.retry_failed_items) == (
        scheduled.run_id, "schedule", True,
    )
    # A later scheduled request does not take the intent back.
    assert (await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="schedule")).retry_failed_items is True


@pytest.mark.asyncio
@pytest.mark.parametrize("running_trigger", ["schedule", "reprocess"])
async def test_a_requested_sync_during_a_running_run_gets_a_successor_that_retries_every_failed_document(
    db, running_trigger,
):
    await _jira_source(db)
    await db.enqueue_source_sync_run(
        source_id=SOURCE_ID,
        trigger=running_trigger,
        reprocess_document_ids=(doc_id("PAY-1"),) if running_trigger == "reprocess" else (),
    )
    leased = await db.lease_next_source_sync_run(worker_id="worker", workspace_id="default", lease_seconds=60)
    assert leased is not None and leased.retry_failed_items is False

    coalesced = await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")
    assert (coalesced.run_id, coalesced.rerun_requested, coalesced.rerun_retry_failed_items) == (
        leased.run_id, True, True,
    )
    assert await db.complete_source_sync_run(
        leased.run_id,
        worker_id="worker",
        lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(source=SOURCE_ID, last_sync_at=datetime.now(timezone.utc), last_sync_status="success"),
    )

    successor = await db.get_latest_source_sync_run(source_id=SOURCE_ID)
    assert (successor.trigger, successor.status, successor.retry_failed_items) == ("rerun", "pending", True)
    assert failed_item_retry_for_run(successor) is FailedItemRetry.ALL


@pytest.mark.asyncio
async def test_a_requested_sync_during_a_running_requested_sync_needs_no_successor(db):
    await _jira_source(db)
    await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")
    leased = await db.lease_next_source_sync_run(worker_id="worker", workspace_id="default", lease_seconds=60)

    coalesced = await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")

    assert (coalesced.run_id, coalesced.rerun_requested) == (leased.run_id, False)


@pytest.mark.asyncio
async def test_a_local_agent_job_a_person_requested_keeps_its_intent_when_a_schedule_coalesces_into_it(db):
    await db.upsert_source(
        id="src-teams", type="teams", name="Teams", config_json='{"conversation_ids":["conversation-a"]}',
        access_policy="workspace", owner_user_id="dev",
    )
    job = {
        "source_id": "src-teams", "source_type": "teams", "operation": "teams_sync",
        "created_by_user_id": "dev", "execution_owner_user_id": "dev",
    }
    requested, created = await db.enqueue_local_agent_job(
        job_id="laj-requested", payload={"source_config_revision": "rev-a", "retry_failed_items": True}, **job,
    )
    assert created
    coalesced, created = await db.enqueue_local_agent_job(
        job_id="laj-scheduled", payload={"source_config_revision": "rev-a"}, trigger="scheduled", **job,
    )

    assert (coalesced, created) == (requested, False)
    assert (await db.get_local_agent_job(requested))["payload"]["retry_failed_items"] is True


@pytest.mark.asyncio
async def test_failed_item_health_counts_the_documents_a_new_release_retries(db) -> None:
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    for document_id, release in (("doc-1", current_release()), ("doc-2", "0.0.1"), ("doc-3", "0.0.1")):
        await db.upsert_source_item_failure(
            SourceItemFailure(
                source_id=SOURCE_ID, document_id=document_id, stage="fetch", error_class="builtins.ValueError",
                message="failed", retryable=False, release=release, failures_in_release=1,
                first_failed_at=now, last_failed_at=now,
            )
        )

    health = await failed_item_health(db)

    assert health.status == "warning"
    assert {key: health.payload[key] for key in ("failed_items", "retrying_after_upgrade", "sources")} == {
        "failed_items": 3, "retrying_after_upgrade": 2, "sources": 1,
    }



@pytest.mark.asyncio
@pytest.mark.parametrize("retried_by", ["requested_sync", "new_release"])
async def test_a_source_that_cannot_ask_for_one_document_retries_the_revision_that_failed(
    db, monkeypatch, retried_by,
):
    monkeypatch.setattr("memforge.pipeline.sync.source_rediscovers_documents", lambda *args: False)
    monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-1")
    harness = await breakable(db)
    entry_id = _change(harness, "PAY-1")
    harness.provider.broken["PAY-1"] = ("normalize", ValueError)
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    assert entry_id not in await harness.current_changelog("PAY-1")
    # The cursor moved past the change, so discovery does not return PAY-1 again.
    harness.provider.changed.clear()
    harness.provider.broken.clear()

    if retried_by == "requested_sync":
        state = await harness.sync()
    else:
        monkeypatch.setenv("MEMFORGE_DEPLOYMENT_REVISION", "release-2")
        service = SyncService(db, AppConfig(sync=SyncConfig(max_extraction_workers=1)))
        assert await service.enqueue_release_recovery() == 1
        state = await harness.reprocess("PAY-1")

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert set(await harness.current_changelog("PAY-1")) == {"9001", entry_id}
    assert await _ledger(db) == {}


class ExistenceListedProvider(BreakableProvider):
    """A provider whose device manifest is an existence listing: an item it leaves out is gone."""

    listing_kinds = frozenset({ScopeListingKind.EXISTENCE})


async def _gone_before_it_committed(db, provider: BreakableProvider):
    """A Source whose PAY-1 failed in its first fetch, then left the provider."""
    await _jira_source(db)
    harness = Harness(db)
    harness.provider = provider
    harness.provider.broken["PAY-1"] = ("fetch", ValueError)
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "partial"
    assert set(await _ledger(db)) == {"PAY-1"}
    harness.provider.broken.clear()
    harness.provider.removed.add("PAY-1")
    harness.provider.fetches.clear()
    return harness


async def _device_sync(harness):
    listed = frozenset(doc_id(key) for key in ISSUES if key not in harness.provider.removed)
    return await harness.orchestrator().sync_gene(
        gene=harness.provider, source_name="Jira", source_id=SOURCE_ID,
        failed_item_retry=SCHEDULED, input_snapshot=SnapshotBinding("snapshot-1", listed),
    )


@pytest.mark.asyncio
async def test_a_server_query_listing_settles_a_failed_document_its_rediscovery_finds_gone(db):
    harness = await _gone_before_it_committed(db, BreakableProvider())

    # A scheduled run leaves this release's deterministic failure alone.
    assert (await harness.sync(retry=SCHEDULED)).last_sync_status == "success"
    assert set(await _ledger(db)) == {"PAY-1"}
    assert harness.provider.rediscovered == []

    state = await harness.sync()

    assert state.last_sync_status == "success"
    assert harness.provider.rediscovered == ["PAY-1"]
    assert harness.provider.fetches == []
    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_server_query_listing_keeps_untried_a_failed_document_it_cannot_rediscover(db, monkeypatch):
    monkeypatch.setattr("memforge.pipeline.sync.source_rediscovers_documents", lambda *args: False)
    harness = await _gone_before_it_committed(db, BreakableProvider())

    state = await harness.sync()

    assert state.last_sync_status == "success"
    assert harness.provider.fetches == []
    assert set(await _ledger(db)) == {"PAY-1"}


@pytest.mark.asyncio
async def test_a_device_existence_manifest_settles_a_failed_document_it_does_not_list(db):
    harness = await _gone_before_it_committed(db, ExistenceListedProvider())

    state = await _device_sync(harness)

    assert state.last_sync_status == "success"
    assert state.absence_check_skipped_reason is None
    assert "PAY-1" not in harness.provider.fetches
    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_device_query_manifest_keeps_untried_a_failed_document_it_does_not_list(db):
    harness = await _gone_before_it_committed(db, BreakableProvider())

    state = await _device_sync(harness)

    # Only the device can confirm by identifier whether PAY-1 left the query
    # or is gone, and it has not.
    assert state.last_sync_status == "success"
    assert state.absence_check_skipped_reason == "absence_confirmation_missing"
    assert state.unlisted_present_count is None
    assert "PAY-1" not in harness.provider.fetches
    assert set(await _ledger(db)) == {"PAY-1"}


@pytest.mark.asyncio
async def test_a_document_leaves_the_ledger_when_it_commits_even_if_the_run_then_stops(db, monkeypatch):
    harness = await breakable(db)
    _change(harness, "PAY-1")
    harness.provider.broken["PAY-1"] = ("fetch", ValueError)
    await harness.sync(retry=SCHEDULED)
    harness.provider.broken.clear()

    async def stopped(self, *args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(GeneSyncOrchestrator, "_record_failed_items", stopped)
    with pytest.raises(asyncio.CancelledError):
        await harness.sync()

    assert await _ledger(db) == {}


@pytest.mark.asyncio
async def test_a_ledger_write_without_the_current_source_activity_lease_is_refused(db):
    await _jira_source(db)
    now = datetime.now(timezone.utc)
    row = SourceItemFailure(
        source_id=SOURCE_ID, document_id="doc-1", stage="fetch", error_class="builtins.ValueError",
        message="failed", retryable=False, release=current_release(), failures_in_release=1,
        first_failed_at=now, last_failed_at=now,
    )
    await db.upsert_source_item_failure(row)
    lost = SourceActivityLease(
        id="activity-lost", source_id=SOURCE_ID, kind=SourceActivityKind.SYNC, capability=None,
        lease_until=now + timedelta(minutes=5),
    )

    with pytest.raises(SourceActivityConflict):
        await db.upsert_source_item_failure(replace(row, message="later"), source_activity=lost)
    with pytest.raises(SourceActivityConflict):
        await db.delete_source_item_failures(SOURCE_ID, ["doc-1"], source_activity=lost)

    assert await db.list_source_item_failures(source_id=SOURCE_ID) == (row,)


def test_a_ledger_row_keeps_the_rename_an_earlier_item_attested():
    now = datetime(2026, 10, 17, tzinfo=timezone.utc)
    renamed = JiraProvider()._item("PAY-1")
    renamed.renamed_from = "PAY:old-unit"
    fresh = JiraProvider()._item("PAY-1")
    first = SourceItemFailure.recorded(
        None,
        RunItemFailure(
            document_id=doc_id("PAY-1"), stage="fetch", error_class="builtins.ValueError", message="failed",
            retryable=False, item=renamed,
        ),
        source_id=SOURCE_ID, release=current_release(), failed_at=now,
    )

    again = SourceItemFailure.recorded(
        first,
        RunItemFailure(
            document_id=doc_id("PAY-1"), stage="fetch", error_class="builtins.ValueError", message="failed",
            retryable=False, item=fresh,
        ),
        source_id=SOURCE_ID, release=current_release(), failed_at=now,
    )

    assert again.item is not None and again.item.renamed_from == "PAY:old-unit"
    assert fresh.renamed_from is None
    assert with_inherited_rename(fresh, first).renamed_from == "PAY:old-unit"
    assert with_inherited_rename(fresh, None) is fresh
