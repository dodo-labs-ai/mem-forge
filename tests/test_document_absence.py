"""A document Source removes a Unit only when its scope listing proves the provider no longer has it."""

from __future__ import annotations

import asyncio
import logging
import json
from datetime import datetime, timezone

import httpx
import pytest

from memforge.local_agent.snapshot_manifest import SnapshotConfirmation, SnapshotConfirmationAnswer
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, ScopeListingKind
import memforge.pipeline.collection_family as collection_family_module
from memforge.pipeline.collection_family import SnapshotBinding
from memforge.pipeline.stored_document import StoredDocumentUnavailable, StoredDocumentUnavailableReason
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.source_item_failures import FailedItemRetry
from memforge.storage.database import Database
from tests.test_sync_bookkeeping import (
    BlockingFetchGene,
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    StubDocumentStore,
    _OrderRecordingMemoryEngine,
    _skip_retry_delay,
)

SCHEDULED = FailedItemRetry.RETRYABLE

SOURCE_ID = "src-document-absence"
FIRST_ISSUE_ID = 100000
JIRA_CONFIG = {"query_mode": "advanced", "jql": "project in (PAY, OPS)"}


def _issue_number(item: ContentItem) -> int:
    return int(item.extra["issue_id"]) - FIRST_ISSUE_ID


class JiraProject(BlockingFetchGene):
    """Issues as Jira holds them, the ones the configured query matches, and the ones changed since the last sync."""

    listing_kinds = frozenset({ScopeListingKind.EXISTENCE, ScopeListingKind.QUERY})

    def __init__(
        self,
        *,
        existing: set[int],
        in_query: set[int] | None = None,
        listing_kind: ScopeListingKind = ScopeListingKind.QUERY,
    ) -> None:
        release = asyncio.Event()
        release.set()
        super().__init__(item_count=0, release=release)
        self.existing = set(existing)
        self.in_query = set(existing if in_query is None else in_query)
        self.changed: set[int] = set()
        self.listing_kind = listing_kind
        self.listing_error: Exception | None = None
        self.confirm_error: Exception | None = None
        self.discovery_error: Exception | None = None
        self.seen_since: list[datetime | None] = []
        self.fetched: list[str] = []
        self.confirmed: list[str] = []
        self.listings = 0
        # The step at which an issue's processing fails, and the error it raises.
        self.broken: dict[int, tuple[str, Exception]] = {}
        # The Document each issue was renamed from, reported by the next discovery only.
        self.renamed: dict[int, str] = {}
        # Issues whose summary was edited, so processing them derives Memories again.
        self.edited: set[int] = set()

    def _item(self, number: int) -> ContentItem:
        return ContentItem(
            item_id=f"jira-{number}",
            title=f"Jira {number}" + (" (edited)" if number in self.edited else ""),
            source_url=f"https://jira.example/browse/{number}",
            last_modified=datetime.now(timezone.utc),
            content_type="application/json",
            space_or_project="PAY",
            version=str(number),
            extra={"issue_id": str(FIRST_ISSUE_ID + number), "issue_key": f"PAY-{number}"},
        )

    async def discover(self, since=None):
        self.seen_since.append(since)
        if self.discovery_error is not None:
            raise self.discovery_error
        renamed, self.renamed = self.renamed, {}
        for number in sorted(self.in_query):
            if since is None or number in self.changed:
                item = self._item(number)
                item.renamed_from = renamed.get(number)
                yield item

    async def rediscover(self, item):
        number = _issue_number(item)
        return self._item(number) if number in self.in_query else None

    def _break(self, stage: str, item: ContentItem) -> None:
        broken = self.broken.get(_issue_number(item))
        if broken is not None and broken[0] == stage:
            raise broken[1]

    async def fetch(self, item):
        self.fetched.append(item.item_id)
        self._break("fetch", item)
        return await super().fetch(item)

    async def normalize(self, raw):
        self._break("normalize", raw.item)
        return await super().normalize(raw)

    async def list_scope(self):
        self.listings += 1
        if self.listing_error is not None:
            raise self.listing_error
        return ScopeListing(kind=self.listing_kind, doc_ids=frozenset(f"jira-{number}" for number in self.in_query))

    async def confirm_absent(self, items):
        if self.confirm_error is not None:
            raise self.confirm_error
        self.confirmed.extend(item.item_id for item in items)
        return {
            item.item_id: AbsenceAnswer.EXISTS if _issue_number(item) in self.existing else AbsenceAnswer.ABSENT
            for item in items
        }

    def move_on(self, *, existing: set[int], in_query: set[int] | None = None, changed: set[int] = frozenset()) -> None:
        self.existing = set(existing)
        self.in_query = set(existing if in_query is None else in_query)
        self.changed = set(changed)
        self.fetched.clear()
        self.confirmed.clear()


class UnlistedJiraProject(JiraProject):
    """A configuration that cannot list its scope, so its listing is never asked for."""

    @classmethod
    def lists_scope(cls, config) -> bool:
        return False

    async def list_scope(self):
        raise AssertionError("a configuration that cannot list is never listed")


class Workspace:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.engine = _OrderRecordingMemoryEngine(defer=False)

    def orchestrator(self, extractor=None) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=StubDocumentStore(),
            memory_extractor=extractor or ProjectionFragmentRecordingExtractor(),
            memory_engine=self.engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.db),
            max_concurrent=1,
            retry_sleep=_skip_retry_delay,
        )

    async def sync(
        self,
        gene: JiraProject,
        *,
        force_full_sync: bool = False,
        retry: FailedItemRetry = FailedItemRetry.ALL,
        manifest: set[int] | None = None,
        answers: dict[int, SnapshotConfirmationAnswer] | None = None,
        confirmation_failure: str | None = None,
        extractor=None,
        excluded_scope_gene=None,
    ):
        """Sync as a person requests it; ``manifest`` binds the run to a device snapshot listing those issues.

        ``answers`` are the device's answers for the issues the manifest left
        out; ``confirmation_failure`` is the cause it gave instead.
        """
        confirmations = tuple(
            SnapshotConfirmation(
                f"jira-{number}",
                str(FIRST_ISSUE_ID + number),
                SnapshotConfirmationAnswer.FAILED if confirmation_failure else answer,
                confirmation_failure,
            )
            for number, answer in sorted((answers or {}).items())
        )
        return await self.orchestrator(extractor).sync_gene(
            gene=gene,
            source_name="Jira",
            source_id=SOURCE_ID,
            force_full_sync=force_full_sync,
            failed_item_retry=retry,
            input_snapshot=(
                SnapshotBinding("snapshot-1", frozenset(f"jira-{number}" for number in manifest), confirmations)
                if manifest is not None
                else None
            ),
            excluded_scope_gene=excluded_scope_gene,
        )

    async def reprocess(self, gene: JiraProject, *numbers: int):
        return await self.orchestrator().sync_gene(
            gene=gene,
            source_name="Jira",
            source_id=SOURCE_ID,
            execution_mode=SourceSyncMode.REPROCESS,
            reprocess_doc_ids=frozenset(f"jira-{number}" for number in numbers),
        )

    async def ledger(self) -> dict[str, ContentItem | None]:
        return {failure.document_id: failure.item for failure in await self.db.list_source_item_failures(source_id=SOURCE_ID)}

    async def unit_id(self, doc_id: str) -> str:
        unit = await self.db.find_source_unit_by_document_id(SOURCE_ID, doc_id, current_only=True)
        assert unit is not None
        return unit.id

    async def recorded_unlisted_present_count(self) -> int | None:
        latest = (await self.db.get_sync_history(source=SOURCE_ID, limit=1))[0]
        return latest["unlisted_present_count"]

    async def held(self) -> set[str]:
        return await self.db.list_indexed_doc_ids(SOURCE_ID)

    async def recorded_skip_reason(self) -> str | None:
        latest = (await self.db.get_sync_history(source=SOURCE_ID, limit=1))[0]
        return latest["absence_check_skipped_reason"]

    def tombstoned(self) -> list[str]:
        return [doc_id for event, doc_id in self.engine.events if event == "tombstone"]


@pytest.fixture
async def workspace(tmp_path):
    database = Database(str(tmp_path / "absence.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    await database.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(JIRA_CONFIG),
        access_policy="workspace",
        owner_user_id="dev",
    )
    try:
        yield Workspace(database)
    finally:
        NoopMemoryEngine.db = None
        await database.close()


async def _synced(workspace: Workspace, gene: JiraProject) -> None:
    first = await workspace.sync(gene)
    assert first.last_sync_status == "success"
    workspace.engine.events.clear()


@pytest.mark.asyncio
async def test_a_renamed_file_is_committed_then_its_old_unit_is_tombstoned_in_an_incremental_run(workspace):
    repository = JiraProject(existing={0, 2}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)

    # The file behind jira-2 was renamed to jira-1.
    repository.move_on(existing={0, 1}, changed={1})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert repository.seen_since[-1] is not None
    assert workspace.engine.events == [("commit", "jira-1"), ("tombstone", "jira-2")]
    assert repository.confirmed == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert await workspace.db.get_document("jira-2") is None


@pytest.mark.asyncio
async def test_a_deleted_issue_is_tombstoned_in_an_incremental_run(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert project.seen_since[-1] is not None
    assert project.confirmed == ["jira-1"]
    assert workspace.tombstoned() == ["jira-1"]
    assert await workspace.held() == {"jira-0"}
    assert state.absence_check_skipped_reason is None
    assert await workspace.recorded_skip_reason() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("force_full_sync", [False, True], ids=["incremental", "force-full"])
async def test_an_issue_that_leaves_the_query_but_still_exists_keeps_its_unit(workspace, force_full_sync):
    """An issue older than a relative-date window, closed under a status filter, or a page moved out of its tree."""
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0, 1}, in_query={0})
    state = await workspace.sync(project, force_full_sync=force_full_sync)

    assert state.last_sync_status == "success"
    assert project.confirmed == ["jira-1"]
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    # It is not refreshed while it stays outside the query.
    assert "jira-1" not in project.fetched


@pytest.mark.asyncio
async def test_an_issue_that_reenters_the_query_is_refreshed_and_a_later_deletion_is_still_found(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    project.move_on(existing={0, 1}, in_query={0})
    await workspace.sync(project)

    project.move_on(existing={0, 1}, changed={1})
    await workspace.sync(project)
    assert project.fetched == ["jira-1"]

    project.move_on(existing={0}, in_query={0})
    await workspace.sync(project)
    assert workspace.tombstoned() == ["jira-1"]


@pytest.mark.asyncio
async def test_an_incomplete_listing_removes_nothing_and_the_run_still_succeeds(workspace, caplog):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    project.listing_error = RuntimeError("Jira search total changed during pagination")
    with caplog.at_level(logging.WARNING, logger="memforge.pipeline.sync"):
        state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.absence_check_skipped_reason == "scope_listing_failed"
    assert await workspace.recorded_skip_reason() == "scope_listing_failed"
    assert "Jira search total changed during pagination" in caplog.text

    project.listing_error = None
    state = await workspace.sync(project)
    assert workspace.tombstoned() == ["jira-1"]
    assert state.absence_check_skipped_reason is None
    assert await workspace.recorded_skip_reason() is None


@pytest.mark.asyncio
async def test_a_rejected_credential_fails_the_run_and_removes_nothing(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    request = httpx.Request("GET", "https://jira.example/rest/api/2/search")
    project.discovery_error = httpx.HTTPStatusError(
        "unauthorized", request=request, response=httpx.Response(httpx.codes.UNAUTHORIZED, request=request)
    )
    state = await workspace.sync(project)

    assert state.last_sync_status == "failed"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}


@pytest.mark.asyncio
async def test_a_failed_confirmation_removes_nothing_and_the_run_still_succeeds(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    project.confirm_error = httpx.ConnectError("connection reset")
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.absence_check_skipped_reason == "absence_confirmation_failed"
    assert await workspace.recorded_skip_reason() == "absence_confirmation_failed"


@pytest.mark.asyncio
async def test_a_source_that_cannot_list_its_scope_skips_no_absence_check(workspace):
    project = UnlistedJiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert state.absence_check_skipped_reason is None
    assert await workspace.recorded_skip_reason() is None


@pytest.mark.asyncio
async def test_runs_where_nothing_changed_at_the_provider_remove_nothing(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    for _ in range(2):
        project.move_on(existing={0, 1})
        state = await workspace.sync(project)
        assert state.last_sync_status == "success"

    assert project.seen_since[-1] is not None
    assert project.fetched == []
    assert project.confirmed == []
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}


@pytest.mark.asyncio
async def test_a_run_without_an_input_snapshot_records_no_complete_snapshot_in_its_projections(workspace):
    await _synced(workspace, JiraProject(existing={0, 1}))

    async with workspace.db.db.execute(
        "SELECT scope_json FROM source_projection_runs WHERE source_id = ?", (SOURCE_ID,)
    ) as cursor:
        scopes = [json.loads(row[0]) async for row in cursor]

    assert scopes and all(scope["authoritative_snapshot"] is False for scope in scopes)


class RaisingExtractor(ProjectionFragmentRecordingExtractor):
    async def extract_projection_fragment_memories(self, catalog, **kwargs):
        raise ValueError("extraction raised")


class FailingCommitEngine(_OrderRecordingMemoryEngine):
    def __init__(self, doc_id: str) -> None:
        super().__init__(defer=False)
        self.failing_doc_id = doc_id

    async def prepare_and_commit_projected_lifecycle(self, **kwargs):
        if kwargs["doc_id"] == self.failing_doc_id:
            raise ValueError("lifecycle commit failed")
        return await super().prepare_and_commit_projected_lifecycle(**kwargs)


class DeviceManifest(JiraProject):
    """A device-collected Source whose snapshot manifest is an existence listing."""

    listing_kinds = frozenset({ScopeListingKind.EXISTENCE})

    async def list_scope(self):
        raise AssertionError("a device-collected Source is listed by its snapshot manifest")


class DeviceJira(DeviceManifest):
    """Jira read through the local agent: its manifest is the query result, confirmed on the device."""

    listing_kinds = frozenset({ScopeListingKind.QUERY})

    async def confirm_absent(self, items):
        raise AssertionError("only the device confirms a device query listing")


@pytest.mark.asyncio
@pytest.mark.parametrize("listing_kind", [ScopeListingKind.EXISTENCE, ScopeListingKind.QUERY])
@pytest.mark.parametrize("stage", ["fetch", "normalize", "projection", "derivation", "lifecycle"])
async def test_a_failed_document_does_not_hold_back_the_removal_of_one_the_listing_left_out(
    workspace, monkeypatch, listing_kind, stage,
):
    project = JiraProject(existing={0, 1, 2}, listing_kind=listing_kind)
    await _synced(workspace, project)

    # jira-1 was edited and fails at ``stage``; jira-2 was deleted.
    project.move_on(existing={0, 1}, changed={1})
    project.edited.add(1)
    extractor = None
    if stage in {"fetch", "normalize"}:
        project.broken[1] = (stage, ValueError(f"{stage} failed"))
    elif stage == "projection":
        project_item = GeneSyncOrchestrator._project_fetched_item

        async def failing_projection(self, **kwargs):
            if kwargs["item"].item_id == "jira-1":
                raise ValueError("projection failed")
            return await project_item(self, **kwargs)

        monkeypatch.setattr(GeneSyncOrchestrator, "_project_fetched_item", failing_projection)
    elif stage == "derivation":
        extractor = RaisingExtractor()
    else:
        workspace.engine = FailingCommitEngine("jira-1")
    state = await workspace.sync(project, extractor=extractor)

    assert state.last_sync_status == "partial"
    assert {failed.doc_id for failed in state.failed_docs} == {"jira-1"}
    assert workspace.tombstoned() == ["jira-2"]
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert project.confirmed == ([] if listing_kind is ScopeListingKind.EXISTENCE else ["jira-2"])
    assert state.absence_check_skipped_reason is None


@pytest.mark.asyncio
async def test_a_query_listing_records_how_many_unlisted_issues_still_exist(workspace):
    project = JiraProject(existing={0, 1, 2})
    await _synced(workspace, project)

    # jira-1 left the query but still exists; jira-2 was deleted.
    project.move_on(existing={0, 1}, in_query={0})
    state = await workspace.sync(project)

    assert workspace.tombstoned() == ["jira-2"]
    assert state.unlisted_present_count == 1
    assert await workspace.recorded_unlisted_present_count() == 1

    project.move_on(existing={0, 1}, in_query={0, 1})
    state = await workspace.sync(project)
    assert state.unlisted_present_count is None
    assert await workspace.recorded_unlisted_present_count() is None


@pytest.mark.asyncio
async def test_an_existence_listing_records_no_unlisted_present_count(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)

    repository.move_on(existing={0})
    state = await workspace.sync(repository)

    assert workspace.tombstoned() == ["jira-1"]
    assert state.unlisted_present_count is None
    assert await workspace.recorded_unlisted_present_count() is None


@pytest.mark.asyncio
async def test_a_run_lists_its_scope_once_and_keeps_that_listing(workspace):
    project = JiraProject(existing={0, 1, 2, 3})
    project.broken[3] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(project)
    assert set(await workspace.ledger()) == {"jira-3"}
    project.listings = 0

    # The ledger row is retried through rediscovery after the listing was
    # taken, and the absence step reads that same listing.
    project.move_on(existing={0, 1, 2, 3}, in_query={0, 1, 3})
    project.broken.clear()
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert project.listings == 1
    assert "jira-3" in project.fetched
    assert project.confirmed == ["jira-2"]
    assert await workspace.ledger() == {}


@pytest.mark.asyncio
async def test_a_failed_listing_keeps_ledger_rows_an_existence_listing_would_settle(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    repository.broken[1] = ("fetch", ValueError("fetch failed"))
    assert (await workspace.sync(repository)).last_sync_status == "partial"
    assert set(await workspace.ledger()) == {"jira-1"}

    # jira-1 is gone; a scheduled run leaves this release's deterministic failure alone.
    repository.move_on(existing={0})
    repository.listing_error = RuntimeError("tree truncated")
    state = await workspace.sync(repository, retry=SCHEDULED)

    assert state.absence_check_skipped_reason == "scope_listing_failed"
    assert set(await workspace.ledger()) == {"jira-1"}

    repository.listing_error = None
    state = await workspace.sync(repository, retry=SCHEDULED)

    assert state.absence_check_skipped_reason is None
    assert repository.fetched == []
    assert await workspace.ledger() == {}


@pytest.mark.asyncio
async def test_a_reprocess_never_lists_its_scope(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    project.listings = 0
    project.listing_error = AssertionError("a reprocess does not list")
    state = await workspace.reprocess(project, 0)

    assert state.last_sync_status == "success"
    assert project.listings == 0
    assert workspace.tombstoned() == []
    assert state.absence_check_skipped_reason is None


@pytest.mark.asyncio
async def test_a_device_existence_manifest_removes_unlisted_units_and_settles_unlisted_rows(workspace):
    device = DeviceManifest(existing={0, 1, 2})
    device.broken[2] = ("fetch", ValueError("package unreadable"))
    assert (await workspace.sync(device, manifest={0, 1, 2})).last_sync_status == "partial"
    assert set(await workspace.ledger()) == {"jira-2"}
    workspace.engine.events.clear()

    # jira-1 and jira-2, which never became a Unit, left the device's folder.
    device.move_on(existing={0})
    state = await workspace.sync(device, manifest={0}, retry=SCHEDULED)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-1"]
    assert await workspace.held() == {"jira-0"}
    assert await workspace.ledger() == {}
    assert state.unlisted_present_count is None


@pytest.mark.asyncio
async def test_a_device_query_manifest_without_confirmations_removes_nothing(workspace):
    device = DeviceJira(existing={0, 1})
    assert (await workspace.sync(device, manifest={0, 1})).last_sync_status == "success"
    workspace.engine.events.clear()

    device.move_on(existing={0, 1}, in_query={0})
    state = await workspace.sync(device, manifest={0})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.absence_check_skipped_reason == "absence_confirmation_missing"
    assert await workspace.recorded_skip_reason() == "absence_confirmation_missing"
    assert state.unlisted_present_count is None


@pytest.mark.asyncio
async def test_a_device_query_manifest_listing_everything_needs_no_confirmation(workspace):
    device = DeviceJira(existing={0, 1})
    await workspace.sync(device, manifest={0, 1})

    state = await workspace.sync(device, manifest={0, 1})

    assert state.absence_check_skipped_reason is None


ABSENT = SnapshotConfirmationAnswer.ABSENT
EXISTS = SnapshotConfirmationAnswer.EXISTS
HIDDEN = SnapshotConfirmationAnswer.HIDDEN


@pytest.mark.asyncio
async def test_a_device_query_manifest_removes_only_the_issues_the_device_answered_absent(workspace):
    device = DeviceJira(existing={0, 1, 2, 3, 4})
    await workspace.sync(device, manifest={0, 1, 2, 3, 4})
    workspace.engine.events.clear()

    # jira-1 was deleted, jira-2 left the JQL, jira-3 is hidden from the
    # device's session, and the device was not asked about jira-4.
    device.move_on(existing={0, 2, 3, 4}, in_query={0})
    state = await workspace.sync(device, manifest={0}, answers={1: ABSENT, 2: EXISTS, 3: HIDDEN})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-1"]
    assert await workspace.held() == {"jira-0", "jira-2", "jira-3", "jira-4"}
    assert state.absence_check_skipped_reason is None
    assert state.unlisted_present_count == 2
    assert await workspace.recorded_unlisted_present_count() == 2


@pytest.mark.asyncio
async def test_a_device_answer_settles_or_keeps_the_failed_issues_its_manifest_left_out(workspace):
    device = DeviceJira(existing={0, 1, 2, 3})
    device.broken[2] = ("fetch", ValueError("issue unreadable"))
    device.broken[3] = ("fetch", ValueError("issue unreadable"))
    assert (await workspace.sync(device, manifest={0, 1, 2, 3})).last_sync_status == "partial"
    assert set(await workspace.ledger()) == {"jira-2", "jira-3"}
    workspace.engine.events.clear()

    # jira-2 was deleted; jira-1 and jira-3 left the JQL but still exist.
    device.move_on(existing={0, 1, 3}, in_query={0})
    state = await workspace.sync(device, manifest={0}, answers={1: EXISTS, 2: ABSENT, 3: EXISTS})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert set(await workspace.ledger()) == {"jira-3"}
    assert "jira-3" not in device.fetched
    assert state.unlisted_present_count == 1

    # The issue that still exists is retried once a manifest lists it again.
    device.broken.clear()
    device.move_on(existing={0, 1, 3}, in_query={0, 3})
    state = await workspace.sync(device, manifest={0, 3}, answers={1: EXISTS}, retry=SCHEDULED)

    assert state.last_sync_status == "success"
    assert "jira-3" in device.fetched
    assert await workspace.ledger() == {}
    assert await workspace.held() == {"jira-0", "jira-1", "jira-3"}


@pytest.mark.asyncio
async def test_a_confirm_set_too_large_to_issue_removes_nothing(workspace):
    device = DeviceJira(existing={0, 1, 2})
    await workspace.sync(device, manifest={0, 1, 2})
    workspace.engine.events.clear()

    # The server issued no confirmations because the set was larger than a
    # manifest, so the device answered nothing.
    device.move_on(existing={0}, in_query={0})
    state = await workspace.sync(device, manifest={0}, answers={})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert state.absence_check_skipped_reason == "absence_confirmation_missing"
    assert await workspace.recorded_skip_reason() == "absence_confirmation_missing"


@pytest.mark.asyncio
async def test_a_failed_device_confirmation_removes_and_settles_nothing(workspace):
    device = DeviceJira(existing={0, 1, 2})
    device.broken[2] = ("fetch", ValueError("issue unreadable"))
    await workspace.sync(device, manifest={0, 1, 2})
    workspace.engine.events.clear()

    device.move_on(existing={0}, in_query={0})
    state = await workspace.sync(
        device, manifest={0}, answers={1: None, 2: None}, confirmation_failure="HTTP 401 from Jira"
    )

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert set(await workspace.ledger()) == {"jira-2"}
    assert state.absence_check_skipped_reason == "absence_confirmation_failed"
    assert await workspace.recorded_skip_reason() == "absence_confirmation_failed"
    assert state.unlisted_present_count is None


def test_an_item_payload_names_its_predecessor_only_when_it_has_one():
    item = ContentItem(
        item_id="github-new",
        title="Design",
        source_url="https://github.example/acme/pay/new/design.md",
        last_modified=datetime(2026, 10, 1, tzinfo=timezone.utc),
        extra={"relative_path": "new/design.md"},
    )
    payload = item.to_payload()

    assert "renamed_from" not in payload
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        {
            "item_id": "github-new",
            "title": "Design",
            "source_url": "https://github.example/acme/pay/new/design.md",
            "last_modified": "2026-10-01T00:00:00+00:00",
            "content_type": "text/html",
            "space_or_project": "",
            "version": "",
            "author": None,
            "labels": [],
            "extra": {"relative_path": "new/design.md"},
        },
        sort_keys=True,
    )
    assert ContentItem.from_payload(payload) == item

    item.renamed_from = "acme/pay:old/design.md"
    assert item.to_payload()["renamed_from"] == "acme/pay:old/design.md"
    assert ContentItem.from_payload(item.to_payload()) == item


async def _renamed_and_failed(workspace) -> JiraProject:
    """jira-2 and jira-3 are held; jira-2 was renamed to jira-1, whose fetch fails, and jira-3 was deleted."""
    repository = JiraProject(existing={0, 2, 3}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    repository.move_on(existing={0, 1}, changed={1})
    repository.renamed[1] = "jira-2"
    repository.broken[1] = ("fetch", ValueError("fetch failed"))
    return repository


@pytest.mark.asyncio
async def test_a_failed_renamed_item_keeps_its_predecessor_and_nothing_else(workspace):
    repository = await _renamed_and_failed(workspace)

    state = await workspace.sync(repository)

    assert state.last_sync_status == "partial"
    assert workspace.tombstoned() == ["jira-3"]
    assert await workspace.held() == {"jira-0", "jira-2"}
    assert (await workspace.ledger())["jira-1"].renamed_from == "jira-2"


@pytest.mark.asyncio
async def test_a_predecessor_is_kept_until_its_renamed_item_commits(workspace):
    repository = await _renamed_and_failed(workspace)
    predecessor = "jira-2"
    predecessor_unit_id = await workspace.unit_id("jira-2")
    await workspace.sync(repository)

    # The provider reported the rename once: a fresh item of jira-1, which
    # names no predecessor, fails again and keeps the rename its row recorded.
    repository.move_on(existing={0, 1}, changed={1})
    state = await workspace.sync(repository)
    assert state.last_sync_status == "partial"
    assert (await workspace.ledger())["jira-1"].renamed_from == predecessor
    assert await workspace.held() == {"jira-0", "jira-2"}

    # A scheduled run leaves jira-1 alone; only its ledger row names jira-2.
    repository.move_on(existing={0, 1})
    state = await workspace.sync(repository, retry=SCHEDULED)
    assert repository.fetched == []
    assert workspace.tombstoned() == ["jira-3"]
    assert await workspace.held() == {"jira-0", "jira-2"}

    # Once jira-1 commits, it continues jira-2's Unit.
    repository.broken.clear()
    repository.move_on(existing={0, 1}, changed={1})
    state = await workspace.sync(repository)
    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-3"]
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert await workspace.unit_id("jira-1") == predecessor_unit_id
    assert await workspace.ledger() == {}


@pytest.mark.asyncio
async def test_a_renamed_item_whose_commit_is_still_deferred_keeps_its_predecessor(workspace):
    repository = await _renamed_and_failed(workspace)
    repository.broken.clear()
    # jira-0 commits first; jira-1's commit then waits on a Unit outside this run.
    repository.edited.add(0)
    repository.changed.add(0)
    workspace.engine = _OrderRecordingMemoryEngine(defer=True)
    workspace.engine.external_blocker = True

    state = await workspace.sync(repository)

    assert state.docs_failed == 1
    assert ("tombstone", "jira-2") not in workspace.engine.events
    assert ("tombstone", "jira-3") in workspace.engine.events
    assert "jira-2" in await workspace.held()


@pytest.mark.asyncio
async def test_a_chained_rename_keeps_the_unit_its_last_name_holds(workspace):
    repository = JiraProject(existing={0, 2}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    unit_id = await workspace.unit_id("jira-2")
    # jira-2 was renamed to jira-1, which committed to jira-2's Unit.
    repository.move_on(existing={0, 1}, changed={1})
    repository.renamed[1] = "jira-2"
    assert (await workspace.sync(repository)).last_sync_status == "success"
    assert await workspace.unit_id("jira-1") == unit_id

    # Then jira-1 was renamed to jira-3, whose fetch fails.
    repository.move_on(existing={0, 3}, changed={3})
    repository.renamed[3] = "jira-1"
    repository.broken[3] = ("fetch", ValueError("fetch failed"))
    state = await workspace.sync(repository)

    assert state.last_sync_status == "partial"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}

    repository.broken.clear()
    repository.move_on(existing={0, 3}, changed={3})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-3"}
    assert await workspace.unit_id("jira-3") == unit_id


@pytest.mark.asyncio
async def test_a_query_listing_keeps_a_predecessor_without_asking_the_provider_about_it(workspace):
    project = JiraProject(existing={0, 2, 3})
    await _synced(workspace, project)

    # jira-2 was renamed to jira-1, whose fetch fails; jira-3 was deleted.
    project.move_on(existing={0, 1}, changed={1})
    project.renamed[1] = "jira-2"
    project.broken[1] = ("fetch", ValueError("fetch failed"))
    state = await workspace.sync(project)

    assert project.confirmed == ["jira-3"]
    assert workspace.tombstoned() == ["jira-3"]
    assert await workspace.held() == {"jira-0", "jira-2"}
    assert state.unlisted_present_count == 0


@pytest.mark.asyncio
async def test_an_unlisted_document_whose_item_cannot_be_rebuilt_is_kept_and_counted(workspace, monkeypatch):
    project = JiraProject(existing={0, 1, 2})
    await _synced(workspace, project)
    rebuild = collection_family_module.stored_source_item

    async def stored_source_item(db, *, source_id, document_id):
        if document_id == "jira-1":
            raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING, document_id)
        return await rebuild(db, source_id=source_id, document_id=document_id)

    monkeypatch.setattr(collection_family_module, "stored_source_item", stored_source_item)
    project.move_on(existing={0})
    state = await workspace.sync(project)

    assert project.confirmed == ["jira-2"]
    assert workspace.tombstoned() == ["jira-2"]
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.unlisted_present_count == 1


@pytest.mark.asyncio
async def test_a_snapshot_bound_configuration_that_cannot_list_removes_nothing(workspace):
    project = UnlistedJiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await workspace.sync(project, manifest={0, 1})
    workspace.engine.events.clear()

    project.move_on(existing={0})
    state = await workspace.sync(project, manifest={0})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.absence_check_skipped_reason is None
