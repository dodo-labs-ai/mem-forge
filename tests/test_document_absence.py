"""A document Source removes a Unit only when its scope listing proves the provider no longer has it (ADR 0045)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from memforge.models import ContentItem, ScopeListing, ScopeListingKind
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.source_projection import ProjectionScopeTransition, ProjectionScopeTransitionStatus
from memforge.source_projection_config import canonical_projection_scope
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

SOURCE_ID = "src-document-absence"
FIRST_ISSUE_ID = 100000
JIRA_CONFIG = {"query_mode": "advanced", "jql": "project in (PAY, OPS)"}


def _issue_number(item: ContentItem) -> int:
    return int(item.extra["issue_id"]) - FIRST_ISSUE_ID


class JiraProject(BlockingFetchGene):
    """Issues as Jira holds them, the ones the configured query matches, and the ones changed since the last sync."""

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

    async def discover(self, since=None):
        self.seen_since.append(since)
        if self.discovery_error is not None:
            raise self.discovery_error
        for number in sorted(self.in_query):
            if since is None or number in self.changed:
                yield ContentItem(
                    item_id=f"jira-{number}",
                    title=f"Jira {number}",
                    source_url=f"https://jira.example/browse/{number}",
                    last_modified=datetime.now(timezone.utc),
                    content_type="application/json",
                    space_or_project="PAY",
                    version=str(number),
                    extra={"issue_id": str(FIRST_ISSUE_ID + number), "issue_key": f"PAY-{number}"},
                )

    async def fetch(self, item):
        self.fetched.append(item.item_id)
        return await super().fetch(item)

    async def list_scope(self):
        if self.listing_error is not None:
            raise self.listing_error
        return ScopeListing(kind=self.listing_kind, doc_ids=frozenset(f"jira-{number}" for number in self.in_query))

    async def confirm_absent(self, items):
        if self.confirm_error is not None:
            raise self.confirm_error
        self.confirmed.extend(item.item_id for item in items)
        return frozenset(item.item_id for item in items if _issue_number(item) not in self.existing)

    def move_on(self, *, existing: set[int], in_query: set[int] | None = None, changed: set[int] = frozenset()) -> None:
        self.existing = set(existing)
        self.in_query = set(existing if in_query is None else in_query)
        self.changed = set(changed)
        self.fetched.clear()
        self.confirmed.clear()


class Workspace:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.engine = _OrderRecordingMemoryEngine(defer=False)

    async def sync(self, gene: JiraProject, *, force_full_sync: bool = False):
        return await GeneSyncOrchestrator(
            db=self.db,
            doc_store=StubDocumentStore(),
            memory_extractor=ProjectionFragmentRecordingExtractor(),
            memory_engine=self.engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.db),
            max_concurrent=1,
            retry_sleep=_skip_retry_delay,
        ).sync_gene(gene=gene, source_name="Jira", source_id=SOURCE_ID, force_full_sync=force_full_sync)

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
async def test_an_incomplete_listing_removes_nothing_and_the_run_still_succeeds(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)

    project.move_on(existing={0})
    project.listing_error = RuntimeError("Jira search total changed during pagination")
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert state.absence_check_skipped_reason == "Jira search total changed during pagination"
    assert await workspace.recorded_skip_reason() == "Jira search total changed during pagination"

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
    assert await workspace.recorded_skip_reason() == "connection reset"


@pytest.mark.asyncio
async def test_a_source_that_cannot_list_its_scope_skips_no_absence_check(workspace):
    project = JiraProject(existing={0, 1})
    project.list_scope = None
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
async def test_removing_a_project_from_the_jql_is_a_scope_transition_that_removes_its_units(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    target_config = {"query_mode": "advanced", "jql": "project = PAY"}
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(target_config),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="scope-transition-drop-ops",
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("jira", JIRA_CONFIG),
            target_scope=canonical_projection_scope("jira", target_config),
        )
    )

    # jira-1 is in the removed project; it still exists, but the user no
    # longer asks for it.
    project.move_on(existing={0, 1}, in_query={0})
    project.listing_error = AssertionError("a scope transition removes by its complete discovery")
    state = await workspace.sync(project)
    transition = (await workspace.db.list_projection_scope_transitions(SOURCE_ID))[0]

    assert state.last_sync_status == "success"
    assert project.seen_since[-1] is None
    assert workspace.tombstoned() == ["jira-1"]
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert await workspace.held() == {"jira-0"}
