"""A document Source's scope transition completes once every Document its new scope lists has committed under it.

Commitment accumulates across the transition's runs as member rows. Each run
fetches only the listed Documents that are not members yet, and removes
nothing until the run that completes the transition, which removes what the
new scope leaves out and sets the cursor to the time the transition opened.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.models import ScopeListingKind
from memforge.pipeline.collection_family import DEFAULT_INCREMENTAL_SYNC_OVERLAP
from memforge.source_projection import (
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
    UnitRemovalReason,
)
from memforge.source_projection_config import canonical_projection_scope
from memforge.storage.database import Database
from tests.test_document_absence import (
    JIRA_CONFIG,
    SOURCE_ID,
    DeviceManifest,
    JiraProject,
    Workspace,
    _synced,
)
from tests.test_sync_bookkeeping import NoopMemoryEngine, _OrderRecordingMemoryEngine

OPENED_AT = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
TARGET_CONFIG = {"query_mode": "advanced", "jql": "project = PAY"}
TRANSITION_ID = "scope-transition-drop-ops"


@pytest.fixture
async def workspace(tmp_path):
    database = Database(str(tmp_path / "transition.db"))
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


async def _change_scope(workspace) -> None:
    """The user changes the Source's scope: the Source holds the new configuration and an open transition."""
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(TARGET_CONFIG),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION_ID,
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("jira", JIRA_CONFIG),
            target_scope=canonical_projection_scope("jira", TARGET_CONFIG),
            created_at=OPENED_AT.isoformat(),
        )
    )


async def _transition(workspace) -> ProjectionScopeTransition:
    [transition] = await workspace.db.list_projection_scope_transitions(SOURCE_ID)
    return transition


async def _members(workspace) -> tuple[str, ...]:
    return await workspace.db.list_projection_scope_transition_member_doc_ids(TRANSITION_ID)


async def _cursor(workspace) -> datetime | None:
    state = await workspace.db.get_sync_state(SOURCE_ID)
    return state.last_sync_at if state is not None else None


async def _removal_reason(workspace, doc_id: str) -> str | None:
    unit = await workspace.db.find_source_unit_by_document_id(SOURCE_ID, doc_id)
    assert unit is not None
    return unit.locator.get("tombstone_reason")


@pytest.mark.asyncio
async def test_a_scope_change_with_one_failed_fetch_stays_open_removes_nothing_and_keeps_the_cursor(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    cursor = await _cursor(workspace)
    await _change_scope(workspace)

    # The new scope holds two Documents under new ids, as files do under a new Git ref.
    repository.move_on(existing={2, 3})
    repository.broken[3] = ("fetch", ValueError("fetch failed"))
    state = await workspace.sync(repository)
    transition = await _transition(workspace)

    assert state.last_sync_status == "partial"
    assert repository.seen_since[-1] is None
    assert workspace.tombstoned() == []
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "1 listed Documents have not committed under the transition"
    assert await _members(workspace) == ("jira-2",)
    assert await _cursor(workspace) == cursor


@pytest.mark.asyncio
async def test_members_accumulate_and_the_run_that_completes_fetches_only_non_members(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    repository.move_on(existing={2, 3})
    repository.broken[3] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(repository)

    repository.broken.clear()
    repository.move_on(existing={2, 3})
    state = await workspace.sync(repository)
    transition = await _transition(workspace)

    assert state.last_sync_status == "success"
    assert repository.fetched == ["jira-3"]
    assert sorted(workspace.tombstoned()) == ["jira-0", "jira-1"]
    assert await workspace.held() == {"jira-2", "jira-3"}
    for removed in ("jira-0", "jira-1"):
        assert await _removal_reason(workspace, removed) == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert await _members(workspace) == ()
    # The next run discovers every change made while the transition was open.
    assert await _cursor(workspace) == OPENED_AT
    repository.move_on(existing={2, 3})
    await workspace.sync(repository)
    assert repository.seen_since[-1] == OPENED_AT - DEFAULT_INCREMENTAL_SYNC_OVERLAP


@pytest.mark.asyncio
async def test_an_unchanged_listed_document_becomes_a_member_without_semantic_work(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)

    repository.move_on(existing={0, 2})
    repository.broken[2] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(repository)

    assert "jira-0" in repository.fetched
    assert ("commit", "jira-0") not in workspace.engine.events
    assert await _members(workspace) == ("jira-0",)


@pytest.mark.asyncio
async def test_a_deferred_document_is_not_a_member_while_its_commit_waits(workspace):
    repository = JiraProject(existing={0}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    # jira-0 changed, so it commits through the lifecycle; jira-1 then waits
    # on a Unit nothing in this run commits or removes.
    workspace.engine = _OrderRecordingMemoryEngine(defer=True)
    workspace.engine.external_blocker = True
    repository.move_on(existing={0, 1})
    repository.edited.add(0)
    state = await workspace.sync(repository)

    assert state.last_sync_status == "partial"
    assert await _members(workspace) == ("jira-0",)
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.FAILED


@pytest.mark.asyncio
async def test_a_deferred_document_becomes_a_member_when_its_commit_converges(workspace):
    repository = JiraProject(existing={0}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    # jira-1 waits on jira-0's Unit, which commits in the same run.
    workspace.engine = _OrderRecordingMemoryEngine(defer=True)
    repository.move_on(existing={0, 1})
    repository.edited.add(0)
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert workspace.engine.commit_only_attempts, "jira-1 committed through deferred convergence"
    assert workspace.engine.events == [("commit", "jira-0"), ("commit", "jira-1")]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_listed_document_blocked_only_by_a_unit_completion_removes_commits_in_the_completing_run(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    engine.blocker_unit_id = await workspace.unit_id("jira-1")
    workspace.engine = engine

    # The new scope leaves jira-1 out; jira-2's commit waits on jira-1's Unit.
    repository.move_on(existing={0, 2})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert engine.events == [("tombstone", "jira-1"), ("commit", "jira-2")]
    assert await workspace.held() == {"jira-0", "jira-2"}
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED
    assert await _cursor(workspace) == OPENED_AT


@pytest.mark.asyncio
async def test_a_blocked_document_that_still_fails_after_the_removals_leaves_the_transition_open(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    cursor = await _cursor(workspace)
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    engine.blocker_unit_id = await workspace.unit_id("jira-1")
    retry = engine.retry_deferred_projected_lifecycle

    async def failing_retry(handle, *, eligible_same_run_source_unit_ids):
        raise RuntimeError("the deferred commit failed")

    engine.retry_deferred_projected_lifecycle = failing_retry
    workspace.engine = engine
    repository.move_on(existing={0, 2})
    state = await workspace.sync(repository)
    transition = await _transition(workspace)

    # The removal is applied; the transition waits for jira-2.
    assert state.last_sync_status == "partial"
    assert engine.events == [("tombstone", "jira-1")]
    assert await workspace.held() == {"jira-0"}
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "1 listed Documents did not commit once the Units blocking them were removed"
    assert await _members(workspace) == ("jira-0",)
    assert await _cursor(workspace) == cursor

    # The next run recovers jira-2's staged derivation, whose commit records
    # it as a member, so discovery does not fetch it again.
    engine.retry_deferred_projected_lifecycle = retry
    engine.events.clear()
    repository.move_on(existing={0, 2})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert repository.fetched == []
    assert engine.events == [("commit", "jira-2")]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_full_sync_during_a_transition_fetches_members_again(workspace):
    repository = JiraProject(existing={0}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    repository.move_on(existing={0, 1})
    repository.broken[1] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(repository)

    repository.broken.clear()
    repository.move_on(existing={0, 1})
    await workspace.sync(repository, force_full_sync=True)

    assert repository.fetched == ["jira-0", "jira-1"]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_successful_run_that_cannot_complete_keeps_the_cursor_and_removes_nothing(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    cursor = await _cursor(workspace)
    await _change_scope(workspace)

    project.move_on(existing={0, 1}, in_query={0})
    project.listing_error = RuntimeError("Jira search total changed during pagination")
    state = await workspace.sync(project)
    transition = await _transition(workspace)

    assert state.last_sync_status == "success"
    assert state.absence_check_skipped_reason == "scope_listing_failed"
    assert workspace.tombstoned() == []
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "the listing of the new scope failed"
    assert await _cursor(workspace) == cursor


@pytest.mark.asyncio
async def test_a_successful_run_that_cannot_complete_records_the_opening_time_when_no_cursor_is_left(workspace):
    """A credential change clears the cursor; a successful run still records one."""
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    await _change_scope(workspace)
    await workspace.db.reset_source_sync_cursor(SOURCE_ID)

    project.move_on(existing={0, 1}, in_query={0})
    project.listing_error = RuntimeError("Jira search total changed during pagination")
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.FAILED
    assert await _cursor(workspace) == OPENED_AT


@pytest.mark.asyncio
async def test_a_jql_change_completes_by_its_listing_and_removes_what_the_new_query_leaves_out(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    await _change_scope(workspace)

    # jira-1 is in the removed project; it still exists, but the user no
    # longer asks for it.
    project.move_on(existing={0, 1}, in_query={0})
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert project.seen_since[-1] is None
    assert project.listings == 2, "one listing per run"
    assert project.confirmed == []
    assert workspace.tombstoned() == ["jira-1"]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED
    assert await workspace.held() == {"jira-0"}


@pytest.mark.asyncio
async def test_a_device_manifest_completes_a_transition_and_removes_what_it_does_not_list(workspace):
    repository = DeviceManifest(existing={0, 1})
    first = await workspace.sync(repository, manifest={0, 1})
    assert first.last_sync_status == "success"
    workspace.engine.events.clear()
    await _change_scope(workspace)

    repository.move_on(existing={0, 2})
    state = await workspace.sync(repository, manifest={0, 2})

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-1"]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED
