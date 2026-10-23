"""A document Source's scope transition completes once every Document its new scope lists has committed under it.

Commitment accumulates across the transition's runs as member rows. Each run
fetches only the listed Documents that are not members yet, and removes
nothing until the run that completes the transition, which removes what the
new scope leaves out and sets the cursor to the time the transition opened.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.memory.lifecycle_plan import ProjectedLifecycleBlocker, ProjectedLifecycleDeferredError
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge import runtime
from memforge.genes.jira_gene import JiraQueryRejected
from memforge.local_agent.document_identity import build_jira_doc_id
from memforge.local_agent.snapshot_manifest import (
    ExcludedScopeListing,
    SnapshotConfirmation,
    SnapshotConfirmationAnswer,
)
from memforge.local_agent.source_contract import LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD, local_agent_sync_job_payload
from memforge.models import ScopeListing, ScopeListingKind
from memforge.pipeline.collection_family import DEFAULT_INCREMENTAL_SYNC_OVERLAP, SnapshotBinding
from memforge.source_projection import (
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
    UnitRemovalReason,
)
from memforge.source_projection_config import canonical_projection_scope, projection_scope_fingerprint
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from tests.llm_fixture import AdmittingClient
from tests.memory_store_fixture import OrchestratorMemoryStore
from tests.test_document_absence import (
    JIRA_CONFIG,
    SOURCE_ID,
    DeviceJira,
    DeviceManifest,
    JiraProject,
    Workspace,
    _synced,
)
from tests.test_sync_bookkeeping import NoopMemoryEngine, _OrderRecordingMemoryEngine

ABSENT = SnapshotConfirmationAnswer.ABSENT
EXISTS = SnapshotConfirmationAnswer.EXISTS
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


async def _change_scope(workspace, *, opened_at: datetime = OPENED_AT) -> None:
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
            created_at=opened_at.isoformat(),
        )
    )


async def _transition(workspace) -> ProjectionScopeTransition:
    transition = await workspace.db.get_projection_scope_transition(TRANSITION_ID)
    assert transition is not None
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
    assert transition.listed_count == 2
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
    engine.blocker_unit_ids = (await workspace.unit_id("jira-1"),)
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
    engine.blocker_unit_ids = (await workspace.unit_id("jira-1"),)
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


class ExcludedScopes:
    """Lists each scope a transition excluded, by its JQL, as Jira holds it; records every scope it lists."""

    def __init__(self, listings: dict[str, set[int] | Exception]) -> None:
        self.listings = listings
        self.listed: list[dict[str, object]] = []

    @asynccontextmanager
    async def __call__(self, scope):
        self.listed.append(dict(scope))
        yield _ExcludedGene(self.listings[str(scope["jql"])])


class _ExcludedGene:
    def __init__(self, listing: set[int] | Exception) -> None:
        self.listing = listing

    async def list_scope(self):
        if isinstance(self.listing, Exception):
            raise self.listing
        return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset(f"jira-{n}" for n in self.listing))


OLD_JQL = JIRA_CONFIG["jql"]
NEW_JQL = TARGET_CONFIG["jql"]


async def _issues_that_left_the_query_before_the_change(workspace) -> JiraProject:
    """jira-0 and jira-1 match the old JQL; jira-2 and jira-3 left it before the user changed it, and still exist."""
    project = JiraProject(existing={0, 1, 2, 3})
    await _synced(workspace, project)
    project.move_on(existing={0, 1, 2, 3}, in_query={0, 1})
    await workspace.sync(project)
    assert workspace.tombstoned() == []
    return project


@pytest.mark.asyncio
async def test_a_jql_change_removes_only_what_the_old_query_lists_and_the_new_one_does_not(workspace):
    project = await _issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)
    excluded = ExcludedScopes({OLD_JQL: {0, 1}})

    # The new JQL drops jira-1; jira-3 has since been deleted.
    project.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(project, excluded_scope_gene=excluded)

    assert state.last_sync_status == "success"
    assert project.seen_since[-1] is None
    assert excluded.listed == [canonical_projection_scope("jira", JIRA_CONFIG)]
    assert sorted(workspace.tombstoned()) == ["jira-1", "jira-3"]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await _removal_reason(workspace, "jira-3") == UnitRemovalReason.ABSENT_FROM_SOURCE.value
    # jira-2 had left the old query before the change and still exists.
    assert await workspace.held() == {"jira-0", "jira-2"}
    assert sorted(project.confirmed) == ["jira-2", "jira-3"]
    assert state.unlisted_present_count == 1
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


async def _superseded(
    workspace, transition_id: str, target_jql: str, predecessor: str | None, *, source_id: str = SOURCE_ID,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await workspace.db.db.execute(
        """INSERT INTO projection_scope_transitions (
               id, source_id, previous_scope_json, target_scope_json, status,
               predecessor_transition_id, created_at, updated_at, completed_at
           ) VALUES (?, ?, ?, ?, 'superseded', ?, ?, ?, ?)""",
        (
            transition_id,
            source_id,
            json.dumps(canonical_projection_scope("jira", JIRA_CONFIG)),
            json.dumps({"query_mode": "advanced", "jql": target_jql}),
            predecessor,
            now,
            now,
            now,
        ),
    )
    await workspace.db.db.commit()


@pytest.mark.asyncio
async def test_excluded_scopes_are_listed_once_each_and_only_by_the_run_that_can_complete(workspace):
    project = JiraProject(existing={0, 1, 2, 3})
    await _synced(workspace, project)
    # The open change supersedes one to "project = OPS", which superseded one
    # to the new JQL, which superseded another to "project = OPS".
    await _superseded(workspace, "superseded-0", "project = OPS", None)
    await _superseded(workspace, "superseded-1", NEW_JQL, "superseded-0")
    await _superseded(workspace, "superseded-2", "project = OPS", "superseded-1")
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
            predecessor_transition_id="superseded-2",
            created_at=OPENED_AT.isoformat(),
        )
    )
    excluded = ExcludedScopes({OLD_JQL: {0, 1}, "project = OPS": {2}})

    project.move_on(existing={0, 1, 2, 3}, in_query={0, 3})
    project.broken[3] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(project, excluded_scope_gene=excluded)
    assert excluded.listed == []
    assert workspace.tombstoned() == []

    project.broken.clear()
    project.move_on(existing={0, 1, 2, 3}, in_query={0, 3})
    await workspace.sync(project, excluded_scope_gene=excluded)

    assert [scope["jql"] for scope in excluded.listed] == [OLD_JQL, "project = OPS"]
    assert sorted(workspace.tombstoned()) == ["jira-1", "jira-2"]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_transient_failure_listing_an_excluded_scope_keeps_the_transition_open(workspace):
    project = await _issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)
    excluded = ExcludedScopes({OLD_JQL: RuntimeError("Jira answered 503")})

    project.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(project, excluded_scope_gene=excluded)
    transition = await _transition(workspace)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == []
    assert project.confirmed == []
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "listing a scope the change excluded failed"


@pytest.mark.asyncio
async def test_a_rejected_excluded_jql_completes_the_transition_with_the_ordinary_rule(workspace):
    project = await _issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)
    excluded = ExcludedScopes({OLD_JQL: JiraQueryRejected("Jira rejected the JQL")})

    project.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(project, excluded_scope_gene=excluded)

    # Nothing is known to have left with the change: what still exists is kept.
    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-3"]
    assert await _removal_reason(workspace, "jira-3") == UnitRemovalReason.ABSENT_FROM_SOURCE.value
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_query_completion_lists_the_excluded_scopes_before_judging_a_blocked_document(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    engine.blocker_unit_ids = (await workspace.unit_id("jira-1"),)
    workspace.engine = engine
    excluded = ExcludedScopes({OLD_JQL: {0, 1}})

    # The new JQL drops jira-1, still in Jira, and adds jira-2, whose commit waits on jira-1's Unit.
    project.move_on(existing={0, 1, 2}, in_query={0, 2})
    state = await workspace.sync(project, excluded_scope_gene=excluded)

    assert state.last_sync_status == "success"
    assert len(excluded.listed) == 1
    assert engine.events == [("tombstone", "jira-1"), ("commit", "jira-2")]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_an_excluded_scope_that_cannot_be_built_completes_the_transition_with_the_ordinary_rule(
    workspace, monkeypatch,
):
    project = await _issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)

    def unbuildable(*, name, config, source_id):
        raise ValueError("Jira JQL is required in advanced mode")

    monkeypatch.setattr(runtime, "create_gene", unbuildable)
    project.move_on(existing={0, 1, 2}, in_query={0})
    await workspace.sync(
        project,
        excluded_scope_gene=runtime.excluded_scope_gene_factory({"id": SOURCE_ID, "type": "jira"}, TARGET_CONFIG),
    )

    assert workspace.tombstoned() == ["jira-3"]
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_failed_confirmation_still_completes_with_the_excluded_scope_removals(workspace):
    project = await _issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)

    project.move_on(existing={0, 1, 2}, in_query={0})
    project.confirm_error = RuntimeError("Jira answered 503")
    state = await workspace.sync(project, excluded_scope_gene=ExcludedScopes({OLD_JQL: {0, 1}}))

    # The later ordinary runs confirm jira-2 and jira-3 again.
    assert workspace.tombstoned() == ["jira-1"]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert state.absence_check_skipped_reason == "absence_confirmation_failed"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


async def _open_transition_superseding(workspace, predecessor: str) -> None:
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
            predecessor_transition_id=predecessor,
            created_at=OPENED_AT.isoformat(),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("chain", ["cycle", "missing", "other-source"])
async def test_a_superseded_chain_that_cannot_be_walked_keeps_the_transition_open(workspace, chain):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    if chain == "cycle":
        await _superseded(workspace, "superseded-a", "project = OPS", "superseded-b")
        await _superseded(workspace, "superseded-b", "project = SEC", "superseded-a")
    elif chain == "other-source":
        await workspace.db.upsert_source(
            id="src-other", type="jira", name="Other", config_json=json.dumps(JIRA_CONFIG),
            access_policy="workspace", owner_user_id="dev",
        )
        await _superseded(workspace, "superseded-a", "project = OPS", None, source_id="src-other")
    await _open_transition_superseding(workspace, "superseded-a")
    excluded = ExcludedScopes({OLD_JQL: {0, 1}, "project = OPS": set(), "project = SEC": set()})

    project.move_on(existing={0, 1}, in_query={0})
    await asyncio.wait_for(workspace.sync(project, excluded_scope_gene=excluded), timeout=10)
    transition = await _transition(workspace)

    assert workspace.tombstoned() == []
    assert excluded.listed == []
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    expected = "loop back to superseded-a" if chain == "cycle" else "superseded transition superseded-a is missing"
    assert expected in (transition.error or "")


@pytest.mark.asyncio
async def test_a_document_blocked_by_a_unit_completion_cannot_remove_lists_no_excluded_scope(workspace):
    project = JiraProject(existing={0, 1})
    await _synced(workspace, project)
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    engine.blocker_unit_ids = ("unit-of-another-source",)
    workspace.engine = engine
    excluded = ExcludedScopes({OLD_JQL: {0, 1}})

    project.move_on(existing={0, 1, 2}, in_query={0, 2})
    await workspace.sync(project, excluded_scope_gene=excluded)

    assert excluded.listed == []
    assert project.confirmed == []
    assert workspace.tombstoned() == []
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.FAILED


@pytest.mark.asyncio
async def test_a_device_query_transition_stays_open_while_its_manifest_lacks_an_excluded_listing(workspace):
    project = DeviceJira(existing={0, 1})
    first = await workspace.sync(project, manifest={0, 1})
    assert first.last_sync_status == "success"
    workspace.engine.events.clear()
    await _change_scope(workspace)

    # The job carried no excluded scope: the previous run had not run yet.
    project.move_on(existing={0, 1}, in_query={0})
    await workspace.sync(project, manifest={0}, answers={1: EXISTS})
    transition = await _transition(workspace)

    assert workspace.tombstoned() == []
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "the device has not listed every scope the change excluded"


@pytest.mark.asyncio
async def test_a_device_transition_stays_open_while_one_of_its_excluded_scopes_is_unlisted(workspace):
    device = await _device_issues_that_left_the_query_before_the_change(workspace)
    await _superseded(workspace, "superseded-ops", "project = OPS", None)
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
            predecessor_transition_id="superseded-ops",
            created_at=OPENED_AT.isoformat(),
        )
    )

    # The job carried the old JQL and "project = OPS"; the device listed only the first.
    device.move_on(existing={0, 1, 2, 3}, in_query={0})
    await workspace.sync(
        device,
        manifest={0},
        answers={1: EXISTS, 2: EXISTS, 3: EXISTS},
        excluded_listings=(_device_listing(OLD_JQL, 0, 1),),
    )
    transition = await _transition(workspace)

    assert workspace.tombstoned() == []
    assert transition.status is ProjectionScopeTransitionStatus.FAILED
    assert transition.error == "the device has not listed every scope the change excluded"


@pytest.mark.asyncio
async def test_a_device_transition_blocked_only_by_what_completion_removes_completes_on_the_next_job(workspace):
    device = DeviceJira(existing={0, 1})
    await workspace.sync(device, manifest={0, 1})
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    engine.blocker_unit_ids = (await workspace.unit_id("jira-1"),)
    workspace.engine = engine

    # The new JQL drops jira-1, still in Jira, and adds jira-2, whose commit
    # waits on jira-1's Unit. The job carried no excluded scope yet.
    device.move_on(existing={0, 1, 2}, in_query={0, 2})
    await workspace.sync(device, manifest={0, 2}, answers={1: EXISTS})
    waiting = await _transition(workspace)
    payload = await local_agent_sync_job_payload(workspace.db, await workspace.db.get_source(SOURCE_ID))
    engine.events.clear()
    await workspace.sync(
        device, manifest={0, 2}, answers={1: EXISTS}, excluded_listings=(_device_listing(OLD_JQL, 0, 1),)
    )

    # jira-2 is no member, so only the mark tells the next job to carry the scopes.
    assert waiting.status is ProjectionScopeTransitionStatus.FAILED
    assert waiting.excluded_scopes_due is True
    assert "jira-2" not in await _members(workspace)
    assert [scope["scope_key"] for scope in payload[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD]] == [
        _device_listing(OLD_JQL).scope_key
    ]
    assert sorted(engine.events) == [("commit", "jira-2"), ("tombstone", "jira-1")]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    completed = await _transition(workspace)
    assert (completed.status, completed.excluded_scopes_due) == (ProjectionScopeTransitionStatus.APPLIED, False)


def _device_listing(jql: str, *numbers: int, rejection: str | None = None) -> ExcludedScopeListing:
    """What the device listed in the excluded scope of this JQL, by the key its job named it."""
    return ExcludedScopeListing(
        scope_key=projection_scope_fingerprint(canonical_projection_scope("jira", {**JIRA_CONFIG, "jql": jql})),
        doc_ids=frozenset(f"jira-{number}" for number in numbers),
        rejection=rejection,
    )


async def _device_issues_that_left_the_query_before_the_change(workspace) -> DeviceJira:
    """jira-0 and jira-1 match the old JQL; jira-2 and jira-3 left it before the change, and still exist."""
    device = DeviceJira(existing={0, 1, 2, 3})
    await workspace.sync(device, manifest={0, 1, 2, 3})
    device.move_on(existing={0, 1, 2, 3}, in_query={0, 1})
    await workspace.sync(device, manifest={0, 1}, answers={2: EXISTS, 3: EXISTS})
    assert workspace.tombstoned() == []
    workspace.engine.events.clear()
    return device


@pytest.mark.asyncio
async def test_a_device_jql_change_removes_only_what_the_old_query_lists_and_the_new_one_does_not(workspace):
    device = await _device_issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)

    # The new JQL drops jira-1; jira-3 has since been deleted.
    device.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(
        device,
        manifest={0},
        answers={1: EXISTS, 2: EXISTS, 3: ABSENT},
        excluded_listings=(_device_listing(OLD_JQL, 0, 1),),
    )

    assert state.last_sync_status == "success"
    assert sorted(workspace.tombstoned()) == ["jira-1", "jira-3"]
    assert await _removal_reason(workspace, "jira-1") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await _removal_reason(workspace, "jira-3") == UnitRemovalReason.ABSENT_FROM_SOURCE.value
    # jira-2 had left the old query before the change and still exists.
    assert await workspace.held() == {"jira-0", "jira-2"}
    assert state.unlisted_present_count == 1
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    [
        "JiraQueryRejected: field 'x' does not exist",
        "The excluded scope lists 10001 issues, more than the 10000 a manifest holds",
    ],
    ids=["jql-rejected", "over-the-manifest-bound"],
)
async def test_a_device_jql_change_whose_old_query_cannot_be_listed_completes_with_the_ordinary_rule(
    workspace, rejection,
):
    device = await _device_issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)

    device.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(
        device,
        manifest={0},
        answers={1: EXISTS, 2: EXISTS, 3: ABSENT},
        excluded_listings=(_device_listing(OLD_JQL, rejection=rejection),),
    )

    # Nothing is removed from scope: only the issue Jira no longer has goes.
    assert workspace.tombstoned() == ["jira-3"]
    assert await _removal_reason(workspace, "jira-3") == UnitRemovalReason.ABSENT_FROM_SOURCE.value
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert state.unlisted_present_count == 2
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_device_jql_change_without_answers_removes_from_scope_and_records_why(workspace):
    device = await _device_issues_that_left_the_query_before_the_change(workspace)
    await _change_scope(workspace)

    # An agent that does not confirm lists the excluded scope but answers nothing.
    device.move_on(existing={0, 1, 2}, in_query={0})
    state = await workspace.sync(device, manifest={0}, excluded_listings=(_device_listing(OLD_JQL, 0, 1),))

    assert workspace.tombstoned() == ["jira-1"]
    assert await workspace.held() == {"jira-0", "jira-2", "jira-3"}
    assert state.absence_check_skipped_reason == "absence_confirmation_missing"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


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


@pytest.mark.asyncio
async def test_a_listed_document_blocked_by_a_unit_of_this_run_and_one_completion_removes_completes(workspace):
    repository = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    engine = _OrderRecordingMemoryEngine(defer=True)
    engine.deferred_doc_id = "jira-2"
    # jira-0 is unchanged and recorded by this run; jira-1 is left out of the new scope.
    engine.blocker_unit_ids = (await workspace.unit_id("jira-0"), await workspace.unit_id("jira-1"))
    workspace.engine = engine

    repository.move_on(existing={0, 2})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    assert engine.events == [("tombstone", "jira-1"), ("commit", "jira-2")]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_derivation_staged_before_the_transition_opened_is_fetched_again(workspace):
    """An edit made before the transition opened is older than the cursor its completion sets."""
    repository = JiraProject(existing={0}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    # jira-1's derivation is staged, but its commit fails for good in this run.
    workspace.engine = _OrderRecordingMemoryEngine(defer=True)
    workspace.engine.external_blocker = True
    repository.move_on(existing={0, 1}, changed={0, 1})
    repository.edited.add(0)
    staged = await workspace.sync(repository)
    assert {failed.doc_id for failed in staged.failed_docs} == {"jira-1"}
    await _change_scope(workspace, opened_at=datetime.now(timezone.utc))

    # The next run recovers jira-1's staged derivation, which records no
    # member, so discovery fetches jira-1 again under the transition.
    workspace.engine = _OrderRecordingMemoryEngine(defer=False)
    repository.move_on(existing={0, 1})
    state = await workspace.sync(repository)

    assert ("commit", "jira-1") in workspace.engine.events, "the staged derivation was recovered"
    assert "jira-1" in repository.fetched
    assert state.last_sync_status == "success"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_real_lifecycle_commit_records_its_member_only_when_its_deferred_commit_converges(workspace):
    repository = JiraProject(existing={0}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, repository)
    await _change_scope(workspace)
    jira_0_unit_id = await workspace.unit_id("jira-0")
    apply = workspace.db.apply_source_projection_lifecycle
    commits: list[tuple[str, str | None, tuple[str, ...]]] = []

    async def deferring_apply(projection, plan, **kwargs):
        doc_id = kwargs["unit_input"].document_id if kwargs.get("unit_input") is not None else None
        members = await workspace.db.list_projection_scope_transition_member_doc_ids(TRANSITION_ID)
        commits.append((doc_id, kwargs.get("scope_transition_id"), members))
        if doc_id == "jira-1" and len(commits) == 1:
            # jira-1's plan consumed Support of jira-0's Unit, which this run records afresh.
            raise ProjectedLifecycleDeferredError(
                "same-run cross-Unit Support is stale",
                [
                    ProjectedLifecycleBlocker(
                        memory_id="mem-shared",
                        source_unit_id=jira_0_unit_id,
                        evidence_unit_id="evidence-shared",
                        supported_unit_revision_id="revision-old",
                        current_unit_revision_id="revision-new",
                    )
                ],
            )
        await apply(projection, plan, **kwargs)

    workspace.db.apply_source_projection_lifecycle = deferring_apply
    memory_store = OrchestratorMemoryStore(workspace.db)
    adapters = build_sqlite_adapters(workspace.db, object())
    workspace.engine = MemoryEngine(
        cross_document_candidates=CrossDocumentCandidateRetriever(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
        ),
        db=workspace.db,
        memory_store=memory_store,
        structured_llm_client=AdmittingClient(),
    )

    repository.move_on(existing={0, 1})
    state = await workspace.sync(repository)

    assert state.last_sync_status == "success"
    # The deferred commit wrote no member; its convergence carried the
    # transition from the prepared commit and wrote it.
    assert commits == [
        ("jira-1", TRANSITION_ID, ("jira-0",)),
        ("jira-1", TRANSITION_ID, ("jira-0",)),
    ]
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


SUPERSEDED_CONFIG = {"query_mode": "advanced", "jql": "project = OPS"}
SUPERSEDED_ID = "scope-transition-to-ops"


async def _open_change(workspace, transition_id: str, target_config, *, superseding: str | None = None) -> None:
    """The user saves a scope change from the last applied scope, superseding the open transition when one is named."""
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(target_config),
        access_policy="workspace",
        owner_user_id="dev",
        projection_scope_transition=ProjectionScopeTransition(
            id=transition_id,
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("jira", JIRA_CONFIG),
            target_scope=canonical_projection_scope("jira", target_config),
            predecessor_transition_id=superseding,
            created_at=OPENED_AT.isoformat(),
        ),
        expected_open_transition_id=superseding,
    )


async def _a_superseded_change_that_added_jira_2(workspace, listing_kind: ScopeListingKind) -> JiraProject:
    """The change to OPS commits jira-2, which only OPS selects, and stays open because jira-1 fails."""
    project = JiraProject(existing={0, 1}, listing_kind=listing_kind)
    await _synced(workspace, project)
    await _open_change(workspace, SUPERSEDED_ID, SUPERSEDED_CONFIG)
    project.move_on(existing={0, 1, 2}, in_query={0, 1, 2})
    project.broken[1] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(project)
    assert await workspace.held() == {"jira-0", "jira-1", "jira-2"}
    assert await workspace.db.list_projection_scope_transition_member_doc_ids(SUPERSEDED_ID) == ("jira-0", "jira-2")

    await _open_change(workspace, TRANSITION_ID, TARGET_CONFIG, superseding=SUPERSEDED_ID)
    project.broken.clear()
    return project


@pytest.mark.asyncio
async def test_a_superseding_existence_change_removes_at_completion_what_only_the_superseded_target_added(workspace):
    project = await _a_superseded_change_that_added_jira_2(workspace, ScopeListingKind.EXISTENCE)
    superseded = await workspace.db.get_projection_scope_transition(SUPERSEDED_ID)
    transition = await _transition(workspace)

    assert superseded.status is ProjectionScopeTransitionStatus.SUPERSEDED
    assert await workspace.db.list_projection_scope_transition_member_doc_ids(SUPERSEDED_ID) == ()
    assert transition.predecessor_transition_id == SUPERSEDED_ID
    assert transition.previous_scope == canonical_projection_scope("jira", JIRA_CONFIG)

    # The new scope holds jira-0, jira-1 and jira-3; the superseding transition counts its own members.
    project.move_on(existing={0, 1, 3})
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-2"]
    assert await _removal_reason(workspace, "jira-2") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await workspace.held() == {"jira-0", "jira-1", "jira-3"}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_superseding_query_change_removes_at_completion_what_only_the_superseded_target_added(workspace):
    project = await _a_superseded_change_that_added_jira_2(workspace, ScopeListingKind.QUERY)
    excluded = ExcludedScopes({OLD_JQL: {0, 1}, SUPERSEDED_CONFIG["jql"]: {0, 1, 2}})

    # The new JQL selects jira-0, jira-1 and jira-3; jira-2 still exists in Jira.
    project.move_on(existing={0, 1, 2, 3}, in_query={0, 1, 3})
    state = await workspace.sync(project, excluded_scope_gene=excluded)

    assert state.last_sync_status == "success"
    assert [scope["jql"] for scope in excluded.listed] == [OLD_JQL, SUPERSEDED_CONFIG["jql"]]
    assert workspace.tombstoned() == ["jira-2"]
    assert await _removal_reason(workspace, "jira-2") == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await workspace.held() == {"jira-0", "jira-1", "jira-3"}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_return_to_the_last_applied_scope_removes_at_completion_what_the_superseded_target_added(workspace):
    project = await _a_superseded_change_that_added_jira_2_then_reverts(workspace)

    project.move_on(existing={0, 1})
    state = await workspace.sync(project)

    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == ["jira-2"]
    assert await workspace.held() == {"jira-0", "jira-1"}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


async def _a_superseded_change_that_added_jira_2_then_reverts(workspace) -> JiraProject:
    project = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)
    await _synced(workspace, project)
    await _open_change(workspace, SUPERSEDED_ID, SUPERSEDED_CONFIG)
    project.move_on(existing={0, 1, 2})
    project.broken[1] = ("fetch", ValueError("fetch failed"))
    await workspace.sync(project)
    project.broken.clear()
    # Back to the last applied scope: A→A, after the change to OPS.
    await _open_change(workspace, TRANSITION_ID, JIRA_CONFIG, superseding=SUPERSEDED_ID)
    transition = await _transition(workspace)
    assert transition.previous_scope == transition.target_scope
    assert transition.predecessor_transition_id == SUPERSEDED_ID
    return project


@pytest.mark.asyncio
async def test_a_transition_stored_before_a_scope_field_was_added_matches_its_configured_target(workspace):
    """A Jira target stored without ``query_mode`` reads the field's default, as the configuration does."""
    simple_config = {"projects": ["PAY"]}
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(simple_config),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION_ID,
            source_id=SOURCE_ID,
            previous_scope={"projects": ["OPS"]},
            target_scope={"projects": ["PAY"]},
            created_at=OPENED_AT.isoformat(),
        )
    )
    project = JiraProject(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)

    state = await workspace.sync(project)

    assert canonical_projection_scope("jira", simple_config) == {"projects": ["PAY"], "query_mode": "simple"}
    assert state.last_sync_status == "success"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


class SitePages(JiraProject):
    """The pages of a GitHub Pages subtree, listed as the repository's branch tree holds them."""

    @classmethod
    def metadata(cls):
        return replace(JiraProject.metadata(), name="github_pages", display_name="GitHub Pages")


@pytest.mark.asyncio
async def test_a_branch_pinned_transition_stored_before_branch_was_a_scope_field_runs_and_completes(workspace):
    """The stored target has no ``branch``; it reads the configured one, in both scopes."""
    site = "https://pages.example.test/org/site"
    pinned = {"sync_mode": "subtree", "root_url": f"{site}/guide", "auth_mode": "github_pat", "branch": "gh-pages"}
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="github_pages",
        name="Pages",
        config_json=json.dumps(pinned),
        access_policy="workspace",
        owner_user_id="dev",
    )
    await workspace.db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id=TRANSITION_ID,
            source_id=SOURCE_ID,
            previous_scope={"sync_mode": "subtree", "root_url": f"{site}/docs"},
            target_scope={"sync_mode": "subtree", "root_url": f"{site}/guide"},
            created_at=OPENED_AT.isoformat(),
        )
    )
    site_pages = SitePages(existing={0, 1}, listing_kind=ScopeListingKind.EXISTENCE)

    state = await workspace.sync(site_pages)

    assert state.last_sync_status == "success"
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED


class _DeviceIdJira(DeviceJira):
    """Device Jira whose Documents carry the ids the agent gives them, as ``build_jira_doc_id`` does."""

    def _item(self, number: int):
        return replace(super()._item(number), item_id=_device_doc(number))


def _device_doc(number: int) -> str:
    return build_jira_doc_id(source_id=SOURCE_ID, issue_key=f"PAY-{number}")


async def _device_run(workspace, device: DeviceJira, binding: SnapshotBinding):
    return await workspace.orchestrator().sync_gene(
        gene=device, source_name="Jira", source_id=SOURCE_ID, input_snapshot=binding
    )


@pytest.mark.asyncio
async def test_a_device_jql_change_completes_end_to_end_from_job_to_listing_to_manifest(workspace, monkeypatch):
    import memforge.genes.jira_gene as jira_gene
    import memforge.main as main

    device = _DeviceIdJira(existing={0, 1, 2})
    await _device_run(workspace, device, SnapshotBinding("snap-0", frozenset(map(_device_doc, (0, 1, 2)))))
    await _change_scope(workspace)
    # The Source reads Jira through the owner's agent and browser session.
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="jira",
        name="Jira",
        config_json=json.dumps(
            {
                **TARGET_CONFIG,
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": "local_agent",
            }
        ),
        access_policy="workspace",
        owner_user_id="dev",
    )
    # The new JQL lists PAY-0. The job carried no excluded scope, so the run
    # that could complete stays open and marks the transition.
    device.move_on(existing={0, 1, 2}, in_query={0})
    await _device_run(workspace, device, SnapshotBinding("snap-1", frozenset({_device_doc(0)})))
    workspace.engine.events.clear()

    # 1. The next job carries the old JQL.
    payload = await local_agent_sync_job_payload(workspace.db, await workspace.db.get_source(SOURCE_ID))

    # 2. The device lists it with its own session: PAY-0 and PAY-1 match the old JQL.
    built: list = []

    class OldQuery:
        def __init__(self, config, source_id):
            assert config["jql"] == OLD_JQL
            self.closed = False
            built.append(self)

        async def close(self):
            self.closed = True

        async def authenticate(self):
            return None

        async def list_scope(self):
            return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"jira-PAY-0", "jira-PAY-1"}))

    monkeypatch.setattr(jira_gene, "JiraGene", OldQuery)
    listed = await main._list_excluded_jira_scopes(payload, source_id=SOURCE_ID, jira_cookie="JSESSIONID=local")
    assert [gene.closed for gene in built] == [True]

    # 3. The manifest stores the listing, and the device answers for PAY-2, which left the query earlier.
    listings = tuple(
        ExcludedScopeListing(entry["scope_key"], frozenset(entry["doc_ids"])) for entry in listed
    )
    snapshot = {"source_id": SOURCE_ID, "workspace_id": "default", "snapshot_id": "snap-2"}
    await workspace.db.attach_source_sync_inputs_to_snapshot(
        **snapshot,
        memberships=[],
        coverage="complete_snapshot",
        manifest_count=1,
        manifest_sha256="b" * 64,
        manifest_items=[(_device_doc(0), "r1", "upsert")],
        local_agent_job_id="laj-2",
        local_agent_attempt_count=1,
        source_config_revision="rev-2",
        confirm=(SnapshotConfirmation(_device_doc(2), "100002"),),
        excluded_listings=listings,
    )
    await workspace.db.write_source_sync_snapshot_confirmations(
        **snapshot,
        manifest_sha256="b" * 64,
        local_agent_job_id="laj-2",
        local_agent_attempt_count=1,
        answers=(SnapshotConfirmation(_device_doc(2), "100002", EXISTS),),
    )

    # 4. The run bound to it completes the transition.
    state = await _device_run(
        workspace,
        device,
        SnapshotBinding(
            "snap-2",
            frozenset({_device_doc(0)}),
            await workspace.db.list_source_sync_snapshot_confirmations(**snapshot),
            await workspace.db.list_source_sync_snapshot_excluded_listings(**snapshot),
        ),
    )

    assert [entry["scope_key"] for entry in payload[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD]] == [
        _device_listing(OLD_JQL).scope_key
    ]
    assert listed == [{"scope_key": _device_listing(OLD_JQL).scope_key, "doc_ids": sorted(map(_device_doc, (0, 1)))}]
    assert state.last_sync_status == "success"
    assert workspace.tombstoned() == [_device_doc(1)]
    assert await _removal_reason(workspace, _device_doc(1)) == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await workspace.held() == {_device_doc(0), _device_doc(2)}
    assert (await _transition(workspace)).status is ProjectionScopeTransitionStatus.APPLIED
