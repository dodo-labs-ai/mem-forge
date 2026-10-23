"""A Notion Source synced end to end: the real Gene against the fake workspace, through the sync orchestrator."""

from __future__ import annotations

import json
from weakref import WeakValueDictionary

import pytest

from memforge import runtime
from memforge.genes import notion_gene
from memforge.genes.notion_gene import NotionGene
from memforge.pipeline.sync import SourceSyncMode
from memforge.source_artifacts import SOURCE_ARTIFACT_OBSERVATION_TYPE
from memforge.source_projection import (
    ProjectionCoverage,
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
    UnitRemovalReason,
)
from memforge.source_projection_config import canonical_projection_scope
from memforge.storage.database import Database
from tests.connector_http_fixture import FakeConnectorHttp
from tests.notion_fake import TOKEN, NotionFake
from tests.test_document_absence import Workspace
from tests.test_sync_bookkeeping import (
    NoopMemoryEngine,
    StubDocumentStore,
    _OrderRecordingMemoryEngine,
    _valid_png_bytes,
)

SOURCE_ID = "src-notion-sync"
TRANSITION_ID = "scope-transition-drop-a-root"


class _CommitRecordingEngine(_OrderRecordingMemoryEngine):
    """Records the projection each Document last committed, as the sync handed it over."""

    def __init__(self) -> None:
        super().__init__(defer=False)
        self.committed: dict[str, object] = {}

    async def prepare_and_commit_projected_lifecycle(self, **kwargs):
        self.committed[kwargs["doc_id"]] = kwargs["projection"]
        return await super().prepare_and_commit_projected_lifecycle(**kwargs)


@pytest.fixture
async def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())
    database = Database(str(tmp_path / "notion.db"), document_store=StubDocumentStore())
    await database.connect()
    NoopMemoryEngine.db = database
    workspace = Workspace(database)
    workspace.engine = _CommitRecordingEngine()
    try:
        yield workspace
    finally:
        NoopMemoryEngine.db = None
        await database.close()


async def _configure(workspace: Workspace, config: dict) -> None:
    await workspace.db.upsert_source(
        id=SOURCE_ID,
        type="notion",
        name="Notion",
        config_json=json.dumps(config),
        access_policy="workspace",
        owner_user_id="dev",
    )


async def _sync(workspace: Workspace, fake: NotionFake, config: dict, **options):
    """One run of the Source against ``fake``; every client the run builds, its excluded scopes' included, is the fake's."""
    with FakeConnectorHttp(fake.handle).serving(notion_gene):
        return await workspace.orchestrator().sync_gene(
            gene=NotionGene(dict(config), SOURCE_ID), source_name="Notion", source_id=SOURCE_ID, **options
        )


def _projection(workspace: Workspace, page_id: str):
    """The projection the page's Document last committed."""
    return workspace.engine.committed[f"notion-{page_id}"]


def _contents(projection) -> dict[str, str]:
    """Each Observation's current content, by Observation type (one Observation per type)."""
    contents = {revision.observation_id: revision.content for revision in projection.observation_revisions}
    return {observation.observation_type: contents[observation.id] for observation in projection.observations}


def _row_with_status(fake: NotionFake, status: str) -> tuple[str, str, str]:
    """A root page holding a database whose one row has a ``Status`` property; returns (root, database, row)."""
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "Release checklist")
    fake.pages[row]["properties"]["Status"] = {"id": "st", "type": "status", "status": {"name": status}}
    return root, database, row


@pytest.mark.asyncio
async def test_reprocess_rediscovers_and_refetches_a_partial_row(workspace) -> None:
    fake = NotionFake()
    root, database, row = _row_with_status(fake, "Done")
    # Notion says the body is truncated and names no block to read again: the row is partial.
    fake.set_markdown(row, "Ship after the review.", truncated=True)
    config = {"token": TOKEN, "root_pages": [root]}
    await _configure(workspace, config)
    assert (await _sync(workspace, fake, config)).last_sync_status == "success"
    fake.requests.clear()

    state = await _sync(
        workspace,
        fake,
        config,
        execution_mode=SourceSyncMode.REPROCESS,
        reprocess_doc_ids=frozenset({f"notion-{row}"}),
    )

    assert state.last_sync_status == "success"
    # Rediscovery read the row and climbed to its database; the fetch read the row and its body again.
    assert fake.count("GET", f"/v1/pages/{row}") == 2
    assert fake.count("GET", f"/v1/databases/{database}") == 1
    assert fake.count("GET", f"/v1/pages/{row}/markdown") == 1
    projection = _projection(workspace, row)
    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION
    properties = json.loads(_contents(projection)["properties"])["properties"]
    assert {"name": "Status", "type": "status", "value": "Done"} in properties


@pytest.mark.asyncio
async def test_a_row_with_an_empty_body_commits_with_its_properties(workspace) -> None:
    fake = NotionFake()
    root, _database, row = _row_with_status(fake, "In progress")
    config = {"token": TOKEN, "root_pages": [root]}
    await _configure(workspace, config)

    state = await _sync(workspace, fake, config)

    assert state.last_sync_status == "success"
    assert state.docs_failed == 0
    projection = _projection(workspace, row)
    assert projection.coverage is ProjectionCoverage.COMPLETE_SNAPSHOT
    contents = _contents(projection)
    assert contents["page_body"] == ""
    assert {"name": "Status", "type": "status", "value": "In progress"} in json.loads(contents["properties"])["properties"]


@pytest.mark.asyncio
async def test_a_notion_page_with_a_file_commits_with_its_artifact(workspace) -> None:
    fake = NotionFake()
    root = fake.add_page("Architecture")
    url = fake.add_file("diagram.png", _valid_png_bytes())
    fake.set_markdown(root, f"The services talk through one queue.\n\n![Diagram]({url})")
    config = {"token": TOKEN, "root_pages": [root]}
    await _configure(workspace, config)

    state = await _sync(workspace, fake, config)

    assert state.last_sync_status == "success"
    projection = _projection(workspace, root)
    observations = {observation.id: observation for observation in projection.observations}
    (artifact,) = [
        revision
        for revision in projection.observation_revisions
        if observations[revision.observation_id].observation_type == SOURCE_ARTIFACT_OBSERVATION_TYPE
    ]
    parent = observations[artifact.metadata["source_artifact"]["parent_observation_id"]]
    assert (parent.observation_type, parent.provider_key) == ("page_body", f"{root}:body")
    assert await workspace.db.get_source_artifact_revision(artifact.id) is not None
    body = _contents(projection)["page_body"]
    assert "notion-file:" in body
    assert "X-Amz-" not in json.dumps([body, artifact.metadata])


class _DropRootB:
    """Roots A and B, synced, then a scope change to A alone, as the user saves it."""

    def __init__(self) -> None:
        self.fake = NotionFake()
        self.a = self.fake.add_page("A")
        self.a_child = self.fake.add_page("A child", parent=("page", self.a))
        self.toggle = self.fake.add_block(self.a, "toggle")
        self.in_toggle = self.fake.add_page("In A's toggle", parent=("block", self.toggle))
        self.b = self.fake.add_page("B")
        self.b_child = self.fake.add_page("B child", parent=("page", self.b))
        self.before = {"token": TOKEN, "root_pages": [self.a, self.b]}
        self.after = {"token": TOKEN, "root_pages": [self.a]}

    async def synced_then_changed(self, workspace: Workspace) -> None:
        await _configure(workspace, self.before)
        assert (await _sync(workspace, self.fake, self.before)).last_sync_status == "success"
        workspace.engine.events.clear()
        await _configure(workspace, self.after)
        await workspace.db.create_projection_scope_transition(
            ProjectionScopeTransition(
                id=TRANSITION_ID,
                source_id=SOURCE_ID,
                previous_scope=canonical_projection_scope("notion", self.before),
                target_scope=canonical_projection_scope("notion", self.after),
                created_at="2026-10-20T09:00:00+00:00",
            )
        )

    async def sync_after(self, workspace: Workspace):
        return await _sync(
            workspace,
            self.fake,
            self.after,
            excluded_scope_gene=runtime.excluded_scope_gene_factory({"id": SOURCE_ID, "type": "notion"}, self.after),
        )

    def docs(self, *page_ids: str) -> set[str]:
        return {f"notion-{page_id}" for page_id in page_ids}


@pytest.mark.asyncio
async def test_a_transient_404_during_a_scope_transition_removes_nothing(workspace) -> None:
    tree = _DropRootB()
    await tree.synced_then_changed(workspace)
    # The kept toggle's children answer 404 once, while the toggle itself still reads.
    tree.fake.fail("GET", f"/v1/blocks/{tree.toggle}/children", 404)

    state = await tree.sync_after(workspace)

    assert state.last_sync_status == "failed"
    assert workspace.tombstoned() == []
    assert await workspace.db.list_indexed_doc_ids(SOURCE_ID) == tree.docs(tree.a, tree.a_child, tree.in_toggle, tree.b, tree.b_child)
    transition = await workspace.db.get_projection_scope_transition(TRANSITION_ID)
    assert transition.status is not ProjectionScopeTransitionStatus.APPLIED


@pytest.mark.asyncio
async def test_a_404_the_read_by_id_confirms_lets_the_scope_transition_complete(workspace) -> None:
    tree = _DropRootB()
    await tree.synced_then_changed(workspace)
    # The toggle was deleted with its page during the walk; A's children still name it.
    for path in (f"/v1/blocks/{tree.toggle}/children", f"/v1/blocks/{tree.toggle}"):
        tree.fake.fail("GET", path, 404, times=2)  # the new scope's walk, and the excluded scope's
    tree.fake.pages[tree.in_toggle]["in_trash"] = True

    state = await tree.sync_after(workspace)

    assert state.last_sync_status == "success"
    transition = await workspace.db.get_projection_scope_transition(TRANSITION_ID)
    assert transition.status is ProjectionScopeTransitionStatus.APPLIED
    assert await workspace.db.list_indexed_doc_ids(SOURCE_ID) == tree.docs(tree.a, tree.a_child)
    for removed in (tree.b, tree.b_child):
        assert await _removal_reason(workspace, removed) == UnitRemovalReason.REMOVED_FROM_SCOPE.value
    assert await _removal_reason(workspace, tree.in_toggle) == UnitRemovalReason.ABSENT_FROM_SOURCE.value


async def _removal_reason(workspace: Workspace, page_id: str) -> str | None:
    unit = await workspace.db.find_source_unit_by_document_id(SOURCE_ID, f"notion-{page_id}")
    assert unit is not None
    return unit.locator.get("tombstone_reason")
