"""ADR 0041: stored input belongs to the Source Unit revision, not to the shared Document row."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Memory, MemorySource, content_hash
from memforge.pipeline.stored_document import (
    StoredDocumentUnavailable,
    StoredDocumentUnavailableReason,
    load_stored_source_document,
)
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.source_activity import SourceActivityKind
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore
from memforge.storage.source_cleanup import SourceArtifactCleanupService
from tests.test_stored_input_currency import SPRINT_HISTORY, JiraProvider, authored, doc_id
from tests.test_sync_bookkeeping import (
    FailingMemoryExtractor,
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    RecordingMemoryEngine,
    _hold_document,
    _release_document,
    _unit_input,
)

TEAM = "src-jira-team"
RELEASE = "src-jira-release"
ISSUE = "PAY-1"
LEGACY_STORED_INPUT_COLUMNS = {
    "raw_content_uri": "TEXT",
    "raw_content_type": "TEXT",
    "normalized_content_uri": "TEXT",
    "pdf_content_uri": "TEXT",
}


class ReadRecordingDocumentStore(LocalDocumentStore):
    """A filesystem store that records which raw objects a run reads and deletes."""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.reads: list[str] = []
        self.deletions: list[str] = []

    def read_artifact(self, uri: str) -> bytes:
        self.reads.append(uri)
        return super().read_artifact(uri)

    def delete_artifact(self, uri: str) -> None:
        self.deletions.append(uri)
        super().delete_artifact(uri)


@dataclass
class Workspace:
    db: Database
    store: ReadRecordingDocumentStore
    provider: JiraProvider
    engine: RecordingMemoryEngine

    def orchestrator(self, extractor=None) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db, doc_store=self.store, memory_extractor=extractor or ProjectionFragmentRecordingExtractor(),
            memory_engine=self.engine, memory_store=RecordingDocumentDeleteMemoryStore(self.db), max_concurrent=1,
        )

    async def cleanup(self) -> int:
        return await SourceArtifactCleanupService(self.db, self.store).run_pending(limit=100)

    def stored(self, unit_input) -> bool:
        return all(
            self.store.get_artifact(uri, "application/octet-stream") is not None
            for uri in (unit_input.raw_content_uri, unit_input.normalized_content_uri)
        )

    async def add_source(self, source_id: str, *, collected_locally: bool) -> None:
        config = {"sync_mode": "local_agent"} if collected_locally else {}
        await self.db.upsert_source(
            id=source_id, type="jira", name=source_id, config_json=json.dumps(config),
            access_policy="workspace", owner_user_id="dev",
        )

    async def sync(self, source_id: str, *, force_full_sync: bool = False, extractor=None):
        state = await self.orchestrator(extractor).sync_gene(
            gene=self.provider, source_name="Jira", source_id=source_id, force_full_sync=force_full_sync,
        )
        if extractor is None:
            assert state.last_sync_status == "success"
        return state

    async def reprocess(self, source_id: str):
        self.store.reads.clear()
        return await self.orchestrator().sync_gene(
            gene=self.provider, source_name="Jira", source_id=source_id,
            execution_mode=SourceSyncMode.REPROCESS, reprocess_doc_ids=frozenset({doc_id(ISSUE)}),
        )

    async def changelog(self, source_id: str) -> set[str]:
        """The provider keys of the current changelog Observations of the Source's Unit."""
        unit = await self.db.find_source_unit_by_document_id(source_id, doc_id(ISSUE), current_only=True)
        projection = await self.db.get_current_source_unit_projection(unit.id)
        observations = {observation.id: observation for observation in projection.observations}
        return {
            observations[revision.observation_id].provider_key
            for revision in projection.observation_revisions
            if observations[revision.observation_id].observation_type == "changelog"
        }

    def stored_histories(self, uri: str) -> list[str]:
        return [history["id"] for history in json.loads(self.store.read_artifact(uri))["changelog"]["histories"]]


@pytest.fixture
async def workspace(tmp_path):
    store = ReadRecordingDocumentStore(str(tmp_path / "documents"))
    database = Database(str(tmp_path / "unit-input.db"), document_store=store)
    await database.connect()
    NoopMemoryEngine.db = database
    try:
        yield Workspace(
            db=database,
            store=store,
            provider=JiraProvider(),
            engine=RecordingMemoryEngine(),
        )
    finally:
        NoopMemoryEngine.db = None
        await database.close()


async def overlapping(workspace: Workspace, *, collected_locally: bool) -> None:
    """Two Sources include the same issue; the release Source syncs it after a changelog entry was added."""
    await workspace.add_source(TEAM, collected_locally=collected_locally)
    await workspace.add_source(RELEASE, collected_locally=collected_locally)
    await workspace.sync(TEAM)
    workspace.provider.histories[ISSUE].append(authored(SPRINT_HISTORY, "Ann"))
    await workspace.sync(RELEASE)


@pytest.mark.asyncio
async def test_overlapping_sources_keep_separate_stored_inputs(workspace):
    await overlapping(workspace, collected_locally=False)

    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    release = await _unit_input(workspace.db, RELEASE, doc_id(ISSUE))
    assert team.source_unit_id != release.source_unit_id
    assert team.raw_content_uri != release.raw_content_uri
    assert workspace.stored_histories(team.raw_content_uri) == ["9001"]
    assert workspace.stored_histories(release.raw_content_uri) == ["9001", "9002"]
    for stored in (team, release):
        unit = await workspace.db.get_current_source_unit_revision(stored.source_unit_id)
        assert stored.unit_revision_id == unit.id
    # The shared row describes the Document as the last Source to sync saw it; it owns nothing.
    assert (await workspace.db.get_document(doc_id(ISSUE))).source == RELEASE
    assert {stored.source_id for stored in await workspace.db.list_document_source_unit_inputs(doc_id(ISSUE))} == {
        TEAM,
        RELEASE,
    }
    assert await workspace.db.list_indexed_doc_ids(TEAM) == {doc_id("PAY-1"), doc_id("PAY-2")}
    assert await workspace.db.count_documents(source=TEAM) == len(workspace.provider.histories)


@pytest.mark.asyncio
async def test_a_rediscovering_source_reprocesses_a_document_whose_row_the_other_source_wrote(workspace):
    await overlapping(workspace, collected_locally=False)

    state = await workspace.reprocess(TEAM)

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert workspace.provider.rediscovered == [ISSUE]
    assert await workspace.changelog(TEAM) == {"9001", "9002"}


@pytest.mark.asyncio
async def test_a_stored_input_reprocess_reads_its_own_unit_revision_input(workspace):
    await overlapping(workspace, collected_locally=True)
    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))

    state = await workspace.reprocess(TEAM)

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert workspace.provider.rediscovered == []
    assert workspace.store.reads == [team.raw_content_uri]
    # The release Source's newer raw content never reaches the team Source's Unit.
    assert await workspace.changelog(TEAM) == {"9001"}
    assert await workspace.changelog(RELEASE) == {"9001", "9002"}


@pytest.mark.asyncio
async def test_removing_a_document_from_one_source_keeps_the_other_sources_input(workspace):
    await overlapping(workspace, collected_locally=False)
    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    release = await _unit_input(workspace.db, RELEASE, doc_id(ISSUE))

    await _release_document(workspace.db, source_id=RELEASE, source_type="jira", doc_id=doc_id(ISSUE))
    await workspace.db.delete_projected_document(doc_id(ISSUE), source_id=RELEASE)
    await SourceArtifactCleanupService(workspace.db, workspace.store).run_pending(limit=10)

    assert await _unit_input(workspace.db, TEAM, doc_id(ISSUE)) == team
    assert workspace.store.get_artifact(team.raw_content_uri, team.raw_content_type) is not None
    assert workspace.store.get_artifact(release.raw_content_uri, release.raw_content_type) is None
    # The team Source still holds the Document, so the row stays and names it.
    assert (await workspace.db.get_document(doc_id(ISSUE))).source == TEAM


@pytest.mark.asyncio
async def test_deleting_a_source_keeps_the_other_sources_stored_input(workspace):
    await overlapping(workspace, collected_locally=False)
    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))

    await workspace.db.delete_source_cascade(RELEASE)
    await SourceArtifactCleanupService(workspace.db, workspace.store).run_pending(limit=10)

    assert await _unit_input(workspace.db, TEAM, doc_id(ISSUE)) == team
    assert workspace.store.get_artifact(team.raw_content_uri, team.raw_content_type) is not None
    assert (await workspace.db.get_document(doc_id(ISSUE))).source == TEAM
    assert (await workspace.reprocess(TEAM)).docs_failed == 0


@pytest.mark.asyncio
async def test_content_links_read_the_input_of_the_supporting_sources_unit(workspace, tmp_path):
    await overlapping(workspace, collected_locally=False)
    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    release = await _unit_input(workspace.db, RELEASE, doc_id(ISSUE))
    await workspace.db.upsert_source(
        id=RELEASE, type="jira", name=RELEASE, config_json="{}",
        access_policy="private", owner_user_id="someone-else",
    )
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    memory = Memory(
        id="mem-team-issue",
        memory_type="fact",
        content="The payroll run issue belongs to the team board.",
        content_hash=content_hash("The payroll run issue belongs to the team board."),
        created_at=now,
        updated_at=now,
        status="active",
    )
    await workspace.db.insert_memory(memory)
    await workspace.db.restore_memory_source_snapshot(
        MemorySource(
            memory_id=memory.id, doc_id=doc_id(ISSUE), source_id=TEAM, source_type="jira",
            excerpt="payroll run", source_updated_at=now,
        )
    )
    config = AppConfig(base_dir=tmp_path / "mem")
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(
        db=workspace.db, config=config, document_store=workspace.store, principal_resolver=lambda request: "dev",
    )
    with TestClient(app) as client:
        detail = client.get(f"/api/v1/memories/{memory.id}")
        team_raw = client.get(f"/api/v1/source-units/{team.source_unit_id}/artifacts/raw_source")
        release_raw = client.get(f"/api/v1/source-units/{release.source_unit_id}/artifacts/raw_source")
        document_raw = client.get(f"/api/v1/documents/{doc_id(ISSUE)}/artifacts/raw_source")

    [evidence] = detail.json()["evidence"]
    assert evidence["document"]["content_url"] == f"/api/v1/source-units/{team.source_unit_id}/content"
    assert [history["id"] for history in team_raw.json()["changelog"]["histories"]] == ["9001"]
    # The release Source is private to another user: its copy is not served, by Unit or by Document.
    assert release_raw.status_code == 404
    assert document_raw.json() == team_raw.json()


@pytest.mark.asyncio
async def test_the_upgrade_gives_each_unit_the_input_its_own_source_wrote(tmp_path):
    path = tmp_path / "upgrade.db"
    store = ReadRecordingDocumentStore(str(tmp_path / "documents"))
    database = Database(str(path))
    await database.connect()
    NoopMemoryEngine.db = database
    workspace = Workspace(db=database, store=store, provider=JiraProvider(), engine=RecordingMemoryEngine())
    try:
        await overlapping(workspace, collected_locally=True)
        release = await _unit_input(database, RELEASE, doc_id(ISSUE))
        document = await database.get_document(doc_id(ISSUE))
        # The workspace as it was before the upgrade: the Document row holds the
        # stored input of the Source that wrote it last, and no Unit records any.
        for column, column_type in LEGACY_STORED_INPUT_COLUMNS.items():
            await database.db.execute(f"ALTER TABLE documents ADD COLUMN {column} {column_type}")
        await database.db.execute(
            """UPDATE documents SET raw_content_uri = ?, raw_content_type = ?, normalized_content_uri = ?
                WHERE doc_id = ?""",
            (release.raw_content_uri, release.raw_content_type, release.normalized_content_uri, doc_id(ISSUE)),
        )
        await database.db.execute("DELETE FROM source_unit_inputs")
        await database.db.execute("DELETE FROM schema_migrations WHERE version = 104")
        await database.db.commit()
    finally:
        await database.close()

    for _ in range(2):
        upgraded = Database(str(path))
        await upgraded.connect()
        try:
            recorded = await _unit_input(upgraded, RELEASE, doc_id(ISSUE))
            release_unit = await upgraded.find_source_unit_by_document_id(RELEASE, doc_id(ISSUE))
            assert recorded.unit_revision_id == (await upgraded.get_current_source_unit_revision(release_unit.id)).id
            assert (recorded.raw_content_uri, recorded.normalized_content_uri) == (
                release.raw_content_uri,
                release.normalized_content_uri,
            )
            assert recorded.raw_content_sha256 is None
            assert recorded.normalized_content_hash == document.content_hash
            assert recorded.item.extra == document.item_extra
            # The team Unit's Document row was overwritten by the release Source, so it gets no input.
            assert await _unit_input(upgraded, TEAM, doc_id(ISSUE)) is None
            with pytest.raises(StoredDocumentUnavailable) as missing:
                await load_stored_source_document(
                    upgraded, store, workspace.provider, source_id=TEAM, document_id=doc_id(ISSUE),
                )
            assert missing.value.reason is StoredDocumentUnavailableReason.RAW_CONTENT_MISSING
            columns = {row[1] for row in await upgraded.db.execute_fetchall("PRAGMA table_info(documents)")}
            assert columns.isdisjoint(LEGACY_STORED_INPUT_COLUMNS)
        finally:
            await upgraded.close()



@pytest.mark.asyncio
async def test_a_renamed_document_releases_the_objects_its_previous_input_named(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    await workspace.sync(TEAM)
    before = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    discovered_item = workspace.provider._item

    def renamed(issue_key):
        item = discovered_item(issue_key)
        item.title = f"{issue_key}: Payroll run renamed"
        return item

    workspace.provider._item = renamed
    await workspace.sync(TEAM)
    await SourceArtifactCleanupService(workspace.db, workspace.store).run_pending(limit=10)

    after = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    assert after.raw_content_uri != before.raw_content_uri
    assert workspace.store.get_artifact(after.raw_content_uri, after.raw_content_type) is not None
    assert workspace.store.get_artifact(before.raw_content_uri, before.raw_content_type) is None
    assert workspace.store.get_artifact(before.normalized_content_uri, "text/markdown") is None


def _renamed(provider: JiraProvider, suffix: str):
    discovered_item = provider._item

    def renamed(issue_key):
        item = discovered_item(issue_key)
        item.title = f"{issue_key}: Payroll run {suffix}"
        return item

    return renamed


@pytest.mark.asyncio
async def test_a_document_renamed_back_before_cleanup_keeps_its_current_objects(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    await workspace.sync(TEAM)
    original = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    discovered_item = workspace.provider._item
    workspace.provider._item = _renamed(workspace.provider, "renamed")
    await workspace.sync(TEAM)
    renamed = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    workspace.provider._item = discovered_item
    # The original keys are written again and named by the current input
    # while their release from the rename is still queued.
    await workspace.sync(TEAM)

    await workspace.cleanup()

    current = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    assert current.raw_content_uri == original.raw_content_uri
    assert workspace.stored(current)
    assert not workspace.store.get_artifact(renamed.raw_content_uri, renamed.raw_content_type)
    assert await workspace.db.list_source_artifact_cleanup_tasks(limit=10) == []


@pytest.mark.asyncio
async def test_a_removed_document_that_returns_before_cleanup_keeps_its_objects(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    await workspace.sync(TEAM)
    workspace.provider.removed.add(ISSUE)
    await workspace.sync(TEAM)
    assert await _unit_input(workspace.db, TEAM, doc_id(ISSUE)) is None
    assert await workspace.db.list_source_artifact_cleanup_tasks(limit=10)
    workspace.provider.removed.clear()
    await workspace.sync(TEAM)

    await workspace.cleanup()

    returned = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    assert workspace.stored(returned)
    assert workspace.store.deletions == []


@pytest.mark.asyncio
async def test_cleanup_waits_while_the_source_has_an_active_activity(workspace):
    await overlapping(workspace, collected_locally=False)
    release = await _unit_input(workspace.db, RELEASE, doc_id(ISSUE))
    await _release_document(workspace.db, source_id=RELEASE, source_type="jira", doc_id=doc_id(ISSUE))
    await workspace.db.delete_projected_document(doc_id(ISSUE), source_id=RELEASE)
    await workspace.db.acquire_source_activity(
        activity_id="sync-in-progress", source_id=RELEASE, kind=SourceActivityKind.SYNC,
    )

    assert await workspace.cleanup() == 0
    assert workspace.stored(release)

    await workspace.db.release_source_activity(activity_id="sync-in-progress")
    await workspace.cleanup()
    assert not workspace.store.get_artifact(release.raw_content_uri, release.raw_content_type)


@pytest.mark.asyncio
async def test_released_inputs_sharing_an_object_delete_it_once(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    shared = workspace.store.store_raw(
        source_id=TEAM, doc_id=doc_id(ISSUE), title="Shared", content=b"{}", content_type="application/json",
    )
    held = await _hold_document(
        workspace.db, source_id=TEAM, source_type="jira", doc_id=doc_id(ISSUE), title=f"{ISSUE}: Payroll run",
        markdown="# Payroll run", version="1", source_url="https://jira.example/browse/PAY-1",
        space_or_project="PAY", raw_content_uri=shared, item_extra={"issue_id": "400001", "issue_key": ISSUE},
    )
    # A second Unit of the Source that once held the Document through the same object.
    await workspace.db.db.execute(
        """INSERT INTO source_units (id, source_id, unit_type, provider_key, locator_json, current_revision_id, updated_at)
           VALUES ('unit-previous', ?, 'jira_issue', 'issue:previous', '{}', 'revision-previous', ?)""",
        (TEAM, "2026-09-27T00:00:00+00:00"),
    )
    await workspace.db.db.execute(
        """INSERT INTO source_unit_inputs (source_unit_id, source_id, document_id, unit_revision_id, item_json,
               raw_content_uri, raw_content_type, recorded_at)
           SELECT 'unit-previous', source_id, document_id, 'revision-previous', item_json,
               raw_content_uri, raw_content_type, recorded_at
             FROM source_unit_inputs WHERE source_unit_id = ?""",
        (held.source_unit_id,),
    )
    await workspace.db.db.commit()
    await _release_document(workspace.db, source_id=TEAM, source_type="jira", doc_id=doc_id(ISSUE))

    await workspace.db.delete_projected_document(doc_id(ISSUE), source_id=TEAM)
    await workspace.cleanup()

    assert workspace.store.deletions == [shared]
    assert not workspace.store.get_artifact(shared, "application/json")
    assert await workspace.db.list_source_artifact_cleanup_tasks(limit=10) == []


@pytest.mark.asyncio
async def test_a_document_link_falls_back_to_the_next_copy_when_the_newest_is_missing(workspace, tmp_path):
    await overlapping(workspace, collected_locally=False)
    team = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    release = await _unit_input(workspace.db, RELEASE, doc_id(ISSUE))
    newest = await workspace.db.list_document_source_unit_inputs(doc_id(ISSUE))
    assert newest[0].source_unit_id == release.source_unit_id
    workspace.store.delete_artifact(release.raw_content_uri)
    config = AppConfig(base_dir=tmp_path / "mem")
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(db=workspace.db, config=config, document_store=workspace.store)
    with TestClient(app) as client:
        document_raw = client.get(f"/api/v1/documents/{doc_id(ISSUE)}/artifacts/raw_source")
        document_content = client.get(f"/api/v1/documents/{doc_id(ISSUE)}/content")

    assert [history["id"] for history in document_raw.json()["changelog"]["histories"]] == ["9001"]
    assert workspace.stored_histories(team.raw_content_uri) == ["9001"]
    # The release copy's normalized markdown is still stored, so content keeps the newest copy.
    assert document_content.text == workspace.store.read_artifact(release.normalized_content_uri).decode("utf-8")


@pytest.mark.asyncio
async def test_a_staged_derivation_keeps_the_objects_it_will_commit(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    await workspace.sync(TEAM)
    original = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    discovered_item = workspace.provider._item
    workspace.provider._item = _renamed(workspace.provider, "renamed")
    await workspace.sync(TEAM)
    # Renamed back with a change that needs extraction: the sync writes the
    # original keys again and stages its derivation, then extraction fails.
    workspace.provider._item = discovered_item
    workspace.provider.histories[ISSUE].append(authored(SPRINT_HISTORY, "Ann"))
    failed = await workspace.sync(TEAM, extractor=FailingMemoryExtractor())
    assert failed.docs_failed >= 1
    staged = await workspace.db.list_source_derivation_attempts(source_id=TEAM, statuses=("retryable_failure",))
    assert [attempt.context.unit_input.raw_content_uri for attempt in staged] == [original.raw_content_uri]

    await workspace.cleanup()
    assert original.raw_content_uri not in workspace.store.deletions

    await workspace.sync(TEAM)
    committed = await _unit_input(workspace.db, TEAM, doc_id(ISSUE))
    assert committed.raw_content_uri == original.raw_content_uri
    assert workspace.stored(committed)
    assert workspace.stored_histories(committed.raw_content_uri) == ["9001", "9002"]


@pytest.mark.asyncio
async def test_a_busy_source_does_not_hold_up_other_sources_cleanup(workspace):
    await workspace.add_source(TEAM, collected_locally=False)
    await workspace.add_source(RELEASE, collected_locally=False)
    busy = workspace.store.store_raw(
        source_id=TEAM, doc_id="doc-busy", title="Busy", content=b"{}", content_type="application/json",
    )
    idle = workspace.store.store_raw(
        source_id=RELEASE, doc_id="doc-idle", title="Idle", content=b"{}", content_type="application/json",
    )
    await workspace.db.enqueue_source_artifact_cleanup_task(source_id=TEAM, artifact_uri=busy)
    await workspace.db.enqueue_source_artifact_cleanup_task(source_id=RELEASE, artifact_uri=idle)
    await workspace.db.acquire_source_activity(activity_id="team-sync", source_id=TEAM, kind=SourceActivityKind.SYNC)

    assert await SourceArtifactCleanupService(workspace.db, workspace.store).run_pending(limit=1) == 1

    assert workspace.store.deletions == [idle]
    assert [task.artifact_uri for task in await workspace.db.list_source_artifact_cleanup_tasks(limit=10)] == [busy]


@pytest.mark.asyncio
async def test_the_scheduler_cleans_up_released_objects(workspace, tmp_path):
    from memforge.runtime import SyncService
    from memforge.scheduler import ARTIFACT_CLEANUP_JOB_ID, SyncScheduler

    await workspace.add_source(TEAM, collected_locally=False)
    released = workspace.store.store_raw(
        source_id=TEAM, doc_id=doc_id(ISSUE), title="Released", content=b"{}", content_type="application/json",
    )
    await workspace.db.enqueue_source_artifact_cleanup_task(source_id=TEAM, artifact_uri=released)
    scheduler = SyncScheduler(
        workspace.db, SyncService(workspace.db, AppConfig(base_dir=tmp_path / "mem")), document_store=workspace.store,
    )
    await scheduler.start()
    try:
        assert scheduler.scheduler.get_job(ARTIFACT_CLEANUP_JOB_ID) is not None
        await scheduler._clean_up_released_artifacts()
    finally:
        await scheduler.shutdown()

    assert workspace.store.deletions == [released]
