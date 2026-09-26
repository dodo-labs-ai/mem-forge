"""Operator reprocessing of Source Units at their current revision."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig, SyncConfig
from memforge.models import ContentItem, GeneMetadata, NormalizedContent, RawContent, SyncState
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.runtime import SourceSyncWorker, SyncService
from memforge.source_activity import SourceSyncRunActive
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload
from memforge.source_projection import ProjectionScopeTransition
from memforge.storage.database import Database
from memforge.storage.document_store import StoredDocumentArtifact
from tests.test_sync_bookkeeping import (
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingMemoryEngine,
    StubDocumentStore,
    _jira_raw_content,
    _valid_png_bytes,
)

SOURCE_ID = "src-reprocess"
ISSUES = {"PAY-1": "300001", "PAY-2": "300002"}
ARTIFACT_ISSUE = "PAY-1"
FIRST_SYNC_AT = datetime(2026, 9, 20, tzinfo=timezone.utc)


def doc_id(issue_key: str) -> str:
    return f"jira-{issue_key}"


class RawKeepingDocumentStore(StubDocumentStore):
    """Keeps every stored body, as the real store does."""

    def store_raw(self, *, source_id, doc_id, title, content, content_type, extension=None):
        uri = super().store_raw(
            source_id=source_id, doc_id=doc_id, title=title, content=content,
            content_type=content_type, extension=extension,
        )
        self.source_artifacts[uri] = content
        return uri

    def get_artifact(self, uri, media_type):
        if uri not in self.source_artifacts:
            return None
        return StoredDocumentArtifact(uri=uri, filename=uri.rsplit("/", 1)[-1], media_type=media_type)


class JiraGene:
    """Serves the provider until it is closed; a reprocess must never reach it."""

    discovery_complete = True

    def __init__(self) -> None:
        self.provider_open = True

    def _provider(self) -> None:
        if not self.provider_open:
            raise AssertionError("reprocess must not contact the provider")

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira", display_name="Jira", description="", default_sync_interval_minutes=60,
            auth_method="pat", data_shape="ticket",
        )

    def requires_pdf_artifact(self, **kwargs) -> bool:
        return False

    async def authenticate(self) -> None:
        self._provider()

    async def discover(self, since=None):
        self._provider()
        for issue_key, issue_id in ISSUES.items():
            yield ContentItem(
                item_id=doc_id(issue_key), title=f"{issue_key}: Payroll run",
                source_url=f"https://jira.example/browse/{issue_key}", last_modified=FIRST_SYNC_AT,
                content_type="application/json", space_or_project="PAY", version="1",
                extra={"issue_id": issue_id, "issue_key": issue_key},
            )

    async def fetch(self, item):
        self._provider()
        raw = _jira_raw_content(item)
        if item.extra["issue_key"] != ARTIFACT_ISSUE:
            return raw
        return replace(raw, artifacts=(
            RawSourceArtifact(
                provider_key="attachment-7001", parent_observation_type="issue_core",
                parent_provider_key=f"{ISSUES[ARTIFACT_ISSUE]}:core", provider_revision="1",
                filename="architecture.png", media_type="image/png",
                declared_size_bytes=len(_valid_png_bytes()), locator={"attachment_id": "attachment-7001"},
            ),
        ))

    @asynccontextmanager
    async def open_source_artifact(self, artifact):
        self._provider()
        payload = _valid_png_bytes()

        async def chunks():
            yield payload

        yield SourceArtifactDownload(chunks=chunks(), media_type="image/png", content_length=len(payload))

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\nBody")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "reprocess.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    try:
        yield database
    finally:
        NoopMemoryEngine.db = None
        await database.close()


class Harness:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.store = RawKeepingDocumentStore()
        self.gene = JiraGene()
        self.extractor = ProjectionFragmentRecordingExtractor()
        self.engine = RecordingMemoryEngine()

    def orchestrator(self) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db, doc_store=self.store, memory_extractor=self.extractor,
            memory_engine=self.engine, memory_store=None, max_concurrent=1,
        )

    async def unit(self, issue_key: str):
        return await self.db.find_source_unit_by_document_id(SOURCE_ID, doc_id(issue_key), current_only=True)

    async def reprocess(self, *issue_keys: str) -> SyncState:
        self.extractor.fragment_calls.clear()
        self.engine.projected_lifecycle_calls.clear()
        return await self.orchestrator().sync_gene(
            gene=self.gene, source_name="Jira", source_id=SOURCE_ID,
            execution_mode=SourceSyncMode.REPROCESS,
            reprocess_doc_ids=frozenset(doc_id(key) for key in issue_keys),
        )


async def synced(db: Database) -> Harness:
    """A Jira Source a local agent collects, synced once; its only copy is then the stored input."""
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json=json.dumps({"sync_mode": "local_agent"}),
        access_policy="workspace", owner_user_id="dev",
    )
    harness = Harness(db)
    state = await harness.orchestrator().sync_gene(gene=harness.gene, source_name="Jira", source_id=SOURCE_ID)
    assert state.last_sync_status == "success"
    harness.gene.provider_open = False
    return harness


def primary_anchors(fragments):
    return {fragment.anchor for fragment in fragments if fragment.primary_eligible}


@pytest.mark.asyncio
async def test_reprocess_reads_the_stored_unit_at_its_current_revision_without_the_provider(db):
    harness = await synced(db)
    cursor = (await db.get_sync_state(SOURCE_ID)).last_sync_at
    untouched_revision = await db.get_current_source_unit_revision((await harness.unit("PAY-2")).id)
    committed = await db.get_current_source_unit_revision((await harness.unit(ARTIFACT_ISSUE)).id)

    state = await harness.reprocess(ARTIFACT_ISSUE)

    assert state.last_sync_status == "success"
    assert (state.docs_processed, state.docs_failed) == (1, 0)
    # The sync cursor stays where the last sync left it.
    assert state.last_sync_at == cursor
    assert (await db.get_sync_state(SOURCE_ID)).last_sync_at == cursor
    [lifecycle] = harness.engine.projected_lifecycle_calls
    assert lifecycle["doc_id"] == doc_id(ARTIFACT_ISSUE)
    assert lifecycle["derivation_reprocess_all_current_observations"] is True
    assert lifecycle["derivation_reprocess_operation_id"]
    assert lifecycle["derivation_support_without_baseline"] is True
    projection = lifecycle["projection"]
    delta = projection.deltas[0]
    assert delta.previous_unit_revision_id == delta.current_unit_revision_id == committed.id
    # Extraction reads every ReadingGroup that holds Primary content, the Artifact included.
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    read = {anchor for catalog in harness.extractor.fragment_calls for anchor in primary_anchors(catalog.fragments)}
    assert read == primary_anchors(context.full_fragments)
    # The committed Artifact is carried, so the complete snapshot keeps it.
    artifact_types = {
        observation.observation_type
        for observation in (await db.get_current_source_unit_projection(delta.source_unit_id)).observations
    }
    assert "binary_artifact" in artifact_types
    # Other Units are not read, and nothing is inferred removed.
    assert await db.get_current_source_unit_revision((await harness.unit("PAY-2")).id) == untouched_revision
    assert await db.get_document(doc_id("PAY-2")) is not None


@pytest.mark.asyncio
async def test_reprocessing_the_same_revision_again_runs_as_a_new_operation(db):
    harness = await synced(db)

    await harness.reprocess("PAY-2")
    [first] = harness.engine.projected_lifecycle_calls
    await harness.reprocess("PAY-2")
    [second] = harness.engine.projected_lifecycle_calls

    assert harness.extractor.fragment_calls
    assert first["derivation_reprocess_operation_id"] != second["derivation_reprocess_operation_id"]
    assert first["derivation_id"] != second["derivation_id"]


@pytest.mark.asyncio
async def test_a_unit_without_stored_input_fails_with_its_reason_and_the_others_continue(db):
    harness = await synced(db)
    document = await db.get_document(doc_id("PAY-2"))
    del harness.store.source_artifacts[document.raw_content_uri]

    state = await harness.reprocess("PAY-1", "PAY-2", "PAY-404")

    assert state.last_sync_status == "partial"
    assert (state.docs_processed, state.docs_failed) == (1, 2)
    assert {failed.doc_id: failed.error.split(":")[0] for failed in state.failed_docs} == {
        doc_id("PAY-2"): "stored_raw_content_missing",
        doc_id("PAY-404"): "stored_document_missing",
    }
    assert [call["doc_id"] for call in harness.engine.projected_lifecycle_calls] == [doc_id("PAY-1")]


@pytest.mark.asyncio
async def test_stored_input_that_would_move_the_unit_fails_closed(db):
    harness = await synced(db)
    await db.db.execute(
        "UPDATE documents SET source_url = ? WHERE doc_id = ?",
        ("https://jira.example/browse/elsewhere", doc_id("PAY-2")),
    )
    await db.db.commit()

    state = await harness.reprocess("PAY-2")

    assert state.last_sync_status == "failed"
    [failed] = state.failed_docs
    assert failed.error.startswith("stored_input_incomplete")
    assert harness.engine.projected_lifecycle_calls == []
    assert harness.extractor.fragment_calls == []


class ConfluenceGene:
    """Places a child page under its parent only from the item metadata fetched with it."""

    discovery_complete = True

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="confluence", display_name="Confluence", description="", default_sync_interval_minutes=60,
            auth_method="pat", data_shape="document",
        )

    def requires_pdf_artifact(self, **kwargs) -> bool:
        return False

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        yield self._page()

    async def rediscover(self, item):
        assert item.item_id == CHILD_PAGE
        return self._page()

    @staticmethod
    def _page() -> ContentItem:
        return ContentItem(
            item_id=CHILD_PAGE, title="Payroll runbook", source_url="https://wiki.example/pages/42",
            last_modified=FIRST_SYNC_AT, content_type="text/html", space_or_project="PAY", version="3",
            extra={"page_id": "42", "space_key": "PAY"},
        )

    async def fetch(self, item):
        item.extra["parent_page_id"] = PARENT_PAGE_ID
        return RawContent(item=item, body=b"<p>Retain A7 for regular payroll.</p>", content_type="text/html")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="Retain A7 for regular payroll.")


CHILD_PAGE = "confluence-42"
PARENT_PAGE_ID = "7"


async def synced_confluence(db: Database) -> Harness:
    await db.upsert_source(
        id=SOURCE_ID, type="confluence", name="Wiki", config_json="{}", access_policy="workspace",
        owner_user_id="dev",
    )
    harness = Harness(db)
    harness.gene = ConfluenceGene()
    state = await harness.orchestrator().sync_gene(gene=harness.gene, source_name="Wiki", source_id=SOURCE_ID)
    assert state.last_sync_status == "success"
    return harness


@pytest.mark.asyncio
async def test_a_child_page_is_reprocessed_from_the_provider_where_its_gene_placed_it(db):
    harness = await synced_confluence(db)
    unit = await db.find_source_unit_by_document_id(SOURCE_ID, CHILD_PAGE, current_only=True)
    committed = await db.get_current_source_unit_revision(unit.id)
    assert unit.locator["parent_page_id"] == PARENT_PAGE_ID

    state = await harness.orchestrator().sync_gene(
        gene=harness.gene, source_name="Wiki", source_id=SOURCE_ID,
        execution_mode=SourceSyncMode.REPROCESS, reprocess_doc_ids=frozenset({CHILD_PAGE}),
    )

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    [lifecycle] = harness.engine.projected_lifecycle_calls[-1:]
    assert lifecycle["projection"].source_unit_revisions[0].location_hash == committed.location_hash


async def _rewrite_committed_artifact_metadata(db: Database, harness: Harness, rewrite) -> None:
    projection = await db.get_current_source_unit_projection((await harness.unit(ARTIFACT_ISSUE)).id)
    [artifact] = [item for item in projection.observations if item.observation_type == "binary_artifact"]
    [revision] = [item for item in projection.observation_revisions if item.observation_id == artifact.id]
    metadata = json.loads(json.dumps(revision.metadata))
    rewrite(metadata["source_artifact"])
    await db.db.execute(
        "UPDATE source_observation_revisions SET metadata_json = ? WHERE id = ?", (json.dumps(metadata), revision.id),
    )
    await db.db.commit()


@pytest.mark.asyncio
async def test_an_artifact_stored_before_its_eligibility_fields_is_reprocessed(db):
    harness = await synced(db)
    await _rewrite_committed_artifact_metadata(
        db, harness, lambda artifact: [artifact.pop(key) for key in ("inference_eligible", "inference_ineligible_reason")],
    )

    state = await harness.reprocess(ARTIFACT_ISSUE)

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    [lifecycle] = harness.engine.projected_lifecycle_calls
    assert any(item.observation_type == "binary_artifact" for item in lifecycle["projection"].observations)


@pytest.mark.asyncio
async def test_unreadable_artifact_metadata_fails_only_its_unit(db):
    harness = await synced(db)
    await _rewrite_committed_artifact_metadata(db, harness, lambda artifact: artifact.pop("sha256"))

    state = await harness.reprocess(ARTIFACT_ISSUE, "PAY-2")

    assert (state.docs_processed, state.docs_failed) == (1, 1)
    [failed] = state.failed_docs
    assert (failed.doc_id, failed.error.split(":")[0]) == (doc_id(ARTIFACT_ISSUE), "stored_artifact_invalid")


@pytest.mark.asyncio
async def test_reprocess_waits_for_an_open_projection_scope_transition(db):
    harness = await synced(db)
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="transition-1", source_id=SOURCE_ID, previous_scope={"projects": ["PAY"]},
            target_scope={"projects": ["PAY", "HR"]},
        )
    )

    state = await harness.reprocess("PAY-1")

    assert state.last_sync_status == "failed"
    assert "Projection Scope transition" in state.error_message
    assert harness.engine.projected_lifecycle_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "doc_ids"),
    [(SourceSyncMode.REPROCESS, None), (SourceSyncMode.NORMAL, frozenset({"jira-PAY-1"}))],
)
async def test_reprocess_documents_belong_to_the_reprocess_mode_only(db, mode, doc_ids):
    harness = await synced(db)

    with pytest.raises(ValueError, match="REPROCESS"):
        await harness.orchestrator().sync_gene(
            gene=harness.gene, source_name="Jira", source_id=SOURCE_ID,
            execution_mode=mode, reprocess_doc_ids=doc_ids,
        )


async def _source_with_cursor(db: Database) -> None:
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace", owner_user_id="dev",
    )
    await db.upsert_sync_state(SyncState(source=SOURCE_ID, last_sync_at=FIRST_SYNC_AT, last_sync_status="success"))


@pytest.mark.asyncio
async def test_reprocess_runs_are_queued_alone_and_a_sync_requested_meanwhile_runs_after(db):
    await _source_with_cursor(db)
    ordinary = await db.enqueue_source_sync_run(source_id=SOURCE_ID)
    assert ordinary.reprocess_document_ids == ()
    with pytest.raises(SourceSyncRunActive):
        await db.enqueue_source_sync_run(
            source_id=SOURCE_ID, trigger="reprocess", reprocess_document_ids=("jira-PAY-1",),
        )
    leased = await db.lease_next_source_sync_run(worker_id="worker", lease_seconds=60)
    await db.complete_source_sync_run(
        leased.run_id, worker_id="worker", lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(source=SOURCE_ID, last_sync_at=FIRST_SYNC_AT, last_sync_status="success"),
    )

    reprocess = await db.enqueue_source_sync_run(
        source_id=SOURCE_ID, trigger="reprocess", reprocess_document_ids=("jira-PAY-2", "jira-PAY-1", "jira-PAY-2"),
    )
    assert (reprocess.trigger, reprocess.reprocess_document_ids) == ("reprocess", ("jira-PAY-1", "jira-PAY-2"))
    requested = await db.enqueue_source_sync_run(source_id=SOURCE_ID, force_full_sync=True)
    assert requested.run_id == reprocess.run_id and requested.coalesced and requested.rerun_requested
    assert requested.reprocess_document_ids == reprocess.reprocess_document_ids

    leased = await db.lease_next_source_sync_run(worker_id="worker", lease_seconds=60)
    await db.complete_source_sync_run(
        leased.run_id, worker_id="worker", lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(source=SOURCE_ID, last_sync_at=FIRST_SYNC_AT, last_sync_status="success"),
    )
    successor = await db.get_latest_source_sync_run(source_id=SOURCE_ID)
    assert successor.run_id != reprocess.run_id
    assert (successor.trigger, successor.status, successor.reprocess_document_ids) == ("rerun", "pending", ())
    assert successor.force_full_sync


class _CapturingRuntimeProvider:
    def __init__(self, status: str) -> None:
        self.status = status
        self.calls: list[dict] = []

    async def build_sync_runtime(self, db, config, **kwargs):
        return object()

    async def run_source_sync(self, **kwargs):
        self.calls.append(kwargs)
        return SyncState(
            source=SOURCE_ID, last_sync_at=FIRST_SYNC_AT, last_sync_status=self.status,
            error_message=None if self.status == "success" else "stored_raw_content_missing: jira-PAY-1",
            failure_retryable=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["success", "failed"])
async def test_the_worker_runs_a_reprocess_from_storage_once(db, status):
    await _source_with_cursor(db)
    config = AppConfig(sync=SyncConfig(max_extraction_workers=1))
    provider = _CapturingRuntimeProvider(status)
    run = await SyncService(db, config, runtime_provider=provider).enqueue_reprocess(SOURCE_ID, ("jira-PAY-1",))

    await SourceSyncWorker(db, config, runtime_provider=provider, worker_id="worker").run_once()

    [call] = provider.calls
    assert call["execution_mode"] is SourceSyncMode.REPROCESS
    assert call["reprocess_doc_ids"] == frozenset({"jira-PAY-1"})
    assert call["force_full_sync"] is False and call["reusable_projection_doc_ids"] == frozenset()
    finished = await db.get_source_sync_run(run.run_id)
    assert finished.status == status
    # An operator requests a failed reprocess again; it is not retried.
    assert finished.next_attempt_at is None


@pytest.mark.asyncio
async def test_a_source_without_a_completed_sync_cannot_be_reprocessed(db):
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace", owner_user_id="dev",
    )
    service = SyncService(db, AppConfig(), runtime_provider=_CapturingRuntimeProvider("success"))

    with pytest.raises(Exception, match="has not completed a sync"):
        await service.enqueue_reprocess(SOURCE_ID, ("jira-PAY-1",))
    assert await db.get_latest_source_sync_run(source_id=SOURCE_ID) is None


def _admin_config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "mem")
    config.server.jwt_secret = "test-secret"
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


@pytest.mark.asyncio
async def test_the_reprocess_route_previews_then_queues_one_run(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    harness = await synced(db)
    app = create_admin_app(db=db, config=_admin_config(tmp_path), document_store=harness.store)
    route = f"/api/v1/sources/{SOURCE_ID}/reprocess"
    with TestClient(app) as client:
        preview = client.post(route, json={"document_ids": ["jira-PAY-1", "jira-PAY-404"], "dry_run": True})
        assert preview.status_code == 200
        assert await db.get_latest_source_sync_run(source_id=SOURCE_ID) is None
        queued = client.post(route, json={"document_ids": ["jira-PAY-1"]})
        again = client.post(route, json={"document_ids": ["jira-PAY-2"]})

    report = preview.json()
    units = {unit["document_id"]: unit for unit in report["units"]}
    assert {key: value for key, value in units["jira-PAY-404"].items() if value is not None} == {
        "document_id": "jira-PAY-404", "available": False, "reason": "stored_document_missing",
    }
    unit = units["jira-PAY-1"]
    assert unit["available"] and unit["artifact_count"] == 1
    assert unit["unit_revision_id"] == (await db.get_current_source_unit_revision(unit["source_unit_id"])).id
    assert unit["extraction_item_count"] >= 1 and unit["reading_group_count"] >= unit["extraction_item_count"]
    assert unit["estimated_model_calls"] == unit["extraction_item_count"] + 1
    assert report["estimated_model_calls"] == unit["estimated_model_calls"]
    assert (report["available_unit_count"], report["unavailable_unit_count"]) == (1, 1)

    assert queued.status_code == 202
    run = await db.get_source_sync_run(queued.json()["run_id"])
    assert (run.trigger, run.reprocess_document_ids) == ("reprocess", ("jira-PAY-1",))
    assert again.status_code == 409


@pytest.mark.asyncio
async def test_the_reprocess_route_refuses_an_open_projection_scope_transition(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    harness = await synced(db)
    await db.create_projection_scope_transition(
        ProjectionScopeTransition(
            id="transition-1", source_id=SOURCE_ID, previous_scope={"projects": ["PAY"]},
            target_scope={"projects": ["PAY", "HR"]},
        )
    )
    app = create_admin_app(db=db, config=_admin_config(tmp_path), document_store=harness.store)
    with TestClient(app) as client:
        response = client.post(f"/api/v1/sources/{SOURCE_ID}/reprocess", json={"document_ids": ["jira-PAY-1"]})

    assert response.status_code == 409
    assert response.json()["detail"] == "projection_scope_transition_open"
    assert await db.get_latest_source_sync_run(source_id=SOURCE_ID) is None


@pytest.mark.asyncio
async def test_existing_sync_runs_read_no_reprocess_documents(db):
    await _source_with_cursor(db)
    columns = {row[1] for row in await db.db.execute_fetchall("PRAGMA table_info(source_sync_runs)")}
    assert "reprocess_document_ids_json" in columns
    run = await db.enqueue_source_sync_run(source_id=SOURCE_ID)
    stored = await db.db.execute_fetchall(
        "SELECT reprocess_document_ids_json FROM source_sync_runs WHERE run_id = ?", (run.run_id,),
    )
    assert stored[0][0] is None
    assert (await db.get_source_sync_run(run.run_id)).reprocess_document_ids == ()
