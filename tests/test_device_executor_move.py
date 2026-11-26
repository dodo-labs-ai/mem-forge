"""A server Source moved to the device keeps syncing incrementally (ADR 0058, design §10.8, gate of A3).

Each type that runs on both executors is created on the server, synced there
against its fake provider, and moved to the device through
``PUT /sources/{id}/executor``. Its owner stores the credential on the device;
**Sync now** then queues a ``document_acquisition`` job, the daemon acquires
with the type's own Gene against the same provider, and the server
interprets what it acquired. The first device run is an ordinary incremental
run from the cursor: it extracts only the items that changed.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import ModuleType
from weakref import WeakValueDictionary

import pytest
from fastapi.testclient import TestClient

import memforge.server.admin_api as admin_api
from memforge.config import AppConfig
from memforge.genes import confluence_gene, github_pages_gene, notion_gene
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.local_agent.source_contract import DOCUMENT_ACQUISITION_OPERATION
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.runtime import DefaultRuntimeProvider, SourceSyncWorker
from memforge.storage.database import Database
from tests.contracts.document_fakes import ProviderListing
from tests.contracts.document_fakes import confluence as confluence_fakes
from tests.contracts.document_fakes import github_pages as github_pages_fakes
from tests.contracts.document_fakes import notion as notion_fakes
from tests.device_report_fixture import device_report
from tests.test_device_acquisition import DAEMON_SERVER, _routed_client
from tests.test_local_adapter_api import _config
from tests.test_sync_bookkeeping import (
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    _OrderRecordingMemoryEngine,
    _skip_retry_delay,
)

OWNER = "owner-a"
TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="


@dataclass(frozen=True)
class DeviceType:
    """A type that runs on both executors, its fake provider and the configuration it is created with."""

    source_type: str
    fakes: ModuleType
    gene_module: ModuleType
    config: dict
    secret: str

    @property
    def token(self) -> str:
        """The token the fake provider accepts, held by the server and then by the device."""

        return str(self.config[self.secret])


TYPES = [
    DeviceType(
        "confluence",
        confluence_fakes,
        confluence_gene,
        {
            "base_url": confluence_fakes.BASE_URL,
            "pat": "wiki-token",
            "sync_mode": "space",
            "spaces": ["PAY"],
            "exclude_labels": ["draft"],
        },
        "pat",
    ),
    DeviceType(
        "notion",
        notion_fakes,
        notion_gene,
        {"token": notion_fakes.TOKEN, "root_pages": [notion_fakes.ROOT]},
        "token",
    ),
    DeviceType(
        "github_pages",
        github_pages_fakes,
        github_pages_gene,
        dict(github_pages_fakes.FAKES.listing_configs["branch-tree"].config),
        "pat",
    ),
]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    from tests.test_daemon_service import FakeKeyring

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


@pytest.fixture(autouse=True)
def rendered(monkeypatch) -> list[str]:
    """The server's PDF renderer, without WeasyPrint's system libraries: it reads the material's own images."""

    titles: list[str] = []

    def renderer():
        async def render(document: str, fetch) -> bytes:
            titles.append(document)
            images = [fetch(f"memforge-material:/image/{index}") for index in range(document.count("memforge-material:"))]
            return b"%PDF-1.4\n" + document.encode() + b"".join(image.body for image in images) + b"\n%%EOF"

        return render

    monkeypatch.setattr("memforge.genes.toolkit.pdf.default_pdf_renderer", renderer)
    return titles


class _Runtime:
    def __init__(self, db: Database, doc_store, extractor: ProjectionFragmentRecordingExtractor) -> None:
        self.db = db
        self.doc_store = doc_store
        self.extractor = extractor
        self.runtime_event_trace_sink = None

    def orchestrator(self) -> GeneSyncOrchestrator:
        engine = _OrderRecordingMemoryEngine(defer=False)
        engine.db = self.db
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=self.doc_store,
            memory_extractor=self.extractor,
            memory_engine=engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.db),
            max_concurrent=1,
            retry_sleep=_skip_retry_delay,
        )


class _RuntimeProvider(DefaultRuntimeProvider):
    def __init__(self, runtime: _Runtime) -> None:
        self._runtime = runtime

    async def build_sync_runtime(self, db, config, **kwargs):
        return self._runtime


class Deployment:
    """One server with its routes, a worker that runs its syncs, and the owner's daemon."""

    def __init__(self, tmp_path, kind: DeviceType) -> None:
        self.kind = kind
        self.db = Database(str(tmp_path / "server.db"))
        asyncio.run(self.db.connect())
        self.app = admin_api.create_admin_app(
            db=self.db,
            config=_config(tmp_path),
            principal_resolver=lambda request: OWNER,
            workspace_role_resolver=lambda request: "workspace_admin",
        )
        self.extractor = ProjectionFragmentRecordingExtractor()
        self.provider = kind.fakes._Provider(ProviderListing.COMPLETE)
        self.client: TestClient

    @contextmanager
    def serving(self) -> Iterator[None]:
        """While the block runs, the type's Gene on either executor reaches the fake provider."""

        with self.provider.http.serving(self.kind.gene_module):
            yield

    def sync(self) -> None:
        """Run the Source's queued sync on the server's worker."""

        runtime = _Runtime(self.db, self.app.state.document_store, self.extractor)
        with self.serving():
            asyncio.run(
                SourceSyncWorker(
                    self.db, AppConfig(), runtime_provider=_RuntimeProvider(runtime), worker_id="worker-test"
                ).run_once()
            )

    def run_device_job(self, source_id: str) -> dict:
        """The owner's daemon leases the Source's job and acquires it with the credential this device holds."""

        import memforge.main as main

        [job] = asyncio.run(
            self.db.lease_local_agent_jobs(
                user_id=OWNER,
                device=device_report([source_id]),
                acquisition_contracts=declared_acquisition_contracts(),
                limit=1,
                lease_seconds=300,
            )
        )
        assert job["operation"] == DOCUMENT_ACQUISITION_OPERATION
        with self.serving():
            result = main._run_cloud_document_acquisition_job(job, _routed_client(self.client))
        assert result.get("error") is None, result
        assert result["sync_started"] is True
        return job

    def extracted_documents(self, source_id: str) -> list[str]:
        """The Documents whose Units were extracted, in extraction order."""

        async def read() -> list[str]:
            documents = {}
            for doc_id in await self.db.list_indexed_doc_ids(source_id):
                unit = await self.db.find_source_unit_by_document_id(source_id, doc_id, current_only=True)
                documents[unit.id] = doc_id
            return [documents[catalog.source_unit_id] for catalog in self.extractor.fragment_calls]

        return asyncio.run(read())

    def runs(self, source_id: str) -> list[tuple[str, str | None]]:
        """Each sync run of the Source, oldest first: its status and the device acquisition it interpreted."""

        async def read() -> list[tuple[str, str | None]]:
            async with self.db.db.execute(
                "SELECT status, input_snapshot_id FROM source_sync_runs WHERE source_id = ? ORDER BY created_at",
                (source_id,),
            ) as cursor:
                return [(str(row["status"]), row["input_snapshot_id"]) async for row in cursor]

        return asyncio.run(read())

    def documents(self, source_id: str) -> dict[str, tuple]:
        async def read() -> dict[str, tuple]:
            return {
                doc.doc_id: (doc.title, doc.version, doc.content_hash)
                for doc in await self.db.list_documents(source=source_id, limit=1000)
            }

        return asyncio.run(read())


@pytest.fixture(params=TYPES, ids=[kind.source_type for kind in TYPES])
def deployment(request, tmp_path) -> Iterator[Deployment]:
    deployment = Deployment(tmp_path, request.param)
    try:
        with TestClient(deployment.app) as client:
            deployment.client = client
            yield deployment
    finally:
        asyncio.run(deployment.db.close())


def _create_and_sync_on_the_server(deployment: Deployment) -> str:
    created = deployment.client.post(
        "/api/v1/sources",
        json={
            "type": deployment.kind.source_type,
            "name": "Team docs",
            "config": deployment.kind.config,
            "access_policy": "workspace",
        },
    )
    assert created.status_code == 200, created.text
    source_id = created.json()["id"]
    queued = deployment.client.post(f"/api/v1/sources/{source_id}/sync")
    assert queued.status_code == 202, queued.text
    assert "run_id" in queued.json()
    deployment.sync()
    assert deployment.documents(source_id)
    return source_id


def _move_to_the_device(deployment: Deployment, source_id: str) -> None:
    from memforge.local_agent.device_credentials import device_credentials

    moved = deployment.client.put(f"/api/v1/sources/{source_id}/executor", json={"executor": "device"})
    assert moved.status_code == 200, moved.text
    # The owner stores the token on the device: `memforge source credentials set`.
    device_credentials(DAEMON_SERVER).set(source_id, {deployment.kind.secret: deployment.kind.token})


def _sync_now_on_the_device(deployment: Deployment, source_id: str) -> None:
    queued = deployment.client.post(f"/api/v1/sources/{source_id}/sync")
    assert queued.status_code == 202, queued.text
    assert queued.json()["status"] == "queued" and "job_id" in queued.json()
    job = deployment.run_device_job(source_id)
    # The job carried the configuration without the secret.
    assert deployment.kind.secret not in job["payload"]["config"]
    deployment.sync()


def test_the_first_device_run_of_a_moved_source_extracts_nothing_unchanged(deployment):
    source_id = _create_and_sync_on_the_server(deployment)
    on_the_server = deployment.documents(source_id)
    deployment.extractor.fragment_calls.clear()

    _move_to_the_device(deployment, source_id)
    _sync_now_on_the_device(deployment, source_id)

    assert deployment.extracted_documents(source_id) == []
    assert {doc_id: doc[2] for doc_id, doc in deployment.documents(source_id).items()} == {
        doc_id: doc[2] for doc_id, doc in on_the_server.items()
    }
    latest = asyncio.run(deployment.db.get_sync_history(source=source_id, limit=1))[0]
    assert latest["status"] == "success", latest
    # The run interpreted what the device acquired, and opened no scope transition.
    assert asyncio.run(deployment.db.get_open_projection_scope_transition(source_id)) is None


@pytest.mark.parametrize("deployment", [TYPES[0]], ids=["confluence"], indirect=True)
def test_a_confluence_source_moved_to_the_device_extracts_only_the_page_that_changed(deployment, monkeypatch):
    source_id = _create_and_sync_on_the_server(deployment)
    # The server's run extracted every page it holds.
    assert sorted(deployment.extracted_documents(source_id)) == sorted(deployment.documents(source_id))
    assert len(deployment.documents(source_id)) > 1
    deployment.extractor.fragment_calls.clear()
    _move_to_the_device(deployment, source_id)

    # Page 101 is edited after the server's run; nothing else changes.
    space, parent, labels, _changed_at = confluence_fakes.PAGES["101"]
    monkeypatch.setitem(confluence_fakes.PAGES, "101", (space, parent, labels, datetime.now(timezone.utc)))
    page = confluence_fakes._page

    def edited(page_id: str) -> dict:
        answer = page(page_id)
        if page_id == "101":
            answer["body"]["storage"]["value"] = "<h1>Page 101</h1><p>The PAY rules of page 101, revised.</p>"
        return answer

    monkeypatch.setattr(confluence_fakes, "_page", edited)
    deployment.provider.requests.clear()
    _sync_now_on_the_device(deployment, source_id)

    assert deployment.extracted_documents(source_id) == ["confluence-101"]
    # The device read the provider from the cursor: only the changed page's body and its export.
    reads = set(
        request.split("?")[0].rsplit("/", 1)[-1]
        for request in deployment.provider.requests
        if "/rest/api/content/" in request and "/child/" not in request
    )
    assert reads == {"101"}


@pytest.mark.parametrize("deployment", [TYPES[0]], ids=["confluence"], indirect=True)
def test_a_scheduled_run_of_a_device_source_queues_its_acquisition_job(deployment):
    from memforge.runtime import SyncService
    from memforge.scheduler import SyncScheduler

    source_id = _create_and_sync_on_the_server(deployment)
    _move_to_the_device(deployment, source_id)
    due = datetime.now(timezone.utc) - timedelta(minutes=1)
    asyncio.run(deployment.db.set_source_sync_schedule(source_id, enabled=True, interval_minutes=60, next_run_at=due))
    scheduler = SyncScheduler(
        deployment.db,
        SyncService(deployment.db, AppConfig(), workspace_id="default"),
        document_store=deployment.app.state.document_store,
    )

    asyncio.run(scheduler._sync_due_sources())

    [job] = [
        job
        for job in asyncio.run(deployment.db.list_current_local_agent_jobs(workspace_id="default", user_id=OWNER))
        if job["status"] == "queued"
    ]
    assert job["operation"] == DOCUMENT_ACQUISITION_OPERATION
    assert job["payload"]["since"] is not None
    # The schedule admitted no server run for the device Source.
    assert deployment.runs(source_id) == [("success", None)]
    deployment.extractor.fragment_calls.clear()
    deployment.run_device_job(source_id)
    deployment.sync()
    [server_run, device_run] = deployment.runs(source_id)
    assert device_run[0] == "success" and device_run[1] is not None
    assert deployment.extracted_documents(source_id) == []
