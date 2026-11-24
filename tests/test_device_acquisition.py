"""A device run of a document Source: the type's own Gene on the device, interpreted on the server (ADR 0058).

The device makes the provider calls a server run makes and hands the server
its manifest, its acquired inputs and its answers through the acquisition
routes. The server answers the manifest with what its run will ask beyond
discovery, by the server run's own rules, processes the attempt once it is
ready, and runs the sync with the type's recorded Gene. A device run and a
server run against the same provider end alike.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest
from PIL import Image

import memforge.server.admin_api as admin_api
from memforge.config import AppConfig
from memforge.genes.notion_gene import NotionGene
from memforge.local_agent.acquisition_daemon import acquire_document_source
from memforge.local_agent.document_acquisition import (
    AcquiredArtifact,
    AcquiredInputPackage,
    AcquiredListing,
    AcquisitionAnswers,
    AcquisitionError,
    AcquisitionManifest,
    AcquisitionManifestTooLarge,
    AcquisitionRequests,
)
from memforge.local_agent.snapshot_manifest import SNAPSHOT_MANIFEST_MAX_ITEMS
from memforge.local_agent.source_contract import (
    DOCUMENT_ACQUISITION_OPERATION,
    LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD,
    LocalAgentLease,
    local_agent_source_config_revision,
)
from memforge.models import (
    AbsenceAnswer,
    ContentItem,
    DocumentRecord,
    NormalizedContent,
    RawContent,
    ScopeListing,
    ScopeListingKind,
    SyncState,
)
from memforge.pipeline.device_acquisition import document_acquisition_job_payload
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.runtime import DefaultRuntimeProvider, SourceSyncWorker
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload
from memforge.source_derivation import SourceUnitDerivationContext, source_derivation_manifest
from memforge.source_item_failures import FailedItemRetry, SourceItemFailure
from memforge.source_projection import ProjectionScopeTransition
from memforge.source_projection_config import canonical_projection_scope, projection_scope_fingerprint
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore
from tests.gene_fixture import FakeGene
from tests.source_type_fixture import register_test_source_type
from tests.test_local_adapter_api import _config
from tests.test_sync_bookkeeping import (
    ProjectionFragmentRecordingExtractor,
    RecordingDocumentDeleteMemoryStore,
    _OrderRecordingMemoryEngine,
    _skip_retry_delay,
)

SOURCE_ID = "src-wiki"
WORKSPACE = "local"
PAGE_ROOT = "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
CONFIG = {"token": "wiki-token-secret", "root_pages": [PAGE_ROOT]}
FIRST_RUN = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)


@dataclass
class Page:
    title: str
    body: str
    edited_at: datetime


class Wiki:
    """The provider: its pages, the ones in the configured tree, their attachments, and what it refuses to read."""

    def __init__(self) -> None:
        self.pages: dict[str, Page] = {}
        self.in_tree: set[str] = set()
        self.refused: set[str] = set()
        self.attachments: dict[str, bytes] = {}
        self.unreadable_attachments: set[str] = set()
        self.calls: list[str] = []

    def write(self, page_id: str, body: str, edited_at: datetime) -> None:
        self.pages[page_id] = Page(title=f"Page {page_id}", body=body, edited_at=edited_at)
        self.in_tree.add(page_id)

    def item(self, page_id: str) -> ContentItem:
        page = self.pages[page_id]
        return ContentItem(
            item_id=f"wiki-{page_id}",
            title=page.title,
            source_url=f"https://wiki.example.com/{page_id}",
            last_modified=page.edited_at,
            content_type="text/markdown",
            space_or_project="team_alpha",
            version=page.edited_at.isoformat(),
            extra={"page_id": page_id},
        )


class WikiUnavailable(RuntimeError):
    """The wiki could not serve a page this time."""

    retryable = True


class AttachmentUnreadable(RuntimeError):
    """The wiki could not serve an attachment; the error says nothing of whether it may pass."""


def _png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color=(20, 40, 60)).save(output, format="PNG")
    return output.getvalue()


class WikiGene(FakeGene):
    """The type's Gene: acquisition reads ``provider``; interpretation is its own."""

    provider: Wiki | None = None
    listing_kinds = frozenset({ScopeListingKind.QUERY})

    @classmethod
    def metadata(cls):
        return replace(NotionGene.metadata(), name="wiki", display_name="Wiki")

    @classmethod
    def config_schema(cls):
        return NotionGene.config_schema()

    @classmethod
    def lists_scope(cls, config) -> bool:
        return True

    @classmethod
    def rediscovers_documents(cls, config: Mapping[str, object]) -> bool:
        return bool(config.get("rediscovers", True))

    def _wiki(self, call: str) -> Wiki:
        assert self.provider is not None, "the provider is reached only where acquisition runs"
        self.provider.calls.append(call)
        return self.provider

    async def discover(self, since=None):
        wiki = self._wiki("discover")
        for page_id in sorted(wiki.in_tree):
            if since is None or wiki.pages[page_id].edited_at >= since:
                yield wiki.item(page_id)

    async def list_scope(self):
        wiki = self._wiki("list_scope")
        return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset(f"wiki-{page_id}" for page_id in wiki.in_tree))

    async def confirm_absent(self, items):
        wiki = self._wiki("confirm_absent")
        return {
            item.item_id: AbsenceAnswer.EXISTS if item.extra["page_id"] in wiki.pages else AbsenceAnswer.ABSENT
            for item in items
        }

    async def rediscover(self, item):
        wiki = self._wiki("rediscover")
        page_id = str(item.extra["page_id"])
        if page_id in wiki.refused:
            raise WikiUnavailable(f"the wiki could not look up page {page_id}")
        return wiki.item(page_id) if page_id in wiki.in_tree else None

    async def fetch(self, item):
        wiki = self._wiki("fetch")
        page_id = str(item.extra["page_id"])
        if page_id in wiki.refused:
            raise WikiUnavailable(f"the wiki could not serve page {page_id}")
        # What fetch read is noted on the item, as a Gene that hydrates during fetch does.
        item.extra["read_from"] = "wiki"
        artifacts: tuple[RawSourceArtifact, ...] = ()
        if page_id in wiki.attachments:
            artifacts = (
                RawSourceArtifact(
                    provider_key=f"attachment-{page_id}",
                    parent_observation_type="document",
                    parent_provider_key=item.item_id,
                    provider_revision=wiki.pages[page_id].edited_at.isoformat(),
                    filename="diagram.png",
                    media_type="image/png",
                    declared_size_bytes=len(wiki.attachments[page_id]),
                    locator={"page_id": page_id},
                ),
            )
        return RawContent(
            item=item, body=wiki.pages[page_id].body.encode(), content_type="text/markdown", artifacts=artifacts
        )

    @asynccontextmanager
    async def open_source_artifact(self, artifact):
        wiki = self._wiki("open_source_artifact")
        page_id = str(artifact.locator["page_id"])
        if page_id in wiki.unreadable_attachments:
            raise AttachmentUnreadable(f"the wiki could not serve the attachment of page {page_id}")
        body = wiki.attachments[page_id]

        async def chunks():
            yield body

        yield SourceArtifactDownload(chunks=chunks(), media_type=artifact.media_type, content_length=len(body))

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\n{raw.body.decode()}")


class _Lease:
    """The leased ``document_acquisition`` job attempt, as the job broker validates it for each request."""

    def __init__(self) -> None:
        self.current = True
        self.held: LocalAgentLease | None = None

    @property
    def payload(self) -> Mapping[str, object]:
        assert self.held is not None
        return self.held.payload

    def lease(self, payload: Mapping[str, object], *, leased_at: datetime) -> None:
        self.held = LocalAgentLease(payload=dict(payload), leased_at=leased_at)

    async def __call__(self, *args, **kwargs) -> LocalAgentLease | None:
        return self.held if self.current else None


class _Side:
    """One deployment: its database, document store and how it syncs."""

    def __init__(self, db: Database, doc_store: LocalDocumentStore) -> None:
        self.db = db
        self.doc_store = doc_store
        self.engine = _OrderRecordingMemoryEngine(defer=False)
        self.engine.db = db
        self.retry_sleeps: list[float] = []

    async def _retry_sleep(self, delay: float) -> None:
        self.retry_sleeps.append(delay)
        await _skip_retry_delay(delay)

    def orchestrator(self) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=self.doc_store,
            memory_extractor=ProjectionFragmentRecordingExtractor(),
            memory_engine=self.engine,
            memory_store=RecordingDocumentDeleteMemoryStore(self.db),
            max_concurrent=1,
            retry_sleep=self._retry_sleep,
        )

    async def documents(self) -> dict[str, tuple]:
        """Each Document as its row describes it, without the times the write itself sets."""

        return {
            doc.doc_id: (
                doc.title,
                doc.source_url,
                doc.space_or_project,
                doc.last_modified,
                tuple(doc.labels),
                doc.version,
                doc.content_hash,
                doc.item_extra,
            )
            for doc in await self.db.list_documents(source=SOURCE_ID, limit=1000)
        }

    async def units(self) -> dict[str, tuple]:
        """The Source's current Units by id."""

        return {
            unit.id: (unit.unit_type, unit.provider_key, dict(unit.locator))
            for unit in await self.db.list_current_source_units(SOURCE_ID)
        }

    async def held(self) -> dict[str, tuple]:
        """Each held Document's stored input: the item the Unit records, its raw bytes and normalized hash."""

        held: dict[str, tuple] = {}
        for doc_id in sorted(await self.db.list_indexed_doc_ids(SOURCE_ID)):
            unit = await self.db.find_source_unit_by_document_id(SOURCE_ID, doc_id, current_only=True)
            assert unit is not None
            stored = await self.db.get_source_unit_input(unit.id)
            assert stored is not None
            held[doc_id] = (stored.item.to_payload(), stored.raw_content_sha256, stored.normalized_content_hash)
        return held

    async def artifacts(self) -> list[tuple[str, dict]]:
        """Each current Document's stored artifacts, without where each deployment's store keeps the bytes."""

        return [
            (doc_id, {key: value for key, value in metadata["source_artifact"].items() if key != "uri"})
            for doc_id, metadata in await self.db.list_current_document_artifacts(SOURCE_ID)
        ]

    async def ledger(self) -> dict[str, tuple]:
        return {
            failure.document_id: (failure.stage, failure.error_class, failure.message, failure.retryable)
            for failure in await self.db.list_source_item_failures(source_id=SOURCE_ID)
        }

    async def cursor(self) -> datetime | None:
        state = await self.db.get_sync_state(SOURCE_ID)
        return state.last_sync_at if state is not None else None

    async def last_run(self) -> dict:
        latest = (await self.db.get_sync_history(source=SOURCE_ID, limit=1))[0]
        return {key: latest[key] for key in ("status", "unlisted_present_count", "absence_check_skipped_reason")}


class _RecordedRuntime:
    def __init__(self, side: _Side) -> None:
        self._side = side
        self.doc_store = side.doc_store
        self.runtime_event_trace_sink = None

    def orchestrator(self) -> GeneSyncOrchestrator:
        return self._side.orchestrator()


class _RuntimeProvider(DefaultRuntimeProvider):
    def __init__(self, side: _Side) -> None:
        self._side = side

    async def build_sync_runtime(self, db, config, **kwargs):
        return _RecordedRuntime(self._side)


class _RouteTransport:
    """The daemon's transport, over the server's acquisition routes."""

    def __init__(self, client, *, job: str, attempt: int = 1) -> None:
        self._client = client
        self._attempt = {"local_agent_job_id": job, "local_agent_attempt_count": attempt}
        self.manifest_sha256 = ""
        self.manifest_status: int | None = None

    def _post(self, path: str, body: dict) -> dict:
        response = self._client.post(f"/api/v1/sources/{SOURCE_ID}/acquisition/{path}", json={**self._attempt, **body})
        assert response.status_code == 200, response.text
        return response.json()

    async def submit_manifest(self, manifest: AcquisitionManifest) -> AcquisitionRequests:
        answer = await asyncio.to_thread(self._post, "manifest", {"manifest": manifest.to_payload()})
        self.manifest_sha256 = answer["manifest_sha256"]
        return AcquisitionRequests.from_payload(answer["requests"])

    def _upload_artifact(self, item, artifact, body) -> str:
        response = self._client.post(
            f"/api/v1/sources/{SOURCE_ID}/adapter/artifacts",
            params={
                "source_unit_key": item.item_id,
                "provider_key": artifact.provider_key,
                "provider_revision": artifact.provider_revision,
                "parent_observation_type": artifact.parent_observation_type,
                "parent_provider_key": artifact.parent_provider_key,
                "filename": artifact.filename,
                "media_type": artifact.media_type,
                **self._attempt,
            },
            content=body,
            headers={"content-type": artifact.media_type},
        )
        assert response.status_code == 200, response.text
        return str(response.json()["input_sha256"])

    async def upload_artifact(self, item, artifact, body) -> str:
        return await asyncio.to_thread(self._upload_artifact, item, artifact, body)

    async def upload_input(self, package: AcquiredInputPackage) -> None:
        await asyncio.to_thread(self._post, "inputs", {"package": package.to_payload()})

    async def submit_answers(self, answers: AcquisitionAnswers) -> None:
        await asyncio.to_thread(
            self._post, "answers", {"manifest_sha256": self.manifest_sha256, "answers": answers.to_payload()}
        )


async def _no_pause(_delay: float) -> None:
    return None


def _connect(path) -> Database:
    database = Database(str(path))
    asyncio.run(database.connect())
    asyncio.run(
        database.upsert_source(
            id=SOURCE_ID,
            type="wiki",
            name="Team wiki",
            config_json=json.dumps(CONFIG),
            created_by_user_id=LOCAL_DEV_USER_ID,
            access_policy="workspace",
            owner_user_id=LOCAL_DEV_USER_ID,
        )
    )
    return database


@pytest.fixture
def wiki(monkeypatch) -> Wiki:
    provider = Wiki()
    register_test_source_type(monkeypatch, "wiki", WikiGene, unit_noun="page", scope_fields=("root_pages",))
    monkeypatch.setattr(WikiGene, "provider", provider)
    return provider


@pytest.fixture
def deployments(tmp_path, wiki):
    """A server deployment, and a device deployment whose daemon talks to its routes."""

    server = _Side(_connect(tmp_path / "server.db"), LocalDocumentStore(str(tmp_path / "server-docs")))
    device_db = _connect(tmp_path / "device.db")
    lease = _Lease()
    app = admin_api.create_admin_app(db=device_db, config=_config(tmp_path), local_agent_lease_validator=lease)
    from fastapi.testclient import TestClient

    try:
        with TestClient(app) as client:
            device = _Side(device_db, app.state.document_store)
            yield server, device, client, lease
    finally:
        asyncio.run(server.db.close())
        asyncio.run(device_db.close())


def _server_run(server: _Side, started_at: datetime, config: Mapping[str, object] = CONFIG) -> None:
    """A scheduled server run, as the device job's run is."""

    gene = WikiGene(dict(config), SOURCE_ID)
    state = asyncio.run(
        server.orchestrator().sync_gene(
            gene=gene,
            source_name="Team wiki",
            source_id=SOURCE_ID,
            failed_item_retry=FailedItemRetry.RETRYABLE,
            started_at=started_at,
        )
    )
    assert state.last_sync_status in {"success", "partial"}, state.error_message


def _device_run(device: _Side, client, lease: _Lease, started_at: datetime, *, job: str) -> AcquisitionRequests:
    """Lease a ``document_acquisition`` job at ``started_at``, acquire on the device, process on the server."""

    source = asyncio.run(device.db.get_source(SOURCE_ID))
    payload = asyncio.run(document_acquisition_job_payload(device.db, device.doc_store, source))
    lease.lease(payload, leased_at=started_at)
    transport = _RouteTransport(client, job=job)
    requests: list[AcquisitionRequests] = []

    async def submit_manifest(manifest):
        requests.append(await _RouteTransport.submit_manifest(transport, manifest))
        return requests[0]

    transport.submit_manifest = submit_manifest  # type: ignore[method-assign]
    asyncio.run(
        acquire_document_source(
            payload,
            open_gene=lambda config: WikiGene(dict(config), SOURCE_ID),
            transport=transport,
            exports_pdf=False,
            sleep=_no_pause,
        )
    )
    processed = client.post(
        f"/api/v1/sources/{SOURCE_ID}/process",
        json={"local_agent_job_id": job, "local_agent_attempt_count": 1},
    )
    assert processed.status_code == 202, processed.text
    asyncio.run(
        SourceSyncWorker(device.db, AppConfig(), runtime_provider=_RuntimeProvider(device), worker_id="worker-test").run_once()
    )
    return requests[0]


async def _end_state(side: _Side) -> tuple:
    return (
        await side.documents(),
        await side.units(),
        await side.held(),
        await side.artifacts(),
        await side.ledger(),
        await side.cursor(),
        await side.last_run(),
    )


def _configure(side: _Side, config: Mapping[str, object]) -> None:
    asyncio.run(
        side.db.upsert_source(
            id=SOURCE_ID,
            type="wiki",
            name="Team wiki",
            config_json=json.dumps(config),
            access_policy="workspace",
            owner_user_id=LOCAL_DEV_USER_ID,
        )
    )


@pytest.mark.parametrize("rediscovers", [True, False], ids=["rediscovers", "does-not-rediscover"])
def test_a_device_run_and_a_server_run_against_the_same_provider_end_alike(deployments, wiki, rediscovers):
    server, device, client, lease = deployments
    config = {**CONFIG, "rediscovers": rediscovers}
    for side in (server, device):
        _configure(side, config)
    for page_id in ("a", "b", "c", "d", "g"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    wiki.refused = {"d"}
    wiki.attachments["a"] = _png()
    runs = [FIRST_RUN, FIRST_RUN + timedelta(hours=1), FIRST_RUN + timedelta(hours=2)]

    def changes(index: int) -> None:
        if index == 1:
            # a is edited, b leaves the tree but stays, c is deleted, e is new, d reads again.
            wiki.write("a", "body of a, edited", runs[1] - timedelta(minutes=30))
            wiki.in_tree.discard("b")
            del wiki.pages["c"]
            wiki.in_tree.discard("c")
            wiki.write("e", "body of e", runs[1] - timedelta(minutes=20))
            wiki.refused = set()
        if index == 2:
            # g is touched without a change to its body, and f is new.
            wiki.write("g", "body of g", runs[2] - timedelta(minutes=10))
            wiki.write("f", "body of f", runs[2] - timedelta(minutes=5))

    device_requests: list[AcquisitionRequests] = []
    before_touch: tuple = ()
    for index, started_at in enumerate(runs):
        if index == 2:
            before_touch = asyncio.run(device.documents())["wiki-g"]
        changes(index)
        _server_run(server, started_at, config)
        server_listing = asyncio.run(WikiGene(dict(config), SOURCE_ID).list_scope())
        device_requests.append(_device_run(device, client, lease, started_at, job=f"job-{index}"))
        stored = asyncio.run(
            device.db.get_source_acquisition(source_id=SOURCE_ID, workspace_id=WORKSPACE, snapshot_id=f"job-{index}:attempt:1")
        )

        assert stored is not None and stored.started_at == started_at
        assert stored.manifest.listing.scope_listing() == server_listing
        assert asyncio.run(_end_state(device)) == asyncio.run(_end_state(server))

    # The first run left the refused page in the ledger.
    assert list(device_requests[0].confirm + device_requests[0].rediscover + device_requests[0].fetch) == []
    # The second run asked the device to confirm what the listing left out, and
    # to read the failed page again: rediscovered by a type that rediscovers,
    # fetched by its stored item by one that does not.
    second = device_requests[1]
    assert sorted(item.item_id for item in second.confirm) == ["wiki-b", "wiki-c"]
    retried = second.rediscover if rediscovers else second.fetch
    assert [item.item_id for item in retried] == ["wiki-d"]
    assert (second.fetch if rediscovers else second.rediscover) == ()
    assert asyncio.run(device.ledger()) == {}

    documents = asyncio.run(device.documents())
    held = asyncio.run(device.held())
    # b left the listing and is kept; c is confirmed absent and its Document is gone.
    assert sorted(documents) == sorted(held) == ["wiki-a", "wiki-b", "wiki-d", "wiki-e", "wiki-f", "wiki-g"]
    assert len(asyncio.run(device.units())) == len(held)
    # The touched page has a new version and the same content; each Unit holds
    # the item as the device's fetch left it.
    assert documents["wiki-g"][5] != before_touch[5]
    assert documents["wiki-g"][6] == before_touch[6]
    assert held["wiki-a"][0]["extra"]["read_from"] == "wiki"
    # The page's attachment went through the artifact route and is stored as the server stores it.
    assert [(doc_id, artifact["filename"]) for doc_id, artifact in asyncio.run(device.artifacts())] == [
        ("wiki-a", "diagram.png")
    ]
    assert asyncio.run(device.last_run())["unlisted_present_count"] == 1
    assert asyncio.run(device.cursor()) == runs[2]


def _fail(database: Database, doc_id: str, item: ContentItem | None) -> None:
    failed_at = datetime(2026, 11, 23, tzinfo=timezone.utc)
    asyncio.run(
        database.upsert_source_item_failure(
            SourceItemFailure(
                source_id=SOURCE_ID,
                document_id=doc_id,
                stage="fetch",
                error_class="builtins.RuntimeError",
                message="the wiki refused it",
                retryable=True,
                release="test",
                failures_in_release=1,
                first_failed_at=failed_at,
                last_failed_at=failed_at,
                item=item,
            )
        )
    )


def _manifest_route(client, manifest: AcquisitionManifest, *, job: str = "job-1"):
    return client.post(
        f"/api/v1/sources/{SOURCE_ID}/acquisition/manifest",
        json={"local_agent_job_id": job, "local_agent_attempt_count": 1, "manifest": manifest.to_payload()},
    )


def _lease_job(device: _Side, lease: _Lease, **payload) -> None:
    source = asyncio.run(device.db.get_source(SOURCE_ID))
    lease.lease(
        {
            "operation": DOCUMENT_ACQUISITION_OPERATION,
            "source_config_revision": local_agent_source_config_revision(source),
            **payload,
        },
        leased_at=FIRST_RUN,
    )


def test_the_manifest_response_asks_what_the_server_run_will_ask_by_its_own_rules(deployments, wiki):
    _server, device, client, lease = deployments
    for page_id in ("a", "b", "c"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    # b failed after it committed, x never became a Unit, a failed and is discovered again.
    _fail(device.db, "wiki-b", wiki.item("b"))
    never_committed = ContentItem(
        item_id="wiki-x", title="x", source_url="", last_modified=FIRST_RUN, extra={"page_id": "x"}
    )
    _fail(device.db, "wiki-x", never_committed)
    _fail(device.db, "wiki-a", wiki.item("a"))
    _lease_job(device, lease)
    listing = AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-a"}))
    manifest = AcquisitionManifest(listing=listing, discovered=(wiki.item("a"),))

    planned = _manifest_route(client, manifest)
    # A replay answers with the requests the manifest was issued, even when the ledger changed since.
    _fail(device.db, "wiki-c", wiki.item("c"))
    replayed = _manifest_route(client, manifest)
    other = _manifest_route(client, AcquisitionManifest(listing=listing))

    assert planned.status_code == 200, planned.text
    requests = AcquisitionRequests.from_payload(planned.json()["requests"])
    # The query listing leaves b and c out: both are confirmed by the items their Units record.
    assert [(item.item_id, item.extra.get("read_from")) for item in requests.confirm] == [
        ("wiki-b", "wiki"),
        ("wiki-c", "wiki"),
    ]
    # The retried rows less the one discovered: b by its Unit's item, x by its own.
    assert [item.item_id for item in requests.rediscover] == ["wiki-b", "wiki-x"]
    assert requests.fetch == ()
    assert replayed.json() == planned.json()
    assert other.status_code == 409


def test_a_source_that_cannot_rediscover_asks_the_device_to_fetch_each_failed_item(deployments, wiki):
    _server, device, client, lease = deployments
    asyncio.run(
        device.db.upsert_source(
            id=SOURCE_ID,
            type="wiki",
            name="Team wiki",
            config_json=json.dumps({**CONFIG, "rediscovers": False}),
            access_policy="workspace",
            owner_user_id=LOCAL_DEV_USER_ID,
        )
    )
    wiki.write("b", "body of b", FIRST_RUN)
    _fail(device.db, "wiki-b", wiki.item("b"))
    _fail(device.db, "wiki-y", None)
    _lease_job(device, lease)
    listing = AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-b", "wiki-y"}))

    response = _manifest_route(client, AcquisitionManifest(listing=listing))

    requests = AcquisitionRequests.from_payload(response.json()["requests"])
    # A row without an item is replayed from its Unit's stored input on the server.
    assert [item.item_id for item in requests.fetch] == ["wiki-b"]
    assert (requests.confirm, requests.rediscover) == ((), ())


def test_processing_waits_until_the_attempt_is_ready(deployments, wiki):
    _server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN - timedelta(hours=1))
    wiki.write("b", "body of b", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    wiki.in_tree.discard("b")
    _lease_job(device, lease)
    transport = _RouteTransport(client, job="job-1")

    def process():
        return client.post(
            f"/api/v1/sources/{SOURCE_ID}/process", json={"local_agent_job_id": "job-1", "local_agent_attempt_count": 1}
        )

    before_manifest = process()
    item = wiki.item("a")
    manifest = AcquisitionManifest(
        listing=AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-a"})), discovered=(item,)
    )
    requests = asyncio.run(transport.submit_manifest(manifest))
    awaiting_answers = process()
    asyncio.run(
        transport.submit_answers(AcquisitionAnswers(absence={entry.item_id: AbsenceAnswer.ABSENT for entry in requests.confirm}))
    )
    awaiting_inputs = process()
    lease.current = False
    lost_lease = client.post(
        f"/api/v1/sources/{SOURCE_ID}/acquisition/inputs",
        json={
            "local_agent_job_id": "job-1",
            "local_agent_attempt_count": 1,
            "package": AcquiredInputPackage(item=item, body=b"body of a", content_type="text/markdown").to_payload(),
        },
    )
    lease.current = True
    asyncio.run(transport.upload_input(AcquiredInputPackage(item=item, body=b"body of a", content_type="text/markdown")))
    ready = process()

    assert [entry.item_id for entry in requests.confirm] == ["wiki-b"]
    assert (before_manifest.status_code, before_manifest.json()["detail"]) == (409, "acquisition_manifest_required")
    assert (awaiting_answers.status_code, awaiting_answers.json()["detail"]) == (409, "acquisition_answers_pending")
    assert (awaiting_inputs.status_code, awaiting_inputs.json()["detail"]) == (409, "acquisition_inputs_incomplete")
    assert (lost_lease.status_code, lost_lease.json()["detail"]) == (409, "local_agent_lease_not_current")
    assert ready.status_code == 202, ready.text


def test_answers_must_answer_exactly_what_was_asked_and_cannot_change(deployments, wiki):
    _server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN - timedelta(hours=1))
    wiki.write("b", "body of b", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    _lease_job(device, lease)
    listing = AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-a"}))
    planned = _manifest_route(client, AcquisitionManifest(listing=listing)).json()

    def answer(absence: dict[str, AbsenceAnswer], digest: str = planned["manifest_sha256"]):
        return client.post(
            f"/api/v1/sources/{SOURCE_ID}/acquisition/answers",
            json={
                "local_agent_job_id": "job-1",
                "local_agent_attempt_count": 1,
                "manifest_sha256": digest,
                "answers": AcquisitionAnswers(absence=absence).to_payload(),
            },
        )

    unasked = answer({"wiki-b": AbsenceAnswer.ABSENT, "wiki-z": AbsenceAnswer.ABSENT})
    other_digest = answer({"wiki-b": AbsenceAnswer.ABSENT}, digest="f" * 64)
    answered = answer({"wiki-b": AbsenceAnswer.EXISTS})
    replayed = answer({"wiki-b": AbsenceAnswer.EXISTS})
    changed = answer({"wiki-b": AbsenceAnswer.ABSENT})

    assert unasked.status_code == 409 and "exactly" in unasked.json()["detail"]
    assert other_digest.status_code == 409
    assert (answered.status_code, replayed.status_code, changed.status_code) == (200, 200, 409)
    stored = asyncio.run(
        device.db.get_source_acquisition(source_id=SOURCE_ID, workspace_id=WORKSPACE, snapshot_id="job-1:attempt:1")
    )
    assert stored is not None and stored.answers is not None
    assert dict(stored.answers.absence) == {"wiki-b": AbsenceAnswer.EXISTS}


def test_a_manifest_above_the_bound_is_refused_with_its_count(deployments, wiki):
    _server, device, client, lease = deployments
    _lease_job(device, lease)
    count = SNAPSHOT_MANIFEST_MAX_ITEMS + 1
    listing = AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset(f"wiki-{n}" for n in range(count)))

    response = _manifest_route(client, AcquisitionManifest(listing=listing))

    assert response.status_code == 422
    assert str(count) in response.json()["detail"]


def test_the_device_fails_a_listing_above_the_bound_before_it_fetches_anything(wiki):
    for page_id in range(SNAPSHOT_MANIFEST_MAX_ITEMS + 1):
        wiki.write(str(page_id), "body", FIRST_RUN)

    class Untouched:
        async def submit_manifest(self, manifest):
            raise AssertionError("nothing is submitted")

        async def upload_input(self, package):
            raise AssertionError("nothing is uploaded")

    with pytest.raises(AcquisitionManifestTooLarge) as raised:
        asyncio.run(
            acquire_document_source(
                {"config": dict(CONFIG), "since": FIRST_RUN.isoformat()},
                open_gene=lambda config: WikiGene(dict(config), SOURCE_ID),
                transport=Untouched(),  # type: ignore[arg-type]
                exports_pdf=False,
            )
        )

    assert raised.value.count == SNAPSHOT_MANIFEST_MAX_ITEMS + 1
    assert "fetch" not in wiki.calls


def test_the_device_makes_the_server_runs_calls_in_order_with_one_gene(wiki):
    for page_id in ("a", "b"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN)
    opened: list[WikiGene] = []
    events: list[str] = []

    class Recording:
        async def submit_manifest(self, manifest):
            events.append(f"manifest:{[item.item_id for item in manifest.discovered]}")
            # The manifest names the items as discovery yielded them, not as fetch changed them.
            assert all("read_from" not in item.extra for item in manifest.discovered)
            return AcquisitionRequests(confirm=(wiki.item("a"),), rediscover=(wiki.item("b"),))

        async def upload_input(self, package):
            events.append(f"input:{package.item.item_id}")

        async def submit_answers(self, answers):
            events.append(f"answers:{sorted(answers.absence)}:{sorted(answers.rediscovered)}")

    def open_gene(config):
        gene = WikiGene(dict(config), SOURCE_ID)
        opened.append(gene)
        return gene

    summary = asyncio.run(
        acquire_document_source(
            {"config": dict(CONFIG), "since": None},
            open_gene=open_gene,
            transport=Recording(),  # type: ignore[arg-type]
            exports_pdf=False,
        )
    )

    assert len(opened) == 1
    assert wiki.calls == ["discover", "list_scope", "fetch", "fetch", "confirm_absent", "rediscover", "fetch"]
    assert events == [
        "input:wiki-a",
        "input:wiki-b",
        "manifest:['wiki-a', 'wiki-b']",
        "input:wiki-b",
        "answers:['wiki-a']:['wiki-b']",
    ]
    assert (summary.discovered, summary.uploaded, summary.failed) == (2, 3, 0)


def test_the_job_carries_the_configuration_without_secrets_since_and_the_due_excluded_scopes(tmp_path, wiki):
    database = _connect(tmp_path / "jobs.db")
    store = LocalDocumentStore(str(tmp_path / "docs"))
    previous = {**CONFIG, "root_pages": ["0a0b0c0d0e0f40718293a4b5c6d7e8f9"]}

    async def scenario():
        fresh = await document_acquisition_job_payload(database, store, await database.get_source(SOURCE_ID))
        await database.upsert_sync_state(SyncState(source=SOURCE_ID, last_sync_at=FIRST_RUN, last_sync_status="success"))
        reprocess = await document_acquisition_job_payload(
            database, store, await database.get_source(SOURCE_ID), reprocess_document_ids=("wiki-b", "wiki-a")
        )
        transition = ProjectionScopeTransition(
            id="t1",
            source_id=SOURCE_ID,
            previous_scope=canonical_projection_scope("wiki", previous),
            target_scope=canonical_projection_scope("wiki", CONFIG),
            created_at=FIRST_RUN.isoformat(),
        )
        await database.upsert_source(
            id=SOURCE_ID,
            type="wiki",
            name="Team wiki",
            config_json=json.dumps(CONFIG),
            access_policy="workspace",
            owner_user_id=LOCAL_DEV_USER_ID,
            projection_scope_transition=transition,
        )
        not_due = await document_acquisition_job_payload(database, store, await database.get_source(SOURCE_ID))
        await database.start_projection_scope_transition("t1", run_id="run-1")
        await database.fail_projection_scope_transition(
            "t1", run_id="run-1", error="excluded listings missing", excluded_scopes_due=True
        )
        due = await document_acquisition_job_payload(database, store, await database.get_source(SOURCE_ID))
        return fresh, reprocess, not_due, due

    try:
        fresh, reprocess, not_due, due = asyncio.run(scenario())
    finally:
        asyncio.run(database.close())

    assert fresh["operation"] == DOCUMENT_ACQUISITION_OPERATION
    assert fresh["config"] == {"root_pages": [PAGE_ROOT]}
    assert fresh["since"] is None
    assert (reprocess["reprocess"], reprocess["since"]) == (["wiki-a", "wiki-b"], None)
    # A transition run discovers everything; the excluded scopes wait for the run that reaches them.
    assert not_due["since"] is None and LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD not in not_due
    assert due[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD] == [
        {
            "scope_key": projection_scope_fingerprint(canonical_projection_scope("wiki", previous)),
            "config": {"root_pages": ["0a0b0c0d0e0f40718293a4b5c6d7e8f9"]},
        }
    ]


def _own(database: Database) -> None:
    """Make the local user the Source's execution owner, whose device runs its jobs."""

    asyncio.run(
        database.upsert_source(
            id=SOURCE_ID,
            type="wiki",
            name="Team wiki",
            config_json=json.dumps(CONFIG),
            access_policy="workspace",
            owner_user_id=LOCAL_DEV_USER_ID,
            execution_owner_user_id=LOCAL_DEV_USER_ID,
        )
    )


def _process(client, *, job: str, attempt: int = 1):
    return client.post(
        f"/api/v1/sources/{SOURCE_ID}/process",
        json={"local_agent_job_id": job, "local_agent_attempt_count": attempt},
    )


def test_a_lost_lease_rejects_the_attempts_later_submissions_and_the_retried_attempt_acquires_again(
    deployments, wiki
):
    _server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    wiki.write("a", "body of a, edited", FIRST_RUN + timedelta(minutes=30))
    _lease_job(device, lease, since=FIRST_RUN.isoformat(), config={"root_pages": [PAGE_ROOT]})
    first = _RouteTransport(client, job="job-1", attempt=1)
    item = wiki.item("a")
    manifest = AcquisitionManifest(
        listing=AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-a"})), discovered=(item,)
    )
    asyncio.run(first.submit_manifest(manifest))

    lease.current = False
    package = AcquiredInputPackage(item=item, body=b"body of a, edited", content_type="text/markdown")
    attempt_one = {"local_agent_job_id": "job-1", "local_agent_attempt_count": 1}
    refused = {
        "manifest": _manifest_route(client, manifest),
        "inputs": client.post(
            f"/api/v1/sources/{SOURCE_ID}/acquisition/inputs", json={**attempt_one, "package": package.to_payload()}
        ),
        "answers": client.post(
            f"/api/v1/sources/{SOURCE_ID}/acquisition/answers",
            json={
                **attempt_one,
                "manifest_sha256": first.manifest_sha256,
                "answers": AcquisitionAnswers().to_payload(),
            },
        ),
        "process": _process(client, job="job-1"),
    }
    lease.current = True
    # The retried attempt starts again: the first attempt's manifest is not its own.
    before_its_manifest = _process(client, job="job-1", attempt=2)
    second = _RouteTransport(client, job="job-1", attempt=2)
    asyncio.run(
        acquire_document_source(
            lease.payload,
            open_gene=lambda config: WikiGene(dict(config), SOURCE_ID),
            transport=second,
            exports_pdf=False,
            sleep=_no_pause,
        )
    )
    processed = _process(client, job="job-1", attempt=2)

    assert {name: (response.status_code, response.json()["detail"]) for name, response in refused.items()} == {
        name: (409, "local_agent_lease_not_current") for name in refused
    }
    assert (before_its_manifest.status_code, before_its_manifest.json()["detail"]) == (
        409,
        "acquisition_manifest_required",
    )
    assert processed.status_code == 202, processed.text
    assert wiki.calls.count("list_scope") == 2
    stored = asyncio.run(
        device.db.get_source_acquisition(source_id=SOURCE_ID, workspace_id=WORKSPACE, snapshot_id="job-1:attempt:2")
    )
    assert stored is not None and [entry.item_id for entry in stored.manifest.discovered] == ["wiki-a"]


def test_a_run_that_disagrees_with_its_acquisition_fails_and_a_new_job_acquires_again(deployments, wiki):
    _server, device, client, lease = deployments
    _own(device.db)
    for page_id in ("a", "b"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    held_before = asyncio.run(device.held())
    second_run = FIRST_RUN + timedelta(hours=1)
    source = asyncio.run(device.db.get_source(SOURCE_ID))
    payload = asyncio.run(document_acquisition_job_payload(device.db, device.doc_store, source))
    lease.lease(payload, leased_at=second_run)
    asyncio.run(
        acquire_document_source(
            payload,
            open_gene=lambda config: WikiGene(dict(config), SOURCE_ID),
            transport=_RouteTransport(client, job="job-1"),
            exports_pdf=False,
            sleep=_no_pause,
        )
    )
    receipt = _process(client, job="job-1")
    # a fails after the manifest response was issued, so the run retries a Document the device was not asked about.
    _fail(device.db, "wiki-a", wiki.item("a"))

    asyncio.run(
        SourceSyncWorker(device.db, AppConfig(), runtime_provider=_RuntimeProvider(device), worker_id="worker-test").run_once()
    )

    assert receipt.status_code == 202, receipt.text
    run = asyncio.run(device.db.get_source_sync_run(receipt.json()["run_id"]))
    assert run is not None and run.status == "failed"
    assert "was not asked to rediscover wiki-a" in (run.error_message or "")
    # The run is not retried on the same acquisition, and nothing it read is kept.
    assert run.next_attempt_at is None
    assert asyncio.run(device.cursor()) == FIRST_RUN
    assert asyncio.run(device.held()) == held_before
    jobs = asyncio.run(device.db.list_current_local_agent_jobs(workspace_id=WORKSPACE, user_id=LOCAL_DEV_USER_ID))
    assert [(job["operation"], job["status"]) for job in jobs] == [(DOCUMENT_ACQUISITION_OPERATION, "queued")]
    assert jobs[0]["payload"]["since"] == payload["since"]
    assert jobs[0]["payload"]["config"] == {"root_pages": [PAGE_ROOT]}


def test_the_daemon_lists_each_excluded_scope_with_an_instance_of_its_own_after_the_fetches(wiki):
    wiki.write("a", "body of a", FIRST_RUN)
    opened: list[dict] = []
    events: list[str] = []

    class Recording:
        async def submit_manifest(self, manifest):
            events.append("manifest")
            events.extend(
                f"excluded:{listing.scope_key}:{sorted(listing.doc_ids)}" for listing in manifest.excluded_listings
            )
            return AcquisitionRequests()

        async def upload_input(self, package):
            events.append(f"input:{package.item.item_id}")

        async def submit_answers(self, answers):
            raise AssertionError("nothing was asked")

    def open_gene(config):
        opened.append(dict(config))
        return WikiGene(dict(config), SOURCE_ID)

    other_root = {"root_pages": ["0a0b0c0d0e0f40718293a4b5c6d7e8f9"]}
    asyncio.run(
        acquire_document_source(
            {
                "config": dict(CONFIG),
                "since": None,
                LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD: [{"scope_key": "scope-1", "config": other_root}],
            },
            open_gene=open_gene,
            transport=Recording(),  # type: ignore[arg-type]
            exports_pdf=False,
        )
    )

    assert opened == [dict(CONFIG), other_root]
    assert wiki.calls == ["discover", "list_scope", "fetch", "list_scope"]
    assert events == ["input:wiki-a", "manifest", "excluded:scope-1:['wiki-a']"]


def _routed_client(client):
    """A daemon API client whose requests reach the device deployment's routes."""

    from urllib.parse import urlsplit

    from memforge.api_target import build_target
    from memforge.tool_client import ToolClient

    tool_client = ToolClient(target=build_target(origin="http://127.0.0.1:8765", edition="oss"), api_token=None)

    def http_json(method: str, url: str, body: dict | None) -> dict:
        response = client.request(method, urlsplit(url).path, json=body)
        if response.status_code >= 400:
            detail = response.json().get("detail")
            return {"error": "MemForge API request failed", "status_code": response.status_code, "detail": detail}
        return response.json() if response.content else {}

    tool_client._http_json = http_json  # type: ignore[method-assign]
    return tool_client


def test_the_daemon_job_acquires_through_the_api_and_starts_the_run(deployments, wiki):
    import memforge.main as main

    server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN - timedelta(hours=1))
    _server_run(server, FIRST_RUN)
    source = asyncio.run(device.db.get_source(SOURCE_ID))
    payload = asyncio.run(document_acquisition_job_payload(device.db, device.doc_store, source))
    lease.lease(payload, leased_at=FIRST_RUN)
    job = {"job_id": "job-0", "attempt_count": 1, "operation": DOCUMENT_ACQUISITION_OPERATION, "payload": payload}

    result = main._run_cloud_document_acquisition_job(job, _routed_client(client))
    asyncio.run(
        SourceSyncWorker(device.db, AppConfig(), runtime_provider=_RuntimeProvider(device), worker_id="worker-test").run_once()
    )

    assert result.get("error") is None, result
    assert result["counts"] == {"discovered": 1, "uploaded": 1, "failed": 0}
    assert result["sync_started"] is True
    assert asyncio.run(_end_state(device)) == asyncio.run(_end_state(server))


def test_the_daemon_job_stops_when_its_lease_is_lost(deployments, wiki):
    import memforge.main as main
    from memforge.local_agent.runner import CloudJobLeaseLost

    _server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN - timedelta(hours=1))
    source = asyncio.run(device.db.get_source(SOURCE_ID))
    payload = asyncio.run(document_acquisition_job_payload(device.db, device.doc_store, source))
    lease.lease(payload, leased_at=FIRST_RUN)
    lease.current = False
    job = {"job_id": "job-0", "attempt_count": 1, "operation": DOCUMENT_ACQUISITION_OPERATION, "payload": payload}

    with pytest.raises(CloudJobLeaseLost):
        main._run_cloud_document_acquisition_job(job, _routed_client(client))


def test_a_daemon_job_above_the_bound_fails_with_the_count_and_is_not_retried(wiki, tmp_path):
    import memforge.main as main
    from memforge.api_target import build_target
    from memforge.local_agent.runner import LocalAgentRunner
    from memforge.local_agent.state import LocalAgentStateStore
    from memforge.tool_client import ToolClient

    count = SNAPSHOT_MANIFEST_MAX_ITEMS + 1
    for page_id in range(count):
        wiki.write(str(page_id), "body", FIRST_RUN)
    client = ToolClient(target=build_target(origin="http://127.0.0.1:8765", edition="oss"), api_token=None)
    completed: list[tuple] = []
    runner = LocalAgentRunner(
        state_store=LocalAgentStateStore(tmp_path / "state.json"),
        cloud_job_handler=lambda job: main._run_cloud_local_agent_job(job, client),
        cloud_jobs_provider=lambda: {
            "jobs": [
                {
                    "job_id": "job-large",
                    "operation": DOCUMENT_ACQUISITION_OPERATION,
                    "source_id": SOURCE_ID,
                    "attempt_count": 1,
                    "payload": {"source_type": "wiki", "config": {"root_pages": [PAGE_ROOT]}, "since": None},
                }
            ]
        },
        cloud_job_completer=lambda job_id, attempt_count, status, result, error=None: (
            completed.append((job_id, status, result, error)) or {"ok": True}
        ),
    )

    runner.run_once(now=FIRST_RUN)

    [(job_id, status, result, error)] = completed
    assert (job_id, status) == ("job-large", "failed")
    assert (result["item_count"], result["retryable"]) == (count, False)
    assert str(count) in (error or "")
    assert "fetch" not in wiki.calls


def test_a_reprocess_on_the_device_ends_as_a_reprocess_on_the_server(deployments, wiki):
    from memforge.pipeline.sync import SourceSyncMode

    server, device, client, lease = deployments
    for page_id in ("a", "b"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    _server_run(server, FIRST_RUN)
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    before = asyncio.run(device.held())
    # The provider changed a's body without a new edit time, so only a reprocess reads it.
    wiki.pages["a"].body = "body of a, corrected"

    asyncio.run(
        server.orchestrator().sync_gene(
            gene=WikiGene(dict(CONFIG), SOURCE_ID),
            source_name="Team wiki",
            source_id=SOURCE_ID,
            execution_mode=SourceSyncMode.REPROCESS,
            reprocess_doc_ids=frozenset({"wiki-a"}),
        )
    )
    source = asyncio.run(device.db.get_source(SOURCE_ID))
    payload = asyncio.run(
        document_acquisition_job_payload(device.db, device.doc_store, source, reprocess_document_ids=("wiki-a",))
    )
    lease.lease(payload, leased_at=FIRST_RUN + timedelta(hours=1))
    wiki.calls.clear()
    asyncio.run(
        acquire_document_source(
            payload,
            open_gene=lambda config: WikiGene(dict(config), SOURCE_ID),
            transport=_RouteTransport(client, job="job-1"),
            exports_pdf=False,
            sleep=_no_pause,
        )
    )
    processed = _process(client, job="job-1")
    asyncio.run(
        SourceSyncWorker(device.db, AppConfig(), runtime_provider=_RuntimeProvider(device), worker_id="worker-test").run_once()
    )

    assert processed.status_code == 202, processed.text
    # A reprocess discovers and lists nothing; it reads the Document again by id.
    assert wiki.calls == ["rediscover", "fetch"]
    held = asyncio.run(device.held())
    assert held == asyncio.run(server.held())
    assert held["wiki-a"][1] != before["wiki-a"][1]
    assert held["wiki-b"] == before["wiki-b"]
    assert asyncio.run(device.cursor()) == asyncio.run(server.cursor()) == FIRST_RUN


def test_an_attachment_the_device_cannot_read_fails_its_document_as_on_the_server(deployments, wiki):
    server, device, client, lease = deployments
    for page_id in ("a", "b"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    wiki.attachments["a"] = _png()
    wiki.unreadable_attachments = {"a"}

    _server_run(server, FIRST_RUN)
    _device_run(device, client, lease, FIRST_RUN, job="job-0")

    assert asyncio.run(_end_state(device)) == asyncio.run(_end_state(server))
    # The attachment fails its page, not the run: b commits and a is in the ledger.
    assert sorted(asyncio.run(device.held())) == ["wiki-b"]
    stage, error_class, message, retryable = asyncio.run(device.ledger())["wiki-a"]
    assert error_class.endswith("AttachmentUnreadable")
    assert message == "the wiki could not serve the attachment of page a"
    assert retryable is False
    # The server run tries the page again within its run; the device run tried
    # the provider again on the device, so its run replays the recorded error once.
    assert server.retry_sleeps == [2, 4]
    assert device.retry_sleeps == []
    package = AcquiredInputPackage.decode(
        asyncio.run(
            asyncio.to_thread(
                device.doc_store.read_artifact,
                asyncio.run(
                    device.db.get_source_sync_snapshot_input(
                        source_id=SOURCE_ID, workspace_id=WORKSPACE, snapshot_id="job-0:attempt:1", doc_id="wiki-a"
                    )
                ).raw_uri,
            )
        )
    )
    assert [(entry.input_sha256, entry.failure is not None) for entry in package.artifacts] == [(None, True)]


def test_an_acquired_input_must_name_only_artifacts_that_were_uploaded(deployments, wiki):
    _server, device, client, lease = deployments
    wiki.write("a", "body of a", FIRST_RUN)
    _lease_job(device, lease)
    artifact = RawSourceArtifact(
        provider_key="attachment-a",
        parent_observation_type="document",
        parent_provider_key="wiki-a",
        provider_revision="1",
        filename="diagram.png",
        media_type="image/png",
        declared_size_bytes=None,
        locator={"page_id": "a"},
    )

    def upload(acquired: AcquiredArtifact):
        package = AcquiredInputPackage(
            item=wiki.item("a"), body=b"body of a", content_type="text/markdown", artifacts=(acquired,)
        )
        return client.post(
            f"/api/v1/sources/{SOURCE_ID}/acquisition/inputs",
            json={"local_agent_job_id": "job-1", "local_agent_attempt_count": 1, "package": package.to_payload()},
        )

    never_uploaded = upload(AcquiredArtifact(artifact, input_sha256="0" * 64))
    failed = upload(
        AcquiredArtifact(
            artifact,
            failure=AcquisitionError(failure_class="tests.AttachmentUnreadable", message="gone", transient=False),
        )
    )

    assert never_uploaded.status_code == 409
    assert never_uploaded.json()["detail"] == "acquired input names an artifact that was not uploaded"
    # An artifact the device could not read names its error, and no upload.
    assert failed.status_code == 200, failed.text


def test_a_run_that_fails_on_a_recorded_provider_error_acquires_again(deployments, wiki):
    _server, device, client, lease = deployments
    _own(device.db)
    for page_id in ("a", "b"):
        wiki.write(page_id, f"body of {page_id}", FIRST_RUN - timedelta(hours=1))
    _device_run(device, client, lease, FIRST_RUN, job="job-0")
    _fail(device.db, "wiki-b", wiki.item("b"))
    # The provider cannot look b up this time, so the run cannot read the Document it retries.
    wiki.refused = {"b"}

    requests = _device_run(device, client, lease, FIRST_RUN + timedelta(hours=1), job="job-1")

    assert [item.item_id for item in requests.rediscover] == ["wiki-b"]
    run = asyncio.run(device.db.get_latest_source_sync_run(source_id=SOURCE_ID, workspace_id=WORKSPACE))
    assert run is not None and run.status == "failed"
    assert "could not look up page b" in (run.error_message or "")
    # The same acquisition answers the same error, so the run is not retried on
    # it: a new job asks the provider again.
    assert run.next_attempt_at is None
    jobs = asyncio.run(device.db.list_current_local_agent_jobs(workspace_id=WORKSPACE, user_id=LOCAL_DEV_USER_ID))
    assert [(job["operation"], job["status"]) for job in jobs] == [(DOCUMENT_ACQUISITION_OPERATION, "queued")]


def test_the_manifest_response_asks_for_the_documents_of_staged_derivations(deployments, wiki):
    _server, device, client, lease = deployments
    _configure(device, {**CONFIG, "rediscovers": False})
    wiki.write("s", "body of s", FIRST_RUN)
    staged_item = wiki.item("s")
    body = "body of s"
    projection = project_source_item(
        source_id=SOURCE_ID,
        source_type="wiki",
        run_id="run-1",
        item=staged_item,
        raw=RawContent(item=staged_item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=staged_item, markdown_body=body),
        scope={},
        access_context={"visibility": "workspace"},
    )
    document = DocumentRecord(
        doc_id=staged_item.item_id,
        source=SOURCE_ID,
        source_url=staged_item.source_url,
        title=staged_item.title,
        space_or_project="team_alpha",
        author=None,
        last_modified=FIRST_RUN,
        labels=[],
        version=staged_item.version,
        content_hash="hash",
        token_count=20,
        last_synced=FIRST_RUN,
    )
    context = SourceUnitDerivationContext(
        document=document,
        doc_type="document",
        project_key=None,
        repo_identifier=None,
        document_content=projection.observation_revisions[0].content,
        update_mode="new",
        changed_hunks=None,
        update_plan_stats=None,
        source_updated_at=FIRST_RUN.isoformat(),
        user_id=None,
    )
    asyncio.run(device.db.stage_source_derivation(source_derivation_manifest(projection, (), context=context)))
    _lease_job(device, lease)
    listing = AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"wiki-s"}))

    response = _manifest_route(client, AcquisitionManifest(listing=listing))

    assert response.status_code == 200, response.text
    # Recovery may end the staged derivation, and a requested run then fetches its Document again.
    requests = AcquisitionRequests.from_payload(response.json()["requests"])
    assert [item.item_id for item in requests.fetch] == ["wiki-s"]
