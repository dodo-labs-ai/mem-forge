"""The server's side of a device run of a document Source (ADR 0058).

A device run makes the provider calls a server run makes, on the owner's
device, and the server's run then interprets what they returned with the
type's recorded Gene (:mod:`memforge.genes.recorded`). This module holds what
the server decides and keeps for such a run:

- :func:`document_acquisition_job_payload` builds the job: the configuration
  without secrets, the ``since`` a server run would discover from, the
  excluded scopes once the open transition's ``excluded_scopes_due`` gate
  says so, and the Documents of a reprocess.
- :func:`enqueue_document_acquisition_job` queues that job for the Source's
  execution owner.
- :func:`acquisition_requests` answers a submitted manifest with what the
  server run will ask beyond discovery, by the server run's own rules.
- :func:`retain_acquired_input` keeps one acquired input for its attempt.
- :func:`ready_acquisition` decides whether the attempt can be processed.
- :func:`recorded_device_gene` and :func:`device_excluded_scope_gene_factory`
  give the run the type's Gene with acquisition answered from the attempt.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import uuid
from collections.abc import AsyncIterator, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from memforge.genes import gene_class, source_rediscovers_documents
from memforge.genes.base import SourceConfigurationError
from memforge.genes.document_gene import DocumentGene
from memforge.genes.recorded import (
    AcquiredInput,
    AcquisitionRecordMismatch,
    recorded_gene_class,
)
from memforge.local_agent.document_acquisition import (
    CONFIG_PAYLOAD_FIELD,
    DOCUMENT_ACQUISITION_OPERATION,
    REPROCESS_PAYLOAD_FIELD,
    SINCE_PAYLOAD_FIELD,
    AcquiredInputPackage,
    AcquisitionManifest,
    AcquisitionRequests,
    StoredAcquisition,
    acquisition_job_config,
    expected_input_ids,
)
from memforge.local_agent.snapshot_manifest import ExcludedScopeListing
from memforge.local_agent.source_contract import (
    LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD,
    LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS,
    LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD,
    execution_owner_user_id,
    local_agent_source_config_revision,
)
from memforge.models import AbsenceAnswer, ContentItem, ScopeListing, ScopeListingKind
from memforge.pipeline.collection_family import DocumentFamily, ExcludedScopeGeneFactory, listing_partition
from memforge.pipeline.stored_document import (
    StoredDocumentUnavailable,
    StoredDocumentUnavailableReason,
    committed_source_item,
    document_content_item,
    stored_source_item,
)
from memforge.pipeline.sync import STAGED_DERIVATION_STATUSES, select_failed_items
from memforge.release import current_release
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload
from memforge.source_item_failures import FailedItemRetry
from memforge.source_projection_config import (
    ExcludedScopesUnavailable,
    canonical_projection_scope,
    canonical_scope_transition,
    excluded_projection_scopes,
    excluded_scope_config,
    projection_scope_fingerprint,
)

if TYPE_CHECKING:
    from memforge.storage.database import Database
    from memforge.storage.document_store import DocumentStore

__all__ = [
    "AcquisitionNotReady",
    "DeviceAcquisitionRecordStore",
    "acquisition_requests",
    "device_excluded_scope_gene_factory",
    "document_acquisition_job_payload",
    "enqueue_document_acquisition_job",
    "ready_acquisition",
    "recorded_device_gene",
    "retain_acquired_input",
]

ACQUIRED_INPUT_METADATA_KEY = "acquired_input"
"""The metadata key of a Source Sync Input that holds one acquired input package."""


def _document_gene_class(source_type: str) -> type[DocumentGene]:
    return cast(type[DocumentGene], gene_class(source_type))


async def document_acquisition_job_payload(
    db: Database,
    doc_store: DocumentStore,
    source: Mapping[str, Any],
    request_payload: Mapping[str, Any] | None = None,
    *,
    reprocess_document_ids: Collection[str] = (),
) -> dict[str, Any]:
    """The payload of one ``document_acquisition`` job, as of the moment it is enqueued.

    - ``config``: the configuration the device runs the type's Gene with,
      without secrets.
    - ``since``: the instant a server run would discover from now, by the
      document family's own rule, as an ISO time, or ``None`` to discover
      everything.
    - ``excluded_scopes``: while a query transition is open and its latest
      run reached the step that lists its excluded scopes
      (``excluded_scopes_due``), the scope key and configuration of each
      distinct excluded scope, which the device lists with its own Gene.
    - ``reprocess``: the Documents of a Reprocess at Current Revision, which
      discovers and lists nothing.

    It also names the Source, its type and configuration revision, and the
    request's control fields (``force_full_sync``, ``retry_failed_items``).
    """

    source_id = str(source["id"])
    source_type = str(source["type"])
    config = dict(source.get("config") or {})
    payload: dict[str, Any] = {
        key: value for key, value in (request_payload or {}).items() if key in LOCAL_AGENT_SYNC_PAYLOAD_CONTROL_FIELDS
    }
    payload.update(
        {
            "operation": DOCUMENT_ACQUISITION_OPERATION,
            "source_id": source_id,
            "source_type": source_type,
            "source_config_revision": local_agent_source_config_revision(source),
            CONFIG_PAYLOAD_FIELD: acquisition_job_config(source_type, config),
        }
    )
    if reprocess_document_ids:
        payload[REPROCESS_PAYLOAD_FIELD] = sorted(reprocess_document_ids)
        payload[SINCE_PAYLOAD_FIELD] = None
        return payload
    transition = await db.get_open_projection_scope_transition(source_id)
    if transition is not None:
        transition = canonical_scope_transition(source_type, transition, canonical_projection_scope(source_type, config))
    family = DocumentFamily(
        _document_gene_class(source_type)(config=config, source_id=source_id),
        db=db,
        doc_store=doc_store,
        source_id=source_id,
    )
    family.prepare(collection_attempt=None, input_snapshot=None)
    since = await family.since(
        existing_state=await db.get_sync_state(source_id),
        indexed_doc_ids=await db.list_indexed_doc_ids(source_id),
        full_discovery=bool(payload.get("force_full_sync")) or transition is not None,
    )
    payload[SINCE_PAYLOAD_FIELD] = since.isoformat() if since is not None else None
    if transition is None:
        return payload
    payload[LOCAL_AGENT_TRANSITION_PAYLOAD_FIELD] = {
        "id": transition.id,
        "previous_scope": dict(transition.previous_scope),
        "target_scope": dict(transition.target_scope),
    }
    if transition.excluded_scopes_due and ScopeListingKind.QUERY in _document_gene_class(source_type).listing_kinds:
        try:
            scopes = await excluded_projection_scopes(db, transition, source_type=source_type, config=config)
        except ExcludedScopesUnavailable:
            scopes = []
        if scopes:
            payload[LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD] = [
                {
                    "scope_key": projection_scope_fingerprint(scope),
                    "config": acquisition_job_config(source_type, excluded_scope_config(source_type, config, scope)),
                }
                for scope in scopes
            ]
    return payload


async def enqueue_document_acquisition_job(
    db: Database,
    doc_store: DocumentStore,
    source: Mapping[str, Any],
    request_payload: Mapping[str, Any] | None = None,
    *,
    workspace_id: str,
    trigger: str,
) -> tuple[str, bool] | None:
    """Queue a ``document_acquisition`` job for the Source's execution owner, as of now.

    The job coalesces with one already queued or leased, as every local-agent
    job does. Returns its id and whether a new job was created, or ``None``
    for a Source without an execution owner, whose runs no device takes.
    """

    owner = execution_owner_user_id(source)
    if owner is None:
        return None
    payload = await document_acquisition_job_payload(db, doc_store, source, request_payload)
    return await db.enqueue_local_agent_job(
        job_id=f"laj-{uuid.uuid4().hex}",
        workspace_id=workspace_id,
        source_id=str(source["id"]),
        source_type=str(source["type"]),
        operation=DOCUMENT_ACQUISITION_OPERATION,
        payload=payload,
        created_by_user_id=owner,
        execution_owner_user_id=owner,
        trigger=trigger,
    )


async def acquisition_requests(
    db: Database,
    source: Mapping[str, Any],
    manifest: AcquisitionManifest,
    *,
    reprocess_document_ids: Collection[str] = (),
) -> AcquisitionRequests:
    """What the server run of this manifest will ask beyond discovery, by the server run's own rules.

    - ``confirm``: when the listing is a query, every Document with a current
      Unit the listing leaves out, by the item its Unit records. The run
      confirms these, or the ones of them a completing transition does not
      remove by its excluded listings.
    - ``rediscover``, for a Source that can ask its provider for one Document
      by id: each Document the run may retry, by the item its committed Unit
      records or, for one that never became a Unit, by the item its failure
      recorded. Those are the failure-ledger rows a sync a person requested
      retries, and the Documents of the staged derivations its recovery may
      end, less the Documents this manifest discovered and the rows its
      listing settles or keeps untried. A run of an open transition retries
      none.
    - ``fetch``, for any other Source: the item each such Document's failure
      recorded, as discovery returned it. One without such an item is
      replayed from its Unit's stored input on the server, and asks nothing
      of the device.

    The response is a superset of what the run asks. A scheduled run retries
    fewer rows than a requested one, but a person's request that coalesces
    into the run asks for every row, and a run that asks for an item the
    device was not asked about fails and acquires again. Recovery may end a
    staged derivation, or a run of a later release may record its Document
    as failed, only once the run starts, so each of them is asked for.

    A reprocess asks for its Documents the same way, and confirms nothing.
    """

    source_id = str(source["id"])
    rediscovering = source_rediscovers_documents(str(source["type"]), source.get("config") or {})
    ledger = {failure.document_id: failure for failure in await db.list_source_item_failures(source_id=source_id)}
    failed_items = {doc_id: failure.item for doc_id, failure in ledger.items() if failure.item is not None}
    confirm: list[ContentItem] = []
    if reprocess_document_ids:
        retried = sorted(reprocess_document_ids)
    else:
        listing = manifest.listing.scope_listing()
        retried_doc_ids: set[str] = set()
        if await db.get_open_projection_scope_transition(source_id) is None:
            staged = {
                attempt.context.document.doc_id: document_content_item(attempt.context.document)
                for attempt in await db.list_source_derivation_attempts(
                    source_id=source_id, statuses=STAGED_DERIVATION_STATUSES
                )
            }
            retried_doc_ids, _left_alone = select_failed_items(
                ledger, retry=FailedItemRetry.ALL, release=current_release(), recovered_doc_ids=set(), ended=staged
            )
            failed_items = {**staged, **failed_items}
        partition = listing_partition(listing, ledger, rediscovering=rediscovering)
        discovered = {item.item_id for item in manifest.discovered}
        retried = sorted(retried_doc_ids - discovered - partition.settled - partition.untried)
        if listing is not None and listing.kind is ScopeListingKind.QUERY:
            for doc_id in sorted(await db.list_indexed_doc_ids(source_id) - listing.doc_ids):
                with contextlib.suppress(StoredDocumentUnavailable):
                    confirm.append(await stored_source_item(db, source_id=source_id, document_id=doc_id))
    rediscover: list[ContentItem] = []
    fetch: list[ContentItem] = []
    for doc_id in retried:
        failed_item = failed_items.get(doc_id)
        if not rediscovering:
            if failed_item is not None:
                fetch.append(failed_item)
            continue
        item = await _rediscovery_item(db, source_id, doc_id, failed_item)
        if item is not None:
            rediscover.append(item)
    return AcquisitionRequests(confirm=tuple(confirm), rediscover=tuple(rediscover), fetch=tuple(fetch))


async def _rediscovery_item(
    db: Database, source_id: str, doc_id: str, failed_item: ContentItem | None
) -> ContentItem | None:
    """The item a server run asks ``rediscover`` with for one Document, ``None`` when it asks nothing."""

    try:
        return await committed_source_item(db, source_id=source_id, document_id=doc_id)
    except StoredDocumentUnavailable as exc:
        if exc.reason is StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING:
            return failed_item
        return None


async def retain_acquired_input(
    db: Database,
    doc_store: DocumentStore,
    *,
    source_id: str,
    workspace_id: str,
    snapshot_id: str,
    package: AcquiredInputPackage,
) -> str:
    """Keep one acquired input as an immutable input of the Source, held by its attempt, and return its hash.

    The same package uploaded again, by this attempt or a later one, is the
    same input.
    """

    body = package.encode()
    digest = hashlib.sha256(body).hexdigest()
    doc_id = package.item.item_id
    metadata = {"doc_id": doc_id, ACQUIRED_INPUT_METADATA_KEY: {"doc_id": doc_id, "version": package.item.version}}
    existing = await db.get_source_sync_inputs_by_raw_hashes(
        source_id=source_id, workspace_id=workspace_id, raw_sha256s=(digest,)
    )
    if existing:
        raw_uri = existing[0].raw_uri
    else:
        # This write does not hold the Source activity lease, so the attempt
        # writes a key of its own: a cleanup task queued by an earlier failed
        # attempt can never name the object written here.
        raw_uri = await asyncio.to_thread(
            doc_store.store_source_artifact,
            source_id=source_id,
            artifact_id=f"acquired-input-{digest[:24]}-{uuid.uuid4().hex}",
            filename="acquired-input.json",
            content=io.BytesIO(body),
            content_type="application/json",
            size_bytes=len(body),
            sha256=digest,
        )
    try:
        retained = await db.create_source_sync_input(
            source_id=source_id,
            workspace_id=workspace_id,
            raw_uri=raw_uri,
            raw_sha256=digest,
            raw_content_type="application/json",
            metadata=metadata,
            sync_snapshot_id=snapshot_id,
        )
    except Exception:
        if not existing:
            await db.enqueue_source_artifact_cleanup_task(source_id=source_id, artifact_uri=raw_uri)
        raise
    if not existing and retained.raw_uri != raw_uri:
        await db.enqueue_source_artifact_cleanup_task(source_id=source_id, artifact_uri=raw_uri)
    return digest


class AcquisitionNotReady(Exception):
    """A device run's attempt cannot be processed yet; ``code`` says what it waits for."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def ready_acquisition(
    db: Database,
    *,
    source_id: str,
    workspace_id: str,
    snapshot_id: str,
    local_agent_job_id: str,
    local_agent_attempt_count: int,
    source_config_revision: str,
) -> StoredAcquisition:
    """The acquisition of one job attempt, once the server run can be started on it.

    It is ready once its manifest is kept for this attempt and configuration,
    the device answered what the manifest response asked, and every
    discovered item, every ``fetch`` item and every ``rediscover`` item the
    provider still returned has an acquired input or the error ``fetch``
    raised for it. Raises :class:`AcquisitionNotReady` with what it waits for:
    ``acquisition_manifest_required``, ``acquisition_manifest_stale``,
    ``acquisition_answers_pending`` or ``acquisition_inputs_incomplete``.
    """

    stored = await db.get_source_acquisition(source_id=source_id, workspace_id=workspace_id, snapshot_id=snapshot_id)
    if stored is None:
        raise AcquisitionNotReady("acquisition_manifest_required")
    if (stored.local_agent_job_id, stored.local_agent_attempt_count, stored.source_config_revision) != (
        local_agent_job_id,
        local_agent_attempt_count,
        source_config_revision,
    ):
        raise AcquisitionNotReady("acquisition_manifest_stale")
    if stored.requests.awaits_answers and stored.answers is None:
        raise AcquisitionNotReady("acquisition_answers_pending")
    held = await db.list_source_sync_snapshot_doc_ids(
        source_id=source_id, workspace_id=workspace_id, snapshot_id=snapshot_id
    )
    if expected_input_ids(stored.manifest, stored.requests, stored.answers) - held:
        raise AcquisitionNotReady("acquisition_inputs_incomplete")
    return stored


class DeviceAcquisitionRecord:
    """One job attempt's acquisition, open as the acquisition record of the run that interprets it."""

    def __init__(
        self,
        db: Database,
        doc_store: DocumentStore,
        stored: StoredAcquisition,
        *,
        source_id: str,
    ) -> None:
        self._db = db
        self._doc_store = doc_store
        self._stored = stored
        self._source_id = source_id
        self._workspace_id = stored.workspace_id
        self._packages: dict[str, AcquiredInputPackage] = {}

    def discovered(self) -> Sequence[ContentItem]:
        return self._stored.manifest.discovered

    def listing(self) -> ScopeListing:
        acquired = self._stored.manifest.listing
        if acquired.failure is not None:
            raise acquired.failure.raised()
        listing = acquired.scope_listing()
        if listing is None:
            raise AcquisitionRecordMismatch("the device took no listing of the scope")
        return listing

    def absence_answer(self, item_id: str) -> AbsenceAnswer:
        answers = self._stored.answers
        if answers is not None and answers.absence_failure is not None:
            raise answers.absence_failure.raised()
        if answers is None or item_id not in answers.absence:
            raise AcquisitionRecordMismatch(f"the device was not asked to confirm {item_id}")
        return answers.absence[item_id]

    def rediscovered(self, item_id: str) -> ContentItem | None:
        answers = self._stored.answers
        if answers is not None and item_id in answers.rediscover_failures:
            raise answers.rediscover_failures[item_id].raised()
        if answers is None or item_id not in answers.rediscovered:
            raise AcquisitionRecordMismatch(f"the device was not asked to rediscover {item_id}")
        return answers.rediscovered[item_id]

    async def _package(self, item_id: str) -> AcquiredInputPackage:
        package = self._packages.get(item_id)
        if package is not None:
            return package
        source_input = await self._db.get_source_sync_snapshot_input(
            source_id=self._source_id,
            workspace_id=self._workspace_id,
            snapshot_id=self._stored.snapshot_id,
            doc_id=item_id,
        )
        if source_input is None:
            raise AcquisitionRecordMismatch(f"item {item_id} was not acquired")
        body = await asyncio.to_thread(self._doc_store.read_artifact, source_input.raw_uri)
        package = AcquiredInputPackage.decode(body)
        if package.item.item_id != item_id:
            raise AcquisitionRecordMismatch(f"the input acquired for {item_id} holds {package.item.item_id}")
        self._packages[item_id] = package
        return package

    async def acquired(self, item_id: str) -> AcquiredInput:
        package = await self._package(item_id)
        if package.failure is not None:
            raise package.failure.raised()
        return AcquiredInput(
            item=package.item,
            body=package.body,
            content_type=package.content_type,
            pdf=package.pdf,
            artifacts=tuple(acquired.artifact for acquired in package.artifacts),
            pdf_failure=package.pdf_failure.raised() if package.pdf_failure is not None else None,
        )

    @contextlib.asynccontextmanager
    async def open_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
        acquired = next(
            (
                acquired
                for package in self._packages.values()
                for acquired in package.artifacts
                if acquired.artifact == artifact
            ),
            None,
        )
        if acquired is None:
            raise AcquisitionRecordMismatch(f"artifact {artifact.provider_key} was not acquired")
        if acquired.failure is not None:
            raise acquired.failure.raised()
        inputs = await self._db.get_source_sync_inputs_by_raw_hashes(
            source_id=self._source_id, workspace_id=self._workspace_id, raw_sha256s=(str(acquired.input_sha256),)
        )
        if not inputs:
            raise AcquisitionRecordMismatch(f"the input of artifact {artifact.provider_key} is not kept")
        body = await asyncio.to_thread(self._doc_store.read_artifact, inputs[0].raw_uri)

        async def chunks() -> AsyncIterator[bytes]:
            yield body

        yield SourceArtifactDownload(chunks=chunks(), media_type=artifact.media_type, content_length=len(body))

    async def close(self) -> None:
        self._packages.clear()


class DeviceAcquisitionRecordStore:
    """The acquisition one job attempt handed over, as the record store of its server run."""

    def __init__(
        self,
        db: Database,
        doc_store: DocumentStore,
        stored: StoredAcquisition,
        *,
        source_id: str,
    ) -> None:
        self._db = db
        self._doc_store = doc_store
        self._stored = stored
        self._source_id = source_id

    async def open(self) -> DeviceAcquisitionRecord:
        return DeviceAcquisitionRecord(self._db, self._doc_store, self._stored, source_id=self._source_id)


def recorded_device_gene(
    db: Database,
    doc_store: DocumentStore,
    source: Mapping[str, Any],
    source_config: dict[str, Any],
    stored: StoredAcquisition,
) -> DocumentGene:
    """The Source type's Gene with acquisition answered from one job attempt's acquisition."""

    source_id = str(source["id"])
    recorded = recorded_gene_class(_document_gene_class(str(source["type"])))
    return recorded(
        config=source_config,
        source_id=source_id,
        record_store=DeviceAcquisitionRecordStore(db, doc_store, stored, source_id=source_id),
    )


class ExcludedScopeNotListed(RuntimeError):
    """The device did not list a scope the open transition excludes, so the transition stays open."""


class _ExcludedScopeRecord:
    """What the device listed in one excluded scope; a run asks an excluded scope only for its listing."""

    def __init__(self, scope_key: str, listing: ExcludedScopeListing | None) -> None:
        self._scope_key = scope_key
        self._listing = listing

    def discovered(self) -> Sequence[ContentItem]:
        return ()

    def listing(self) -> ScopeListing:
        if self._listing is None:
            raise ExcludedScopeNotListed(f"the device did not list excluded scope {self._scope_key}")
        if self._listing.rejection is not None:
            raise SourceConfigurationError(self._listing.rejection)
        return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=self._listing.doc_ids)

    def absence_answer(self, item_id: str) -> AbsenceAnswer:
        raise AcquisitionRecordMismatch("an excluded scope is only listed")

    def rediscovered(self, item_id: str) -> ContentItem | None:
        raise AcquisitionRecordMismatch("an excluded scope is only listed")

    async def acquired(self, item_id: str) -> AcquiredInput:
        raise AcquisitionRecordMismatch("an excluded scope is only listed")

    def open_artifact(self, artifact: RawSourceArtifact) -> contextlib.AbstractAsyncContextManager[SourceArtifactDownload]:
        raise AcquisitionRecordMismatch("an excluded scope is only listed")

    async def close(self) -> None:
        return None


class _ExcludedScopeRecordStore:
    def __init__(self, record: _ExcludedScopeRecord) -> None:
        self._record = record

    async def open(self) -> _ExcludedScopeRecord:
        return self._record


def device_excluded_scope_gene_factory(
    source: Mapping[str, Any], source_config: Mapping[str, Any], stored: StoredAcquisition
) -> ExcludedScopeGeneFactory:
    """Genes that list the scopes a Source's scope change excluded, as the device listed them.

    Each is the type's recorded Gene over the excluded scope's configuration.
    Its listing is the one the device took of that scope; a scope the device
    reported rejected raises ``SourceConfigurationError``, and one the device
    did not list raises :class:`ExcludedScopeNotListed`, which keeps the
    transition open.
    """

    source_type = str(source["type"])
    source_id = str(source["id"])
    recorded = recorded_gene_class(_document_gene_class(source_type))
    listings = {listing.scope_key: listing for listing in stored.manifest.excluded_listings}

    @contextlib.asynccontextmanager
    async def excluded_scope_gene(scope: Mapping[str, object]) -> AsyncIterator[DocumentGene]:
        try:
            config = excluded_scope_config(source_type, source_config, scope)
        except ValueError as exc:
            raise SourceConfigurationError(f"The excluded scope {dict(scope)} cannot be configured: {exc}") from exc
        scope_key = projection_scope_fingerprint(scope)
        gene = recorded(
            config=config,
            source_id=source_id,
            record_store=_ExcludedScopeRecordStore(_ExcludedScopeRecord(scope_key, listings.get(scope_key))),
        )
        try:
            await gene.authenticate()
            yield gene
        finally:
            await gene.close()

    return excluded_scope_gene
