"""The device's side of a document Source's run: the type's own Gene, run on the owner's machine (ADR 0058).

:func:`acquire_document_source` runs one ``document_acquisition`` job. One
Gene instance serves the Source's configuration for the whole attempt,
authenticated once, because a Gene may keep what discovery read for
``fetch``; each excluded scope is listed by an instance of its own, as on the
server. The calls are the ones a server run makes, in this order:

1. ``discover(since)``, then ``list_scope`` when the configuration lists;
2. ``fetch`` of each discovered item, with its PDF export when the type
   exports PDFs and the artifacts it describes, each uploaded as an acquired
   input whose body travels only when the server does not hold it for the
   Source; an artifact that cannot be read is the item's failure, as it is in
   a server run, and not the attempt's. Each item ``fetch`` reads carries the
   ``stored_extra`` a server run gives it, read from the server first;
3. ``list_scope`` of each excluded scope the job carries;
4. the manifest is submitted, and its response names what the server run
   will ask beyond discovery;
5. ``confirm_absent`` and ``rediscover`` for the items the response named,
   ``fetch`` of each item ``rediscover`` still finds and of each ``fetch``
   item, and the answers.

A server run retries failed Documents before it fetches the discovered
items; the device does so after the manifest response. The calls are
independent, so the order changes no result.

A reprocess job discovers and lists nothing; it answers only what its
manifest response names. The server run retries a Document whose processing
failed unless the error declares it may not pass, so the device fetches
again the same way before it records an error as the item's acquired input.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Protocol, TypeVar

from memforge.genes.base import SourceConfigurationError
from memforge.genes.document_gene import DocumentGene
from memforge.genes.toolkit.pdf import PdfMaterial
from memforge.local_agent.document_acquisition import (
    CONFIG_PAYLOAD_FIELD,
    REPROCESS_PAYLOAD_FIELD,
    SINCE_PAYLOAD_FIELD,
    AcquiredArtifact,
    AcquiredBodyRequired,
    AcquiredInputPackage,
    AcquiredListing,
    AcquisitionAnswers,
    AcquisitionError,
    AcquisitionManifest,
    AcquisitionRequests,
    body_sha256,
)
from memforge.local_agent.device_manifest import DEVICE_MANIFEST_MAX_ITEMS, ExcludedScopeListing
from memforge.local_agent.source_contract import LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD
from memforge.models import AbsenceAnswer, ContentItem
from memforge.pipeline.sync import MAX_RETRIES
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT,
    RawSourceArtifact,
    read_source_artifact_bytes,
)

__all__ = ["AcquisitionSummary", "AcquisitionTransport", "acquire_document_source"]

logger = logging.getLogger(__name__)

T = TypeVar("T")

GeneOpener = Callable[[Mapping[str, Any]], DocumentGene]
"""Creates the type's Gene for one configuration, with the device's own credentials."""


class AcquisitionTransport(Protocol):
    """How one job attempt hands the server what its acquisition returned."""

    async def submit_manifest(self, manifest: AcquisitionManifest) -> AcquisitionRequests:
        """Submit the manifest; the response names what the server run will ask beyond discovery."""

    async def upload_artifact(self, item: ContentItem, artifact: RawSourceArtifact, body: bytes) -> str:
        """Upload one artifact body an item's fetch described, and return the hash of the input that holds it."""

    async def held_bodies(self, body_sha256s: Sequence[str]) -> Collection[str]:
        """Which of the named bodies the server already holds for the Source."""

    async def upload_input(self, package: AcquiredInputPackage, body: bytes | None) -> None:
        """Upload one acquired input, before or after the manifest, with its body unless the server holds it.

        Raises :class:`~memforge.local_agent.document_acquisition.AcquiredBodyRequired`
        when ``body`` is ``None`` and the server no longer holds the body the
        package names.
        """

    async def stored_extras(self, item_ids: Sequence[str]) -> Mapping[str, Mapping[str, Any]]:
        """The ``extra`` the Source stored with each named item, for the items it has stored."""

    async def submit_answers(self, answers: AcquisitionAnswers) -> None:
        """Submit the answers to what the manifest response asked."""


@dataclass(frozen=True, slots=True)
class AcquisitionSummary:
    """What one attempt acquired: items discovered, inputs uploaded, and fetches that failed."""

    discovered: int
    uploaded: int
    failed: int


async def acquire_document_source(
    payload: Mapping[str, Any],
    *,
    open_gene: GeneOpener,
    transport: AcquisitionTransport,
    exports_pdf: bool,
    sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
) -> AcquisitionSummary:
    """Acquire one document Source's run on the device and hand it to the server.

    A listing or discovery above the bound every device manifest has raises
    :class:`~memforge.local_agent.document_acquisition.AcquisitionManifestTooLarge`
    with the count, before any item is fetched. An error of ``discover`` or
    ``authenticate`` fails the attempt, as it fails a server run.
    """

    config = dict(payload.get(CONFIG_PAYLOAD_FIELD) or {})
    since_value = payload.get(SINCE_PAYLOAD_FIELD)
    since = datetime.fromisoformat(str(since_value)) if since_value else None
    reprocessing = bool(payload.get(REPROCESS_PAYLOAD_FIELD))
    gene = open_gene(config)
    try:
        await gene.authenticate()
        acquirer = _Acquirer(gene, transport, exports_pdf=exports_pdf, sleep=sleep)
        manifest = AcquisitionManifest()
        if not reprocessing:
            discovered = [item async for item in gene.discover(since=since)]
            listing = AcquiredListing()
            if type(gene).lists_scope(config):
                try:
                    listing = AcquiredListing.of(await gene.list_scope())
                except Exception as exc:
                    listing = AcquiredListing(failure=AcquisitionError.from_exception(exc))
            # The manifest names the items as discovery yielded them; ``fetch`` may change them.
            manifest = AcquisitionManifest(listing=listing, discovered=tuple(copy.deepcopy(discovered)))
            manifest.check_bound()
            await acquirer.upload_each(discovered)
            excluded: list[ExcludedScopeListing] = []
            for entry in payload.get(LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD) or ():
                scope_listing = await _list_excluded_scope(open_gene, entry)
                if scope_listing is not None:
                    excluded.append(scope_listing)
            manifest = replace(manifest, excluded_listings=tuple(excluded))
        requests = await transport.submit_manifest(manifest)
        answers, found = await _answer(gene, requests)
        await acquirer.upload_each([*found, *requests.fetch])
        if requests.awaits_answers:
            await transport.submit_answers(answers)
        return AcquisitionSummary(discovered=len(manifest.discovered), uploaded=acquirer.uploaded, failed=acquirer.failed)
    finally:
        await gene.close()


async def _list_excluded_scope(open_gene: GeneOpener, entry: Mapping[str, Any]) -> ExcludedScopeListing | None:
    """List one excluded scope with an instance of its own, or report it rejected.

    A configuration the provider or the Gene rejects, and a scope that lists
    more items than a manifest may name, are definitive rejections. A scope
    the device cannot list for any other reason is left out, and the
    transition stays open until a manifest lists it.
    """

    scope_key = str(entry["scope_key"])
    try:
        gene = open_gene(dict(entry.get("config") or {}))
    except ValueError as exc:
        return ExcludedScopeListing(scope_key=scope_key, rejection=f"the scope cannot be configured: {exc}")
    try:
        await gene.authenticate()
        listing = await gene.list_scope()
    except SourceConfigurationError as exc:
        return ExcludedScopeListing(scope_key=scope_key, rejection=str(exc) or type(exc).__name__)
    except Exception as exc:
        logger.warning("Could not list excluded scope %s; its transition stays open: %s", scope_key, exc)
        return None
    finally:
        await gene.close()
    if len(listing.doc_ids) > DEVICE_MANIFEST_MAX_ITEMS:
        return ExcludedScopeListing(
            scope_key=scope_key,
            rejection=f"the scope lists {len(listing.doc_ids)} items, more than a manifest may name",
        )
    return ExcludedScopeListing(scope_key=scope_key, doc_ids=frozenset(listing.doc_ids))


async def _answer(gene: DocumentGene, requests: AcquisitionRequests) -> tuple[AcquisitionAnswers, list[ContentItem]]:
    """Answer ``confirm`` and ``rediscover``; return the answers and each item ``rediscover`` still finds as itself.

    An item ``confirm_absent`` leaves unanswered is answered ``exists``: the
    server run keeps every item it is not told is absent, and counts it as
    present. The server run fetches each item ``rediscover`` still finds, so
    the device fetches it too.
    """

    absence: dict[str, AbsenceAnswer] = {}
    absence_failure: AcquisitionError | None = None
    if requests.confirm:
        try:
            answered = await gene.confirm_absent(list(requests.confirm))
            absence = {item.item_id: answered.get(item.item_id, AbsenceAnswer.EXISTS) for item in requests.confirm}
        except Exception as exc:
            absence_failure = AcquisitionError.from_exception(exc)
    rediscovered: dict[str, ContentItem | None] = {}
    rediscover_failures: dict[str, AcquisitionError] = {}
    found: list[ContentItem] = []
    for item in requests.rediscover:
        try:
            current = await gene.rediscover(item)
        except Exception as exc:
            rediscover_failures[item.item_id] = AcquisitionError.from_exception(exc)
            continue
        # The answer is the item as rediscover returned it; fetch may change the item it reads.
        rediscovered[item.item_id] = copy.deepcopy(current)
        if current is not None and current.item_id == item.item_id:
            found.append(current)
    answers = AcquisitionAnswers(
        absence=absence,
        absence_failure=absence_failure,
        rediscovered=rediscovered,
        rediscover_failures=rediscover_failures,
    )
    return answers, found


class _Acquirer:
    """Fetches one item at a time with the attempt's Gene and uploads its acquired input."""

    def __init__(
        self,
        gene: DocumentGene,
        transport: AcquisitionTransport,
        *,
        exports_pdf: bool,
        sleep: Callable[[float], Awaitable[object]],
    ) -> None:
        self._gene = gene
        self._transport = transport
        self._exports_pdf = exports_pdf
        self._sleep = sleep
        self.uploaded = 0
        self.failed = 0

    async def upload_each(self, items: Sequence[ContentItem]) -> None:
        """Fetch and upload each item, with the ``stored_extra`` a server run's ``fetch`` reads."""

        if not items:
            return
        extras = await self._transport.stored_extras([item.item_id for item in items])
        for item in items:
            item.stored_extra = dict(extras.get(item.item_id) or {})
            await self._upload(item)

    async def _upload(self, item: ContentItem) -> None:
        """Upload the item's acquired input, and its body only when the server does not hold it."""

        package, body = await self._acquire(item)
        if package.body_sha256 is None:
            await self._transport.upload_input(package, None)
        elif package.body_sha256 in await self._transport.held_bodies([package.body_sha256]):
            try:
                await self._transport.upload_input(package, None)
            except AcquiredBodyRequired:
                await self._transport.upload_input(package, body)
        else:
            await self._transport.upload_input(package, body)
        self.uploaded += 1
        if package.failure is not None:
            self.failed += 1

    async def _acquire(self, item: ContentItem) -> tuple[AcquiredInputPackage, bytes]:
        """The item's acquired input and its body: what ``fetch`` returned, with the item as ``fetch`` left it."""

        try:
            raw = await self._again(lambda: self._gene.fetch(item))
        except Exception as exc:
            return AcquiredInputPackage(item=item, failure=AcquisitionError.from_exception(exc)), b""
        pdf_material: PdfMaterial | None = None
        pdf_failure: AcquisitionError | None = None
        if self._exports_pdf and not raw.authoritative_empty:
            try:
                pdf_material = await self._again(lambda: self._gene.fetch_pdf_material(item))
            except Exception as exc:
                pdf_failure = AcquisitionError.from_exception(exc)
        package = AcquiredInputPackage(
            item=item,
            body_sha256=body_sha256(raw.body),
            content_type=raw.content_type,
            pdf_material=pdf_material,
            pdf_failure=pdf_failure,
            artifacts=await self._acquire_artifacts(item, raw.artifacts),
        )
        return package, raw.body

    async def _acquire_artifacts(
        self, item: ContentItem, artifacts: tuple[RawSourceArtifact, ...]
    ) -> tuple[AcquiredArtifact, ...]:
        """Read and upload each artifact ``fetch`` described, as the server run materializes them.

        Each body is read through the bounds the server's store applies, so a
        body the server would refuse fails here with the error the server run
        raises. An artifact that cannot be read keeps its error, which the
        server run raises where it opens that artifact and fails the Document.
        """

        acquired: list[AcquiredArtifact] = []
        remaining_unit_bytes = MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT
        for artifact in artifacts:
            try:
                body = await self._again(
                    lambda artifact=artifact, remaining=remaining_unit_bytes: read_source_artifact_bytes(
                        artifact=artifact,
                        remaining_unit_bytes=remaining,
                        open_artifact=self._gene.open_source_artifact,
                    )
                )
            except Exception as exc:
                acquired.append(AcquiredArtifact(artifact, failure=AcquisitionError.from_exception(exc)))
                continue
            remaining_unit_bytes -= len(body)
            input_sha256 = await self._transport.upload_artifact(item, artifact, body)
            acquired.append(AcquiredArtifact(artifact, input_sha256=input_sha256))
        return tuple(acquired)

    async def _again(self, call: Callable[[], Awaitable[T]]) -> T:
        """Make one provider call, again after a pause when it fails, as the server run retries a Document."""

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await call()
            except Exception as exc:
                if attempt == MAX_RETRIES or not bool(getattr(exc, "retryable", True)):
                    raise
                await self._sleep(2**attempt)
        raise AssertionError("unreachable")
