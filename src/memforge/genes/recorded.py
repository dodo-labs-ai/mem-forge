"""The recorded Gene: a document type's own Gene, with acquisition answered from a record.

A document Gene is two parts (see :mod:`memforge.genes.document_gene`):

- **Acquisition** reaches the provider: ``authenticate``, ``discover``,
  ``list_scope``, ``confirm_absent``, ``rediscover``, ``fetch``, ``fetch_pdf``
  and ``open_source_artifact``.
- **Interpretation** reads only what acquisition returned: ``normalize``,
  ``raw_from_stored_input``, ``requires_pdf_artifact``, ``project_native`` and
  the readings, together with the class-level answers such as ``lists_scope``
  and ``rediscovers_documents``.

When acquisition ran somewhere else, such as on the owner's device, its
results are an :class:`AcquisitionRecord`. :func:`recorded_gene_class` returns
the type's Gene with every acquisition method answered from that record and
everything else inherited, so the server's run interprets a recorded run
exactly as it interprets one that read the provider itself. One class serves
every document type, whatever store holds the record.

A run and its record must agree. A call the record holds no answer to, such
as ``fetch`` of an item that was never acquired, raises
:class:`AcquisitionRecordMismatch`, which fails the run instead of one
Document, so the run is retried with a record that does answer it. A call
whose recorded answer is the error acquisition raised, such as a ``fetch``
the provider refused, raises :class:`AcquisitionFailed`, so the run treats it
as it treats that error from the provider itself.
"""

from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime
from typing import Protocol, TypeVar

from memforge.genes.document_gene import DocumentGene
from memforge.models import AbsenceAnswer, ContentItem, RawContent, ScopeListing
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload

__all__ = [
    "AcquiredInput",
    "AcquisitionFailed",
    "AcquisitionRecord",
    "AcquisitionRecordMismatch",
    "AcquisitionRecordStore",
    "recorded_gene_class",
]

G = TypeVar("G", bound=DocumentGene)


class AcquisitionRecordMismatch(RuntimeError):
    """The run asked its acquisition record something the record holds no answer to.

    The run and the record disagree about what was acquired, so no Document
    of the run can be trusted to the record: the run fails.
    """


class AcquisitionFailed(RuntimeError):
    """The error acquisition raised for one call, raised again where the record answers that call.

    The run treats it as it treats that error from the provider itself.
    ``failure_class`` names the class of the error and ``error_code`` its
    code, which the failure ledger records. ``retryable`` is set only when the
    error declared it, because the run reads that declaration to decide
    whether to try the Document again. Whether the error was transient, which
    the ledger records, is its cause's answer
    (:func:`memforge.llm.structured.failure_retryable`).
    """

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        transient: bool,
        declared_retryable: bool | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        if declared_retryable is not None:
            self.retryable = declared_retryable
        if error_code is not None:
            self.error_code = error_code
        self.__cause__ = _Transience(transient)


class _Transience(Exception):
    """Whether the error acquisition raised was transient, as the cause of its :class:`AcquisitionFailed`."""

    def __init__(self, transient: bool) -> None:
        super().__init__("transient" if transient else "not transient")
        self.retryable = transient


@dataclass(frozen=True, slots=True)
class AcquiredInput:
    """What acquisition read for one item: the input its Unit stores when the item commits.

    ``item`` is the item as ``fetch`` left it. ``body`` and ``content_type``
    are the bytes ``fetch`` returned, ``pdf`` the item's PDF export when its
    type exports one, or ``pdf_failure`` the error that export raised, and
    ``artifacts`` the bodies ``fetch`` described, each opened through
    :meth:`AcquisitionRecord.open_artifact`.
    """

    item: ContentItem
    body: bytes
    content_type: str
    pdf: bytes | None = None
    artifacts: tuple[RawSourceArtifact, ...] = ()
    pdf_failure: AcquisitionFailed | None = None


class AcquisitionRecord(Protocol):
    """What one run's acquisition returned, open for the run that interprets it.

    Each question the record holds no answer to raises
    :class:`AcquisitionRecordMismatch`; each one whose answer is the error
    acquisition raised raises it as :class:`AcquisitionFailed`.
    """

    def discovered(self) -> Sequence[ContentItem]:
        """The items ``discover`` yielded, in the order it yielded them."""

    def listing(self) -> ScopeListing:
        """The scope listing acquisition took."""

    def absence_answer(self, item_id: str) -> AbsenceAnswer:
        """The answer ``confirm_absent`` gave for one item."""

    def rediscovered(self, item_id: str) -> ContentItem | None:
        """The item ``rediscover`` returned for one item, ``None`` when it found none."""

    async def acquired(self, item_id: str) -> AcquiredInput:
        """The acquired input of one item."""

    def open_artifact(self, artifact: RawSourceArtifact) -> AbstractAsyncContextManager[SourceArtifactDownload]:
        """Open the body of one artifact an acquired input describes."""

    async def close(self) -> None:
        """Release what the open record holds."""


class AcquisitionRecordStore(Protocol):
    """Where the records of one Source's runs are kept."""

    async def open(self) -> AcquisitionRecord:
        """Open the record the next run interprets."""


@functools.cache
def recorded_gene_class(gene_cls: type[G]) -> type[G]:
    """The Gene of ``gene_cls``'s type with acquisition answered from an :class:`AcquisitionRecord`.

    The class is created once per type. Its instances take the store of their
    records as ``record_store``; ``authenticate`` opens the record and
    ``close`` releases it.
    """

    class RecordedGene(gene_cls):  # type: ignore[valid-type, misc]
        __doc__ = f"{gene_cls.__name__} with acquisition answered from an acquisition record."

        def __init__(self, config: dict, source_id: str, *, record_store: AcquisitionRecordStore) -> None:
            super().__init__(config, source_id)
            self._record_store = record_store
            self._record: AcquisitionRecord | None = None

        def _opened(self) -> AcquisitionRecord:
            if self._record is None:
                raise RuntimeError(f"{type(self).__name__} reads its acquisition record only after authenticate()")
            return self._record

        async def authenticate(self) -> None:
            """Open the record; there is no provider client."""
            await self.close()
            self._record = await self._record_store.open()

        async def close(self) -> None:
            record, self._record = self._record, None
            try:
                if record is not None:
                    await record.close()
            finally:
                await super().close()

        async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
            """The recorded items, in order; acquisition already applied ``since``."""
            for item in self._opened().discovered():
                yield _copy(item)

        async def list_scope(self) -> ScopeListing:
            return self._opened().listing()

        async def confirm_absent(self, items: Sequence[ContentItem]) -> Mapping[str, AbsenceAnswer]:
            record = self._opened()
            return {item.item_id: record.absence_answer(item.item_id) for item in items}

        async def rediscover(self, item: ContentItem) -> ContentItem | None:
            current = self._opened().rediscovered(item.item_id)
            return _copy(current) if current is not None else None

        async def fetch(self, item: ContentItem) -> RawContent:
            """The item's acquired input; ``item`` takes the state ``fetch`` left it in where it was acquired."""
            acquired = await self._opened().acquired(item.item_id)
            _take_fetched_state(item, acquired.item)
            raw = self.raw_from_stored_input(item, acquired.body, acquired.content_type)
            return replace(raw, artifacts=acquired.artifacts)

        async def fetch_pdf(self, item: ContentItem) -> bytes | None:
            acquired = await self._opened().acquired(item.item_id)
            if acquired.pdf_failure is not None:
                raise acquired.pdf_failure
            return acquired.pdf

        @asynccontextmanager
        async def open_source_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
            async with self._opened().open_artifact(artifact) as download:
                yield download

    RecordedGene.__name__ = RecordedGene.__qualname__ = f"Recorded{gene_cls.__name__}"
    return RecordedGene


def _copy(item: ContentItem) -> ContentItem:
    """A recorded item as acquisition hands it out: a copy the run may change."""

    return replace(item, labels=list(item.labels), extra=dict(item.extra), stored_extra=dict(item.stored_extra))


def _take_fetched_state(item: ContentItem, fetched: ContentItem) -> None:
    """Give ``item`` every field ``fetched`` holds, except the Source's own ``stored_extra``."""

    copied = _copy(fetched)
    for field in fields(ContentItem):
        if field.name != "stored_extra":
            setattr(item, field.name, getattr(copied, field.name))
