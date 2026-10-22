"""The steps of a sync run where the document and conversation families differ.

A sync run chooses its family once, from its Gene, through
:func:`collection_family`. The orchestrator calls the family at each step
where the families differ and runs everything else, such as fetching,
projection, extraction and the Memory lifecycle, the same way for both.

- :class:`DocumentFamily` reads a provider's items. It discovers changes
  since the incremental cursor, and proves a Document gone from a submitted
  snapshot or from a listing of the configured scope.
- :class:`ConversationFamily` plans each run from the collection attempt it
  is bound to. Each Unit the conversation layer removes carries its own
  evidence.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast

from memforge.conversations.collections import StoredCollections
from memforge.genes import gene_exports_pdf
from memforge.genes.conversation_gene import CollectionAttempt, ConversationGene
from memforge.genes.document_gene import DocumentGene
from memforge.genes.family import CollectionFamilyKind, gene_family
from memforge.models import (
    AbsenceCheckSkippedReason,
    ContentItem,
    ScopeListingKind,
    SyncState,
)
from memforge.pipeline.stored_document import (
    StoredDocumentUnavailable,
    StoredDocumentUnavailableReason,
    StoredSourceDocument,
    load_stored_source_document,
    rediscover_source_document,
    stored_source_item,
)
from memforge.source_item_failures import SourceItemFailure
from memforge.source_projection import (
    ProjectionScopeTransition,
    SourceRemoval,
    UnitRemovalReason,
)

if TYPE_CHECKING:
    from memforge.genes.base import Gene
    from memforge.storage.adapters.protocols import RelationalStore
    from memforge.storage.document_store import DocumentStore

__all__ = [
    "DEFAULT_INCREMENTAL_SYNC_OVERLAP",
    "AbsenceCheckSkipped",
    "AbsenceOutcome",
    "CollectionFamily",
    "ConversationFamily",
    "DocumentFamily",
    "SnapshotBinding",
    "collection_family",
]

logger = logging.getLogger(__name__)

# Discovery reads from the cursor minus this overlap, which covers provider
# timestamp precision and clock skew.
DEFAULT_INCREMENTAL_SYNC_OVERLAP = timedelta(minutes=10)


class AbsenceCheckSkipped(Exception):
    """A sync run's absence check did not complete; ``reason`` says which step failed."""

    def __init__(self, reason: AbsenceCheckSkippedReason) -> None:
        super().__init__(reason)
        self.reason: AbsenceCheckSkippedReason = reason


@dataclass(frozen=True, slots=True)
class SnapshotBinding:
    """The complete input snapshot a device submitted for one document sync run.

    ``doc_ids`` are the Documents its manifest names: everything the scope
    holds, as the device read it.
    """

    snapshot_id: str
    doc_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class AbsenceOutcome:
    """What a run's absence step found.

    ``removals`` are the Documents to remove with their reasons; ``None``
    means the run proved nothing absent. ``skipped_reason`` names the step
    that failed when the check could not complete.
    """

    removals: Mapping[str, SourceRemoval] | None = None
    skipped_reason: AbsenceCheckSkippedReason | None = None


def _absent(doc_ids: set[str], *, evidence: str) -> dict[str, SourceRemoval]:
    """Documents the provider no longer has, as the run's evidence shows."""
    return {
        doc_id: SourceRemoval(reason=UnitRemovalReason.ABSENT_FROM_SOURCE, evidence=evidence) for doc_id in doc_ids
    }


class CollectionFamily(ABC):
    """One sync run's family steps, bound to the run's Gene and Source."""

    def __init__(
        self,
        gene: Gene,
        *,
        db: RelationalStore,
        doc_store: DocumentStore,
        source_id: str,
    ) -> None:
        self.gene = gene
        self.db = db
        self.doc_store = doc_store
        self.source_id = source_id
        self.input_snapshot: SnapshotBinding | None = None

    @abstractmethod
    def prepare(
        self,
        *,
        collection_attempt: CollectionAttempt | None,
        input_snapshot: SnapshotBinding | None,
    ) -> None:
        """Bind the run to what it collects, before anything else runs."""

    def begin_scope_transition(self, transition: ProjectionScopeTransition) -> None:
        """Start the run of an open Projection Scope transition."""

    async def since(
        self,
        *,
        existing_state: SyncState | None,
        indexed_doc_ids: set[str],
        full_discovery: bool,
    ) -> datetime | None:
        """The time discovery reads changes from; ``None`` discovers everything.

        It is the last cursor minus
        :data:`DEFAULT_INCREMENTAL_SYNC_OVERLAP`. A run that must discover
        everything, a run bound to a complete input snapshot, a Source that
        never synced or holds no Document, and a family-specific reason
        (:meth:`_requires_full_discovery`) read from the start.
        """

        if full_discovery or self.input_snapshot is not None:
            return None
        last_sync_time = existing_state.last_sync_at if existing_state else None
        if last_sync_time and await self._requires_full_discovery():
            last_sync_time = None
        # A sync state without a held Document is a broken earlier run.
        if last_sync_time and not indexed_doc_ids:
            logger.warning(
                "Sync state exists but 0 indexed docs for %s — forcing full re-sync",
                self.source_id,
            )
            return None
        return last_sync_time - DEFAULT_INCREMENTAL_SYNC_OVERLAP if last_sync_time else None

    async def _requires_full_discovery(self) -> bool:
        return False

    @abstractmethod
    async def document_input(
        self,
        document_id: str,
        *,
        rediscovering: bool,
        failure: SourceItemFailure | None,
    ) -> tuple[ContentItem, StoredSourceDocument | None]:
        """The input of a Document the run processes without discovering it."""

    @abstractmethod
    async def absence(
        self,
        *,
        reprocessing: bool,
        transition_open: bool,
        coverage_proves_absence: bool,
        run_committed: bool,
        indexed_doc_ids: set[str],
        crawled_doc_ids: set[str],
    ) -> AbsenceOutcome:
        """The Documents this run proves the Source no longer has."""

    async def completes_transition(self, *, docs_failed: int) -> bool:
        """Whether this family completes the open transition, whatever the run's coverage proves."""
        return False

    def unreadable_conversations(self, *, reprocessing: bool) -> tuple[str, ...]:
        """Selected conversations the Source cannot read; each makes the run partial."""
        return ()


class DocumentFamily(CollectionFamily):
    """A document Source: provider items, discovered since the cursor and listed by scope."""

    gene: DocumentGene

    def prepare(
        self,
        *,
        collection_attempt: CollectionAttempt | None,
        input_snapshot: SnapshotBinding | None,
    ) -> None:
        self.input_snapshot = input_snapshot

    async def _requires_full_discovery(self) -> bool:
        if not gene_exports_pdf(self.gene):
            return False
        missing_pdf_count = await self.db.count_missing_pdf_uris(self.source_id)
        if missing_pdf_count:
            logger.info(
                "Found %d documents missing required PDF provenance for %s; forcing full sync",
                missing_pdf_count,
                self.source_id,
            )
        return bool(missing_pdf_count)

    async def document_input(
        self,
        document_id: str,
        *,
        rediscovering: bool,
        failure: SourceItemFailure | None,
    ) -> tuple[ContentItem, StoredSourceDocument | None]:
        """The input of a Document the run processes as a reprocess.

        A Source that can ask its provider for one Document by id reads its
        current item, asking with the item the Document's failure ledger row
        kept when the Document never committed a Unit. Any other Source
        fetches the item the failure ledger row kept, as discovery returned
        it: it names the revision that failed, which the Unit's stored input
        does not hold. A Document without such a row reads the stored input of
        its Unit's current revision.
        """

        if not rediscovering:
            if failure is not None and failure.item is not None:
                return failure.item, None
            stored = await load_stored_source_document(
                self.db, self.doc_store, self.gene, source_id=self.source_id, document_id=document_id,
            )
            return stored.item, stored
        try:
            return (
                await rediscover_source_document(
                    self.db, self.gene, source_id=self.source_id, document_id=document_id,
                ),
                None,
            )
        except StoredDocumentUnavailable as exc:
            if (
                exc.reason is not StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING
                or failure is None
                or failure.item is None
            ):
                raise
        current = await self.gene.rediscover(failure.item)
        if current is None or current.item_id != document_id:
            raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.PROVIDER_DOCUMENT_MISSING, document_id)
        return current, None

    async def absence(
        self,
        *,
        reprocessing: bool,
        transition_open: bool,
        coverage_proves_absence: bool,
        run_committed: bool,
        indexed_doc_ids: set[str],
        crawled_doc_ids: set[str],
    ) -> AbsenceOutcome:
        """Remove what a submitted snapshot did not return, or what the scope listing proves gone.

        Discovery narrowed by ``since`` returns changed items only, so a
        Document it did not return proves nothing. A submitted snapshot, or a
        complete discovery of a newly configured scope, is the whole scope:
        what it did not return is removed. Every other run lists the
        configured scope by identifier and removes only what that listing
        proves absent. Absence is proven only after every Unit of the run has
        committed.
        """

        if coverage_proves_absence and run_committed:
            return AbsenceOutcome(removals=_absent(indexed_doc_ids - crawled_doc_ids, evidence="complete snapshot"))
        if not run_committed or reprocessing or transition_open:
            return AbsenceOutcome()
        try:
            absent_doc_ids = await self._absent_from_scope_listing(indexed_doc_ids)
        except AbsenceCheckSkipped as exc:
            # The run's content is committed; only the absence proof is
            # missing, so every unlisted Document keeps its Support and the
            # next run lists again. The run records which step failed, so its
            # sync activity shows that nothing was removed; the provider's
            # error stays in the log.
            logger.warning(
                "Absence check skipped for %s (%s): %s",
                self.source_id,
                exc.reason,
                exc.__cause__,
            )
            return AbsenceOutcome(skipped_reason=exc.reason)
        if absent_doc_ids is None:
            return AbsenceOutcome()
        return AbsenceOutcome(removals=_absent(absent_doc_ids, evidence="scope listing"))

    async def _absent_from_scope_listing(self, indexed_doc_ids: set[str]) -> set[str] | None:
        """The held Documents this run's scope listing proves the provider no longer has.

        ``None`` when the Gene cannot list its scope. A listing or confirmation
        that fails raises ``AbsenceCheckSkipped`` naming the step, and the
        caller removes nothing in this run. An unlisted Document of an
        existence listing is absent. One of a query listing is absent only
        when the provider reports it not found; one that still exists keeps
        its Unit and Support and is not refreshed while it stays outside the
        query.
        """

        try:
            listing = await self.gene.list_scope()
        except Exception as exc:
            raise AbsenceCheckSkipped("scope_listing_failed") from exc
        if listing is None:
            return None
        unlisted = indexed_doc_ids - listing.doc_ids
        if not unlisted or listing.kind is ScopeListingKind.EXISTENCE:
            return unlisted
        try:
            items = []
            for doc_id in sorted(unlisted):
                try:
                    items.append(await stored_source_item(self.db, source_id=self.source_id, document_id=doc_id))
                except StoredDocumentUnavailable as exc:
                    # Nothing identifies the Document to ask its provider for,
                    # so its absence cannot be proven and it is kept.
                    logger.warning("Cannot confirm whether unlisted Document %s still exists: %s", doc_id, exc)
            absent = set(await self.gene.confirm_absent(items)) & unlisted
        except Exception as exc:
            raise AbsenceCheckSkipped("absence_confirmation_failed") from exc
        logger.info(
            "Scope listing of %s left %d held Documents unlisted; %d are absent at the provider",
            self.source_id,
            len(unlisted),
            len(absent),
        )
        return absent


class ConversationFamily(CollectionFamily):
    """A conversation Source: each run plans from its bound collection attempt."""

    gene: ConversationGene

    def prepare(
        self,
        *,
        collection_attempt: CollectionAttempt | None,
        input_snapshot: SnapshotBinding | None,
    ) -> None:
        if input_snapshot is not None:
            raise ValueError("a conversation Source plans from its collection attempt, not an input snapshot")
        if collection_attempt is None:
            raise ValueError("a conversation Source syncs only within a collection attempt")
        self.gene.bind_conversation_run(
            projection_store=self.db,
            collections=StoredCollections(self.db, workspace_id=collection_attempt.workspace_id),
            collection_attempt_id=collection_attempt.collection_attempt_id,
        )

    def begin_scope_transition(self, transition: ProjectionScopeTransition) -> None:
        self.gene.begin_scope_transition(
            previous_scope=transition.previous_scope,
            target_scope=transition.target_scope,
        )

    async def document_input(
        self,
        document_id: str,
        *,
        rediscovering: bool,
        failure: SourceItemFailure | None,
    ) -> tuple[ContentItem, StoredSourceDocument | None]:
        """The stored input of the Unit's current revision.

        A conversation Source plans every run afresh from committed Units.
        When the Unit never committed, discovery plans it again while its
        content is in reach, so nothing is left to retry.
        """

        try:
            stored = await load_stored_source_document(
                self.db, self.doc_store, self.gene, source_id=self.source_id, document_id=document_id,
            )
        except StoredDocumentUnavailable as exc:
            if exc.reason is StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING:
                raise StoredDocumentUnavailable(
                    StoredDocumentUnavailableReason.PROVIDER_DOCUMENT_MISSING, document_id
                ) from exc
            raise
        return stored.item, stored

    async def absence(
        self,
        *,
        reprocessing: bool,
        transition_open: bool,
        coverage_proves_absence: bool,
        run_committed: bool,
        indexed_doc_ids: set[str],
        crawled_doc_ids: set[str],
    ) -> AbsenceOutcome:
        """The removals the conversation layer planned for this attempt.

        Each conversation Unit carries the evidence that removes it, so its
        removal does not wait on other Units of the run.
        """

        if reprocessing:
            return AbsenceOutcome()
        return AbsenceOutcome(removals=self.gene.removals())

    async def completes_transition(self, *, docs_failed: int) -> bool:
        """Complete once deselected conversations are removed and every readable target was polled."""

        return docs_failed == 0 and await self.gene.completes_scope_transition()

    def unreadable_conversations(self, *, reprocessing: bool) -> tuple[str, ...]:
        return () if reprocessing else self.gene.unreadable_conversations()


def collection_family(
    gene: Gene,
    *,
    db: RelationalStore,
    doc_store: DocumentStore,
    source_id: str,
) -> CollectionFamily:
    """The family of a sync run's Gene; a Gene of neither family has no sync run."""

    kind = gene_family(type(gene))
    if kind is CollectionFamilyKind.DOCUMENT:
        return DocumentFamily(cast(DocumentGene, gene), db=db, doc_store=doc_store, source_id=source_id)
    if kind is CollectionFamilyKind.CONVERSATION:
        return ConversationFamily(cast(ConversationGene, gene), db=db, doc_store=doc_store, source_id=source_id)
    raise ValueError(f"{type(gene).__name__} belongs to no collection family and has no sync run")
