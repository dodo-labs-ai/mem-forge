"""The steps of a sync run where the document and conversation families differ.

A sync run chooses its family once, from its Gene, through
:func:`collection_family`. The orchestrator calls the family at each step
where the families differ and runs everything else, such as fetching,
projection, extraction and the Memory lifecycle, the same way for both.

- :class:`DocumentFamily` reads a provider's items. It discovers changes
  since the incremental cursor, and takes one listing of the configured scope
  per run: a device's snapshot manifest, or the connector's own listing. That
  listing decides which failed Documents are retried and which held
  Documents are gone, and when an open scope transition completes.
- :class:`ConversationFamily` plans each run from the collection attempt it
  is bound to. Each Unit the conversation layer removes carries its own
  evidence.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar, Literal, cast

from memforge.conversations.collections import StoredCollections
from memforge.genes import gene_exports_pdf
from memforge.genes.conversation_gene import CollectionAttempt, ConversationGene
from memforge.genes.document_gene import DocumentGene
from memforge.genes.family import CollectionFamilyKind, gene_family
from memforge.local_agent.source_contract import local_agent_sync_operation
from memforge.models import (
    AbsenceAnswer,
    AbsenceCheckSkippedReason,
    ContentItem,
    ScopeListing,
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
    "LedgerPartition",
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
    that failed when the check could not complete. ``unlisted_present_count``
    counts the unlisted Documents the provider confirmed it still has or
    hides, and is ``None`` when the run confirmed nothing.
    """

    removals: Mapping[str, SourceRemoval] | None = None
    skipped_reason: AbsenceCheckSkippedReason | None = None
    unlisted_present_count: int | None = None


@dataclass(frozen=True, slots=True)
class LedgerPartition:
    """The failure ledger rows a run's scope listing does not name, by what the run does with them.

    ``settled`` rows leave the ledger: their item is gone. ``untried`` rows
    stay in the ledger and are not retried in this run.
    """

    settled: frozenset[str] = frozenset()
    untried: frozenset[str] = frozenset()


def _absent(doc_ids: set[str], *, evidence: str) -> dict[str, SourceRemoval]:
    """Documents the provider no longer has, as the run's evidence shows."""
    return {
        doc_id: SourceRemoval(reason=UnitRemovalReason.ABSENT_FROM_SOURCE, evidence=evidence) for doc_id in doc_ids
    }


class CollectionFamily(ABC):
    """One sync run's family steps, bound to the run's Gene and Source."""

    records_transition_members: ClassVar[bool] = False
    """Whether a Document that commits during an open scope transition is recorded as one of its members."""

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

    async def take_scope_listing(self) -> None:
        """Take the run's one listing of its scope, after discovery and before failed Documents are retried."""

    def partition_ledger(self, ledger_doc_ids: Collection[str], *, rediscovering: bool) -> LedgerPartition:
        """The failure ledger rows the run's listing settles or leaves untried; the rest follow the retry rules."""
        return LedgerPartition()

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
        transition: ProjectionScopeTransition | None,
        indexed_doc_ids: set[str],
        uncommitted: Sequence[ContentItem],
        deferred_blockers: Mapping[str, frozenset[str]],
    ) -> AbsenceOutcome:
        """The Documents this run removes from the Source, and why.

        ``transition`` is the open scope transition the run belongs to.
        ``uncommitted`` holds the item of every Document of this run, and of
        every failure ledger row the run leaves open, that has not committed.
        ``deferred_blockers`` maps each Document whose commit is still
        deferred to the Units outside this run that block it.
        """

    @abstractmethod
    async def transition_outstanding(self, *, docs_failed: int, removals_failed: int) -> str | None:
        """What keeps the open scope transition from completing after this run; ``None`` completes it.

        Called once the run's removals and the deferred commits they unblock
        are done. ``removals_failed`` counts the removals that failed.
        """

    def cursor(
        self,
        *,
        status: Literal["success", "partial", "failed"],
        started_at: datetime,
        previous: datetime | None,
        transition: ProjectionScopeTransition | None,
        transition_completed: bool,
    ) -> datetime | None:
        """The incremental cursor after this run, given the one it started from.

        A run whose failures are all per-Document ones, or that failed none,
        moves it to its start, which precedes its discovery, so an item the
        provider changes after discovery read it is read again by the next
        run. A partial run of an open transition and a failed run keep it.
        """

        if status == "success" or (status == "partial" and transition is None):
            return started_at
        return previous

    def unreadable_conversations(self, *, reprocessing: bool) -> tuple[str, ...]:
        """Selected conversations the Source cannot read; each makes the run partial."""
        return ()


class DocumentFamily(CollectionFamily):
    """A document Source: provider items, discovered since the cursor and listed by scope.

    A run takes at most one :class:`~memforge.models.ScopeListing`, right
    after discovery and before it retries failed Documents, and keeps it for
    the rest of the run: a connector such as Confluence resets its discovery
    walk when it rediscovers one Document, so listing again later could
    answer differently. A run bound to a device's input snapshot reads the
    snapshot manifest as its listing, with the Gene's single listing kind.

    During an open scope transition, every Document that commits is recorded
    as one of the transition's members, run after run, and each run fetches
    only the listed Documents that are not members yet. The transition
    completes once the listing of the new scope is complete and every listed
    Document is a member; only then does it remove what the new scope leaves
    out.
    """

    gene: DocumentGene
    records_transition_members = True

    def __init__(
        self,
        gene: Gene,
        *,
        db: RelationalStore,
        doc_store: DocumentStore,
        source_id: str,
    ) -> None:
        super().__init__(gene, db=db, doc_store=doc_store, source_id=source_id)
        self.scope_listing: ScopeListing | None = None
        self._listing_skipped_reason: AbsenceCheckSkippedReason | None = None
        self._unconfirmed_ledger_doc_ids: frozenset[str] = frozenset()
        self._transition: ProjectionScopeTransition | None = None
        self._transition_outstanding: str | None = None

    def prepare(
        self,
        *,
        collection_attempt: CollectionAttempt | None,
        input_snapshot: SnapshotBinding | None,
    ) -> None:
        self.input_snapshot = input_snapshot
        self.gene.reset_scope_listing()

    def begin_scope_transition(self, transition: ProjectionScopeTransition) -> None:
        self._transition = transition

    async def take_scope_listing(self) -> None:
        """Take the run's listing: the bound snapshot manifest, or the connector's own listing.

        A configuration that cannot list, and one collected on a device whose
        run is bound to no snapshot, has no listing. A connector listing that
        raises leaves the run without one and records ``scope_listing_failed``,
        so nothing is removed or settled by listing in this run.
        """

        if not self.gene.lists_scope(self.gene.config):
            return
        if self.input_snapshot is not None:
            self.scope_listing = ScopeListing(kind=self._device_listing_kind(), doc_ids=self.input_snapshot.doc_ids)
            return
        if self._collected_on_device():
            return
        try:
            self.scope_listing = await self.gene.list_scope()
        except Exception as exc:
            # The provider's error stays in the log; the run records which
            # step failed, so its sync activity shows that nothing was removed.
            logger.warning("Scope listing of %s failed; nothing is removed in this run: %s", self.source_id, exc)
            self._listing_skipped_reason = "scope_listing_failed"

    def partition_ledger(self, ledger_doc_ids: Collection[str], *, rediscovering: bool) -> LedgerPartition:
        """Settle or keep the failure ledger rows the listing does not name.

        - Existence listing: every unlisted row is settled, whatever the retry
          rules say, including a row whose Document never became a Unit.
        - Query listing read from the provider: an unlisted row is retried
          through ``rediscover``, which settles it when the item is gone or has
          left the query. A Source that cannot rediscover keeps it untried.
        - Query listing from a device: an unlisted row is kept untried until
          the device confirms it by identifier.

        A listed row, and every row of a run without a listing, follows the
        retry rules. The rows a device has yet to confirm also keep the run
        from removing anything (:meth:`absence`).
        """

        listing = self.scope_listing
        if listing is None:
            return LedgerPartition()
        unlisted = frozenset(ledger_doc_ids) - listing.doc_ids
        if listing.kind is ScopeListingKind.EXISTENCE:
            return LedgerPartition(settled=unlisted)
        if self.input_snapshot is not None:
            self._unconfirmed_ledger_doc_ids = unlisted
            return LedgerPartition(untried=unlisted)
        return LedgerPartition() if rediscovering else LedgerPartition(untried=unlisted)

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
        transition: ProjectionScopeTransition | None,
        indexed_doc_ids: set[str],
        uncommitted: Sequence[ContentItem],
        deferred_blockers: Mapping[str, frozenset[str]],
    ) -> AbsenceOutcome:
        """Remove the held Documents the run's scope listing shows gone.

        A reprocess removes nothing. A run of an open scope transition removes
        only when it completes the transition (:meth:`_transition_removals`).
        Any other run reads its listing: a failed Document says nothing about
        one the listing left out, so item failures do not hold absence back.

        - Without a listing, nothing is removed; a listing that failed
          records why.
        - A held Document the listing does not name is kept while it is the
          predecessor of a renamed item in ``uncommitted``.
        - Existence listing: every other unlisted Document is absent.
        - Query listing: each other unlisted Document is read by identifier,
          and only one the provider answers ``absent`` for is absent. A
          confirmation that fails, or a device listing whose confirmations
          are missing, removes nothing and records why.
        """

        if reprocessing:
            return AbsenceOutcome()
        if transition is not None:
            return await self._transition_removals(transition, indexed_doc_ids, deferred_blockers)
        if self._listing_skipped_reason is not None:
            return AbsenceOutcome(skipped_reason=self._listing_skipped_reason)
        listing = self.scope_listing
        if listing is None:
            return AbsenceOutcome()
        unlisted = indexed_doc_ids - listing.doc_ids
        held = await self._held_by_renames(unlisted, uncommitted)
        if held:
            logger.info(
                "Kept %d unlisted Documents of %s whose renamed successor has not committed",
                len(held),
                self.source_id,
            )
        unlisted -= held
        if listing.kind is ScopeListingKind.EXISTENCE:
            return AbsenceOutcome(removals=_absent(unlisted, evidence="scope listing"))
        if self.input_snapshot is not None:
            if unlisted or self._unconfirmed_ledger_doc_ids:
                logger.warning(
                    "Device listing of %s left %d held and %d failed Documents unlisted, and the device "
                    "confirmed none of them; nothing is removed",
                    self.source_id,
                    len(unlisted),
                    len(self._unconfirmed_ledger_doc_ids),
                )
                return AbsenceOutcome(skipped_reason="absence_confirmation_missing")
            return AbsenceOutcome()
        if not unlisted:
            return AbsenceOutcome()
        try:
            absent = await self._confirmed_absent(unlisted)
        except AbsenceCheckSkipped as exc:
            logger.warning(
                "Absence check skipped for %s (%s): %s",
                self.source_id,
                exc.reason,
                exc.__cause__,
            )
            return AbsenceOutcome(skipped_reason=exc.reason)
        return AbsenceOutcome(
            removals=_absent(absent, evidence="scope listing"),
            unlisted_present_count=len(unlisted) - len(absent),
        )

    async def _transition_removals(
        self,
        transition: ProjectionScopeTransition,
        indexed_doc_ids: set[str],
        deferred_blockers: Mapping[str, frozenset[str]],
    ) -> AbsenceOutcome:
        """What the run removes when it can complete the open transition; nothing otherwise.

        The run can complete it when its listing of the new scope is complete
        and every listed Document has committed under the transition, in this
        run or an earlier one. A listed Document whose commit is still
        deferred counts as committed when every Unit outside this run that
        blocks it is one this completion removes: removing them is what lets
        it commit, and :meth:`transition_outstanding` checks that it did.

        Completion removes every held Document the new listing leaves out, as
        ``removed_from_scope``. A run that cannot complete the transition
        removes nothing: a Unit whose Document id changed with the scope, such
        as a file under a new Git ref, carries over to its new Document only
        when that Document commits.
        """

        if self._listing_skipped_reason is not None:
            self._transition_outstanding = "the listing of the new scope failed"
            return AbsenceOutcome(skipped_reason=self._listing_skipped_reason)
        listing = self.scope_listing
        if listing is None:
            self._transition_outstanding = "the new scope cannot be listed"
            return AbsenceOutcome()
        removals = {
            doc_id: SourceRemoval(reason=UnitRemovalReason.REMOVED_FROM_SCOPE, evidence="scope transition listing")
            for doc_id in indexed_doc_ids - listing.doc_ids
        }
        waiting = listing.doc_ids - set(await self.db.list_projection_scope_transition_member_doc_ids(transition.id))
        if waiting:
            removed_unit_ids = await self._current_unit_ids(removals)
            waiting = {
                doc_id
                for doc_id in waiting
                if not deferred_blockers.get(doc_id) or not deferred_blockers[doc_id] <= removed_unit_ids
            }
        if waiting:
            self._transition_outstanding = f"{len(waiting)} listed Documents have not committed under the transition"
            logger.info(
                "Scope transition %s of %s waits on %d listed Documents; nothing is removed",
                transition.id,
                self.source_id,
                len(waiting),
            )
            return AbsenceOutcome()
        logger.info(
            "Scope transition %s of %s completes: %d Documents the new scope leaves out are removed",
            transition.id,
            self.source_id,
            len(removals),
        )
        return AbsenceOutcome(removals=removals)

    async def transition_outstanding(self, *, docs_failed: int, removals_failed: int) -> str | None:
        """Complete once every listed Document is a member and every removal succeeded.

        A Document failure does not hold the transition back once the
        Document is a member: the failure ledger retries it, and the cursor
        the completing run sets makes the next run discover it again. A
        completion whose removals did not all succeed, or whose deferred
        commits stayed blocked after them, leaves the transition open with
        the removals that succeeded applied; the next run that can complete
        it removes the rest.
        """

        if self._transition_outstanding is not None:
            return self._transition_outstanding
        if removals_failed:
            return f"{removals_failed} Documents the new scope leaves out could not be removed"
        transition = self._transition
        assert transition is not None and self.scope_listing is not None
        waiting = self.scope_listing.doc_ids - set(
            await self.db.list_projection_scope_transition_member_doc_ids(transition.id)
        )
        if waiting:
            return f"{len(waiting)} listed Documents did not commit once the Units blocking them were removed"
        return None

    def cursor(
        self,
        *,
        status: Literal["success", "partial", "failed"],
        started_at: datetime,
        previous: datetime | None,
        transition: ProjectionScopeTransition | None,
        transition_completed: bool,
    ) -> datetime | None:
        """Outside a transition, the shared rule. A transition run moves the cursor only when it completes.

        The completing run sets it to the time the transition opened, not to
        its own start: transition runs fetch only the listed Documents that
        are not members yet, so the next run discovers every change made while
        the transition was open. Any other transition run keeps the cursor. A
        successful one that finds none, as after a credential change cleared
        it, records the opening time too, because a successful run always
        records a cursor and no run reads it until the transition closes.
        """

        if transition is None:
            return super().cursor(
                status=status,
                started_at=started_at,
                previous=previous,
                transition=transition,
                transition_completed=transition_completed,
            )
        if transition_completed or (previous is None and status == "success"):
            return transition.opened_at
        return previous

    async def _current_unit_ids(self, doc_ids: Collection[str]) -> set[str]:
        """The current Unit of each Document."""

        unit_ids: set[str] = set()
        for doc_id in sorted(doc_ids):
            unit = await self.db.find_source_unit_by_document_id(self.source_id, doc_id, current_only=True)
            if unit is not None:
                unit_ids.add(unit.id)
        return unit_ids

    async def _held_by_renames(self, unlisted: set[str], uncommitted: Sequence[ContentItem]) -> set[str]:
        """The unlisted Documents of the current Unit of each Document an uncommitted item was renamed from.

        The Unit is the one the renamed item commits to, found the same way.
        """

        if not unlisted:
            return set()
        held: set[str] = set()
        for document_id in sorted({item.renamed_from for item in uncommitted if item.renamed_from}):
            predecessor = await self.db.find_source_unit_by_document_id(
                self.source_id, document_id, current_only=True,
            )
            if predecessor is not None:
                held.update(set(await self.db.list_source_unit_document_ids(predecessor.id)) & unlisted)
        return held

    async def _confirmed_absent(self, unlisted: set[str]) -> set[str]:
        """The unlisted Documents the provider answers ``absent`` for when read by identifier.

        One that exists or is hidden keeps its Unit and Support and is not
        refreshed while it stays outside the query. One whose stored item
        cannot be rebuilt identifies nothing to ask for, so it is kept. A
        confirmation that fails raises ``AbsenceCheckSkipped``.
        """

        try:
            items = []
            for doc_id in sorted(unlisted):
                try:
                    items.append(await stored_source_item(self.db, source_id=self.source_id, document_id=doc_id))
                except StoredDocumentUnavailable as exc:
                    logger.warning("Cannot confirm whether unlisted Document %s still exists: %s", doc_id, exc)
            answers = await self.gene.confirm_absent(items)
        except Exception as exc:
            raise AbsenceCheckSkipped("absence_confirmation_failed") from exc
        absent = {doc_id for doc_id, answer in answers.items() if answer == AbsenceAnswer.ABSENT} & unlisted
        logger.info(
            "Scope listing of %s left %d held Documents unlisted; %d are absent at the provider",
            self.source_id,
            len(unlisted),
            len(absent),
        )
        return absent

    def _device_listing_kind(self) -> ScopeListingKind:
        """The kind of a device's snapshot manifest: the single kind its Gene declares."""

        kinds = self.gene.listing_kinds
        if len(kinds) != 1:
            raise ValueError(
                f"{type(self.gene).__name__} is listed by a device's snapshot manifest and must declare "
                "exactly one listing kind"
            )
        return next(iter(kinds))

    def _collected_on_device(self) -> bool:
        """Whether the owner's device collects this Source, so its listing is the device's snapshot manifest."""
        return local_agent_sync_operation(self.gene.metadata().name, self.gene.config) is not None


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
        transition: ProjectionScopeTransition | None,
        indexed_doc_ids: set[str],
        uncommitted: Sequence[ContentItem],
        deferred_blockers: Mapping[str, frozenset[str]],
    ) -> AbsenceOutcome:
        """The removals the conversation layer planned for this attempt.

        Each conversation Unit carries the evidence that removes it, so its
        removal does not wait on other Units of the run.
        """

        if reprocessing:
            return AbsenceOutcome()
        return AbsenceOutcome(removals=self.gene.removals())

    async def transition_outstanding(self, *, docs_failed: int, removals_failed: int) -> str | None:
        """Complete once deselected conversations are removed and every readable target was polled."""

        if docs_failed == 0 and await self.gene.completes_scope_transition():
            return None
        return f"the collection attempt did not answer every target conversation (failed_docs={docs_failed})"

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
