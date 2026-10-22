"""The failure ledger: the Documents of a Source whose processing failed.

A sync run records each per-Document failure, whatever its stage, as one row
of the Source's ledger, and deletes the row once the Document commits or
leaves the Source. The ledger, not the sync cursor, owns the retry of a
failed Document, so one failing Document never holds its Source's cursor back.

Each row records the release that produced it. A failure is retried:

- by the next scheduled run of the same release while it may pass, that is,
  while it is retryable and has not failed at the same stage with the same
  error class in as many consecutive runs as the retry budget allows;
- once by each later release, whatever its classification;
- by every sync a person requested for its Source;
- by any run whose discovery returns the Document changed since it failed.

A scheduled run never retries a deterministic failure of its own release:
when discovery returns such a Document with the version and change time of
the item that failed, the run leaves it alone.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Protocol

from memforge.models import ContentItem
from memforge.source_activity import SourceActivityLease

# Consecutive runs of one release in which a Document may fail at the same
# stage with the same error class before only a manual sync or a later release
# retries it.
SOURCE_ITEM_FAILURE_RETRY_BUDGET = 3
MAX_SOURCE_ITEM_FAILURE_MESSAGE_CHARS = 512

# The release of a ledger row whose failure no release recorded: a derivation
# staged before derivations recorded their release.
UNRECORDED_RELEASE = "unrecorded"

# The stage of a Document whose removal failed. The next run whose listing
# proves the Document absent removes it again, so such a row is not retried by
# processing the Document.
REMOVAL_STAGE = "removal"


class FailedItemRetry(str, Enum):
    """Which ledger rows a sync run retries.

    RETRYABLE is the rule of a run nobody asked for: rows of an earlier
    release, and rows of this release that may still pass. ALL is the rule of
    a sync a person requested: every row, whatever its release or
    classification.
    """

    RETRYABLE = "retryable"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class RunItemFailure:
    """One Document's failure as the run that met it observed it.

    ``stage`` is the step that failed (``fetch``, ``normalize``,
    ``projection``, ``derivation``, a lifecycle stage, ``removal``),
    ``error_class`` the qualified class of the failure, and ``retryable`` its
    classification (``memforge.llm.structured.failure_retryable``). ``item``
    is the item discovery returned for the Document, when the run had one.
    """

    document_id: str
    stage: str
    error_class: str
    message: str
    retryable: bool
    title: str | None = None
    error_code: str | None = None
    source_unit_id: str | None = None
    item: ContentItem | None = None


@dataclass(frozen=True, slots=True)
class SourceItemFailure:
    """One ledger row: the latest failure of one Document of one Source."""

    source_id: str
    document_id: str
    stage: str
    error_class: str
    message: str
    retryable: bool
    release: str
    failures_in_release: int
    first_failed_at: datetime
    last_failed_at: datetime
    title: str | None = None
    error_code: str | None = None
    source_unit_id: str | None = None
    item: ContentItem | None = None

    @classmethod
    def recorded(
        cls,
        previous: SourceItemFailure | None,
        failure: RunItemFailure,
        *,
        source_id: str,
        release: str,
        failed_at: datetime,
    ) -> SourceItemFailure:
        """The row after ``failure``, given the Document's ``previous`` row.

        ``failures_in_release`` counts the consecutive failures of this release
        at the same stage with the same error class; a failure of another
        release, stage or class starts the count again. The row keeps the
        rename its earlier item attested, because a provider may report a
        rename only once: the predecessor's Unit is kept until this Document
        commits.
        """

        repeats = (
            previous is not None
            and previous.release == release
            and (previous.stage, previous.error_class) == (failure.stage, failure.error_class)
        )
        return cls(
            source_id=source_id,
            document_id=failure.document_id,
            stage=failure.stage,
            error_class=failure.error_class,
            message=failure.message[:MAX_SOURCE_ITEM_FAILURE_MESSAGE_CHARS],
            retryable=failure.retryable,
            release=release,
            failures_in_release=(previous.failures_in_release + 1 if repeats and previous is not None else 1),
            first_failed_at=previous.first_failed_at if previous is not None else failed_at,
            last_failed_at=failed_at,
            title=failure.title or (previous.title if previous is not None else None),
            error_code=failure.error_code,
            source_unit_id=failure.source_unit_id or (previous.source_unit_id if previous is not None else None),
            item=(
                with_inherited_rename(failure.item, previous)
                if failure.item is not None
                else (previous.item if previous is not None else None)
            ),
        )

    @property
    def budget_exhausted(self) -> bool:
        """The same failure repeated in as many consecutive runs of its release as the budget allows."""

        return self.failures_in_release >= SOURCE_ITEM_FAILURE_RETRY_BUDGET

    def recorded_by_another_release(self, release: str) -> bool:
        return self.release != release

    def retried_by(self, retry: FailedItemRetry, *, release: str) -> bool:
        """Whether a run of ``release`` under ``retry`` processes this Document again."""

        if retry is FailedItemRetry.ALL or self.recorded_by_another_release(release):
            return True
        return self.retryable and not self.budget_exhausted


def with_inherited_rename(item: ContentItem, row: SourceItemFailure | None) -> ContentItem:
    """``item``, naming the predecessor its Document's ledger ``row`` recorded when ``item`` names none.

    A provider may report a rename only once, so a fresh item discovered for
    a failed Document keeps the rename its earlier, failed item attested.
    """

    if item.renamed_from is not None or row is None or row.item is None or row.item.renamed_from is None:
        return item
    return replace(item, renamed_from=row.item.renamed_from)


class SourceItemFailureStore(Protocol):
    """Persistence of the failure ledger; each row is keyed by its Source and Document."""

    async def upsert_source_item_failure(
        self,
        failure: SourceItemFailure,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Insert the row, or replace the Document's row in its Source, while ``source_activity`` is current."""
        ...

    async def list_source_item_failures(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[SourceItemFailure, ...]:
        """The rows of one Source, or of every Source, ordered by Source and Document."""
        ...

    async def delete_source_item_failures(
        self,
        source_id: str,
        document_ids: Collection[str],
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Delete the rows of these Documents of the Source while ``source_activity`` is current.

        A Document without a row is ignored.
        """
        ...


__all__ = [
    "FailedItemRetry",
    "MAX_SOURCE_ITEM_FAILURE_MESSAGE_CHARS",
    "REMOVAL_STAGE",
    "RunItemFailure",
    "SOURCE_ITEM_FAILURE_RETRY_BUDGET",
    "SourceItemFailure",
    "SourceItemFailureStore",
    "UNRECORDED_RELEASE",
    "with_inherited_rename",
]
