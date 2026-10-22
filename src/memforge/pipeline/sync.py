"""Gene Sync Orchestrator — the heart of MemForge's data pipeline.

Coordinates the full lifecycle of syncing data from a gene (external source)
into the memory layer:

    authenticate -> discover -> fetch -> normalize -> store ->
    extract claim-sized memories -> reconcile lifecycle -> detect deletions

Concurrency is managed via an asyncio.Semaphore (for LLM/embedding calls)
and an asyncio.Lock (for SQLite writes). Each content item is processed
independently with retry logic and per-item error isolation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import threading
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import tiktoken

from memforge.evals.agent_evaluation import (
    assessment_sink_for_runtime_sink,
    publish_agent_assessments,
    publish_runtime_events,
)
from memforge.genes import gene_exports_pdf, source_rediscovers_documents
from memforge.genes.conversation_gene import CollectionAttempt
from memforge.genes.base import SourceConfigurationError
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmMetricsCollector,
    StructuredLlmImage,
    structured_llm_metrics_scope,
)
from memforge.llm.structured import failure_retryable as retries_failure
from memforge.models import (
    AbsenceCheckSkippedReason,
    ChangelogEntry,
    ContentItem,
    DocumentRecord,
    FailedDoc,
    MemoryExtractionResult,
    NormalizedContent,
    RawContent,
    SourceUnitInput,
    SyncState,
    content_hash as compute_content_hash,
    require_attested_content,
)
from memforge.pipeline.sync_memory import ProcessMemoryReclaimer, SyncMemoryObserver

from memforge.pipeline.document_update import (
    DocumentUpdatePlan,
    plan_document_update,
)
from memforge.pipeline.source_projection_adapters import (
    DEFAULT_SOURCE_PROJECTION_ADAPTER,
    project_source_unit_tombstone,
)
from memforge.memory.lifecycle_plan import (
    AUTHORITATIVE_SOURCE_UNIT_REMOVAL_REASON,
    ReconciliationScope,
)
from memforge.memory.engine import SourceUnitLifecycleDeferred, SourceUnitLifecycleExecutionError
from memforge.memory.lifecycle_planner import (
    lifecycle_access_context_hash,
    lifecycle_plan_id,
)
from memforge.memory.project_resolver import resolve_project_key
from memforge.source_projection import (
    ProjectionEnvelope,
    ProjectionRequest,
    ProjectionRunMode,
    SourceProjection,
    SourceProjectionAdapter,
    SourceRemoval,
)
from memforge.source_artifacts import (
    StoredSourceArtifact,
    materialize_source_artifacts,
)
from memforge.source_derivation import (
    DERIVATION_DETERMINISTIC_FAILURE,
    DERIVATION_INPUT_SUPERSEDED,
    DERIVATION_RELEASE_SUPERSEDED,
    SourceDerivationAttempt,
    SourceUnitDerivationContext,
    SourceUnitDerivationRequest,
    SourceUnitDeriver,
    aggregate_extraction_metrics,
    plan_source_derivation_recovery_commit,
    safe_derivation_error,
)
from memforge.pipeline.extraction_contract import (
    CONTRACT_SUPERSEDED,
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
)
from memforge.pipeline.projection_context import CommittedSourceUnitSnapshot
from memforge.pipeline.collection_family import (
    SnapshotBinding,
    collection_family,
)
from memforge.pipeline.stored_document import (
    StoredDocumentUnavailable,
    StoredDocumentUnavailableReason,
    StoredSourceDocument,
    document_content_item,
    project_stored_input,
)
from memforge.pipeline.projection_images import (
    load_projection_images,
)
from memforge.release import current_release
from memforge.source_access import memory_visibility_for_source_id
from memforge.source_activity import SourceActivityLease
from memforge.source_item_failures import (
    REMOVAL_STAGE,
    UNRECORDED_RELEASE,
    FailedItemRetry,
    RunItemFailure,
    SourceItemFailure,
    with_inherited_rename,
)
from memforge.source_projection_config import canonical_projection_scope, source_access_context
from memforge.source_time import SOURCE_UPDATED_AT_KEY, parse_source_time, reported_source_time

if TYPE_CHECKING:
    from memforge.evals.agent_evaluation import RuntimeEventTraceSink
    from memforge.genes.base import Gene
    from memforge.memory.engine import MemoryEngine
    from memforge.memory.store import MemoryStore
    from memforge.models import ContentItem
    from memforge.pipeline.memory_extractor import MemoryExtractor
    from memforge.storage.database import Database
    from memforge.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentLifecycleAdmission",
    "ExtractionWorkPool",
    "GeneSyncOrchestrator",
    "SourceSyncMode",
    "SyncMemoryObserver",
    "get_process_document_lifecycle_admission",
]



class SourceSyncMode(str, Enum):
    """What one Source sync run reads.

    NORMAL discovers and fetches from the provider, and retries the Documents
    of the Source's failure ledger that its ``FailedItemRetry`` selects.
    REPROCESS processes the Source Units of named Documents with the current
    adapter and compiler: extraction reads every ReadingGroup and every
    Support is read over the whole Unit. It reads each Document's current
    state from the provider when the Source can ask for one Document by id,
    and its stored input otherwise (see
    :mod:`memforge.pipeline.stored_document`). A NORMAL run processes each
    Document it retries from the ledger the same way. REPROCESS never advances
    the sync cursor or infers removals.
    """

    NORMAL = "normal"
    REPROCESS = "reprocess"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_extraction_metrics(
    results: list[MemoryExtractionResult],
) -> dict[str, object]:
    """Aggregate content-free LLM cost signals across one bounded extraction fan-out."""

    return aggregate_extraction_metrics(results)


@dataclass(frozen=True, slots=True)
class ExtractionAdmission:
    """Content-free observation of one admitted extraction work item."""

    queue_wait_ms: int
    multimodal: bool
    active_multimodal: int


@dataclass(frozen=True, slots=True)
class _SourceDerivationRecovery:
    """Per-Unit recovery results shared by counting and deferred convergence."""

    results: tuple[dict[str, Any], ...]

    @property
    def completed_results(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            result
            for result in self.results
            if result.get("deferred_lifecycle") is None and not result["failed"]
        )

    @property
    def failed_results(self) -> tuple[dict[str, Any], ...]:
        return tuple(result for result in self.results if result["failed"])

    @property
    def deferred_results(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            result
            for result in self.results
            if result.get("deferred_lifecycle") is not None
        )

    @property
    def processed(self) -> int:
        return sum(bool(result["processed"]) for result in self.results)

    @property
    def updated(self) -> int:
        return sum(bool(result["updated"]) for result in self.results)

    @property
    def memories_extracted(self) -> int:
        return sum(int(result["memories_extracted"]) for result in self.results)


class ExtractionWorkPool:
    """Work-conserving fair pool for app-wide heavy extraction work."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self._condition = asyncio.Condition()
        self._multimodal_semaphore = asyncio.Semaphore(1)
        self._active_by_source: dict[str, int] = defaultdict(int)
        self._waiting_by_source: dict[str, int] = defaultdict(int)
        self._total_active = 0
        self._active_multimodal = 0

    @asynccontextmanager
    async def slot(self, source_id: str, *, multimodal: bool = False):
        started_at = asyncio.get_running_loop().time()
        multimodal_acquired = False
        multimodal_active = False
        worker_acquired = False
        try:
            if multimodal:
                await self._multimodal_semaphore.acquire()
                multimodal_acquired = True
            await self.acquire(source_id)
            worker_acquired = True
            if multimodal:
                self._active_multimodal += 1
                multimodal_active = True
            yield ExtractionAdmission(
                queue_wait_ms=max(
                    0,
                    round((asyncio.get_running_loop().time() - started_at) * 1000),
                ),
                multimodal=multimodal,
                active_multimodal=self._active_multimodal,
            )
        finally:
            if worker_acquired:
                await self.release(source_id)
            if multimodal_active:
                self._active_multimodal -= 1
            if multimodal_acquired:
                self._multimodal_semaphore.release()

    async def acquire(self, source_id: str) -> None:
        async with self._condition:
            self._waiting_by_source[source_id] += 1
            try:
                while not self._can_acquire(source_id):
                    await self._condition.wait()
                self._waiting_by_source[source_id] -= 1
                if self._waiting_by_source[source_id] <= 0:
                    self._waiting_by_source.pop(source_id, None)
                self._active_by_source[source_id] += 1
                self._total_active += 1
            except BaseException:
                self._waiting_by_source[source_id] -= 1
                if self._waiting_by_source[source_id] <= 0:
                    self._waiting_by_source.pop(source_id, None)
                self._condition.notify_all()
                raise

    async def release(self, source_id: str) -> None:
        async with self._condition:
            if self._active_by_source[source_id] <= 0:
                raise RuntimeError(f"Extraction work slot was not held by {source_id}")
            self._active_by_source[source_id] -= 1
            if self._active_by_source[source_id] <= 0:
                self._active_by_source.pop(source_id, None)
            self._total_active -= 1
            self._condition.notify_all()

    def _can_acquire(self, source_id: str) -> bool:
        if self._total_active >= self.max_workers:
            return False

        source_count = len(
            {source for source, count in self._active_by_source.items() if count > 0}
            | {source for source, count in self._waiting_by_source.items() if count > 0}
        )
        fair_share = max(1, math.ceil(self.max_workers / max(1, source_count)))

        if self._active_by_source[source_id] < fair_share:
            return True

        for waiting_source, waiting_count in self._waiting_by_source.items():
            if waiting_source == source_id or waiting_count <= 0:
                continue
            if self._active_by_source[waiting_source] < fair_share:
                return False
        return True


_PROCESS_EXTRACTION_POOL_LOCK = threading.Lock()
_PROCESS_EXTRACTION_POOL: ExtractionWorkPool | None = None
_PROCESS_EXTRACTION_POOL_LIMIT: int | None = None


def get_process_extraction_work_pool(max_workers: int) -> ExtractionWorkPool:
    """Return the one process-wide pool for heavy extraction work."""
    global _PROCESS_EXTRACTION_POOL, _PROCESS_EXTRACTION_POOL_LIMIT

    requested_limit = max(1, int(max_workers))
    with _PROCESS_EXTRACTION_POOL_LOCK:
        if _PROCESS_EXTRACTION_POOL is None:
            _PROCESS_EXTRACTION_POOL = ExtractionWorkPool(requested_limit)
            _PROCESS_EXTRACTION_POOL_LIMIT = requested_limit
            return _PROCESS_EXTRACTION_POOL

        if requested_limit != _PROCESS_EXTRACTION_POOL_LIMIT:
            logger.warning(
                "Ignoring sync extraction worker limit %d because process-wide limit %d is already active",
                requested_limit,
                _PROCESS_EXTRACTION_POOL_LIMIT,
            )

        return _PROCESS_EXTRACTION_POOL


class DocumentLifecycleAdmission:
    """Process-wide admission control for memory-heavy document lifecycles."""

    def __init__(self, max_active: int) -> None:
        self.max_active = max(1, int(max_active))
        self._semaphore = asyncio.Semaphore(self.max_active)
        self._lock = asyncio.Lock()
        self._active = 0
        self._max_active_seen = 0

    @asynccontextmanager
    async def slot(self, source_id: str, doc_id: str):
        del source_id, doc_id
        started_at = asyncio.get_running_loop().time()
        await self._semaphore.acquire()
        queue_wait_ms = max(
            0,
            round((asyncio.get_running_loop().time() - started_at) * 1000),
        )
        async with self._lock:
            self._active += 1
            self._max_active_seen = max(self._max_active_seen, self._active)
        try:
            yield queue_wait_ms
        finally:
            async with self._lock:
                self._active -= 1
            self._semaphore.release()

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def max_active_seen(self) -> int:
        return self._max_active_seen


_PROCESS_DOCUMENT_LIFECYCLE_LOCK = threading.Lock()
_PROCESS_DOCUMENT_LIFECYCLE_ADMISSION: DocumentLifecycleAdmission | None = None
_PROCESS_DOCUMENT_LIFECYCLE_LIMIT: int | None = None


def get_process_document_lifecycle_admission(max_active: int) -> DocumentLifecycleAdmission | None:
    """Return the process-wide admission controller for memory-heavy sync work."""
    global _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION, _PROCESS_DOCUMENT_LIFECYCLE_LIMIT

    if max_active <= 0:
        return None

    requested_limit = max(1, int(max_active))
    with _PROCESS_DOCUMENT_LIFECYCLE_LOCK:
        if _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION is None:
            _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION = DocumentLifecycleAdmission(requested_limit)
            _PROCESS_DOCUMENT_LIFECYCLE_LIMIT = requested_limit
            return _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION

        if requested_limit < (_PROCESS_DOCUMENT_LIFECYCLE_LIMIT or requested_limit) and (
            _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION.active_count == 0
        ):
            _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION = DocumentLifecycleAdmission(requested_limit)
            _PROCESS_DOCUMENT_LIFECYCLE_LIMIT = requested_limit
            return _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION

        if requested_limit != _PROCESS_DOCUMENT_LIFECYCLE_LIMIT:
            logger.warning(
                "Ignoring sync document lifecycle limit %d because process-wide limit %d is already active",
                requested_limit,
                _PROCESS_DOCUMENT_LIFECYCLE_LIMIT,
            )

        return _PROCESS_DOCUMENT_LIFECYCLE_ADMISSION


MAX_RETRIES = 3
MAX_LIFECYCLE_CONVERGENCE_ROUNDS = 3
MAX_LIFECYCLE_CONVERGENCE_ATTEMPTS = 128
"""Number of retry attempts per content item before marking as failed."""


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (GPT-4 encoding). Falls back to char/4."""
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_updated_at(source_semantics: dict[str, Any]) -> datetime | None:
    """The source time of a Unit's content as its Gene reports it (design 0.9).

    ``ContentItem.last_modified`` is a discovery marker and may be a fetch or
    submission time, so it is never read as the content's time.
    """

    return parse_source_time(reported_source_time(source_semantics.get(SOURCE_UPDATED_AT_KEY)))


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"


def _is_provider_unreachable(error: str) -> bool:
    normalized = error.lower()
    return any(
        marker in normalized
        for marker in (
            "all connection attempts failed",
            "cannot connect to host",
            "connect call failed",
            "connect timeout",
            "connection refused",
            "connection timed out",
            "failed to connect",
            "name or service not known",
            "network is unreachable",
            "no route to host",
            "nodename nor servname",
            "temporary failure in name resolution",
        )
    )


# How rate-limit failures read: Atlassian's rate-limit error, an HTTP "429 Too Many Requests",
# or a LiteLLM RateLimitError. A bare status number is not matched, because identifiers
# and ranges quoted in other errors contain the same digits.
_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "too many requests")


def _failure_category(error: str) -> str:
    normalized = error.lower()
    if "embedding provider unreachable" in normalized:
        return "embedding_provider_unreachable"
    if "llm provider unreachable" in normalized:
        return "llm_provider_unreachable"
    if _is_provider_unreachable(error) and (
        "litellm" in normalized or "anthropicexception" in normalized or "openaiexception" in normalized
    ):
        return "llm_provider_unreachable"
    if any(marker in normalized for marker in _RATE_LIMIT_MARKERS):
        return "rate_limit"
    if "pdf export" in normalized or "did not produce a pdf" in normalized:
        return "pdf_export"
    if "certificate_verify_failed" in normalized or "certificate verify" in normalized:
        return "certificate"
    return "other"


def summarize_unreadable_conversations(names: Sequence[str]) -> str:
    return (
        f"{_plural(len(names), 'conversation')} could not be read: {', '.join(names)}. "
        "Their memories are kept. Restore access, or deselect a conversation that is no longer used."
    )


def summarize_failed_documents(docs_failed: int, failed_docs: list[FailedDoc]) -> str:
    if docs_failed <= 0:
        return "Sync failed"

    counts: dict[str, int] = {}
    for failed_doc in failed_docs:
        category = _failure_category(failed_doc.error)
        counts[category] = counts.get(category, 0) + 1

    if counts.get("embedding_provider_unreachable") or counts.get("llm_provider_unreachable"):
        parts = [f"{_plural(docs_failed, 'document')} could not be synced."]
        details: list[str] = []
        if counts.get("embedding_provider_unreachable"):
            details.append(
                f"Embedding provider was unreachable for "
                f"{_plural(counts['embedding_provider_unreachable'], 'document')}"
            )
        if counts.get("llm_provider_unreachable"):
            details.append(
                f"LLM provider was unreachable for {_plural(counts['llm_provider_unreachable'], 'document')}"
            )
        if counts.get("pdf_export"):
            details.append(f"PDF export was unavailable for {_plural(counts['pdf_export'], 'document')}")
        if counts.get("rate_limit"):
            details.append(f"Source rate limit was reached for {_plural(counts['rate_limit'], 'document')}")
        if counts.get("certificate"):
            details.append(f"certificate verification failed for {_plural(counts['certificate'], 'document')}")
        if counts.get("other"):
            details.append(f"{_plural(counts['other'], 'document')} failed for other reasons")
        if details:
            parts.append("; ".join(details) + ".")
        if counts.get("rate_limit"):
            parts.append("Wait a few minutes, then retry the sync.")
        parts.append("Check the provider endpoint, network access, and service status, then retry the sync.")
        return " ".join(parts)

    if counts.get("pdf_export") or counts.get("rate_limit") or counts.get("certificate"):
        parts = [f"{_plural(docs_failed, 'document')} could not be imported."]
        details: list[str] = []
        if counts.get("pdf_export"):
            details.append(f"PDF export was unavailable for {_plural(counts['pdf_export'], 'document')}")
        if counts.get("rate_limit"):
            details.append(f"Source rate limit was reached for {_plural(counts['rate_limit'], 'document')}")
        if counts.get("certificate"):
            details.append(f"certificate verification failed for {_plural(counts['certificate'], 'document')}")
        if counts.get("other"):
            details.append(f"{_plural(counts['other'], 'document')} failed for other reasons")
        if details:
            parts.append("; ".join(details) + ".")
        if counts.get("rate_limit"):
            parts.append("Wait a few minutes, then retry the sync.")
        return " ".join(parts)

    return f"{_plural(docs_failed, 'document')} could not be synced. Review the failed document details."


_MAX_RETAINED_DOCUMENT_ERROR_CHARS = 512


class MemoryExtractionFailure(RuntimeError):
    """Extraction exhausted its own retry policy; do not replay the document.

    ``error_code`` is the content-free code of the failure, when extraction
    reported one.
    """

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code


def _fail_deferred_commits_nothing_can_unblock(
    results: list[dict[str, Any]], run_source_unit_ids: set[str],
) -> None:
    """Fail every deferred intent that no Unit outside ``run_source_unit_ids`` holds up.

    Such an intent already had every chance to commit; an intent that waits on
    an outside Unit, directly or through another deferred intent, stays deferred.
    """
    deferred = {
        str(result["source_unit_id"]): result
        for result in results
        if result.get("deferred_lifecycle") is not None and result.get("source_unit_id")
    }
    blockers = {
        unit_id: set(result["deferred_lifecycle"].handle.blocking_source_unit_ids)
        for unit_id, result in deferred.items()
    }
    waiting = {unit_id for unit_id, blocked_by in blockers.items() if not blocked_by <= run_source_unit_ids}
    while newly_waiting := {
        unit_id for unit_id, blocked_by in blockers.items() if unit_id not in waiting and blocked_by & waiting
    }:
        waiting |= newly_waiting
    for unit_id, result in deferred.items():
        if unit_id not in waiting:
            result["terminal_error"] = _retained_document_error(result["deferred_lifecycle"])
            result["terminal_failure"] = result["deferred_lifecycle"]
            result["deferred_lifecycle"] = None


def _retained_document_error(exc: BaseException) -> str:
    """Project an exception into bounded state without retaining its traceback."""

    if len(exc.args) == 1 and isinstance(exc.args[0], str):
        detail = exc.args[0]
        if 0 < len(detail) <= _MAX_RETAINED_DOCUMENT_ERROR_CHARS:
            stripped = detail.strip()
            if stripped:
                return stripped
    return f"{type(exc).__name__}: error details omitted"


def _failure_stage(exc: BaseException, stage: str) -> str:
    """The step at which a Document failed: the lifecycle's own stage when it reports one."""

    if isinstance(exc, SourceUnitLifecycleExecutionError) and exc.stage is not None:
        return exc.stage
    return stage


def _failure_class(exc: BaseException) -> str:
    """The qualified class of a failure: the class of the failure a lifecycle error wraps."""

    if isinstance(exc, SourceUnitLifecycleExecutionError) and exc.__cause__ is not None:
        exc = exc.__cause__
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _run_item_failure(
    exc: BaseException,
    *,
    document_id: str,
    title: str | None,
    stage: str,
    store: Any,
    source_unit_id: str | None = None,
    item: ContentItem | None = None,
) -> RunItemFailure:
    """One Document's failure, classified for the failure ledger."""

    error_code = getattr(exc, "error_code", None)
    return RunItemFailure(
        document_id=document_id,
        stage=_failure_stage(exc, stage),
        error_class=_failure_class(exc),
        message=_retained_document_error(exc),
        retryable=retries_failure(exc, store=store),
        title=title,
        error_code=error_code if isinstance(error_code, str) else None,
        source_unit_id=source_unit_id,
        item=item,
    )


def _derivation_item(context: SourceUnitDerivationContext) -> ContentItem:
    """The item that asks for a staged derivation's Document again."""

    if context.unit_input is not None:
        return context.unit_input.item
    return document_content_item(context.document)


def _select_failed_items(
    ledger: Mapping[str, SourceItemFailure],
    *,
    retry: FailedItemRetry,
    release: str,
    recovered_doc_ids: set[str],
    ended: Mapping[str, ContentItem | None],
) -> tuple[set[str], dict[str, ContentItem | None]]:
    """The ledger Documents a NORMAL run processes again, and those it leaves alone while the provider keeps them unchanged.

    Recovery has already processed the Documents of the derivations it
    resumed. Of those, a Document whose staged derivation recovery ended is
    processed again by a requested sync and left alone by a scheduled one, as
    a deterministic failure of this release is. A failed removal is retried
    by the run's own absence check, not by processing the Document. Each
    Document left alone maps to the item that failed, when one was recorded.
    """

    retried: set[str] = set()
    left_alone: dict[str, ContentItem | None] = {}
    for failure in ledger.values():
        if failure.stage == REMOVAL_STAGE or failure.document_id in recovered_doc_ids:
            continue
        if failure.retried_by(retry, release=release):
            retried.add(failure.document_id)
        else:
            left_alone[failure.document_id] = failure.item
    if retry is FailedItemRetry.ALL:
        retried.update(ended)
    else:
        left_alone.update(ended)
    return retried, left_alone


def _same_provider_revision(item: ContentItem, failed: ContentItem | None) -> bool:
    """Whether discovery returned the provider revision that failed: the same version and change time."""

    return failed is not None and (item.version, item.last_modified) == (failed.version, failed.last_modified)


def _memory_extraction_error(
    *,
    doc_id: str,
    extraction: MemoryExtractionResult,
) -> MemoryExtractionFailure:
    _error_type, error_code, _fields = safe_derivation_error(extraction)
    message = f"memory extraction failed for {doc_id}: {extraction.error_type}"
    detail = extraction.error or ""
    if detail and len(detail) <= _MAX_RETAINED_DOCUMENT_ERROR_CHARS:
        stripped = detail.strip()
        if stripped:
            message = f"{message}: {stripped}"
    return MemoryExtractionFailure(message, error_code=error_code)


def _skipped_reading_group_count(extraction: MemoryExtractionResult) -> int:
    """The ReadingGroups a successful extraction left unread."""

    return int((extraction.metadata or {}).get("skipped_reading_group_count", 0) or 0)


# ---------------------------------------------------------------------------
# GeneSyncOrchestrator
# ---------------------------------------------------------------------------


@dataclass
class _SourceUnitExecutionDiagnostics:
    """Result of one diagnostic scope, never a durable lifecycle state."""

    source_unit_id: str | None = None
    status: str = "failed"
    error_class: str | None = None
    # The content-free code of the failure, when the failing step reported one.
    error_code: str | None = None
    # The step of the Document's work that ran last: ``fetch``, ``normalize``,
    # ``projection``, ``derivation`` or ``lifecycle``.
    stage: str = "derivation"
    # The ReadingGroups the derivation left unread: the model's answer for the
    # group was rejected, or the group does not fit the model.
    skipped_reading_group_count: int = 0

    def bind_source_unit(self, source_unit_id: str) -> None:
        if self.source_unit_id is not None and self.source_unit_id != source_unit_id:
            raise ValueError("a diagnostic scope cannot span different Source Units")
        self.source_unit_id = source_unit_id


class GeneSyncOrchestrator:
    """Orchestrates the full sync pipeline for a single gene (data source).

    Lifecycle per sync run::

        orchestrator = GeneSyncOrchestrator(db, doc_store, memory_extractor, ...)
        state = await orchestrator.sync_gene(gene, source_name, source_id)

    Each content item discovered by the Gene flows through one provider-neutral
    Source Projection, bounded extraction batches, revision-pinned Evidence,
    and one complete atomic Lifecycle Plan. Provider-specific semantics stop at
    the injected Source Projection Adapter.

    Concurrency is bounded by a semaphore (for LLM/embedding API calls)
    and a lock (for SQLite write serialization).
    """

    def __init__(
        self,
        db: Database,
        doc_store: DocumentStore,
        memory_extractor: MemoryExtractor,
        memory_engine: MemoryEngine,
        memory_store: MemoryStore,
        max_concurrent: int = 3,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        memory_observer: SyncMemoryObserver | None = None,
        memory_reclaimer: ProcessMemoryReclaimer | None = None,
        source_projection_adapter: SourceProjectionAdapter | None = None,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        structured_llm_client: LiteLlmStructuredClient | None = None,
        runtime_event_trace_sink: RuntimeEventTraceSink | None = None,
    ) -> None:
        self.db = db
        self.doc_store = doc_store
        self.memory_extractor = memory_extractor
        self.memory_engine = memory_engine
        self.memory_store = memory_store
        self.max_concurrent = max(1, max_concurrent)
        self.extraction_pool = extraction_pool or ExtractionWorkPool(self.max_concurrent)
        self.document_lifecycle_admission = document_lifecycle_admission
        self.memory_observer = memory_observer
        self.memory_reclaimer = memory_reclaimer or ProcessMemoryReclaimer()
        self.source_projection_adapter = source_projection_adapter or DEFAULT_SOURCE_PROJECTION_ADAPTER
        self._retry_sleep = retry_sleep
        self.structured_llm_client = structured_llm_client
        self.runtime_event_trace_sink = runtime_event_trace_sink

        self._db_lock = asyncio.Lock()

    def _source_parallelism_limit(self) -> int:
        return self.max_concurrent

    def _document_parallelism_limit(self) -> int:
        if self.document_lifecycle_admission is None:
            return self.max_concurrent
        return min(self.max_concurrent, self.document_lifecycle_admission.max_active)

    @asynccontextmanager
    async def _heavy_work_slot(self, source_id: str, *, multimodal: bool = False):
        async with self.extraction_pool.slot(
            source_id,
            multimodal=multimodal,
        ) as admission:
            yield admission

    @asynccontextmanager
    async def _document_lifecycle_slot(self, source_id: str, doc_id: str):
        if self.document_lifecycle_admission is None:
            yield 0
            return
        async with self.document_lifecycle_admission.slot(source_id, doc_id) as queue_wait_ms:
            yield queue_wait_ms

    def _memory_sample(
        self,
        stage: str,
        *,
        source_id: str,
        run_id: str | None,
        doc_id: str | None = None,
        ok: bool = True,
        error: Exception | None = None,
        **fields: Any,
    ) -> None:
        if self.memory_observer is None:
            return
        active = None
        max_seen = None
        if self.document_lifecycle_admission is not None:
            active = self.document_lifecycle_admission.active_count
            max_seen = self.document_lifecycle_admission.max_active_seen
        self.memory_observer.sample(
            stage,
            source_id=source_id,
            run_id=run_id or "",
            doc_id=doc_id,
            ok=ok,
            error_class=type(error).__name__ if error is not None else None,
            active_lifecycles=active,
            max_active_seen=max_seen,
            **fields,
        )

    # ==================================================================
    # Public: sync_gene
    # ==================================================================

    async def sync_gene(
        self,
        gene: Gene,
        source_name: str,
        source_id: str,
        progress_callback: Callable[[dict], None] | None = None,
        force_full_sync: bool = False,
        input_snapshot: SnapshotBinding | None = None,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        source_activity: SourceActivityLease | None = None,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
        reusable_projection_doc_ids: frozenset[str] = frozenset(),
        record_terminal_result: bool = True,
        failed_item_retry: FailedItemRetry = FailedItemRetry.ALL,
        collection_attempt: CollectionAttempt | None = None,
    ) -> SyncState:
        """Run the full sync pipeline for a gene.

        Parameters
        ----------
        gene:
            An authenticated gene instance (authenticate() is called here).
        source_name:
            Human-readable name of this source (e.g. "Confluence - Engineering").
        source_id:
            Unique identifier for the source (FK in documents table).
        progress_callback:
            Optional callback invoked with ``{phase, current, total, title}``
            at each processing step.
        force_full_sync:
            When true, ignore the incremental cursor and reprocess discovered
            documents even when their content hash is unchanged.
        input_snapshot:
            The complete input snapshot a device submitted for a document
            Source's run. The run discovers all of it, while still skipping
            Documents whose content is unchanged, and its manifest is the
            run's scope listing.
        reprocess_doc_ids:
            The Documents a REPROCESS run processes; required by and only
            accepted with that mode.
        execution_mode:
            NORMAL discovers from the provider; REPROCESS processes
            ``reprocess_doc_ids`` (see :class:`SourceSyncMode`). Discovery
            options do not apply to it.
        record_terminal_result:
            Persist the terminal SyncState and history here. Durable workers
            disable this so their lease-fenced completion transaction owns the
            complete terminal read model.
        failed_item_retry:
            Which rows of the Source's failure ledger a NORMAL run retries:
            ``ALL`` for a sync a person requested, ``RETRYABLE`` for a
            scheduled one (see :mod:`memforge.source_item_failures`).
        collection_attempt:
            The collection attempt a conversation Source's run processes;
            required for, and only read by, the conversation family.

        Every per-Document failure is recorded in the failure ledger, which
        owns its retry, so a run whose failures are all per-Document ones
        advances the incremental cursor; its status is ``partial``. A
        conversation the Source cannot read also makes the run ``partial``,
        without a ledger row: it keeps its Support until the user restores
        access or deselects it. A failure of the run itself keeps the cursor.

        Returns
        -------
        SyncState
            Final sync result with counts and error details.
        """
        reprocessing = execution_mode is SourceSyncMode.REPROCESS
        if reprocessing != bool(reprocess_doc_ids):
            raise ValueError("reprocess_doc_ids name the Documents of a REPROCESS run")
        if source_activity is not None:
            if source_activity.source_id != source_id:
                raise ValueError("source activity does not belong to the synced Source")
        run_id = uuid.uuid4().hex[:12]
        durable_cycle_id = lifecycle_cycle_id or run_id
        transition_run_id = scope_transition_run_id or durable_cycle_id
        started_at = datetime.now(timezone.utc)
        bind_document_store = getattr(gene, "bind_document_store", None)
        if callable(bind_document_store):
            bind_document_store(self.doc_store)
        family = collection_family(gene, db=self.db, doc_store=self.doc_store, source_id=source_id)
        family.prepare(collection_attempt=collection_attempt, input_snapshot=input_snapshot)
        complete_snapshot = family.input_snapshot is not None

        logger.info(
            "Sync started for %s (source_id=%s, run_id=%s)",
            source_name,
            source_id,
            run_id,
        )
        self._memory_sample("sync_run_start", source_id=source_id, run_id=run_id)

        docs_processed = 0
        docs_updated = 0
        docs_failed = 0
        memories_extracted = 0
        skipped_reading_group_count = 0
        failed_docs: list[FailedDoc] = []
        # The ledger record of each failed Document, and the Documents that
        # committed or were removed, which leave the ledger.
        item_failures: dict[str, RunItemFailure] = {}
        settled_doc_ids: set[str] = set()
        release = current_release()
        runtime_bundles = []
        error_message: str | None = None
        crawled_doc_ids: set[str] = set()
        existing_state = await self.db.get_sync_state(source_id)
        configured_source = await self.db.get_source(source_id)
        configured_source_type = str((configured_source or {}).get("type") or "unknown")
        configured_projection_scope = canonical_projection_scope(
            configured_source_type,
            (configured_source or {}).get("config") or {},
        )
        configured_access_context = source_access_context(configured_source or {})
        scope_transition = await self.db.get_open_projection_scope_transition(source_id)
        if force_full_sync or reprocessing or scope_transition is not None:
            # Reuse is an ordinary incremental optimization only. Full syncs
            # and scope transitions require the full per-document path.
            reusable_projection_doc_ids = frozenset()
        # Each Document that commits during an open transition of a document
        # Source becomes one of the transition's members, in its commit.
        member_transition_id = (
            scope_transition.id if scope_transition is not None and family.records_transition_members else None
        )
        transition_started = False
        transition_completed = False
        reused_projection_count = 0
        total_item_count = 0
        failure_retryable = True
        absence_check_skipped_reason: AbsenceCheckSkippedReason | None = None
        unlisted_present_count: int | None = None
        recovered_completed_results: tuple[dict[str, Any], ...] = ()
        recovered_deferred_results: tuple[dict[str, Any], ...] = ()
        recovered_failed_results: tuple[dict[str, Any], ...] = ()
        unreadable_conversations: tuple[str, ...] = ()

        def fail_document(failure: RunItemFailure, *, error: str | None = None) -> None:
            failed_docs.append(
                FailedDoc(
                    doc_id=failure.document_id,
                    title=failure.title or failure.document_id,
                    error=error if error is not None else failure.message,
                )
            )
            item_failures[failure.document_id] = failure

        try:
            recovered = await self._resume_source_derivations(
                source_id=source_id,
                source_activity=source_activity,
                run_id=run_id,
                lifecycle_execution_owner_id=durable_cycle_id,
                progress_callback=progress_callback,
                scope_transition_id=member_transition_id,
            )
            recovered_completed_results = recovered.completed_results
            recovered_deferred_results = recovered.deferred_results
            recovered_failed_results = recovered.failed_results

            # The failure ledger decides which failed Documents a NORMAL run
            # processes again, and which a scheduled run leaves alone when
            # discovery returns them.
            ledger = {
                failure.document_id: failure
                for failure in await self.db.list_source_item_failures(source_id=source_id)
            }
            retried_doc_ids: set[str] = set()
            left_alone: dict[str, ContentItem | None] = {}
            if not reprocessing and scope_transition is None:
                retried_doc_ids, left_alone = _select_failed_items(
                    ledger,
                    retry=failed_item_retry,
                    release=release,
                    recovered_doc_ids={str(result["doc_id"]) for result in recovered.results},
                    ended={str(result["doc_id"]): result["failure"].item for result in recovered_failed_results},
                )
            skipped_doc_ids: set[str] = set()
            rediscovering = source_rediscovers_documents(
                configured_source_type,
                (configured_source or {}).get("config") or {},
            )

            stored_documents: dict[str, StoredSourceDocument] = {}
            if reprocessing:
                if scope_transition is not None:
                    raise RuntimeError(
                        "reprocess waits for the open Projection Scope transition to complete"
                    )
                if rediscovering:
                    await gene.authenticate()
                items = []
                for doc_id in sorted(reprocess_doc_ids or ()):
                    try:
                        item, stored = await family.document_input(
                            doc_id,
                            rediscovering=rediscovering,
                            failure=ledger.get(doc_id),
                        )
                    except StoredDocumentUnavailable as exc:
                        docs_failed += 1
                        fail_document(
                            _run_item_failure(exc, document_id=doc_id, title=doc_id, stage="fetch", store=self.db),
                            error=str(exc),
                        )
                        continue
                    if stored is not None:
                        stored_documents[doc_id] = stored
                    items.append(item)
                indexed_doc_ids: set[str] = set()
                last_sync_time = None
                total_item_count = len(items)
            else:
                # ----------------------------------------------------------
                # Step 0: Authenticate
                # ----------------------------------------------------------
                await gene.authenticate()
                logger.info("Gene %s authenticated successfully", source_name)
                if scope_transition is not None:
                    if dict(scope_transition.target_scope) != configured_projection_scope:
                        raise RuntimeError("open Projection Scope transition does not match configured target scope")
                    scope_transition = await self.db.start_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                    )
                    transition_started = True
                    family.begin_scope_transition(scope_transition)

                # ----------------------------------------------------------
                # Step 1: Get indexed doc_ids for deletion detection
                # ----------------------------------------------------------
                indexed_doc_ids = await self._get_indexed_doc_ids(source_id)
                logger.info(
                    "Found %d previously indexed documents for %s",
                    len(indexed_doc_ids),
                    source_id,
                )

                # ----------------------------------------------------------
                # Step 2: Get last sync time for incremental discovery
                # ----------------------------------------------------------
                last_sync_time = await family.since(
                    existing_state=existing_state,
                    indexed_doc_ids=indexed_doc_ids,
                    full_discovery=force_full_sync or scope_transition is not None,
                )

                # ----------------------------------------------------------
                # Step 3: Discover content items
                # ----------------------------------------------------------
                items: list[ContentItem] = []
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "discovering",
                            "current": 0,
                            "total": 0,
                            "title": None,
                        }
                    )

                async for item in gene.discover(since=last_sync_time):
                    crawled_doc_ids.add(item.item_id)
                    if item.item_id in left_alone:
                        if _same_provider_revision(item, left_alone[item.item_id]):
                            # A deterministic failure of this release waits for
                            # a requested sync, the next release or a change.
                            skipped_doc_ids.add(item.item_id)
                            continue
                        # The provider changed the Document since it failed.
                        retried_doc_ids.add(item.item_id)
                    items.append(item)
                    if progress_callback:
                        progress_callback(
                            {
                                "phase": "discovering",
                                "current": len(items),
                                "total": 0,
                                "title": None,
                            }
                        )

                logger.info(
                    "Discovered %d content items from %s (since=%s)",
                    len(crawled_doc_ids),
                    source_name,
                    last_sync_time.isoformat() if last_sync_time else "full sync",
                )
                if skipped_doc_ids:
                    logger.info(
                        "Left %d unchanged discovered Documents of %s alone: each failed deterministically "
                        "in this release",
                        len(skipped_doc_ids),
                        source_id,
                    )
                # The run's one scope listing, taken before any failed
                # Document is retried, decides the ledger rows it does not
                # name: settled when their item is gone, or kept untried. In a
                # scope transition it is the listing of the new scope.
                await family.take_scope_listing()
                ledger_partition = family.partition_ledger(ledger, rediscovering=rediscovering)
                settled_doc_ids.update(ledger_partition.settled)
                # A failed Document that discovery did not return is read the
                # way a reprocess reads it.
                for doc_id in sorted(
                    retried_doc_ids - crawled_doc_ids - ledger_partition.settled - ledger_partition.untried
                ):
                    try:
                        item, stored = await family.document_input(
                            doc_id,
                            rediscovering=rediscovering,
                            failure=ledger.get(doc_id),
                        )
                    except StoredDocumentUnavailable as exc:
                        if exc.reason is StoredDocumentUnavailableReason.PROVIDER_DOCUMENT_MISSING:
                            # Nothing is left to retry: the provider no longer
                            # has the item, or it left the query.
                            settled_doc_ids.add(doc_id)
                            continue
                        docs_failed += 1
                        fail_document(
                            _run_item_failure(
                                exc,
                                document_id=doc_id,
                                title=ledger[doc_id].title,
                                stage="fetch",
                                store=self.db,
                            ),
                            error=str(exc),
                        )
                        continue
                    if stored is not None:
                        stored_documents[doc_id] = stored
                    items.append(item)
                discovered_doc_ids = set(crawled_doc_ids)
                unexpected_reuse_ids = reusable_projection_doc_ids - discovered_doc_ids
                if unexpected_reuse_ids and complete_snapshot:
                    raise ValueError(
                        "reusable Source Projection membership is outside provider discovery: "
                        f"{sorted(unexpected_reuse_ids)[0]}"
                    )
                reusable_projection_doc_ids = (reusable_projection_doc_ids & discovered_doc_ids) - (
                    retried_doc_ids | set(left_alone)
                )
                if member_transition_id is not None and not force_full_sync:
                    # A Document already committed under the transition is not
                    # fetched again; one that changed since is discovered by
                    # the first run after the transition completes.
                    members = set(await self.db.list_projection_scope_transition_member_doc_ids(member_transition_id))
                    member_count = len(items)
                    items = [item for item in items if item.item_id not in members]
                    if member_count > len(items):
                        logger.info(
                            "Skipped %d Documents of %s already committed under scope transition %s",
                            member_count - len(items),
                            source_id,
                            member_transition_id,
                        )
                total_item_count = len(items)
                if reusable_projection_doc_ids:
                    items = [item for item in items if item.item_id not in reusable_projection_doc_ids]
                    reused_projection_count = total_item_count - len(items)
                    docs_processed += reused_projection_count
                    logger.info(
                        "Reused %d current Source Projections for %s without per-document materialization",
                        reused_projection_count,
                        source_id,
                    )
                self._memory_sample(
                    "after_discovery",
                    source_id=source_id,
                    run_id=run_id,
                    item_count=total_item_count,
                    reused_projection_count=reused_projection_count,
                    indexed_doc_count=len(indexed_doc_ids),
                    full_sync=last_sync_time is None,
                )

            # A failed Document keeps the rename its ledger row recorded,
            # because its provider may have reported the rename only once.
            items = [with_inherited_rename(item, ledger.get(item.item_id)) for item in items]

            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "current": reused_projection_count,
                        "total": total_item_count,
                        "title": None,
                    }
                )

            # ----------------------------------------------------------
            # Step 4: Process items concurrently (with retry + error isolation)
            # ----------------------------------------------------------
            progress_counter = reused_projection_count
            docs_updated_counter = 0
            memories_extracted_counter = 0
            item_semaphore = asyncio.Semaphore(self._document_parallelism_limit())
            recovered_targets_by_doc_id: dict[str, frozenset[tuple[str, str]]] = {}
            for recovered_result in recovered_deferred_results:
                recovered_doc_id = str(recovered_result["doc_id"])
                existing_targets = recovered_targets_by_doc_id.get(
                    recovered_doc_id,
                    frozenset(),
                )
                recovered_targets_by_doc_id[recovered_doc_id] = existing_targets | {
                    (
                        str(recovered_result["source_unit_id"]),
                        str(recovered_result["target_unit_revision_id"]),
                    )
                }

            async def _process_one(item: ContentItem) -> dict:
                """Process a single item with retry logic and error isolation."""
                nonlocal docs_updated_counter, failure_retryable
                nonlocal memories_extracted_counter, progress_counter
                stats = {
                    "processed": False,
                    "updated": False,
                    "memories_extracted": 0,
                    "skipped_reading_group_count": 0,
                    "failed": False,
                    "runtime_bundle": None,
                    "source_unit_id": None,
                    "provider_target_unit_revision_id": None,
                    "deferred_lifecycle": None,
                    "recovered_deferred_target": None,
                    "doc_id": item.item_id,
                    "title": item.title,
                }
                document_completed = False

                def on_item_target(
                    source_unit_id: str,
                    target_unit_revision_id: str,
                ) -> None:
                    stats["source_unit_id"] = source_unit_id
                    stats["provider_target_unit_revision_id"] = (
                        target_unit_revision_id
                    )

                def on_item_progress(progress: dict) -> None:
                    nonlocal document_completed, progress_counter
                    if progress.get("event") == "document_processed":
                        if document_completed:
                            return
                        document_completed = True
                        progress_counter += 1
                        if progress_callback:
                            progress_callback(
                                {
                                    "phase": "processing",
                                    "current": progress_counter,
                                    "total": total_item_count,
                                    "title": progress.get("title"),
                                    "docs_updated": docs_updated_counter,
                                    "memories_extracted": memories_extracted_counter,
                                }
                            )
                        return
                    if progress_callback:
                        progress_callback(progress)

                last_error: str | None = None
                last_failure: RunItemFailure | None = None
                # A Document retried from the failure ledger is processed as a
                # reprocess, so it is derived again under a new identity.
                reprocess_item = reprocessing or item.item_id in retried_doc_ids
                async with item_semaphore:
                    for attempt in range(1, MAX_RETRIES + 1):
                        attempt_error: str | None = None
                        retry_document = True
                        diagnostics = _SourceUnitExecutionDiagnostics(stage="fetch")
                        try:
                            item_stats = await self._process_item(
                                gene=gene,
                                item=item,
                                source_name=source_name,
                                source_id=source_id,
                                run_id=run_id,
                                progress_callback=on_item_progress,
                                force_reprocess=force_full_sync or reprocess_item,
                                reprocessing=reprocess_item,
                                stored_document=stored_documents.get(item.item_id),
                                diagnostics=diagnostics,
                                projection_scope=configured_projection_scope,
                                scope_transition=(
                                    {
                                        "id": scope_transition.id,
                                        "previous_scope": dict(scope_transition.previous_scope),
                                        "target_scope": dict(scope_transition.target_scope),
                                    }
                                    if scope_transition is not None
                                    else None
                                ),
                                projection_access_context=configured_access_context,
                                complete_snapshot=complete_snapshot,
                                scope_transition_id=member_transition_id,
                                source_activity=source_activity,
                                lifecycle_execution_owner_id=durable_cycle_id,
                                lifecycle_attempt_count=attempt,
                                source_unit_target_callback=on_item_target,
                                recovered_deferred_targets=(
                                    recovered_targets_by_doc_id.get(
                                        item.item_id,
                                        frozenset(),
                                    )
                                ),
                            )
                            recovered_target = item_stats.get(
                                "recovered_deferred_target"
                            )
                            stats["processed"] = recovered_target is None
                            stats["updated"] = item_stats.get("updated", False)
                            stats["memories_extracted"] = item_stats.get(
                                "memories_extracted",
                                0,
                            )
                            stats["skipped_reading_group_count"] = (
                                diagnostics.skipped_reading_group_count
                            )
                            stats["source_unit_id"] = item_stats.get(
                                "source_unit_id"
                            )
                            stats["recovered_deferred_target"] = (
                                recovered_target
                            )
                            last_error = None
                        except SourceUnitLifecycleDeferred as exc:
                            stats["deferred_lifecycle"] = exc
                            stats["skipped_reading_group_count"] = (
                                diagnostics.skipped_reading_group_count
                            )
                            stats["source_unit_id"] = (
                                exc.handle.source_unit_id
                            )
                            stats["runtime_bundle"] = exc.runtime_bundle
                            attempt_error = None
                            retry_document = False
                        except Exception as exc:
                            attempt_error = _retained_document_error(exc)
                            last_failure = _run_item_failure(
                                exc,
                                document_id=item.item_id,
                                title=item.title,
                                stage=diagnostics.stage,
                                store=self.db,
                                source_unit_id=diagnostics.source_unit_id,
                                item=item,
                            )
                            stats["runtime_bundle"] = getattr(exc, "runtime_bundle", None)
                            explicitly_retryable = bool(
                                getattr(exc, "retryable", True)
                            )
                            retry_document = (
                                not isinstance(exc, MemoryExtractionFailure)
                                and explicitly_retryable
                            )
                            if not explicitly_retryable:
                                failure_retryable = False
                        if stats["deferred_lifecycle"] is not None:
                            break
                        if attempt_error is None:
                            stats["runtime_bundle"] = None
                            break
                        last_error = attempt_error
                        if not retry_document:
                            logger.error(
                                "Failed to process %s after extraction retry exhaustion: %s",
                                item.item_id,
                                attempt_error,
                            )
                            break
                        if attempt < MAX_RETRIES:
                            delay = 2**attempt
                            logger.warning(
                                "Retry %d/%d for %s after error: %s",
                                attempt,
                                MAX_RETRIES,
                                item.item_id,
                                attempt_error,
                            )
                            await self._retry_sleep(delay)
                        else:
                            logger.error(
                                "Failed to process %s after %d attempts: %s",
                                item.item_id,
                                MAX_RETRIES,
                                attempt_error,
                            )

                if last_error is not None and last_failure is not None:
                    stats["failed"] = True
                    fail_document(last_failure, error=last_error)
                elif stats["processed"]:
                    if item.item_id in ledger and stats.get("deferred_lifecycle") is None:
                        # The Document committed; its row leaves the ledger
                        # at once, so a run that stops before it ends keeps none.
                        await self.db.delete_source_item_failures(
                            source_id, (item.item_id,), source_activity=source_activity,
                        )
                    if stats["updated"]:
                        docs_updated_counter += 1
                    memories_extracted_counter += stats["memories_extracted"]
                    if progress_callback:
                        progress_callback(
                            {
                                "phase": "processing",
                                "current": progress_counter,
                                "total": total_item_count,
                                "title": item.title,
                                "docs_updated": docs_updated_counter,
                                "memories_extracted": memories_extracted_counter,
                            }
                        )

                # Update progress if the item failed before it reached completion.
                if not document_completed and progress_callback:
                    progress_counter += 1
                    progress_callback(
                        {
                            "phase": "processing",
                            "current": progress_counter,
                            "total": total_item_count,
                            "title": item.title,
                            "docs_updated": docs_updated_counter,
                            "memories_extracted": memories_extracted_counter,
                        }
                    )

                return stats

            item_tasks = [asyncio.create_task(_process_one(item)) for item in items]
            try:
                results = await asyncio.gather(*item_tasks)
            except BaseException:
                # The sync run owns every per-item task.  A child cancellation
                # is propagated by gather without cancelling its siblings, so
                # drain them explicitly before the caller may release runtime
                # resources such as the source database bundle.
                for task in item_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*item_tasks, return_exceptions=True)
                raise

            # A successful recovery has already applied this target, so exact
            # provider rediscovery follows the projected no-op path without
            # semantic work. Retain the recovery outcome as the single source
            # of terminal counts instead of replacing it with that no-op.
            completed_recovery_targets = {
                (
                    str(result["source_unit_id"]),
                    str(result["target_unit_revision_id"]),
                )
                for result in recovered_completed_results
            }
            provider_results = [
                result
                for result in results
                if result.get("recovered_deferred_target") is None
                and (
                    str(result.get("source_unit_id") or ""),
                    str(
                        result.get("provider_target_unit_revision_id")
                        or ""
                    ),
                )
                not in completed_recovery_targets
            ]
            provider_source_unit_ids = {
                str(result["source_unit_id"])
                for result in provider_results
                if result.get("source_unit_id")
            }
            # A recovery that failed counts only when this run's provider
            # pass did not process the same Unit again.
            recovered_results = [
                *recovered_completed_results,
                *recovered_deferred_results,
                *recovered_failed_results,
            ]
            recovered_results_for_run = [
                result
                for result in recovered_results
                if str(result["source_unit_id"])
                not in provider_source_unit_ids
            ]
            for recovered_result in recovered_deferred_results:
                if (
                    str(recovered_result["source_unit_id"])
                    not in provider_source_unit_ids
                ):
                    continue
                await self.db.supersede_source_derivation(
                    str(recovered_result["derivation_id"]),
                    reason_code=DERIVATION_INPUT_SUPERSEDED,
                )
            effective_results = [
                *recovered_results_for_run,
                *provider_results,
            ]
            deferred_results = [
                result
                for result in effective_results
                if result.get("deferred_lifecycle") is not None
            ]
            deferred_result_ids = {id(result) for result in deferred_results}

            # Aggregate completed work after recovery/provider selection. Deferred work is not a
            # failure and must not prevent authoritative tombstones that may
            # heal its blocker later in this same Source run.
            for r in effective_results:
                if id(r) in deferred_result_ids:
                    continue
                if r["processed"]:
                    docs_processed += 1
                if r["updated"]:
                    docs_updated += 1
                if r["failed"]:
                    docs_failed += 1
                elif r["processed"]:
                    settled_doc_ids.add(str(r["doc_id"]))
                if r.get("failure") is not None:
                    # A recovery that failed deterministically; retrying the run cannot apply it.
                    fail_document(r["failure"])
                    failure_retryable = False
                memories_extracted += r["memories_extracted"]
                skipped_reading_group_count += r["skipped_reading_group_count"]
                if r.get("runtime_bundle") is not None:
                    runtime_bundles.append(r["runtime_bundle"])

            # ----------------------------------------------------------
            # Step 5: Remove Documents the provider no longer has
            # ----------------------------------------------------------
            # The family decides what the run removes. Deferred commits
            # blocked only by this run's Units converge first. A Document
            # that has not committed by then keeps the Unit its renamed item
            # names; it holds back nothing else. In a scope transition,
            # removal waits for the run that completes it.
            await self._converge_deferred_projected_lifecycle(effective_results)
            deleted_count = 0
            removals_failed = 0
            tombstoned_source_unit_ids: set[str] = set()
            committed_doc_ids = {
                str(result["doc_id"])
                for result in effective_results
                if result.get("processed")
                and not result.get("failed")
                and result.get("deferred_lifecycle") is None
                and result.get("terminal_error") is None
            }
            left_ledger_doc_ids = committed_doc_ids | settled_doc_ids
            absence = await family.absence(
                reprocessing=reprocessing,
                transition=scope_transition,
                indexed_doc_ids=indexed_doc_ids,
                deferred_blockers={
                    str(result["doc_id"]): frozenset(result["deferred_lifecycle"].handle.blocking_source_unit_ids)
                    for result in deferred_results
                    if result.get("deferred_lifecycle") is not None
                },
                uncommitted=[
                    *(item for item in items if item.item_id not in committed_doc_ids),
                    *(
                        failure.item
                        for failure in ledger.values()
                        if failure.item is not None and failure.document_id not in left_ledger_doc_ids
                    ),
                ],
            )
            removals = absence.removals
            absence_check_skipped_reason = absence.skipped_reason
            unlisted_present_count = absence.unlisted_present_count

            if removals is not None:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "detecting_deletions",
                            "current": 0,
                            "total": 0,
                            "title": None,
                        }
                    )

                (
                    deleted_count,
                    deletion_failures,
                    tombstoned_source_unit_ids,
                ) = await self._detect_deletions(
                    source_id=source_id,
                    source_type=configured_source_type,
                    source_name=source_name,
                    run_id=run_id,
                    lifecycle_cycle_id=(scope_transition.id if scope_transition is not None else durable_cycle_id),
                    removals=removals,
                    source_filter_summary=gene.filter_summary(last_sync_time),
                    source_activity=source_activity,
                )
                for deletion_failure in deletion_failures:
                    fail_document(deletion_failure)
                removals_failed = len(deletion_failures)
                docs_failed += removals_failed
                removed_doc_ids = set(removals) - {failure.document_id for failure in deletion_failures}
                # A removed Document has nothing left to retry, whatever
                # failed for it earlier in this run.
                for removed_doc_id in removed_doc_ids:
                    item_failures.pop(removed_doc_id, None)
                settled_doc_ids.update(removed_doc_ids)
                if deleted_count > 0:
                    logger.info(
                        "Soft-deleted memories from %d removed documents",
                        deleted_count,
                    )
            else:
                logger.debug(
                    "No Document of %s is proven absent: no listing, or its check was skipped (%s)",
                    source_id,
                    absence_check_skipped_reason or "nothing to check",
                )

            # A removed Unit may have blocked a deferred commit of this run.
            if tombstoned_source_unit_ids:
                await self._converge_deferred_projected_lifecycle(
                    effective_results,
                    additional_source_unit_ids=tombstoned_source_unit_ids,
                )
            run_items = {item.item_id: item for item in items}
            for result in deferred_results:
                if result.get("deferred_lifecycle") is not None:
                    result["terminal_error"] = _retained_document_error(result["deferred_lifecycle"])
                    result["terminal_failure"] = result["deferred_lifecycle"]
                terminal_error = result.get("terminal_error")
                if terminal_error is not None:
                    result["failed"] = True
                    failure_retryable = False
                    docs_failed += 1
                    fail_document(
                        _run_item_failure(
                            result["terminal_failure"],
                            document_id=str(result["doc_id"]),
                            title=str(result["title"]),
                            stage="lifecycle_commit",
                            store=self.db,
                            source_unit_id=result.get("source_unit_id"),
                            item=run_items.get(str(result["doc_id"])),
                        ),
                        error=str(terminal_error),
                    )
                    if result.get("runtime_bundle") is not None:
                        runtime_bundles.append(result["runtime_bundle"])
                    continue
                if result["processed"]:
                    settled_doc_ids.add(str(result["doc_id"]))
                    docs_processed += 1
                if result["updated"]:
                    docs_updated += 1
                memories_extracted += result["memories_extracted"]
                skipped_reading_group_count += result["skipped_reading_group_count"]

            if scope_transition is not None:
                outstanding = await family.transition_outstanding(
                    docs_failed=docs_failed,
                    removals_failed=removals_failed,
                )
                if outstanding is None:
                    scope_transition = await self.db.complete_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                    )
                    transition_completed = True
                else:
                    # The transition stays open; the next run continues it.
                    scope_transition = await self.db.fail_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                        error=outstanding,
                    )
                transition_started = False

            # ----------------------------------------------------------
            # Step 6: Update source doc_count
            # ----------------------------------------------------------
            total_docs = await self.db.count_documents(source=source_id)
            await self.db.update_source_doc_count(source_id, total_docs)

            # ----------------------------------------------------------
            # Record failed Documents and determine final status
            # ----------------------------------------------------------
            # Every Document failure left is in the ledger, which retries it,
            # so the run is partial and its cursor advances.
            await self._record_failed_items(
                source_id,
                release=release,
                failures=item_failures,
                settled_doc_ids=settled_doc_ids,
                source_activity=source_activity,
            )
            # A conversation the Source cannot read keeps its Support and
            # fails nothing; the run is partial until the user restores
            # access or deselects it.
            unreadable_conversations = family.unreadable_conversations(reprocessing=reprocessing)
            status = "success" if docs_failed == 0 and not unreadable_conversations else "partial"

        except Exception as e:
            logger.error(
                "Sync failed for %s (run_id=%s): %s",
                source_name,
                run_id,
                e,
                exc_info=True,
            )
            status = "failed"
            error_message = str(e)
            try:
                await self._record_failed_items(
                    source_id,
                    release=release,
                    failures=item_failures,
                    settled_doc_ids=settled_doc_ids,
                    source_activity=source_activity,
                )
            except Exception:
                logger.exception("Failed to record the failed Documents of %s", source_id)
            failure_retryable = (
                not isinstance(e, SourceConfigurationError)
                and bool(getattr(e, "retryable", True))
            )
            if (runtime_bundle := getattr(e, "runtime_bundle", None)) is not None:
                runtime_bundles.append(runtime_bundle)
            if scope_transition is not None and transition_started:
                try:
                    await self.db.fail_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                        error=error_message,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist Projection Scope transition failure for %s",
                        scope_transition.id,
                    )

        try:
            await self.memory_store.attempt_lifecycle_vector_delivery(source_id=source_id)
        except Exception:
            # Relational lifecycle state and its durable outbox are
            # authoritative. Delivery is run-level and independent of
            # whether later source work completed successfully.
            logger.exception(
                "Source sync completed with lifecycle vector delivery still pending for source %s",
                source_id,
            )

        # ------------------------------------------------------------------
        # Step 7: Record SyncState and sync_history
        # ------------------------------------------------------------------
        finished_at = datetime.now(timezone.utc)

        if docs_failed > 0 and error_message is None:
            error_message = summarize_failed_documents(docs_failed, failed_docs)
        if unreadable_conversations:
            error_message = " ".join(
                part for part in (error_message, summarize_unreadable_conversations(unreadable_conversations)) if part
            )

        # The family moves the incremental cursor (see
        # CollectionFamily.cursor). A reprocess discovers nothing, so the
        # cursor stays where the last sync left it.
        previous_sync_at = existing_state.last_sync_at if existing_state else None
        if reprocessing:
            sync_at = previous_sync_at
        else:
            sync_at = family.cursor(
                status=status,
                started_at=started_at,
                previous=previous_sync_at,
                transition=scope_transition,
                transition_completed=transition_completed,
            )

        sync_state = SyncState(
            source=source_id,
            last_sync_at=sync_at,
            last_sync_status=status,
            docs_processed=docs_processed,
            docs_updated=docs_updated,
            docs_failed=docs_failed,
            memories_extracted=memories_extracted,
            skipped_reading_group_count=skipped_reading_group_count,
            error_message=error_message,
            failed_docs=failed_docs,
            failure_retryable=failure_retryable,
            absence_check_skipped_reason=absence_check_skipped_reason,
            unlisted_present_count=unlisted_present_count,
            runtime_bundles=tuple(runtime_bundles),
        )

        if record_terminal_result:
            await self.db.record_source_sync_result(
                sync_state,
                started_at=started_at,
                finished_at=finished_at,
                run_id=run_id,
            )
            if self.runtime_event_trace_sink is not None:
                assessment_sink = assessment_sink_for_runtime_sink(
                    self.runtime_event_trace_sink
                )
                for bundle in sync_state.runtime_bundles:
                    publish_runtime_events(self.runtime_event_trace_sink, bundle.events)
                    publish_agent_assessments(
                        assessment_sink,
                        bundle.assessments,
                        bundle.events,
                    )

        if progress_callback:
            progress_callback(
                {
                    "phase": "complete",
                    "current": total_item_count,
                    "total": total_item_count,
                    "title": None,
                }
            )

        materialized_item_count = len(items) if "items" in locals() else 0
        self._memory_sample(
            "sync_run_end",
            source_id=source_id,
            run_id=run_id,
            status=status,
            docs_processed=docs_processed,
            docs_updated=docs_updated,
            docs_failed=docs_failed,
            memories_extracted=memories_extracted,
            item_count=total_item_count,
            materialized_item_count=materialized_item_count,
            reused_projection_count=reused_projection_count,
        )

        logger.info(
            "Sync complete for %s (run_id=%s): "
            "%d processed, %d updated, %d failed, "
            "%d memories extracted, %d document parts unread, status=%s",
            source_name,
            run_id,
            docs_processed,
            docs_updated,
            docs_failed,
            memories_extracted,
            skipped_reading_group_count,
            status,
        )

        return sync_state

    async def _record_failed_items(
        self,
        source_id: str,
        *,
        release: str,
        failures: Mapping[str, RunItemFailure],
        settled_doc_ids: set[str],
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Record this run's failed Documents in the failure ledger and drop the settled ones.

        A Document that committed, or that left the Source, leaves the ledger;
        a failed Document's row records this failure and its release. Each
        write commits only while the run holds its Source activity lease.
        """

        ledger = {
            failure.document_id: failure
            for failure in await self.db.list_source_item_failures(source_id=source_id)
        }
        failed_at = datetime.now(timezone.utc)
        for document_id, failure in sorted(failures.items()):
            await self.db.upsert_source_item_failure(
                SourceItemFailure.recorded(
                    ledger.get(document_id),
                    failure,
                    source_id=source_id,
                    release=release,
                    failed_at=failed_at,
                ),
                source_activity=source_activity,
            )
        settled = (settled_doc_ids - set(failures)) & set(ledger)
        if settled:
            await self.db.delete_source_item_failures(source_id, settled, source_activity=source_activity)
            logger.info("%d failed Document(s) of %s committed or left the Source", len(settled), source_id)

    async def _converge_deferred_projected_lifecycle(
        self,
        results: list[dict[str, Any]],
        *,
        additional_source_unit_ids: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        """Commit same-run deferred intents without repeating semantic work.

        Only an intent whose blockers are all Units of this run, or Units this
        run removed, is retried. An intent still deferred afterwards has failed
        for good unless it waits, directly or through another deferred intent,
        on a Unit outside that set: only the caller's later removal of such a
        Unit can still unblock it, so that intent is left pending.
        """

        run_source_unit_ids = {
            str(result["source_unit_id"])
            for result in results
            if result.get("source_unit_id")
        } | set(additional_source_unit_ids)
        pending = {
            str(result["source_unit_id"]): result
            for result in results
            if result.get("deferred_lifecycle") is not None
            and result.get("source_unit_id")
            and set(result["deferred_lifecycle"].handle.blocking_source_unit_ids).issubset(run_source_unit_ids)
        }
        if not pending:
            return

        await self._retry_deferred_projected_lifecycle(pending, run_source_unit_ids)
        _fail_deferred_commits_nothing_can_unblock(results, run_source_unit_ids)

    async def _retry_deferred_projected_lifecycle(
        self, pending: dict[str, dict[str, Any]], run_source_unit_ids: set[str],
    ) -> None:
        """Retry eligible intents in rounds while each round commits at least one."""
        attempt_budget = min(
            MAX_LIFECYCLE_CONVERGENCE_ROUNDS * len(pending),
            MAX_LIFECYCLE_CONVERGENCE_ATTEMPTS,
        )
        attempts = 0
        for round_index in range(MAX_LIFECYCLE_CONVERGENCE_ROUNDS):
            if not pending or attempts >= attempt_budget:
                break
            successful = 0
            for source_unit_id in sorted(tuple(pending)):
                if attempts >= attempt_budget:
                    break
                result = pending[source_unit_id]
                deferred = result["deferred_lifecycle"]
                try:
                    lifecycle_stats = (
                        await self.memory_engine.retry_deferred_projected_lifecycle(
                            deferred.handle,
                            eligible_same_run_source_unit_ids=run_source_unit_ids,
                        )
                    )
                except SourceUnitLifecycleDeferred as exc:
                    attempts += 1
                    result["deferred_lifecycle"] = exc
                    result["runtime_bundle"] = exc.runtime_bundle
                    continue
                except Exception as exc:
                    if bool(getattr(exc, "commit_attempted", True)):
                        attempts += 1
                    result["deferred_lifecycle"] = None
                    result["terminal_error"] = _retained_document_error(exc)
                    result["terminal_failure"] = exc
                    result["runtime_bundle"] = getattr(
                        exc,
                        "runtime_bundle",
                        result.get("runtime_bundle"),
                    )
                    pending.pop(source_unit_id)
                    continue

                attempts += 1
                result["processed"] = True
                result["updated"] = True
                result["memories_extracted"] = int(
                    lifecycle_stats.get("added", 0)
                )
                result["failed"] = False
                result["deferred_lifecycle"] = None
                result["runtime_bundle"] = None
                pending.pop(source_unit_id)
                successful += 1
            if successful == 0:
                break

    async def _resume_source_derivations(
        self,
        *,
        source_id: str,
        run_id: str,
        lifecycle_execution_owner_id: str | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        source_activity: SourceActivityLease | None = None,
        scope_transition_id: str | None = None,
    ) -> _SourceDerivationRecovery:
        """Resume durable Source Unit work before reading the provider.

        ``scope_transition_id`` names the open scope transition whose members
        the recovered commits record.

        A derivation another release staged is superseded, not resumed: the
        failure ledger records its Document, so the run derives it again with
        this release's code. A recovery whose failure is transient stops the
        run and leaves the derivation staged, until the same failure has
        repeated in as many consecutive runs as the ledger's retry budget
        allows; every other failure ends the staged derivation, and recovery
        continues with the next, so one Unit never blocks the Source.
        """

        release = current_release()
        ledger = {
            failure.document_id: failure
            for failure in await self.db.list_source_item_failures(source_id=source_id)
        }
        results: list[dict[str, Any]] = []
        attempts = await self.db.list_source_derivation_attempts(
            source_id=source_id,
            statuses=(
                "pending",
                "retryable_failure",
                "completed",
            ),
        )
        latest_by_projection: dict[
            tuple[str, str, str],
            SourceDerivationAttempt,
        ] = {}
        superseded_attempts: list[tuple[SourceDerivationAttempt, str | None]] = []
        release_superseded: list[SourceDerivationAttempt] = []
        for attempt in attempts:
            if attempt.release != release:
                release_superseded.append(attempt)
                continue
            if attempt.extraction_contract_version != PROJECTION_EXTRACTION_CONTRACT_VERSION:
                if attempt.status in {"pending", "retryable_failure"}:
                    superseded_attempts.append(
                        (attempt, CONTRACT_SUPERSEDED)
                    )
                continue
            projection_scope = (
                attempt.source_unit_id,
                attempt.target_unit_revision_id,
                attempt.projection_identity_hash,
            )
            previous = latest_by_projection.get(projection_scope)
            if previous is not None:
                superseded_attempts.append((previous, None))
            latest_by_projection[projection_scope] = attempt
        for attempt, reason_code in superseded_attempts:
            await self.db.supersede_source_derivation(
                attempt.id,
                reason_code=reason_code,
            )
        for attempt in release_superseded:
            await self._supersede_derivation_of_another_release(
                attempt, ledger=ledger, release=release, source_activity=source_activity,
            )
        if release_superseded:
            logger.info(
                "Superseded %d Source derivation(s) staged by another release for source %s; "
                "their Documents are derived again",
                len(release_superseded),
                source_id,
            )
        if superseded_attempts:
            logger.info(
                "Superseded %d Source derivation(s) under another contract or replaced by a newer one "
                "before recovery for source %s",
                len(superseded_attempts),
                source_id,
            )
        attempts = tuple(
            sorted(
                latest_by_projection.values(),
                key=lambda attempt: (attempt.created_at, attempt.id),
            )
        )
        if attempts and progress_callback:
            progress_callback(
                {
                    "phase": "recovering_derivations",
                    "current": 0,
                    "total": len(attempts),
                    "title": None,
                }
            )
        for index, attempt in enumerate(attempts):
            doc_id = attempt.context.document.doc_id
            recovery_error: Exception | None = None
            diagnostics = _SourceUnitExecutionDiagnostics(source_unit_id=attempt.source_unit_id)
            async with self._document_lifecycle_slot(
                source_id,
                doc_id,
            ) as document_queue_wait_ms:
                self._memory_sample(
                    "derivation_recovery_lifecycle_enter",
                    source_id=source_id,
                    run_id=run_id,
                    doc_id=doc_id,
                    document_queue_wait_ms=document_queue_wait_ms,
                )
                try:
                    async with self._source_unit_diagnostics(
                        diagnostics,
                        source_id=source_id,
                        run_id=run_id,
                        doc_id=doc_id,
                    ):
                        lifecycle_stats = await self._resume_source_derivation_attempt(
                            attempt=attempt,
                            source_id=source_id,
                            source_activity=source_activity,
                            lifecycle_execution_owner_id=lifecycle_execution_owner_id,
                            diagnostics=diagnostics,
                            scope_transition_id=scope_transition_id,
                        )
                        if lifecycle_stats is not None:
                            diagnostics.status = "committed"
                        elif diagnostics.error_class is None:
                            diagnostics.status = "skipped"
                except SourceUnitLifecycleDeferred as exc:
                    results.append(
                        {
                            "processed": False,
                            "updated": False,
                            "memories_extracted": 0,
                            "skipped_reading_group_count": (
                                diagnostics.skipped_reading_group_count
                            ),
                            "failed": False,
                            "runtime_bundle": exc.runtime_bundle,
                            "source_unit_id": exc.handle.source_unit_id,
                            "target_unit_revision_id": (
                                attempt.target_unit_revision_id
                            ),
                            "derivation_id": attempt.id,
                            "deferred_lifecycle": exc,
                            "doc_id": doc_id,
                            "title": attempt.context.document.title,
                        }
                    )
                    lifecycle_stats = None
                except Exception as exc:
                    recovery_error = exc
                    failure = _run_item_failure(
                        exc,
                        document_id=doc_id,
                        title=attempt.context.document.title,
                        stage=diagnostics.stage,
                        store=self.db,
                        source_unit_id=attempt.source_unit_id,
                        item=_derivation_item(attempt.context),
                    )
                    recorded = SourceItemFailure.recorded(
                        ledger.get(doc_id),
                        failure,
                        source_id=source_id,
                        release=release,
                        failed_at=datetime.now(timezone.utc),
                    )
                    # A failure whose cause proves it transient (a lost activity
                    # fence, a transient storage or provider error, a model
                    # timeout) stops the run and leaves the derivation staged,
                    # until it has failed the same way in as many consecutive
                    # runs as the ledger's budget allows.
                    if failure.retryable and not recorded.budget_exhausted:
                        try:
                            await self.db.upsert_source_item_failure(recorded, source_activity=source_activity)
                        except Exception:
                            logger.warning(
                                "Failed to record the recovery failure of Source derivation %s",
                                attempt.id,
                                exc_info=True,
                            )
                        raise
                    await self.db.supersede_source_derivation(
                        attempt.id,
                        reason_code=DERIVATION_DETERMINISTIC_FAILURE,
                    )
                    logger.warning(
                        "Recovery of Source derivation %s for %s failed (%d consecutive failure(s) in this release): %s",
                        attempt.id,
                        doc_id,
                        recorded.failures_in_release,
                        exc,
                    )
                    results.append(
                        {
                            "processed": False,
                            "updated": False,
                            "memories_extracted": 0,
                            "skipped_reading_group_count": 0,
                            "failed": True,
                            "error": failure.message,
                            "failure": failure,
                            "runtime_bundle": getattr(exc, "runtime_bundle", None),
                            "source_unit_id": attempt.source_unit_id,
                            "target_unit_revision_id": attempt.target_unit_revision_id,
                            "derivation_id": attempt.id,
                            "deferred_lifecycle": None,
                            "doc_id": doc_id,
                            "title": attempt.context.document.title,
                        }
                    )
                    lifecycle_stats = None
                finally:
                    self._memory_sample(
                        "derivation_recovery_lifecycle_exit",
                        source_id=source_id,
                        run_id=run_id,
                        doc_id=doc_id,
                        ok=diagnostics.status != "failed",
                        error=recovery_error,
                    )
            if progress_callback:
                progress_callback(
                    {
                        "phase": "recovering_derivations",
                        "current": index + 1,
                        "total": len(attempts),
                        "title": None,
                    }
                )
            if lifecycle_stats is None:
                continue
            results.append(
                {
                    "processed": True,
                    "updated": True,
                    "memories_extracted": int(
                        lifecycle_stats.get("added", 0)
                    ),
                    "skipped_reading_group_count": (
                        diagnostics.skipped_reading_group_count
                    ),
                    "failed": False,
                    "runtime_bundle": None,
                    "source_unit_id": attempt.source_unit_id,
                    "target_unit_revision_id": (
                        attempt.target_unit_revision_id
                    ),
                    "derivation_id": attempt.id,
                    "deferred_lifecycle": None,
                    "doc_id": doc_id,
                    "title": attempt.context.document.title,
                }
            )
        return _SourceDerivationRecovery(results=tuple(results))

    async def _supersede_derivation_of_another_release(
        self,
        attempt: SourceDerivationAttempt,
        *,
        ledger: dict[str, SourceItemFailure],
        release: str,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Supersede a derivation another release staged and record its Document in the failure ledger.

        The ledger row carries the staging release, so the next NORMAL run, or
        the release recovery queued at startup, derives the Document again with
        this release's code. A Document that already has a row keeps it.
        """

        await self.db.supersede_source_derivation(attempt.id, reason_code=DERIVATION_RELEASE_SUPERSEDED)
        doc_id = attempt.context.document.doc_id
        if doc_id in ledger:
            return
        now = datetime.now(timezone.utc)
        failure = SourceItemFailure(
            source_id=attempt.source_id,
            document_id=doc_id,
            stage="derivation",
            error_class=DERIVATION_RELEASE_SUPERSEDED,
            error_code=DERIVATION_RELEASE_SUPERSEDED,
            message="Another release staged this Document's derivation; this release derives it again.",
            retryable=True,
            release=attempt.release or UNRECORDED_RELEASE,
            failures_in_release=1,
            first_failed_at=now,
            last_failed_at=now,
            title=attempt.context.document.title,
            source_unit_id=attempt.source_unit_id,
            item=_derivation_item(attempt.context),
        )
        await self.db.upsert_source_item_failure(failure, source_activity=source_activity)
        ledger[doc_id] = failure

    async def _resume_source_derivation_attempt(
        self,
        *,
        attempt: SourceDerivationAttempt,
        source_id: str,
        lifecycle_execution_owner_id: str | None,
        diagnostics: _SourceUnitExecutionDiagnostics,
        source_activity: SourceActivityLease | None = None,
        scope_transition_id: str | None = None,
    ) -> dict | None:
        """Resume one derivation inside the process document admission."""

        context = attempt.context
        current_revision = await self.db.get_current_source_unit_revision(attempt.source_unit_id)
        current_revision_id = current_revision.id if current_revision is not None else None
        if current_revision_id not in {
            attempt.base_unit_revision_id,
            attempt.target_unit_revision_id,
        }:
            await self.db.supersede_source_derivation(attempt.id)
            return None
        projection = attempt.projection
        delta = projection.deltas[0]
        plan_id = lifecycle_plan_id(
            ReconciliationScope(
                id=f"scope:{projection.run_id}",
                source_id=projection.source_id,
                source_unit_id=delta.source_unit_id,
                base_unit_revision_id=delta.previous_unit_revision_id,
                target_unit_revision_id=delta.current_unit_revision_id,
            )
        )
        if await self.db.get_lifecycle_plan_payload(plan_id) is not None:
            await self.db.supersede_source_derivation(attempt.id)
            logger.info(
                "Superseded Source derivation %s because lifecycle plan %s already applied for Source Unit %s",
                attempt.id,
                plan_id,
                attempt.source_unit_id,
            )
            return None
        extraction = await self._extract_source_derivation_work(
            projection=projection,
            source_type=projection.source_type,
            doc_type=context.doc_type,
            source_id=source_id,
            derivation_context=context,
        )
        if extraction.error_type:
            diagnostics.error_class = extraction.error_type
            _error_type, diagnostics.error_code, _fields = safe_derivation_error(extraction)
            logger.warning(
                "Recovery of Source derivation %s for %s failed: %s: %s",
                attempt.id,
                context.document.doc_id,
                diagnostics.error_class,
                extraction.error,
            )
            return None
        diagnostics.skipped_reading_group_count = _skipped_reading_group_count(extraction)
        recovery_commit = plan_source_derivation_recovery_commit(
            stored_derivation_id=attempt.id,
            produced_derivation_id=extraction.derivation_id,
        )
        if recovery_commit.superseded_derivation_id is not None:
            await self.db.supersede_source_derivation(
                recovery_commit.superseded_derivation_id,
                reason_code=DERIVATION_INPUT_SUPERSEDED,
            )
        source_updated_at = (
            datetime.fromisoformat(context.source_updated_at) if context.source_updated_at is not None else None
        )
        diagnostics.stage = "lifecycle"
        return await self.memory_engine.prepare_and_commit_projected_lifecycle(
            projection=projection,
            doc_id=context.document.doc_id,
            raw_memories=extraction.memories,
            doc_type=context.doc_type,
            project_key=context.project_key,
            repo_identifier=context.repo_identifier,
            document_content=context.document_content,
            update_mode=context.update_mode,
            changed_hunks=context.changed_hunks,
            update_plan_stats=(dict(context.update_plan_stats) if context.update_plan_stats is not None else None),
            current_changed_ranges=context.current_changed_ranges,
            source_updated_at=source_updated_at,
            user_id=context.user_id,
            protected_source_observation_ids=(extraction.protected_source_observation_ids),
            document=context.document,
            unit_input=context.unit_input,
            derivation_id=recovery_commit.applied_derivation_id,
            derivation_reprocess_all_current_observations=(
                context.reprocess_all_current_observations
            ),
            derivation_reprocess_operation_id=context.reprocess_operation_id,
            derivation_support_without_baseline=context.support_without_baseline,
            source_activity=source_activity,
            lifecycle_execution_owner_id=lifecycle_execution_owner_id,
            lifecycle_attempt_count=1,
            scope_transition_id=scope_transition_id,
        )

    # ==================================================================
    # Private: _process_item
    # ==================================================================

    @asynccontextmanager
    async def _source_unit_diagnostics(
        self,
        diagnostics: _SourceUnitExecutionDiagnostics,
        *,
        source_id: str,
        run_id: str | None,
        doc_id: str,
    ):
        collector = StructuredLlmMetricsCollector()
        started = asyncio.get_running_loop().time()
        try:
            with structured_llm_metrics_scope(collector):
                yield diagnostics
        except SourceUnitLifecycleDeferred:
            diagnostics.status = "prepared"
            raise
        except BaseException as exc:
            diagnostics.status = "failed"
            diagnostics.error_class = type(exc).__name__
            error_code = getattr(exc, "error_code", None)
            diagnostics.error_code = error_code if isinstance(error_code, str) else None
            raise
        finally:
            if diagnostics.source_unit_id is not None:
                try:
                    await self._record_source_unit_llm_summary(
                        collector=collector,
                        source_unit_id=diagnostics.source_unit_id,
                        source_id=source_id,
                        run_id=run_id,
                        doc_id=doc_id,
                        source_unit_elapsed_ms=max(0, round((asyncio.get_running_loop().time() - started) * 1000)),
                        ok=diagnostics.status != "failed",
                        error_class=diagnostics.error_class,
                        error_code=diagnostics.error_code,
                        status=diagnostics.status,
                    )
                except Exception:
                    logger.warning("Failed to record Source Unit LLM summary", exc_info=True)

    async def _process_item(
        self,
        gene: Gene,
        item: ContentItem,
        source_name: str,
        source_id: str,
        run_id: str | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        force_reprocess: bool = False,
        projection_scope: dict[str, object] | None = None,
        scope_transition: dict[str, object] | None = None,
        projection_access_context: dict[str, object] | None = None,
        complete_snapshot: bool = False,
        scope_transition_id: str | None = None,
        source_activity: SourceActivityLease | None = None,
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
        source_unit_target_callback: Callable[[str, str], None] | None = None,
        recovered_deferred_targets: frozenset[tuple[str, str]] = frozenset(),
        reprocessing: bool = False,
        stored_document: StoredSourceDocument | None = None,
        diagnostics: _SourceUnitExecutionDiagnostics | None = None,
    ) -> dict:
        """Process one Document; ``diagnostics`` records its Source Unit and the step that ran last.

        ``scope_transition_id`` names the open scope transition whose member
        the Document becomes when it commits.
        """

        doc_id = item.item_id
        self._memory_sample("document_wait_start", source_id=source_id, run_id=run_id, doc_id=doc_id)
        lifecycle_error: Exception | None = None
        diagnostics = diagnostics or _SourceUnitExecutionDiagnostics(stage="fetch")

        async with self._document_lifecycle_slot(source_id, doc_id) as document_queue_wait_ms:
            self._memory_sample(
                "document_lifecycle_enter",
                source_id=source_id,
                run_id=run_id,
                doc_id=doc_id,
                document_queue_wait_ms=document_queue_wait_ms,
            )
            try:
                async with self._source_unit_diagnostics(
                    diagnostics,
                    source_id=source_id,
                    run_id=run_id,
                    doc_id=doc_id,
                ):
                    result = await self._process_item_admitted(
                        gene=gene,
                        item=item,
                        source_name=source_name,
                        source_id=source_id,
                        run_id=run_id,
                        progress_callback=progress_callback,
                        force_reprocess=force_reprocess,
                        projection_scope=projection_scope,
                        scope_transition=scope_transition,
                        projection_access_context=projection_access_context,
                        complete_snapshot=complete_snapshot,
                        scope_transition_id=scope_transition_id,
                        source_activity=source_activity,
                        lifecycle_execution_owner_id=lifecycle_execution_owner_id,
                        lifecycle_attempt_count=lifecycle_attempt_count,
                        source_unit_target_callback=source_unit_target_callback,
                        recovered_deferred_targets=recovered_deferred_targets,
                        source_unit_id_callback=diagnostics.bind_source_unit,
                        reprocessing=reprocessing,
                        stored_document=stored_document,
                        diagnostics=diagnostics,
                    )
                    if result.get("recovered_deferred_target") is not None:
                        diagnostics.status = "prepared"
                    else:
                        diagnostics.status = "committed"
                return result
            except Exception as exc:
                lifecycle_error = exc
                raise
            finally:
                lifecycle_ok = diagnostics.status != "failed"
                self._memory_sample(
                    "document_lifecycle_exit",
                    source_id=source_id,
                    run_id=run_id,
                    doc_id=doc_id,
                    ok=lifecycle_ok,
                    error=lifecycle_error,
                )
                reclaim_result = self.memory_reclaimer.reclaim()
                self._memory_sample(
                    "document_memory_reclaimed",
                    source_id=source_id,
                    run_id=run_id,
                    doc_id=doc_id,
                    ok=lifecycle_ok,
                    error=lifecycle_error,
                    **reclaim_result,
                )

    async def _project_fetched_item(
        self,
        *,
        gene: Gene,
        item: ContentItem,
        raw: RawContent,
        normalized: NormalizedContent,
        source_id: str,
        source_type: str,
        projection_scope: dict[str, object] | None,
        scope_transition: dict[str, object] | None,
        projection_access_context: dict[str, object] | None,
        complete_snapshot: bool,
    ) -> tuple[SourceProjection, tuple[StoredSourceArtifact, ...]]:
        """Project fetched content onto its Source Unit and store the Artifacts it returned.

        The Unit is the Document's current Unit, a historical Unit it returns
        to during a scope transition, or the predecessor of an attested rename.
        """

        stored_source_artifacts: tuple[StoredSourceArtifact, ...] = ()
        async with self._db_lock:
            persisted_source_unit = await self.db.find_source_unit_by_document_id(
                source_id,
                item.item_id,
                current_only=True,
            )
            historical_source_unit = (
                None
                if persisted_source_unit is not None
                else await self.db.find_source_unit_by_document_id(
                    source_id,
                    item.item_id,
                )
            )
        probe_scope: dict[str, object] = {
            "configured_scope": dict(projection_scope or {}),
            "document_id": item.item_id,
            # Part of every stored projection's identity: whether the item came
            # from a complete input snapshot.
            "authoritative_snapshot": complete_snapshot,
        }
        if persisted_source_unit is not None:
            # Document lineage is the durable identity bridge after an
            # attested rename: later items of the Document, which no longer
            # name their predecessor, reuse the Unit's id and provider key.
            probe_scope.update(
                {
                    "source_unit_id": persisted_source_unit.id,
                    "source_unit_provider_key": persisted_source_unit.provider_key,
                }
            )
        elif historical_source_unit is not None:
            scope_identity_probe = (
                await self.source_projection_adapter.project(
                    ProjectionEnvelope(
                        request=ProjectionRequest(
                            run_id="projection-scope-identity-probe",
                            source_id=source_id,
                            source_type=source_type,
                            scope=probe_scope,
                            run_mode=ProjectionRunMode.FULL_SNAPSHOT,
                            scope_transition=scope_transition,
                            access_context=dict(projection_access_context or {}),
                        ),
                        item=item,
                        raw=raw,
                        normalized=normalized,
                    )
                )
                if scope_transition is not None
                else None
            )
            if (
                scope_identity_probe is not None
                and scope_identity_probe.source_units[0].provider_key == historical_source_unit.provider_key
            ):
                # Re-entry during an explicit selector transition keeps the
                # provider's stable identity. This includes ref A -> B -> A;
                # absence/reappearance in an unchanged scope remains a new
                # incarnation unless the provider attests a rename.
                persisted_source_unit = historical_source_unit
                probe_scope.update(
                    {
                        "source_unit_id": historical_source_unit.id,
                        "source_unit_provider_key": historical_source_unit.provider_key,
                    }
                )
            else:
                # Reusing a historical locator after its former Unit moved is
                # a new incarnation, not a move back. Seed a distinct identity
                # once; subsequent snapshots bind through the current row.
                probe_scope["source_unit_incarnation"] = f"{historical_source_unit.id}:{item.version}"
        projection_probe = await self.source_projection_adapter.project(
            ProjectionEnvelope(
                request=ProjectionRequest(
                    run_id="projection-probe",
                    source_id=source_id,
                    source_type=source_type,
                    scope=probe_scope,
                    run_mode=ProjectionRunMode.FULL_SNAPSHOT,
                    scope_transition=scope_transition,
                    access_context=dict(projection_access_context or {}),
                ),
                item=item,
                raw=raw,
                normalized=normalized,
            )
        )
        if persisted_source_unit is None and item.renamed_from is not None:
            async with self._db_lock:
                predecessor_unit = await self.db.find_source_unit_by_document_id(
                    source_id,
                    item.renamed_from,
                    current_only=True,
                )
            if predecessor_unit is not None:
                # An attested rename continues the predecessor's Unit.
                persisted_source_unit = predecessor_unit
                probe_scope.update(
                    {
                        "source_unit_id": predecessor_unit.id,
                        "source_unit_provider_key": predecessor_unit.provider_key,
                    }
                )
                projection_probe = await self.source_projection_adapter.project(
                    ProjectionEnvelope(
                        request=ProjectionRequest(
                            run_id="projection-probe",
                            source_id=source_id,
                            source_type=source_type,
                            scope=probe_scope,
                            run_mode=ProjectionRunMode.FULL_SNAPSHOT,
                            scope_transition=scope_transition,
                            access_context=dict(projection_access_context or {}),
                        ),
                        item=item,
                        raw=raw,
                        normalized=normalized,
                    )
                )
        source_unit = projection_probe.source_units[0]

        # Artifact identity belongs to the immutable Source Unit, not to its
        # current document locator. Jira issue keys and repository paths can
        # change while the provider-native unit remains the same. Materialize
        # only after the identity probe, then rebuild the authoritative probe
        # with the revision-pinned Artifact observations.
        if raw.artifacts:
            stored_source_artifacts = await materialize_source_artifacts(
                source_id=source_id,
                source_unit_key=source_unit.id,
                artifacts=raw.artifacts,
                store=self.doc_store,
                open_artifact=gene.open_source_artifact,
            )
        if stored_source_artifacts:
            probe_scope.update(
                {
                    "source_unit_id": source_unit.id,
                    "source_unit_provider_key": source_unit.provider_key,
                }
            )
            projection_probe = await self.source_projection_adapter.project(
                ProjectionEnvelope(
                    request=ProjectionRequest(
                        run_id="projection-probe",
                        source_id=source_id,
                        source_type=source_type,
                        scope=probe_scope,
                        run_mode=ProjectionRunMode.FULL_SNAPSHOT,
                        scope_transition=scope_transition,
                        access_context=dict(projection_access_context or {}),
                    ),
                    item=item,
                    raw=raw,
                    normalized=normalized,
                    artifacts=stored_source_artifacts,
                )
            )
            source_unit = projection_probe.source_units[0]

        return projection_probe, stored_source_artifacts

    async def _process_item_admitted(
        self,
        gene: Gene,
        item: ContentItem,
        source_name: str,
        source_id: str,
        run_id: str | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        force_reprocess: bool = False,
        projection_scope: dict[str, object] | None = None,
        scope_transition: dict[str, object] | None = None,
        projection_access_context: dict[str, object] | None = None,
        complete_snapshot: bool = False,
        scope_transition_id: str | None = None,
        source_activity: SourceActivityLease | None = None,
        source_unit_id_callback: Callable[[str], None] | None = None,
        lifecycle_execution_owner_id: str | None = None,
        lifecycle_attempt_count: int = 1,
        source_unit_target_callback: Callable[[str, str], None] | None = None,
        recovered_deferred_targets: frozenset[tuple[str, str]] = frozenset(),
        reprocessing: bool = False,
        stored_document: StoredSourceDocument | None = None,
        diagnostics: _SourceUnitExecutionDiagnostics | None = None,
    ) -> dict:
        """Process a single content item through the full pipeline.

        ``reprocessing`` processes the Unit as an operator reprocess (see
        :class:`SourceSyncMode`). A ``stored_document`` replaces the provider:
        its stored raw content and committed Artifacts are the input, and
        nothing is fetched or exported. ``diagnostics`` records each step as
        it starts, so a failure names the step it ended.

        Steps:
            1. Fetch raw content
            2. Normalize to markdown
            3. Compare the content hash and inspect stored artifacts
            4. Store new or missing artifacts
            5. Count tokens
            6. Extract current Memory candidates once per bounded Source Unit batch
            7. Reconcile lifecycle, batch-resolve candidate entity mentions, and persist
            8. Record the changelog

        Returns
        -------
        dict
            Stats: ``{updated, memories_extracted}``.
        """
        doc_id = item.item_id
        diagnostics = diagnostics or _SourceUnitExecutionDiagnostics()
        stats = {
            "updated": False,
            "memories_extracted": 0,
            "memory_supports_added": 0,
            "memory_supports_updated": 0,
            "memory_supports_removed": 0,
            "source_unit_id": None,
            "recovered_deferred_target": None,
        }

        # ------------------------------------------------------------------
        # 1. Fetch raw content
        # ------------------------------------------------------------------
        diagnostics.stage = "fetch"
        if stored_document is None:
            previous_input = await self._stored_unit_input(source_id, doc_id)
            item.stored_extra = dict(previous_input.item.extra) if previous_input is not None else {}
        raw = stored_document.raw if stored_document is not None else await gene.fetch(item)
        logger.debug("Fetched %s (%d bytes)", doc_id, len(raw.body))
        self._memory_sample(
            "after_fetch",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_bytes=len(raw.body),
        )

        # ------------------------------------------------------------------
        # 2. Normalize to markdown, and
        # 3. project provider-native content into stable source lineage.
        #
        # A first in-memory projection gives us the stable Source Unit ID. We
        # then load that unit's current revisions and build the authoritative
        # delta before persisting the projection. No lifecycle decision is
        # allowed to infer deletion from the enclosing sync's item list. A
        # stored input is checked against its committed revision before
        # anything is written, the same check the reprocess preview runs.
        # ------------------------------------------------------------------
        source_metadata = gene.metadata()
        source_type = source_metadata.name
        source_shape = source_metadata.data_shape
        exports_pdf = gene_exports_pdf(gene)
        product_name = source_metadata.display_name
        diagnostics.stage = "normalize"
        if stored_document is not None:
            stored_projection = await project_stored_input(
                stored_document,
                gene=gene,
                projection_adapter=self.source_projection_adapter,
                projection_scope=dict(projection_scope or {}),
                access_context=dict(projection_access_context or {}),
            )
            normalized = stored_projection.normalized
            projection_probe = stored_projection.projection
            stored_source_artifacts = stored_document.artifacts
        else:
            normalized = await gene.normalize(raw)
            require_attested_content(raw, normalized)
            diagnostics.stage = "projection"
            projection_probe, stored_source_artifacts = await self._project_fetched_item(
                gene=gene,
                item=item,
                raw=raw,
                normalized=normalized,
                source_id=source_id,
                source_type=source_type,
                projection_scope=projection_scope,
                scope_transition=scope_transition,
                projection_access_context=projection_access_context,
                complete_snapshot=complete_snapshot,
            )
        source_unit = projection_probe.source_units[0]
        markdown_body = normalized.markdown_body
        empty_content = not markdown_body.strip()
        self._memory_sample(
            "after_normalize",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            content_chars=len(markdown_body),
        )

        if source_unit_id_callback is not None:
            source_unit_id_callback(source_unit.id)
        stats["source_unit_id"] = source_unit.id

        current_target = (
            source_unit.id,
            projection_probe.source_unit_revisions[0].id,
        )
        if source_unit_target_callback is not None:
            source_unit_target_callback(*current_target)
        if current_target in recovered_deferred_targets:
            stats["recovered_deferred_target"] = current_target
            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            logger.info(
                "Reused prepared lifecycle intent for unchanged Source Unit target %s (%s)",
                source_unit.id,
                current_target[1],
            )
            return stats
        if recovered_deferred_targets:
            logger.info(
                "Provider target %s supersedes prepared lifecycle target(s) %s",
                current_target,
                sorted(recovered_deferred_targets),
            )

        projection_run_id = f"{run_id or 'direct'}:{source_unit.id}:{projection_probe.source_unit_revisions[0].id}"
        diagnostics.stage = "projection"
        async with self._db_lock:
            projection = await self.db.get_source_projection(projection_run_id)
            if projection is None:
                prior_unit_revision = await self.db.get_current_source_unit_revision(source_unit.id)
                prior_observation_revisions = await self.db.get_current_source_observation_revisions(source_unit.id)
                stored_observation_revisions = await self.db.get_source_observation_revisions(
                    tuple(revision.id for revision in projection_probe.observation_revisions)
                )
                projection = await self.source_projection_adapter.project(
                    ProjectionEnvelope(
                        request=ProjectionRequest(
                            run_id=projection_run_id,
                            source_id=source_id,
                            source_type=source_type,
                            scope={
                                "configured_scope": dict(projection_scope or {}),
                                "source_unit_id": source_unit.id,
                                "source_unit_provider_key": source_unit.provider_key,
                                # Part of the projection identity; see the probe scope.
                                "authoritative_snapshot": complete_snapshot,
                            },
                            run_mode=(
                                ProjectionRunMode.FULL_SNAPSHOT
                                if prior_unit_revision is None
                                else ProjectionRunMode.DELTA
                            ),
                            scope_transition=scope_transition,
                            access_context=dict(projection_access_context or {}),
                        ),
                        item=item,
                        raw=raw,
                        normalized=normalized,
                        artifacts=stored_source_artifacts,
                        prior_unit_revision=prior_unit_revision,
                        prior_observation_revisions=prior_observation_revisions,
                        stored_observation_revisions=stored_observation_revisions,
                    )
                )

            projection_requires_extraction = projection.deltas[0].requires_extraction
            # Location/access-only and idempotent observations carry no Memory
            # mutation, so their lineage advances without semantic work.
            # Semantic work plans against the committed base. Either way the
            # projection is recorded only after its raw content is stored.
            skip_semantic_work = not projection_requires_extraction and not force_reprocess

            lineage_document_ids = await self.db.list_source_unit_document_ids(source_unit.id)

        lineage_predecessor_docs: list[DocumentRecord] = []
        for lineage_doc_id in lineage_document_ids:
            if lineage_doc_id == doc_id:
                continue
            predecessor = await self.db.get_document(lineage_doc_id)
            if predecessor is not None:
                lineage_predecessor_docs.append(predecessor)

        # The Unit's stored input is what this Source stored when it recorded
        # the committed revision, wherever the Unit's Document was then. Its
        # normalized-content hash decides whether the content changed; the
        # projection delta independently controls semantic work, so a
        # provider revision or location-only move does not trigger extraction.
        new_hash = compute_content_hash(markdown_body)
        async with self._db_lock:
            stored_input = await self.db.get_source_unit_input(source_unit.id)
        existing_hash = stored_input.normalized_content_hash if stored_input is not None else None
        content_unchanged = existing_hash == new_hash
        previous_markdown = (
            self._read_previous_normalized_content(stored_input)
            if existing_hash is not None and existing_hash != new_hash
            else None
        )

        requires_pdf_uri = not raw.authoritative_empty and gene.requires_pdf_artifact(
            item=item,
            stored_input=stored_input,
            existing_hash=existing_hash,
            new_hash=(existing_hash if not projection_requires_extraction else new_hash),
        )

        # ------------------------------------------------------------------
        # 3. Store raw + normalized on disk
        # ------------------------------------------------------------------
        # The stored raw content is the input of the committed Unit revision:
        # a projection that moves the Unit to a new revision stores the raw
        # content it was projected from, even when the normalized markdown is
        # unchanged. Every path stores it under this Source's object keys
        # before the revision is recorded with it, so a failed save fails the
        # Document before anything commits. A reprocess from stored input
        # reads its raw content from storage and keeps the provider exports it
        # cannot repeat; only a changed normalization is stored again.
        # A stored object is reused only when it is this Document's own
        # object in this Source; any other URI a stored input carries is
        # written again under this Document's keys.
        def own_object(uri: str | None) -> str | None:
            return uri if self.doc_store.belongs_to_document(uri, source_id=source_id, doc_id=doc_id) else None

        reuse_content_artifacts = content_unchanged and (stored_document is not None or not force_reprocess)
        unit_revision_unchanged = (
            projection.deltas[0].previous_unit_revision_id == projection.source_unit_revisions[0].id
        )
        reused_raw: SourceUnitInput | None = None
        raw_sha256: str | None = None
        if stored_document is not None:
            # A reprocess read these bytes from the stored object.
            reused_raw = stored_document.unit_input
            raw_sha256 = hashlib.sha256(raw.body).hexdigest()
        elif reuse_content_artifacts and unit_revision_unchanged and stored_input is not None:
            reused_raw = stored_input
            raw_sha256 = stored_input.raw_content_sha256
        raw_uri = own_object(reused_raw.raw_content_uri) if reused_raw is not None else None
        raw_content_type = reused_raw.raw_content_type if raw_uri and reused_raw is not None else raw.content_type
        norm_uri = (
            own_object(stored_input.normalized_content_uri)
            if reuse_content_artifacts and stored_input is not None
            else None
        )
        stored_content_artifact = False
        if not raw_uri:
            raw_sha256 = hashlib.sha256(raw.body).hexdigest()
            raw_uri = self.doc_store.store_raw(
                source_id=source_id,
                doc_id=doc_id,
                title=item.title,
                content=raw.body,
                content_type=raw.content_type,
            )
            stored_content_artifact = True
        if not norm_uri:
            norm_uri = self.doc_store.store_normalized(
                source_id=source_id,
                doc_id=doc_id,
                title=item.title,
                markdown=markdown_body,
            )
            stored_content_artifact = True
        if stored_content_artifact:
            self._memory_sample(
                "after_raw_store",
                source_id=source_id,
                run_id=run_id,
                doc_id=doc_id,
                raw_bytes=len(raw.body),
                content_chars=len(markdown_body),
            )

        # ------------------------------------------------------------------
        # 3b. Export PDF (if gene supports it)
        # ------------------------------------------------------------------
        pdf_uri = (
            own_object(stored_input.pdf_content_uri)
            if (reuse_content_artifacts or stored_document is not None) and stored_input is not None
            else None
        )
        should_fetch_pdf = (
            stored_document is None
            and not raw.authoritative_empty
            and exports_pdf
            and (force_reprocess or requires_pdf_uri or not pdf_uri)
        )
        if should_fetch_pdf:
            try:
                pdf_bytes = await gene.fetch_pdf(item)
            except Exception as e:
                if requires_pdf_uri:
                    raise RuntimeError(f"{product_name} PDF export failed for {item.title}: {e}") from e
                logger.warning("PDF export failed for %s: %s", item.title, e)
                pdf_bytes = None
            if pdf_bytes and len(pdf_bytes) > 100:
                pdf_uri = self.doc_store.store_raw(
                    source_id=source_id,
                    doc_id=doc_id,
                    title=item.title,
                    content=pdf_bytes,
                    content_type="application/pdf",
                    extension=".pdf",
                )
                logger.info("Stored PDF for %s (%d bytes)", item.title, len(pdf_bytes))
            elif requires_pdf_uri:
                raise RuntimeError(f"{product_name} PDF export did not produce a PDF for {item.title}")
            self._memory_sample(
                "after_pdf_export",
                source_id=source_id,
                run_id=run_id,
                doc_id=doc_id,
                pdf_bytes=len(pdf_bytes) if pdf_bytes else 0,
            )

        token_count = _count_tokens(markdown_body)
        now = datetime.now(timezone.utc)
        source_row = await self.db.get_source(source_id)
        binding = source_row.get("project_binding") if source_row else None
        project_key = resolve_project_key(
            binding,
            item_field_value=item.space_or_project,
            repo=None,
            workspace=None,
        )
        doc_record = DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=item.source_url,
            title=item.title,
            space_or_project=item.space_or_project,
            author=item.author,
            last_modified=item.last_modified,
            labels=item.labels,
            version=item.version,
            content_hash=new_hash,
            token_count=token_count,
            last_synced=now,
            client=normalized.source_semantics.get("client") or None,
            item_extra=dict(item.extra),
        )
        unit_input = SourceUnitInput(
            source_unit_id=source_unit.id,
            unit_revision_id=projection.source_unit_revisions[0].id,
            source_id=source_id,
            document_id=doc_id,
            item=item,
            raw_content_uri=raw_uri,
            raw_content_type=raw_content_type,
            raw_content_sha256=raw_sha256,
            normalized_content_uri=norm_uri,
            normalized_content_hash=new_hash,
            pdf_content_uri=pdf_uri,
        )

        if skip_semantic_work:
            stats["updated"] = not content_unchanged
            async with self._db_lock:
                await self.db.record_source_projection(
                    projection,
                    unit_input=unit_input,
                    source_activity=source_activity,
                    scope_transition_id=scope_transition_id,
                )
                await self.db.upsert_document(
                    doc_record,
                    require_configured_source=True,
                    source_activity=source_activity,
                )
            await self._finalize_projected_document_moves(
                predecessor_docs=lineage_predecessor_docs,
                current_doc_id=doc_id,
                source_id=source_id,
                source_unit_id=source_unit.id,
                source_activity=source_activity,
            )

            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            logger.debug("Skipping semantic work for projected no-op %s (%s)", item.title, doc_id)
            return stats

        stats["updated"] = True

        # Determine change type for changelog
        change_type = "updated" if existing_hash is not None else "created"
        previous_version = stored_input.item.version if stored_input is not None else None
        update_plan: DocumentUpdatePlan | None = None
        if change_type == "updated":
            update_plan = plan_document_update(
                previous_content=previous_markdown,
                updated_content=markdown_body,
                data_shape=source_shape,
            )
            await self._record_document_update_strategy(
                plan=update_plan,
                doc_id=doc_id,
                source_id=source_id,
                run_id=run_id,
                previous_version=previous_version,
                current_version=item.version,
                previous_hash=existing_hash,
                current_hash=new_hash,
            )

        if empty_content and not stored_source_artifacts:
            diagnostics.stage = "lifecycle"
            memory_stats = await self.memory_engine.prepare_and_commit_projected_lifecycle(
                projection=projection,
                doc_id=doc_id,
                raw_memories=[],
                doc_type=f"{source_type}_empty",
                project_key=project_key,
                repo_identifier=normalized.source_semantics.get("repo_identifier"),
                document_content="",
                update_mode=(update_plan.mode if update_plan else "full_document"),
                changed_hunks=(update_plan.changed_hunks if update_plan else None),
                update_plan_stats=self._document_update_plan_stats(update_plan),
                source_updated_at=_source_updated_at(normalized.source_semantics),
                document=doc_record,
                unit_input=unit_input,
                source_activity=source_activity,
                lifecycle_execution_owner_id=lifecycle_execution_owner_id,
                lifecycle_attempt_count=lifecycle_attempt_count,
                scope_transition_id=scope_transition_id,
            )
            stats["memories_extracted"] = memory_stats.get("added", 0)
            stats["memory_supports_removed"] = memory_stats.get("deleted", 0)
            await self._insert_changelog(
                ChangelogEntry(
                    id=None,
                    doc_id=doc_id,
                    change_type=change_type,
                    previous_version=previous_version,
                    current_version=item.version,
                    content_diff=None,
                    ai_change_summary=f"Document '{item.title}' is now empty.",
                    detected_at=now,
                    title=item.title,
                    source=source_id,
                )
            )
            await self._finalize_projected_document_moves(
                predecessor_docs=lineage_predecessor_docs,
                current_doc_id=doc_id,
                source_id=source_id,
                source_unit_id=source_unit.id,
                source_activity=source_activity,
            )
            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            logger.info(
                "Reconciled explicit empty Source Observation for %s (%s)",
                item.title,
                doc_id,
            )
            return stats

        repo_identifier = normalized.source_semantics.get("repo_identifier")
        source_updated_at = _source_updated_at(normalized.source_semantics)
        derivation_context = SourceUnitDerivationContext(
            document=doc_record,
            doc_type=source_type,
            project_key=project_key,
            repo_identifier=(str(repo_identifier) if repo_identifier is not None else None),
            document_content=markdown_body,
            update_mode=(update_plan.mode if update_plan else "full_document"),
            changed_hunks=(update_plan.changed_hunks if update_plan else None),
            update_plan_stats=self._document_update_plan_stats(update_plan),
            source_updated_at=(source_updated_at.isoformat() if source_updated_at is not None else None),
            user_id=None,
            current_changed_ranges=(update_plan.current_changed_ranges if update_plan is not None else ()),
            reprocess_all_current_observations=force_reprocess,
            reprocess_operation_id=(run_id if force_reprocess else None),
            support_without_baseline=reprocessing,
            unit_input=unit_input,
        )
        # Extraction owns the only document-content model call. Historical
        # cross-document/cross-source discovery remains post-commit Relation
        # work; same-source incumbents are loaded by exact lifecycle lineage.
        diagnostics.stage = "derivation"
        extraction_result = await self._extract_for_document_update(
            projection=projection,
            update_plan=update_plan,
            source_type=source_type,
            doc_type=source_type,
            doc_id=doc_id,
            source_id=source_id,
            run_id=run_id,
            derivation_context=derivation_context,
            reprocess_current_observations=force_reprocess,
        )

        raw_memories = extraction_result.memories
        if extraction_result.error_type:
            raise _memory_extraction_error(doc_id=doc_id, extraction=extraction_result)
        diagnostics.skipped_reading_group_count = _skipped_reading_group_count(extraction_result)
        logger.debug(
            "Extracted %d raw memories from %s",
            len(raw_memories),
            doc_id,
        )
        self._memory_sample(
            "after_extract",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_memory_count=len(raw_memories),
            existing_memory_count=0,
            entity_mention_count=len(
                {entity_ref for raw_memory in raw_memories for entity_ref in raw_memory.entity_refs}
            ),
            content_chars=len(markdown_body),
        )

        # ------------------------------------------------------------------
        # 8. Bind claims to revision-pinned evidence and apply one complete,
        # stale-guarded Lifecycle Plan for this Source Unit.
        # ------------------------------------------------------------------
        diagnostics.stage = "lifecycle"
        memory_stats = await self.memory_engine.prepare_and_commit_projected_lifecycle(
            projection=projection,
            doc_id=doc_id,
            raw_memories=raw_memories,
            doc_type=source_type,
            project_key=project_key,
            repo_identifier=repo_identifier,
            document_content=markdown_body,
            update_mode=update_plan.mode if update_plan else "full_document",
            changed_hunks=update_plan.changed_hunks if update_plan else None,
            update_plan_stats=self._document_update_plan_stats(update_plan),
            current_changed_ranges=(update_plan.current_changed_ranges if update_plan is not None else ()),
            source_updated_at=source_updated_at,
            protected_source_observation_ids=(extraction_result.protected_source_observation_ids),
            document=doc_record,
            unit_input=unit_input,
            derivation_id=extraction_result.derivation_id,
            derivation_reprocess_all_current_observations=(
                derivation_context.reprocess_all_current_observations
            ),
            derivation_reprocess_operation_id=(
                derivation_context.reprocess_operation_id
            ),
            derivation_support_without_baseline=(
                derivation_context.support_without_baseline
            ),
            source_activity=source_activity,
            lifecycle_execution_owner_id=lifecycle_execution_owner_id,
            lifecycle_attempt_count=lifecycle_attempt_count,
            scope_transition_id=scope_transition_id,
        )
        stats["memories_extracted"] = memory_stats.get("added", 0)

        self._memory_sample(
            "after_memory_engine",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_memory_count=len(raw_memories),
            memories_extracted=stats["memories_extracted"],
            entity_mention_count=len(
                {entity_ref for raw_memory in raw_memories for entity_ref in raw_memory.entity_refs}
            ),
            entity_resolution_unique_mentions=memory_stats.get("entity_resolution_unique_mentions", 0),
            entity_resolution_exact_hits=memory_stats.get("entity_resolution_exact_hits", 0),
            entity_resolution_alias_hits=memory_stats.get("entity_resolution_alias_hits", 0),
            entity_resolution_embedded_mentions=memory_stats.get("entity_resolution_embedded_mentions", 0),
            entity_resolution_ambiguous_mentions=memory_stats.get("entity_resolution_ambiguous_mentions", 0),
            entity_resolution_embedding_batches=memory_stats.get("entity_resolution_embedding_batches", 0),
            entity_resolution_llm_calls=memory_stats.get("entity_resolution_llm_calls", 0),
            entity_resolution_validation_retries=memory_stats.get(
                "entity_resolution_validation_retries",
                0,
            ),
            entity_resolution_candidate_count=memory_stats.get("entity_resolution_candidate_count", 0),
            entity_resolution_new_entities=memory_stats.get("entity_resolution_new_entities", 0),
            entity_resolution_elapsed_ms=memory_stats.get("entity_resolution_elapsed_ms", 0),
        )

        # ------------------------------------------------------------------
        # 10. Record changelog
        # ------------------------------------------------------------------
        changelog_entry = ChangelogEntry(
            id=None,
            doc_id=doc_id,
            change_type=change_type,
            previous_version=previous_version,
            current_version=item.version,
            content_diff=None,
            ai_change_summary=(f"New document: {item.title}" if change_type == "created" else f"Updated: {item.title}"),
            detected_at=now,
            title=item.title,
            source=source_id,
        )

        async with self._db_lock:
            await self._insert_changelog(changelog_entry)

        await self._finalize_projected_document_moves(
            predecessor_docs=lineage_predecessor_docs,
            current_doc_id=doc_id,
            source_id=source_id,
            source_unit_id=source_unit.id,
            source_activity=source_activity,
        )

        if progress_callback:
            progress_callback(
                {
                    "phase": "processing",
                    "event": "document_processed",
                    "title": item.title,
                }
            )

        logger.info(
            "Processed %s (%s): %s, %d memories inserted, %d source supports added",
            item.title,
            doc_id,
            change_type,
            stats["memories_extracted"],
            stats["memory_supports_added"],
        )

        return stats

    async def _extract_for_document_update(
        self,
        *,
        projection: SourceProjection,
        update_plan: DocumentUpdatePlan | None,
        source_type: str,
        doc_type: str,
        doc_id: str,
        source_id: str,
        run_id: str | None,
        derivation_context: SourceUnitDerivationContext,
        reprocess_current_observations: bool = False,
    ) -> MemoryExtractionResult:
        """Run durable extraction for the changed Observations of a document."""
        changed_observation_ids = {
            anchor.observation_id for delta in projection.deltas for anchor in delta.changed_anchors
        }
        changed_observation_ids.update(
            observation_id for delta in projection.deltas for observation_id in delta.added_observation_ids
        )
        if not changed_observation_ids and not reprocess_current_observations:
            result = MemoryExtractionResult(
                memories=[],
                metadata={"projection_changed_observation_count": 0},
            )
            await self._record_memory_extraction_result(
                mode="projection_no_changes",
                plan=update_plan,
                doc_id=doc_id,
                source_id=source_id,
                source_type=source_type,
                run_id=run_id,
                result=result,
            )
            return result
        result = await self._extract_source_derivation_work(
            projection=projection,
            source_type=source_type,
            doc_type=doc_type,
            source_id=source_id,
            derivation_context=derivation_context,
        )
        await self._record_memory_extraction_result(
            mode="projection_batches",
            plan=update_plan,
            doc_id=doc_id,
            source_id=source_id,
            source_type=source_type,
            run_id=run_id,
            result=result,
        )
        return result

    async def _extract_source_derivation_work(
        self,
        *,
        projection: SourceProjection,
        source_type: str,
        doc_type: str,
        source_id: str,
        derivation_context: SourceUnitDerivationContext,
    ) -> MemoryExtractionResult:
        """Execute durable extraction work for one Source Unit revision."""

        visibility, owner_user_id = await memory_visibility_for_source_id(
            self.db,
            source_id=source_id,
        )
        access_context_hash = lifecycle_access_context_hash(
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=derivation_context.project_key,
            repo_identifier=derivation_context.repo_identifier,
        )
        from memforge.pipeline.revision_assessment import client_reads_images, revision_inference_capability_hash
        inference_capability_hash = revision_inference_capability_hash(
            self.structured_llm_client, extraction_model=getattr(self.memory_extractor, "model", None),
        )
        committed_base_snapshot = await self._committed_base_snapshot_for(
            projection,
            reprocess_all_current_observations=(
                derivation_context.reprocess_all_current_observations
            ),
        )

        from memforge.pipeline.revision_assessment import RevisionAssessmentContext
        base_projection = (
            await self.db.get_current_source_unit_projection(projection.source_units[0].id)
            if projection.deltas[0].previous_unit_revision_id
            else None
        )
        reads_images = client_reads_images(self.structured_llm_client)
        revision_context = RevisionAssessmentContext(
            projection=projection, base=base_projection,
            access_context_hash=access_context_hash,
            image_loader=lambda ids: self._projection_images(projection=projection, observation_ids=ids),
            reads_images=reads_images,
        )

        async def plan_requests(authority):
            from memforge.pipeline.extraction_requests import plan_extraction_requests

            # Planning reads the bytes of the Artifacts each request cites.
            async with self._heavy_work_slot(source_id, multimodal=any(
                fragment.kind.value == "artifact" for fragment in revision_context.full_fragments
            )):
                return plan_extraction_requests(
                    revision_context, authority, extractor=self.memory_extractor,
                    source_type=source_type, doc_type=doc_type,
                )

        async def extract_request(request):
            async with self._heavy_work_slot(
                source_id,
                multimodal=any(fragment.kind.value == "artifact" for fragment in request.catalog.fragments),
            ) as admission:
                result = await self.memory_extractor.extract_projection_fragment_memories(
                    request.catalog,
                    source_type=source_type,
                    doc_type=doc_type,
                    revision_context=revision_context,
                )
                input_binary_bytes = int((result.metadata or {}).get("image_bytes", 0) or 0)
                result.metadata = {
                    **(result.metadata or {}),
                    "extraction_queue_wait_ms": admission.queue_wait_ms,
                    "input_binary_bytes": input_binary_bytes,
                    "multimodal_calls": int(input_binary_bytes > 0),
                    "max_active_multimodal": admission.active_multimodal,
                }
                return result

        result = await SourceUnitDeriver(
            self.db,
            runtime_event_trace_sink=self.runtime_event_trace_sink,
        ).derive(
            SourceUnitDerivationRequest(
                projection=projection,
                context=derivation_context,
                plan_requests=plan_requests,
                extract_request=extract_request,
                max_concurrent=self._source_parallelism_limit(),
                committed_base_snapshot=committed_base_snapshot,
                access_context_hash=access_context_hash,
                inference_capability_hash=inference_capability_hash,
                reads_images=reads_images,
            )
        )
        result.extraction.derivation_id = result.derivation.id
        result.extraction.metadata = {
            **(result.extraction.metadata or {}),
            "derivation_id": result.derivation.id,
            "source_unit_id": result.derivation.source_unit_id,
            "target_unit_revision_id": result.derivation.target_unit_revision_id,
            "extraction_contract_version": (
                result.derivation.extraction_contract_version
            ),
        }
        return result.extraction

    async def _committed_base_snapshot_for(
        self,
        projection: SourceProjection,
        *,
        reprocess_all_current_observations: bool,
    ) -> CommittedSourceUnitSnapshot | None:
        """Load one complete committed base; the planner owns mismatch policy."""

        delta = projection.deltas[0]
        if (
            delta.previous_unit_revision_id is None
            and not reprocess_all_current_observations
        ):
            return None
        unit_revision = await self.db.get_current_source_unit_revision(
            delta.source_unit_id
        )
        if unit_revision is None:
            return None
        if (
            not reprocess_all_current_observations
            and unit_revision.id != delta.previous_unit_revision_id
        ):
            # The planner persists the typed mismatch instead of turning a
            # deterministic snapshot race into document-level retries.
            return None
        observation_revisions = (
            await self.db.get_current_source_observation_revisions(
                delta.source_unit_id
            )
        )
        return CommittedSourceUnitSnapshot(
            unit_revision=unit_revision,
            observation_revisions=tuple(observation_revisions.values()),
        )

    def _projection_images(
        self,
        *,
        projection: SourceProjection,
        observation_ids: set[str],
    ) -> tuple[StructuredLlmImage, ...]:
        """Load exact stored bytes for image Artifact Observations only."""

        return load_projection_images(
            projection=projection,
            observation_ids=observation_ids,
            document_store=self.doc_store,
        )

    def _document_update_plan_stats(self, plan: DocumentUpdatePlan | None) -> dict[str, int | float | str] | None:
        """Return audit-friendly update plan details for downstream reconciliation."""
        if plan is None:
            return None
        return {
            "reason": plan.reason,
            "data_shape": plan.data_shape,
            "diff_line_count": plan.diff_line_count,
            "added_lines": plan.added_lines,
            "removed_lines": plan.removed_lines,
            "changed_ratio": plan.changed_ratio,
        }

    def _read_previous_normalized_content(self, stored_input: SourceUnitInput | None) -> str | None:
        """Read the normalized markdown of the Unit's stored input before this sync overwrites it.

        Only an object stored under the input's own Source and Document keys
        holds this Unit's previous content; for any other URI the update has
        no previous content to compare against.
        """
        if stored_input is None or not self.doc_store.belongs_to_document(
            stored_input.normalized_content_uri,
            source_id=stored_input.source_id,
            doc_id=stored_input.document_id,
        ):
            return None

        try:
            return self.doc_store.read_normalized(str(stored_input.normalized_content_uri))
        except Exception as e:
            logger.warning("Failed to read previous normalized content via document store: %s", e)
            return None

    async def _record_document_update_strategy(
        self,
        *,
        plan: DocumentUpdatePlan,
        doc_id: str,
        source_id: str,
        run_id: str | None,
        previous_version: str | None,
        current_version: str | None,
        previous_hash: str | None,
        current_hash: str,
    ) -> None:
        """Record how an updated document will be processed."""
        context = self._memory_store_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )
        payload = {
            "previous_version": previous_version,
            "current_version": current_version,
            "previous_content_hash": previous_hash,
            "current_content_hash": current_hash,
            "data_shape": plan.data_shape,
            "diff_line_count": plan.diff_line_count,
            "added_lines": plan.added_lines,
            "removed_lines": plan.removed_lines,
            "changed_ratio": plan.changed_ratio,
        }
        if plan.fallback_from:
            payload["fallback_from"] = plan.fallback_from

        await self.memory_store.record_audit_event(
            "document_update_strategy_selected",
            "committed",
            context=context,
            doc_id=doc_id,
            source_id=source_id,
            decision=plan.mode,
            reason=plan.reason,
            thresholds=plan.thresholds,
            payload=payload,
        )

    async def _record_memory_extraction_result(
        self,
        *,
        mode: str,
        plan: DocumentUpdatePlan | None,
        doc_id: str,
        source_id: str,
        source_type: str,
        run_id: str | None,
        result: MemoryExtractionResult,
    ) -> None:
        """Record whether a memory extraction call produced usable candidates."""
        context = self._memory_store_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )
        if result.error_type:
            event_type = "memory_extraction_failed"
            status = "failed"
            reason = result.error_type
        else:
            event_type = "memory_extraction_completed"
            status = "committed"
            reason = plan.reason if plan else "full_document"

        payload: dict[str, Any] = {
            "extraction_mode": mode,
            "extracted_count": len(result.memories),
            "source_type": source_type,
        }
        if result.metadata:
            payload.update(result.metadata)
        if plan:
            payload.update(
                {
                    "diff_line_count": plan.diff_line_count,
                    "added_lines": plan.added_lines,
                    "removed_lines": plan.removed_lines,
                    "changed_ratio": plan.changed_ratio,
                }
            )

        await self.memory_store.record_audit_event(
            event_type,
            status,
            context=context,
            doc_id=doc_id,
            source_id=source_id,
            decision=mode,
            reason=reason,
            thresholds=plan.thresholds if plan else None,
            payload=payload,
            error=result.error,
        )

    async def _record_source_unit_llm_summary(
        self,
        *,
        collector: StructuredLlmMetricsCollector,
        source_unit_id: str,
        source_id: str,
        run_id: str | None,
        doc_id: str,
        source_unit_elapsed_ms: int,
        ok: bool,
        error_class: str | None,
        error_code: str | None = None,
        status: str | None = None,
    ) -> None:
        """Emit one content-free LLM aggregate for a completed Source Unit scope."""

        summary = collector.summary(
            source_unit_elapsed_ms=source_unit_elapsed_ms,
        )
        outcome = status or ("committed" if ok else "failed")
        payload = {
            "event": "source_unit_llm_summary",
            "source_id": source_id,
            "source_unit_id": source_unit_id,
            "run_id": run_id,
            "doc_id": doc_id,
            "ok": ok,
            "status": outcome,
            "error_class": error_class,
            "error_code": error_code,
            **summary.to_payload(),
        }
        logger.info(
            "source_unit_llm_summary %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        await self.memory_store.record_audit_event(
            "source_unit_llm_summary",
            outcome,
            context=self._memory_store_context(
                run_id=run_id,
                source_id=source_id,
                doc_id=doc_id,
            ),
            doc_id=doc_id,
            source_id=source_id,
            reason="source_unit_lifecycle_"
            + {
                "committed": "completed",
                "failed": "failed",
                "skipped": "skipped",
                "prepared": "deferred",
            }[outcome],
            payload={
                "source_unit_id": source_unit_id,
                "error_class": error_class,
                "error_code": error_code,
                **summary.to_payload(),
            },
        )

    def _memory_store_context(
        self,
        *,
        run_id: str | None,
        source_id: str,
        doc_id: str,
    ):
        return self.memory_store.operation_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )

    async def _finalize_projected_document_moves(
        self,
        *,
        predecessor_docs: list[DocumentRecord],
        current_doc_id: str,
        source_id: str,
        source_unit_id: str,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Atomically rebind this Source's provenance, then remove its obsolete document aliases."""
        if not predecessor_docs:
            return
        for predecessor in predecessor_docs:
            if predecessor.doc_id == current_doc_id:
                continue
            await self.db.rebind_projected_document_support(
                predecessor.doc_id,
                current_doc_id,
                source_id=source_id,
                source_activity=source_activity,
            )
            deletion_context = {
                "deletion_kind": "source_unit_move",
                "reason": "stable Source Unit moved to a new document locator",
                "source_unit_id": source_unit_id,
                "successor_doc_id": current_doc_id,
            }
            await self.memory_store.delete_projected_document(
                predecessor.doc_id,
                source_id=source_id,
                deletion_context=deletion_context,
                source_activity=source_activity,
            )

    # ==================================================================
    # Private: deletion detection
    # ==================================================================

    async def _detect_deletions(
        self,
        source_id: str,
        source_type: str,
        source_name: str,
        run_id: str,
        lifecycle_cycle_id: str,
        removals: Mapping[str, SourceRemoval],
        source_filter_summary: str | None,
        source_activity: SourceActivityLease | None = None,
    ) -> tuple[int, list[RunItemFailure], set[str]]:
        """Remove the held Documents this run proved removable.

        For each document, persist an explicit Source Unit tombstone carrying
        its removal reason, apply a gate-checked complete lifecycle ledger, and
        only then remove document storage when no review is required. The
        deletion context records the reason and the evidence that allowed it.

        Returns the count, failures, and Source Units whose tombstones committed.
        """
        if not removals:
            return 0, [], set()

        logger.info(
            "Detected %d deletions for source %s",
            len(removals),
            source_id,
        )

        deleted_count = 0
        failed_deletions: list[RunItemFailure] = []
        tombstoned_source_unit_ids: set[str] = set()
        for doc_id in sorted(removals):
            title = doc_id
            try:
                # Get existing document info before deletion
                existing_doc = await self.db.get_document(doc_id)
                if existing_doc is None:
                    logger.info(
                        "Document %s was already removed by a completed Source Unit move",
                        doc_id,
                    )
                    deleted_count += 1
                    continue

                source_unit = await self.db.find_source_unit_by_document_id(source_id, doc_id)
                if source_unit is None:
                    raise RuntimeError("source absence cannot be reconciled without persisted Source Unit lineage")
                lineage_document_ids = await self.db.list_source_unit_document_ids(source_unit.id)
                current_document_id = lineage_document_ids[0] if lineage_document_ids else doc_id
                if current_document_id != doc_id:
                    current_document = await self.db.get_document(current_document_id)
                    if current_document is not None:
                        await self.db.rebind_projected_document_support(
                            doc_id,
                            current_document_id,
                            source_id=source_id,
                            source_activity=source_activity,
                        )
                        await self.memory_store.delete_projected_document(
                            doc_id,
                            source_id=source_id,
                            deletion_context={
                                "deletion_kind": "source_unit_move",
                                "reason": "historical document locator is no longer current",
                                "source_unit_id": source_unit.id,
                                "successor_doc_id": current_document_id,
                            },
                            source_activity=source_activity,
                        )
                        logger.info(
                            "Removed historical document locator %s after move to %s",
                            doc_id,
                            current_document_id,
                        )
                        deleted_count += 1
                        continue

                # Record deletion in changelog
                now = datetime.now(timezone.utc)
                title = existing_doc.title if existing_doc else doc_id
                changelog_entry = ChangelogEntry(
                    id=None,
                    doc_id=doc_id,
                    change_type="deleted",
                    previous_version=(existing_doc.version if existing_doc else None),
                    current_version=None,
                    content_diff=None,
                    ai_change_summary=(f"Document '{title}' was deleted from source."),
                    detected_at=now,
                    title=title,
                    source=source_id,
                )
                async with self._db_lock:
                    await self._insert_changelog(changelog_entry)

                prior_unit_revision = await self.db.get_current_source_unit_revision(source_unit.id)
                if prior_unit_revision is None:
                    raise RuntimeError("source absence cannot be reconciled without a current Source Unit revision")
                prior_observation_revisions = await self.db.get_current_source_observation_revisions(source_unit.id)
                removal = removals[doc_id]
                tombstone = project_source_unit_tombstone(
                    source_type=source_type,
                    run_id=(f"{run_id}:tombstone:{uuid.uuid5(uuid.NAMESPACE_URL, f'{source_id}:{doc_id}').hex}"),
                    source_unit=source_unit,
                    prior_unit_revision=prior_unit_revision,
                    prior_observation_revisions=prior_observation_revisions,
                    reason=removal.reason.value,
                )
                target_revision = tombstone.source_unit_revisions[0]
                lifecycle_result = await self.memory_engine.apply_projected_tombstone(
                    projection=tombstone,
                    doc_id=doc_id,
                    reason=AUTHORITATIVE_SOURCE_UNIT_REMOVAL_REASON,
                    lifecycle_cycle_id=lifecycle_cycle_id,
                    source_activity=source_activity,
                )
                await self.db.record_source_projection(
                    tombstone,
                    source_activity=source_activity,
                )
                tombstoned_source_unit_ids.add(source_unit.id)
                if lifecycle_result["can_delete_document"]:
                    await self.memory_store.delete_projected_document(
                        doc_id,
                        source_id=source_id,
                        deletion_context={
                            "deletion_kind": "source_removal",
                            "reason": removal.reason.value,
                            "evidence": removal.evidence,
                            "source_filter_summary": source_filter_summary,
                            "source_unit_id": source_unit.id,
                            "target_unit_revision_id": target_revision.id,
                        },
                        source_activity=source_activity,
                    )
                else:
                    logger.warning(
                        "Retained removed document %s while lifecycle review or "
                        "document support remains (pending_reviews=%d)",
                        doc_id,
                        lifecycle_result["pending_review"],
                    )

                logger.info("Reconciled removed document %s (%s)", doc_id, title)
                deleted_count += 1

            except Exception as e:
                logger.error(
                    "Error handling deletion of %s: %s",
                    doc_id,
                    e,
                )
                failed_deletions.append(
                    _run_item_failure(e, document_id=doc_id, title=title, stage=REMOVAL_STAGE, store=self.db)
                )

        return deleted_count, failed_deletions, tombstoned_source_unit_ids

    # ==================================================================
    # Private: helpers
    # ==================================================================

    async def _get_indexed_doc_ids(self, source_id: str) -> set[str]:
        """The Documents the Source holds: those with a current Source Unit."""
        return await self.db.list_indexed_doc_ids(source_id)

    async def _stored_unit_input(self, source_id: str, doc_id: str) -> SourceUnitInput | None:
        """The stored input of the Source's current Unit for this Document, if it recorded one."""
        source_unit = await self.db.find_source_unit_by_document_id(source_id, doc_id, current_only=True)
        return await self.db.get_source_unit_input(source_unit.id) if source_unit is not None else None

    async def _insert_changelog(self, entry: ChangelogEntry) -> None:
        """Record a Document change; the changelog is an audit trail, so a failure does not fail the sync."""
        try:
            await self.db.insert_changelog(entry)
        except Exception as e:
            logger.error(
                "Failed to insert changelog for %s: %s",
                entry.doc_id,
                e,
            )
