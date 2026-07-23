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
import json
import logging
import math
import threading
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

import tiktoken

from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmMetricsCollector,
)
from memforge.models import (
    ChangelogEntry,
    DocumentRecord,
    FailedDoc,
    MemoryExtractionResult,
    SyncState,
    content_hash as compute_content_hash,
)
from memforge.pipeline.sync_memory import ProcessMemoryReclaimer, SyncMemoryObserver

from memforge.pipeline.document_units import ExtractionContextPacker, UnitizationPolicy, unitize_markdown
from memforge.pipeline.document_update import (
    DocumentUpdatePlan,
    plan_document_update,
    quote_overlaps_current_changes,
)
from memforge.pipeline.source_projection_adapters import (
    DEFAULT_SOURCE_PROJECTION_ADAPTER,
    project_source_unit_tombstone,
    source_run_projection_coverage,
)
from memforge.pipeline.projection_context import plan_projection_extraction_batches
from memforge.memory.lifecycle_plan import AUTHORITATIVE_SOURCE_UNIT_REMOVAL_REASON
from memforge.memory.project_resolver import resolve_project_key
from memforge.source_projection import (
    ProjectionCoverage,
    ProjectionEnvelope,
    ProjectionRequest,
    ProjectionRunMode,
    SourceProjection,
    SourceProjectionAdapter,
    SourceRelationType,
)
from memforge.source_projection_config import canonical_projection_scope

if TYPE_CHECKING:
    from memforge.genes.base import Gene
    from memforge.memory.engine import MemoryEngine
    from memforge.memory.store import MemoryStore
    from memforge.models import ContentItem, Memory
    from memforge.pipeline.memory_extractor import MemoryExtractor
    from memforge.pipeline.source_support_detector import SourceSupportDetector
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

DEFAULT_INCREMENTAL_SYNC_OVERLAP = timedelta(minutes=10)


class SourceSyncMode(str, Enum):
    """Execution contract for one Gene discovery run."""

    NORMAL = "normal"
    PROJECTION_REPAIR = "projection_repair"
    REBASELINE_PREFLIGHT = "rebaseline_preflight"
    REBASELINE_REPLAY = "rebaseline_replay"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_extraction_metrics(
    results: list[MemoryExtractionResult],
) -> dict[str, int]:
    """Aggregate content-free LLM cost signals across one bounded extraction fan-out."""

    keys = (
        "structured_llm_calls",
        "prompt_chars",
        "structured_llm_elapsed_ms",
    )
    return {
        key: sum(int((result.metadata or {}).get(key, 0) or 0) for result in results)
        for key in keys
    }


class ExtractionWorkPool:
    """Work-conserving fair pool for app-wide heavy extraction work."""

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max(1, int(max_workers))
        self._condition = asyncio.Condition()
        self._active_by_source: dict[str, int] = defaultdict(int)
        self._waiting_by_source: dict[str, int] = defaultdict(int)
        self._total_active = 0

    @asynccontextmanager
    async def slot(self, source_id: str):
        await self.acquire(source_id)
        try:
            yield
        finally:
            await self.release(source_id)

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
        await self._semaphore.acquire()
        async with self._lock:
            self._active += 1
            self._max_active_seen = max(self._max_active_seen, self._active)
        try:
            yield
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


def _parse_source_updated_at(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("source_updated_at must be an ISO datetime string")
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("source_updated_at must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _source_updated_at_for_item(item: ContentItem, source_semantics: dict[str, Any]) -> datetime:
    explicit = _parse_source_updated_at(source_semantics.get("source_updated_at"))
    if explicit is not None:
        return explicit
    if item.last_modified.tzinfo is None or item.last_modified.utcoffset() is None:
        raise ValueError("ContentItem.last_modified must include an explicit timezone offset")
    return item.last_modified.astimezone(timezone.utc)


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
    if "rate limit" in normalized or "429" in normalized:
        return "rate_limit"
    if "pdf export" in normalized or "did not produce a pdf" in normalized:
        return "pdf_export"
    if "certificate_verify_failed" in normalized or "certificate verify" in normalized:
        return "certificate"
    return "other"


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
            details.append(f"Confluence rate limited {_plural(counts['rate_limit'], 'document')}")
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
        parts = [f"{_plural(docs_failed, 'Confluence document')} could not be imported."]
        details: list[str] = []
        if counts.get("pdf_export"):
            details.append(f"PDF export was unavailable for {_plural(counts['pdf_export'], 'document')}")
        if counts.get("rate_limit"):
            details.append(f"Confluence rate limited {_plural(counts['rate_limit'], 'document')}")
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


def _source_filter_summary(gene: Gene, since: datetime | None) -> str | None:
    config = getattr(gene, "config", None)
    parts: list[str] = []

    if isinstance(config, dict):
        jql_filter = config.get("jql_filter")
        if isinstance(jql_filter, str) and jql_filter.strip():
            parts.append(jql_filter.strip())

    if since:
        parts.append(f"updated >= {since.isoformat()}")

    return " AND ".join(parts) or None


# ---------------------------------------------------------------------------
# GeneSyncOrchestrator
# ---------------------------------------------------------------------------


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
        source_support_detector: SourceSupportDetector | None = None,
        max_concurrent: int = 3,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        memory_observer: SyncMemoryObserver | None = None,
        memory_reclaimer: ProcessMemoryReclaimer | None = None,
        source_projection_adapter: SourceProjectionAdapter | None = None,
        retry_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        structured_llm_client: LiteLlmStructuredClient | None = None,
    ) -> None:
        self.db = db
        self.doc_store = doc_store
        self.memory_extractor = memory_extractor
        self.memory_engine = memory_engine
        self.memory_store = memory_store
        self.source_support_detector = source_support_detector
        self.max_concurrent = max(1, max_concurrent)
        self.extraction_pool = extraction_pool
        self.document_lifecycle_admission = document_lifecycle_admission
        self.memory_observer = memory_observer
        self.memory_reclaimer = memory_reclaimer or ProcessMemoryReclaimer()
        self.source_projection_adapter = source_projection_adapter or DEFAULT_SOURCE_PROJECTION_ADAPTER
        self._retry_sleep = retry_sleep
        self.structured_llm_client = structured_llm_client

        self._llm_semaphore = asyncio.Semaphore(self.max_concurrent)
        self._db_lock = asyncio.Lock()
        # Rate limiter: token-bucket style, refills at calls_per_minute rate
        self._rate_limit_interval = 60.0 / max(self.max_concurrent * 10, 30)  # seconds between calls
        self._last_llm_call_time: float = 0.0

    async def _acquire_llm_slot(self) -> None:
        """Acquire semaphore + enforce rate limit before an LLM call."""
        await self._llm_semaphore.acquire()
        import time

        now = time.monotonic()
        elapsed = now - self._last_llm_call_time
        if elapsed < self._rate_limit_interval:
            await asyncio.sleep(self._rate_limit_interval - elapsed)
        self._last_llm_call_time = time.monotonic()

    def _release_llm_slot(self) -> None:
        """Release semaphore after an LLM call."""
        self._llm_semaphore.release()

    def _source_parallelism_limit(self) -> int:
        return self.max_concurrent

    def _document_parallelism_limit(self) -> int:
        if self.document_lifecycle_admission is None:
            return self.max_concurrent
        return min(self.max_concurrent, self.document_lifecycle_admission.max_active)

    @asynccontextmanager
    async def _heavy_work_slot(self, source_id: str):
        if self.extraction_pool is not None:
            async with self.extraction_pool.slot(source_id):
                yield
            return
        async with self._llm_semaphore:
            yield

    @asynccontextmanager
    async def _document_lifecycle_slot(self, source_id: str, doc_id: str):
        if self.document_lifecycle_admission is None:
            yield
            return
        async with self.document_lifecycle_admission.slot(source_id, doc_id):
            yield

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
        authoritative_snapshot: bool = False,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        source_activity_epoch: int | None = None,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
        reusable_projection_doc_ids: frozenset[str] = frozenset(),
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
        authoritative_snapshot:
            When true, discover the complete submitted snapshot so removals can
            be reconciled, while still skipping documents whose content is unchanged.
        reprocess_doc_ids:
            Optional document identifiers that should be re-extracted during a
            full discovery. Other unchanged documents remain skipped.
        execution_mode:
            ``projection_repair`` establishes the current provider-neutral
            projection baseline for explicitly requested documents without
            running enrichment, Memory lifecycle, vector writes, deletion
            detection, or advancing the ordinary source sync cursor.
            ``rebaseline_replay`` runs only after a successful preflight and
            lifecycle reset, so it may remove legacy documents that predate
            persisted Source Unit lineage when complete discovery proves their
            absence.

        Returns
        -------
        SyncState
            Final sync result with counts and error details.
        """
        run_id = uuid.uuid4().hex[:12]
        durable_cycle_id = lifecycle_cycle_id or run_id
        transition_run_id = scope_transition_run_id or durable_cycle_id
        projection_repair = execution_mode is SourceSyncMode.PROJECTION_REPAIR
        rebaseline_preflight = execution_mode is SourceSyncMode.REBASELINE_PREFLIGHT
        rebaseline_replay = execution_mode is SourceSyncMode.REBASELINE_REPLAY
        non_mutating_run = projection_repair or rebaseline_preflight
        if projection_repair and not reprocess_doc_ids:
            raise ValueError("projection repair requires explicit document identifiers")
        started_at = datetime.now(timezone.utc)
        bind_document_store = getattr(gene, "bind_document_store", None)
        if callable(bind_document_store):
            bind_document_store(self.doc_store)

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
        memories_corroborated = 0
        failed_docs: list[FailedDoc] = []
        error_message: str | None = None
        crawled_doc_ids: set[str] = set()
        existing_state = await self.db.get_sync_state(source_id)
        configured_source = await self.db.get_source(source_id)
        configured_source_type = str((configured_source or {}).get("type") or "unknown")
        configured_projection_scope = canonical_projection_scope(
            configured_source_type,
            (configured_source or {}).get("config") or {},
        )
        configured_access_context = {
            "access_policy": str((configured_source or {}).get("access_policy") or "workspace"),
            "owner_user_id": (configured_source or {}).get("owner_user_id"),
        }
        scope_transition = None
        if not non_mutating_run:
            scope_transition = await self.db.get_open_projection_scope_transition(source_id)
        if (
            force_full_sync
            or execution_mode is not SourceSyncMode.NORMAL
            or scope_transition is not None
        ):
            # Reuse is an ordinary incremental optimization only. Recovery
            # modes and scope transitions require the full per-document path.
            reusable_projection_doc_ids = frozenset()
        transition_started = False
        run_coverage = ProjectionCoverage.PARTIAL_PROJECTION
        reused_projection_count = 0
        total_item_count = 0

        try:
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
            last_sync_time = (
                None
                if non_mutating_run or force_full_sync or authoritative_snapshot or scope_transition is not None
                else (existing_state.last_sync_at if existing_state else None)
            )
            if last_sync_time and hasattr(gene, "fetch_pdf"):
                missing_pdf_count = await self._count_missing_pdf_uris(source_id)
                if missing_pdf_count:
                    logger.info(
                        "Found %d documents missing PDF provenance for %s; forcing full sync",
                        missing_pdf_count,
                        source_id,
                    )
                    last_sync_time = None

            # If sync_state says "synced" but there are 0 indexed docs,
            # force a full re-sync (handles previously broken sync stubs)
            if last_sync_time and not indexed_doc_ids:
                logger.warning(
                    "Sync state exists but 0 indexed docs for %s — forcing full re-sync",
                    source_id,
                )
                last_sync_time = None
            elif last_sync_time:
                last_sync_time = last_sync_time - DEFAULT_INCREMENTAL_SYNC_OVERLAP

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

            begin_discovery = getattr(gene, "begin_discovery", None)
            if callable(begin_discovery):
                begin_discovery()
            async for item in gene.discover(since=last_sync_time):
                items.append(item)
                crawled_doc_ids.add(item.item_id)
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "discovering",
                            "current": len(items),
                            "total": 0,
                            "title": None,
                        }
                    )

            # Preflight remains non-mutating, but its provider-level discovery
            # attestation still decides whether absent historical units are a
            # proven scope removal or an incomplete replay.
            discovery_coverage = source_run_projection_coverage(
                incremental=last_sync_time is not None,
                authoritative_snapshot=authoritative_snapshot,
                discovery_complete=bool(getattr(gene, "discovery_complete", False)),
            )
            run_coverage = (
                ProjectionCoverage.PARTIAL_PROJECTION
                if non_mutating_run
                else discovery_coverage
            )

            if projection_repair:
                requested_doc_ids = set(reprocess_doc_ids or ())
                discovered_by_id = {item.item_id: item for item in items}
                items = [discovered_by_id[doc_id] for doc_id in sorted(requested_doc_ids) if doc_id in discovered_by_id]
                for missing_doc_id in sorted(requested_doc_ids - discovered_by_id.keys()):
                    failed_docs.append(
                        FailedDoc(
                            doc_id=missing_doc_id,
                            title=missing_doc_id,
                            error="requested document was not returned by provider discovery",
                        )
                    )
                    docs_failed += 1

            logger.info(
                "Discovered %d content items from %s (since=%s)",
                len(items),
                source_name,
                last_sync_time.isoformat() if last_sync_time else "full sync",
            )
            discovered_doc_ids = {item.item_id for item in items}
            unexpected_reuse_ids = reusable_projection_doc_ids - discovered_doc_ids
            if unexpected_reuse_ids and authoritative_snapshot:
                raise ValueError(
                    "reusable Source Projection membership is outside provider discovery: "
                    f"{sorted(unexpected_reuse_ids)[0]}"
                )
            reusable_projection_doc_ids = (
                reusable_projection_doc_ids & discovered_doc_ids
            )
            total_item_count = len(items)
            if reusable_projection_doc_ids:
                items = [
                    item
                    for item in items
                    if item.item_id not in reusable_projection_doc_ids
                ]
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
                projection_coverage=run_coverage.value,
            )

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

            async def _process_one(item: ContentItem) -> dict:
                """Process a single item with retry logic and error isolation."""
                nonlocal docs_updated_counter, memories_extracted_counter, progress_counter
                stats = {
                    "processed": False,
                    "updated": False,
                    "memories_extracted": 0,
                    "memories_corroborated": 0,
                    "failed": False,
                    "preflight_source_unit_id": None,
                    "preflight_observation_ids": (),
                    "projection_scope_attestation": None,
                }
                document_completed = False

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

                last_error: Exception | None = None
                async with item_semaphore:
                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            item_stats = await self._process_item(
                                gene=gene,
                                item=item,
                                source_name=source_name,
                                source_id=source_id,
                                run_id=run_id,
                                progress_callback=on_item_progress,
                                force_reprocess=(
                                    force_full_sync and (not reprocess_doc_ids or item.item_id in reprocess_doc_ids)
                                ),
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
                                authoritative_snapshot=authoritative_snapshot,
                                execution_mode=execution_mode,
                                expected_source_activity_epoch=source_activity_epoch,
                            )
                            stats["processed"] = True
                            stats["updated"] = item_stats.get("updated", False)
                            stats["memories_extracted"] = item_stats.get(
                                "memories_extracted",
                                0,
                            )
                            stats["memories_corroborated"] = item_stats.get(
                                "memories_corroborated",
                                0,
                            )
                            stats["preflight_source_unit_id"] = item_stats.get("preflight_source_unit_id")
                            stats["preflight_observation_ids"] = item_stats.get(
                                "preflight_observation_ids",
                                (),
                            )
                            stats["projection_scope_attestation"] = item_stats.get("projection_scope_attestation")
                            last_error = None
                            break
                        except Exception as e:
                            last_error = e
                            if attempt < MAX_RETRIES:
                                delay = 2**attempt
                                logger.warning(
                                    "Retry %d/%d for %s after error: %s",
                                    attempt,
                                    MAX_RETRIES,
                                    item.item_id,
                                    e,
                                )
                                await self._retry_sleep(delay)
                            else:
                                logger.error(
                                    "Failed to process %s after %d attempts: %s",
                                    item.item_id,
                                    MAX_RETRIES,
                                    e,
                                )

                if last_error is not None:
                    stats["failed"] = True
                    failed_docs.append(
                        FailedDoc(
                            doc_id=item.item_id,
                            title=item.title,
                            error=str(last_error),
                        )
                    )
                elif stats["processed"]:
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

            # Aggregate stats
            for r in results:
                if r["processed"]:
                    docs_processed += 1
                if r["updated"]:
                    docs_updated += 1
                if r["failed"]:
                    docs_failed += 1
                memories_extracted += r["memories_extracted"]
                memories_corroborated += r["memories_corroborated"]

            if rebaseline_preflight and docs_failed == 0:
                current_units = await self.db.list_current_source_unit_observation_ids(source_id)
                replayed_units = {
                    str(result["preflight_source_unit_id"]): frozenset(
                        str(observation_id) for observation_id in result["preflight_observation_ids"]
                    )
                    for result in results
                    if result["preflight_source_unit_id"] is not None
                }
                missing_units = sorted(set(current_units) - set(replayed_units))
                missing_observations = {
                    unit_id: sorted(set(observation_ids) - replayed_units.get(unit_id, frozenset()))
                    for unit_id, observation_ids in current_units.items()
                    if set(observation_ids) - replayed_units.get(unit_id, frozenset())
                }
                if (missing_units or missing_observations) and not discovery_coverage.proves_absence:
                    raise ValueError(
                        "source rebaseline replay closure is incomplete: "
                        f"missing_units={missing_units}, "
                        f"missing_observations={missing_observations}"
                    )

            # ----------------------------------------------------------
            # Step 5: Detect deletions (only on full sync, not incremental)
            # ----------------------------------------------------------
            # When since= is set, the gene only returns CHANGED pages.
            # Pages not returned aren't deleted — they're just unchanged.
            # Only run deletion detection on full syncs (since=None).
            deleted_count = 0
            absence_is_authoritative = not non_mutating_run and run_coverage.proves_absence and docs_failed == 0

            if absence_is_authoritative:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "detecting_deletions",
                            "current": 0,
                            "total": 0,
                            "title": None,
                        }
                    )

                deleted_count, deletion_failures = await self._detect_deletions(
                    source_id=source_id,
                    source_type=configured_source_type,
                    source_name=source_name,
                    run_id=run_id,
                    lifecycle_cycle_id=(
                        scope_transition.id
                        if scope_transition is not None
                        else durable_cycle_id
                    ),
                    indexed_doc_ids=indexed_doc_ids,
                    crawled_doc_ids=crawled_doc_ids,
                    source_filter_summary=_source_filter_summary(gene, last_sync_time),
                    allow_legacy_orphan_cleanup=rebaseline_replay,
                    expected_source_activity_epoch=source_activity_epoch,
                )
                if deletion_failures:
                    failed_docs.extend(deletion_failures)
                    docs_failed += len(deletion_failures)
                if deleted_count > 0:
                    logger.info(
                        "Soft-deleted memories from %d removed documents",
                        deleted_count,
                    )
            else:
                logger.debug(
                    "Skipping deletion detection without authoritative coverage: "
                    "coverage=%s failed_docs=%d returned=%d indexed=%d",
                    run_coverage.value,
                    docs_failed,
                    len(crawled_doc_ids),
                    len(indexed_doc_ids),
                )

            if scope_transition is not None:
                scoped_reconciliation_coverage = None
                if docs_failed == 0:
                    scoped_reconciliation_coverage = self.source_projection_adapter.reconciliation_coverage(
                        source_type=configured_source_type,
                        transition=scope_transition,
                        current_units=await self.db.list_current_source_units(source_id),
                        run_attestations=tuple(
                            result["projection_scope_attestation"]
                            for result in results
                            if result.get("projection_scope_attestation") is not None
                        ),
                    )
                if (absence_is_authoritative or scoped_reconciliation_coverage is not None) and docs_failed == 0:
                    scope_transition = await self.db.complete_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                        coverage=scoped_reconciliation_coverage or run_coverage,
                    )
                else:
                    scope_transition = await self.db.fail_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                        coverage=run_coverage,
                        error=(
                            "target scope did not produce a complete successful snapshot: "
                            f"coverage={run_coverage.value}, failed_docs={docs_failed}"
                        ),
                    )
                transition_started = False

            # ----------------------------------------------------------
            # Step 6: Update source doc_count
            # ----------------------------------------------------------
            if not non_mutating_run:
                total_docs = await self.db.count_documents(source=source_id)
                await self.db.update_source_doc_count(source_id, total_docs)

            # ----------------------------------------------------------
            # Determine final status
            # ----------------------------------------------------------
            if docs_failed == 0:
                status = "success"
            elif docs_processed > 0 or deleted_count > 0:
                status = "partial"
            else:
                status = "failed"

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
            if scope_transition is not None and transition_started:
                try:
                    await self.db.fail_projection_scope_transition(
                        scope_transition.id,
                        run_id=transition_run_id,
                        coverage=run_coverage,
                        error=error_message,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist Projection Scope transition failure for %s",
                        scope_transition.id,
                    )

        if self.memory_store is not None:
            try:
                await self.memory_store.attempt_lifecycle_vector_delivery(
                    source_id=source_id
                )
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

        # Advance the incremental watermark after a successful sync, including
        # no-change runs where discovery returns zero items.
        if status == "success":
            sync_at = finished_at
        elif existing_state and existing_state.last_sync_at:
            sync_at = existing_state.last_sync_at
        else:
            sync_at = None

        sync_state = SyncState(
            source=source_id,
            last_sync_at=sync_at,
            last_sync_status=status,
            docs_processed=docs_processed,
            docs_updated=docs_updated,
            docs_failed=docs_failed,
            memories_extracted=memories_extracted,
            memories_corroborated=memories_corroborated,
            error_message=error_message,
            failed_docs=failed_docs,
        )

        if not non_mutating_run:
            try:
                await self.db.upsert_sync_state(sync_state)
            except Exception as e:
                logger.error("Failed to upsert sync state: %s", e)

        # Record in sync_history for audit trail
        if not non_mutating_run:
            try:
                await self.db.insert_sync_history(
                    source=source_id,
                    status=status,
                    docs_processed=docs_processed,
                    docs_updated=docs_updated,
                    docs_failed=docs_failed,
                    memories_extracted=memories_extracted,
                    error_message=error_message,
                    failed_docs=[{"doc_id": fd.doc_id, "title": fd.title, "error": fd.error} for fd in failed_docs]
                    if failed_docs
                    else None,
                    started_at=started_at.isoformat(),
                    finished_at=finished_at.isoformat(),
                    run_id=run_id,
                )
            except Exception as e:
                logger.error("Failed to insert sync history: %s", e)

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
            memories_corroborated=memories_corroborated,
            item_count=total_item_count,
            materialized_item_count=materialized_item_count,
            reused_projection_count=reused_projection_count,
        )

        logger.info(
            "Sync complete for %s (run_id=%s): "
            "%d processed, %d updated, %d failed, "
            "%d memories extracted, %d corroborated, status=%s",
            source_name,
            run_id,
            docs_processed,
            docs_updated,
            docs_failed,
            memories_extracted,
            memories_corroborated,
            status,
        )

        return sync_state

    # ==================================================================
    # Private: _process_item
    # ==================================================================

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
        authoritative_snapshot: bool = False,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        expected_source_activity_epoch: int | None = None,
    ) -> dict:
        doc_id = item.item_id
        self._memory_sample("document_wait_start", source_id=source_id, run_id=run_id, doc_id=doc_id)
        lifecycle_error: Exception | None = None
        lifecycle_ok = False
        source_unit_id: str | None = None
        metrics_collector = StructuredLlmMetricsCollector() if self.structured_llm_client is not None else None

        def bind_source_unit(candidate_source_unit_id: str) -> None:
            nonlocal source_unit_id
            source_unit_id = candidate_source_unit_id

        async with self._document_lifecycle_slot(source_id, doc_id):
            lifecycle_started = asyncio.get_running_loop().time()
            self._memory_sample("document_lifecycle_enter", source_id=source_id, run_id=run_id, doc_id=doc_id)
            metrics_scope = (
                self.structured_llm_client.metrics_scope(metrics_collector)
                if self.structured_llm_client is not None and metrics_collector is not None
                else nullcontext()
            )
            try:
                with metrics_scope:
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
                        authoritative_snapshot=authoritative_snapshot,
                        execution_mode=execution_mode,
                        expected_source_activity_epoch=expected_source_activity_epoch,
                        source_unit_id_callback=bind_source_unit,
                    )
                lifecycle_ok = True
                return result
            except Exception as exc:
                lifecycle_error = exc
                raise
            finally:
                if metrics_collector is not None and source_unit_id is not None:
                    try:
                        await self._record_source_unit_llm_summary(
                            collector=metrics_collector,
                            source_unit_id=source_unit_id,
                            source_id=source_id,
                            run_id=run_id,
                            doc_id=doc_id,
                            source_unit_elapsed_ms=max(
                                0,
                                round((asyncio.get_running_loop().time() - lifecycle_started) * 1000),
                            ),
                            ok=lifecycle_ok,
                            error_class=(type(lifecycle_error).__name__ if lifecycle_error is not None else None),
                        )
                    except Exception:
                        logger.warning(
                            "Failed to record Source Unit LLM summary",
                            exc_info=True,
                        )
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
        authoritative_snapshot: bool = False,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        expected_source_activity_epoch: int | None = None,
        source_unit_id_callback: Callable[[str], None] | None = None,
    ) -> dict:
        """Process a single content item through the full pipeline.

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
            Stats: ``{updated, memories_extracted, memories_corroborated}``.
        """
        doc_id = item.item_id
        stats = {
            "updated": False,
            "memories_extracted": 0,
            "memories_corroborated": 0,
            "memory_supports_added": 0,
            "memory_supports_updated": 0,
            "memory_supports_removed": 0,
            "projection_scope_attestation": None,
        }

        # ------------------------------------------------------------------
        # 1. Fetch raw content
        # ------------------------------------------------------------------
        raw = await gene.fetch(item)
        logger.debug("Fetched %s (%d bytes)", doc_id, len(raw.body))
        self._memory_sample(
            "after_fetch",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_bytes=len(raw.body),
        )

        # ------------------------------------------------------------------
        # 2. Normalize to markdown
        # ------------------------------------------------------------------
        normalized = await gene.normalize(raw)
        markdown_body = normalized.markdown_body
        self._memory_sample(
            "after_normalize",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            content_chars=len(markdown_body or ""),
        )

        empty_content = not markdown_body or not markdown_body.strip()
        if empty_content and raw.body.strip() and not raw.authoritative_empty:
            raise ValueError(f"normalization produced empty content from a non-empty artifact: {doc_id}")
        if empty_content and not raw.authoritative_empty:
            raise ValueError(f"provider did not attest authoritative empty content: {doc_id}")
        if empty_content and not str(raw.empty_evidence or "").strip():
            raise ValueError(f"authoritative empty content is missing provider evidence: {doc_id}")
        # ------------------------------------------------------------------
        # 3. Project provider-native content into stable source lineage.
        #
        # A first in-memory projection gives us the stable Source Unit ID. We
        # then load that unit's current revisions and build the authoritative
        # delta before persisting the projection. No lifecycle decision is
        # allowed to infer deletion from the enclosing sync's item list.
        # ------------------------------------------------------------------
        source_metadata = gene.metadata()
        source_type = source_metadata.name
        source_shape = source_metadata.data_shape
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
            "authoritative_snapshot": authoritative_snapshot,
        }
        if persisted_source_unit is not None:
            # Document lineage is the durable identity bridge after an
            # authoritative rename.  Reuse both immutable Unit id and provider
            # key on subsequent ordinary snapshots that no longer carry the
            # one-time previous_filename signal.
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
                and scope_identity_probe.source_units[0].provider_key
                == historical_source_unit.provider_key
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
        if persisted_source_unit is None:
            predecessor_document_ids = {
                str(relation.metadata.get("predecessor_document_id"))
                for relation in projection_probe.relations
                if relation.relation_type is SourceRelationType.RENAMED_FROM
                and relation.metadata.get("predecessor_document_id")
            }
            for predecessor_document_id in sorted(predecessor_document_ids):
                async with self._db_lock:
                    predecessor_unit = await self.db.find_source_unit_by_document_id(
                        source_id,
                        predecessor_document_id,
                        current_only=True,
                    )
                if predecessor_unit is None:
                    continue
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
                break
        source_unit = projection_probe.source_units[0]

        if source_unit.unit_type == "teams_scope_attestation":
            # Scope evidence is valid only for this processing run. Persisting
            # it as a document/Source Unit would turn historical selectors into
            # an ever-growing pseudo corpus and make later rebaseline replay
            # stale control state. Return the validated locator directly to the
            # reconciliation step without recording durable lineage.
            stats["projection_scope_attestation"] = dict(source_unit.locator)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            return stats

        if source_unit_id_callback is not None:
            source_unit_id_callback(source_unit.id)
        stats["source_unit_id"] = source_unit.id

        projection_run_id = f"{run_id or 'direct'}:{source_unit.id}:{projection_probe.source_unit_revisions[0].id}"
        async with self._db_lock:
            projection = await self.db.get_source_projection(projection_run_id)
            if projection is None:
                prior_unit_revision = await self.db.get_current_source_unit_revision(source_unit.id)
                prior_observation_revisions = await self.db.get_current_source_observation_revisions(source_unit.id)
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
                                "authoritative_snapshot": authoritative_snapshot,
                            },
                            run_mode=(
                                ProjectionRunMode.FULL_SNAPSHOT
                                if (
                                    execution_mode is SourceSyncMode.REBASELINE_PREFLIGHT or prior_unit_revision is None
                                )
                                else ProjectionRunMode.DELTA
                            ),
                            scope_transition=scope_transition,
                            access_context=dict(projection_access_context or {}),
                        ),
                        item=item,
                        raw=raw,
                        normalized=normalized,
                        prior_unit_revision=(
                            None if execution_mode is SourceSyncMode.REBASELINE_PREFLIGHT else prior_unit_revision
                        ),
                        prior_observation_revisions=(
                            {} if execution_mode is SourceSyncMode.REBASELINE_PREFLIGHT else prior_observation_revisions
                        ),
                    )
                )

            projection_requires_extraction = projection.deltas[0].requires_extraction
            if not projection_requires_extraction and execution_mode is not SourceSyncMode.REBASELINE_PREFLIGHT:
                # Location/access-only and idempotent observations carry no
                # Memory mutation, so their lineage can advance independently.
                await self.db.record_source_projection(
                    projection,
                    expected_source_activity_epoch=expected_source_activity_epoch,
                )

            lineage_document_ids = await self.db.list_source_unit_document_ids(source_unit.id)

        if execution_mode is SourceSyncMode.REBASELINE_PREFLIGHT:
            if not projection.coverage.proves_absence:
                raise ValueError(
                    f"source rebaseline projection is not complete: {doc_id} coverage={projection.coverage.value}"
                )
            stats["preflight_source_unit_id"] = source_unit.id
            stats["preflight_observation_ids"] = tuple(observation.id for observation in projection.observations)
            return stats

        lineage_predecessor_docs: list[DocumentRecord] = []
        for lineage_doc_id in lineage_document_ids:
            if lineage_doc_id == doc_id:
                continue
            predecessor = await self.db.get_document(lineage_doc_id)
            if predecessor is not None:
                lineage_predecessor_docs.append(predecessor)

        # Document artifacts retain their existing normalized-content hash
        # contract. The projection delta independently controls semantic work.
        # This keeps vector/document freshness compatible while ensuring a
        # provider revision or location-only move does not trigger extraction.
        new_hash = compute_content_hash(markdown_body)
        async with self._db_lock:
            existing_hash = await self.db.get_content_hash(doc_id)
            existing_doc = await self.db.get_document(doc_id)
            if existing_doc is None and lineage_predecessor_docs:
                existing_doc = lineage_predecessor_docs[0]
                existing_hash = existing_doc.content_hash
        content_unchanged = existing_hash == new_hash
        unchanged = (
            (content_unchanged or not projection_requires_extraction)
            and not force_reprocess
        )
        previous_markdown = (
            self._read_previous_normalized_content(existing_doc)
            if existing_hash is not None and existing_hash != new_hash
            else None
        )

        requires_pdf_uri = not raw.authoritative_empty and gene.requires_pdf_artifact(
            item=item,
            existing_doc=existing_doc,
            existing_hash=existing_hash,
            new_hash=(existing_hash if not projection_requires_extraction else new_hash),
        )

        # ------------------------------------------------------------------
        # 3. Store raw + normalized on disk
        # ------------------------------------------------------------------
        raw_uri = existing_doc.raw_content_uri if unchanged and existing_doc else None
        norm_uri = existing_doc.normalized_content_uri if unchanged and existing_doc else None
        stored_content_artifact = False
        if not content_unchanged or not raw_uri:
            raw_uri = self.doc_store.store_raw(
                source_id=source_id,
                doc_id=doc_id,
                title=item.title,
                content=raw.body,
                content_type=raw.content_type,
            )
            stored_content_artifact = True
        if not content_unchanged or not norm_uri:
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
        pdf_uri = existing_doc.pdf_content_uri if unchanged and existing_doc else None
        should_fetch_pdf = (
            execution_mode is not SourceSyncMode.PROJECTION_REPAIR
            and not raw.authoritative_empty
            and hasattr(gene, "fetch_pdf")
            and (projection_requires_extraction or requires_pdf_uri or not pdf_uri)
        )
        if should_fetch_pdf:
            try:
                pdf_bytes = await gene.fetch_pdf(item)
            except Exception as e:
                if requires_pdf_uri:
                    raise RuntimeError(f"Confluence PDF export failed for {item.title}: {e}") from e
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
                raise RuntimeError(f"Confluence PDF export did not produce a PDF for {item.title}")
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
            raw_content_uri=raw_uri,
            raw_content_type=(
                existing_doc.raw_content_type if content_unchanged and existing_doc else raw.content_type
            ),
            normalized_content_uri=norm_uri,
            pdf_content_uri=pdf_uri,
            last_synced=now,
            client=normalized.source_semantics.get("client") or None,
        )

        if execution_mode is SourceSyncMode.PROJECTION_REPAIR:
            stats["updated"] = existing_doc is None or not content_unchanged
            async with self._db_lock:
                await self.db.upsert_document(
                    doc_record,
                    require_configured_source=True,
                )
                await self.db.record_source_projection(
                    projection,
                    expected_source_activity_epoch=expected_source_activity_epoch,
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
                "Repaired Source Projection baseline for %s (%s) without Memory lifecycle",
                item.title,
                doc_id,
            )
            return stats

        if unchanged:
            stats["updated"] = not content_unchanged
            async with self._db_lock:
                await self.db.upsert_document(
                    doc_record,
                    require_configured_source=True,
                )
            await self._finalize_projected_document_moves(
                predecessor_docs=lineage_predecessor_docs,
                current_doc_id=doc_id,
                source_unit_id=source_unit.id,
            )

            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            logger.debug("Skipping semantic work for unchanged %s (%s)", item.title, doc_id)
            return stats

        stats["updated"] = True

        # Determine change type for changelog
        change_type = "updated" if existing_hash is not None else "created"
        previous_version = existing_doc.version if existing_doc else None
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

        # ------------------------------------------------------------------
        # 5. Store document record before expensive enrichment/memory work
        # ------------------------------------------------------------------
        async with self._db_lock:
            await self.db.upsert_document(
                doc_record,
                require_configured_source=True,
            )

        if empty_content:
            try:
                memory_stats = await self.memory_engine.apply_projected_lifecycle(
                    projection=projection,
                    doc_id=doc_id,
                    raw_memories=[],
                    doc_type=f"{source_type}_empty",
                    project_key=project_key,
                    repo_identifier=normalized.source_semantics.get("repo_identifier"),
                    document_content="",
                    update_mode=update_plan.mode if update_plan else "full_document",
                    changed_hunks=(update_plan.changed_hunks if update_plan else None),
                    update_plan_stats=self._document_update_plan_stats(update_plan),
                    source_updated_at=_source_updated_at_for_item(
                        item,
                        normalized.source_semantics,
                    ),
                    user_id=(
                        str(normalized.source_semantics.get("uploader_user_id")).strip()
                        if normalized.source_semantics.get("uploader_user_id")
                        else None
                    ),
                    expected_source_activity_epoch=expected_source_activity_epoch,
                )
            except Exception:
                await self._restore_document_processing_snapshot(
                    doc_id=doc_id,
                    existing_doc=existing_doc,
                )
                raise
            await self.db.record_source_projection(
                projection,
                expected_source_activity_epoch=expected_source_activity_epoch,
            )
            stats["memories_extracted"] = memory_stats.get("added", 0)
            stats["memories_corroborated"] = memory_stats.get("updated", 0)
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
                source_unit_id=source_unit.id,
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
        # Extraction owns the only document-content model call. Historical
        # cross-document/cross-source discovery remains post-commit Relation
        # work; same-source incumbents are loaded by exact lifecycle lineage.
        extraction_result = await self._extract_for_document_update(
            projection=projection,
            update_plan=update_plan,
            markdown_body=markdown_body,
            source_type=source_type,
            doc_type=source_type,
            doc_id=doc_id,
            source_id=source_id,
            run_id=run_id,
            document_title=item.title,
            document_url=item.source_url,
        )

        raw_memories = extraction_result.memories
        if extraction_result.error_type:
            await self._restore_document_processing_snapshot(
                doc_id=doc_id,
                existing_doc=existing_doc,
            )
            error_detail = extraction_result.error or ""
            raise RuntimeError(f"memory extraction failed for {doc_id}: {extraction_result.error_type}: {error_detail}")
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
                {
                    entity_ref
                    for raw_memory in raw_memories
                    for entity_ref in raw_memory.entity_refs
                }
            ),
            content_chars=len(markdown_body),
        )

        # ------------------------------------------------------------------
        # 8. Bind claims to revision-pinned evidence and apply one complete,
        # stale-guarded Lifecycle Plan for this Source Unit.
        # ------------------------------------------------------------------
        source_updated_at = _source_updated_at_for_item(item, normalized.source_semantics)
        uploader_user_id = normalized.source_semantics.get("uploader_user_id")
        actor_user_id = (
            str(uploader_user_id).strip() if isinstance(uploader_user_id, str) and uploader_user_id.strip() else None
        )
        try:
            memory_stats = await self.memory_engine.apply_projected_lifecycle(
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
                source_updated_at=source_updated_at,
                user_id=actor_user_id,
                expected_source_activity_epoch=expected_source_activity_epoch,
            )
        except Exception:
            # Projection and lifecycle state roll back together in the engine;
            # keep the durable document snapshot on the same prior revision so
            # an ordinary retry still observes the semantic delta.
            await self._restore_document_processing_snapshot(
                doc_id=doc_id,
                existing_doc=existing_doc,
            )
            raise
        # The production engine commits this projection with its Lifecycle
        # Plan. The idempotent write also keeps narrow test/custom engines on
        # the same success-only projection contract.
        await self.db.record_source_projection(
            projection,
            expected_source_activity_epoch=expected_source_activity_epoch,
        )
        stats["memories_extracted"] = memory_stats.get("added", 0)
        stats["memories_corroborated"] = memory_stats.get("updated", 0)

        self._memory_sample(
            "after_memory_engine",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_memory_count=len(raw_memories),
            memories_extracted=stats["memories_extracted"],
            memories_corroborated=stats["memories_corroborated"],
            entity_mention_count=len(
                {
                    entity_ref
                    for raw_memory in raw_memories
                    for entity_ref in raw_memory.entity_refs
                }
            ),
            entity_resolution_unique_mentions=memory_stats.get(
                "entity_resolution_unique_mentions", 0
            ),
            entity_resolution_exact_hits=memory_stats.get(
                "entity_resolution_exact_hits", 0
            ),
            entity_resolution_alias_hits=memory_stats.get(
                "entity_resolution_alias_hits", 0
            ),
            entity_resolution_embedded_mentions=memory_stats.get(
                "entity_resolution_embedded_mentions", 0
            ),
            entity_resolution_ambiguous_mentions=memory_stats.get(
                "entity_resolution_ambiguous_mentions", 0
            ),
            entity_resolution_embedding_batches=memory_stats.get(
                "entity_resolution_embedding_batches", 0
            ),
            entity_resolution_llm_calls=memory_stats.get(
                "entity_resolution_llm_calls", 0
            ),
            entity_resolution_candidate_count=memory_stats.get(
                "entity_resolution_candidate_count", 0
            ),
            entity_resolution_new_entities=memory_stats.get(
                "entity_resolution_new_entities", 0
            ),
            entity_resolution_elapsed_ms=memory_stats.get(
                "entity_resolution_elapsed_ms", 0
            ),
            identity_resolution_pair_count=memory_stats.get(
                "identity_resolution_pair_count", 0
            ),
            identity_resolution_llm_calls=memory_stats.get(
                "identity_resolution_llm_calls", 0
            ),
            identity_resolution_prompt_chars=memory_stats.get(
                "identity_resolution_prompt_chars", 0
            ),
            identity_resolution_elapsed_ms=memory_stats.get(
                "identity_resolution_elapsed_ms", 0
            ),
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
            ai_change_summary=(
                f"New document: {item.title}"
                if change_type == "created"
                else f"Updated: {item.title}"
            ),
            detected_at=now,
            title=item.title,
            source=source_id,
        )

        async with self._db_lock:
            await self._insert_changelog(changelog_entry)

        await self._finalize_projected_document_moves(
            predecessor_docs=lineage_predecessor_docs,
            current_doc_id=doc_id,
            source_unit_id=source_unit.id,
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
            "Processed %s (%s): %s, %d memories inserted, %d corroborated, %d source supports added",
            item.title,
            doc_id,
            change_type,
            stats["memories_extracted"],
            stats["memories_corroborated"],
            stats["memory_supports_added"],
        )

        return stats

    async def _extract_for_document_update(
        self,
        *,
        projection: SourceProjection,
        update_plan: DocumentUpdatePlan | None,
        markdown_body: str,
        source_type: str,
        doc_type: str,
        doc_id: str,
        source_id: str,
        run_id: str | None,
        document_title: str,
        document_url: str,
    ) -> MemoryExtractionResult:
        """Run full extraction or diff-guided extraction for a document."""
        projection_batches = plan_projection_extraction_batches(projection)
        prefer_single_observation_diff = (
            len(projection.observations) == 1
            and update_plan is not None
            and update_plan.mode == "diff_guided"
            and hasattr(self.memory_extractor, "extract_memory_changes")
        )
        changed_observation_ids = {
            anchor.observation_id
            for delta in projection.deltas
            for anchor in delta.changed_anchors
        }
        changed_observation_ids.update(
            observation_id
            for delta in projection.deltas
            for observation_id in delta.added_observation_ids
        )
        if not projection_batches and not changed_observation_ids:
            result = MemoryExtractionResult(
                memories=[],
                metadata={"projection_changed_observation_count": 0},
            )
            await self._record_memory_extraction_result(
                mode="projection_no_changes",
                plan=update_plan,
                doc_id=doc_id,
                source_id=source_id,
                run_id=run_id,
                result=result,
                extraction_metadata=result.metadata,
            )
            return result
        if (
            (len(projection.observations) > 1 or len(projection_batches) > 1)
            and not prefer_single_observation_diff
            and projection_batches
            and hasattr(self.memory_extractor, "extract_projection_batch_memories")
        ):
            result = await self._extract_projection_batches(
                projection_batches=projection_batches,
                source_type=source_type,
                doc_type=doc_type,
                source_id=source_id,
            )
            await self._record_memory_extraction_result(
                mode="projection_batches",
                plan=update_plan,
                doc_id=doc_id,
                source_id=source_id,
                run_id=run_id,
                result=result,
                extraction_metadata=result.metadata,
            )
            return result

        if (
            update_plan
            and update_plan.mode == "diff_guided"
            and hasattr(self.memory_extractor, "extract_memory_changes")
        ):
            try:
                async with self._heavy_work_slot(source_id):
                    result = await self.memory_extractor.extract_memory_changes(
                        changed_hunks=update_plan.changed_hunks or "",
                        updated_document=markdown_body,
                        source_type=source_type,
                        doc_type=doc_type,
                    )
                if not result.error_type:
                    result = self._enforce_diff_guided_evidence_boundary(
                        result=result,
                        updated_document=markdown_body,
                        plan=update_plan,
                        source_id=source_id,
                        doc_id=doc_id,
                    )
                await self._record_memory_extraction_result(
                    mode=update_plan.mode,
                    plan=update_plan,
                    doc_id=doc_id,
                    source_id=source_id,
                    run_id=run_id,
                    result=result,
                    extraction_metadata={
                        "current_changed_range_count": len(update_plan.current_changed_ranges),
                        "rejected_outside_changed_range_count": result.metadata.get(
                            "rejected_outside_changed_range_count",
                            0,
                        ),
                    },
                )
                if not result.error_type:
                    return result
                await self._record_document_update_strategy_fallback(
                    plan=update_plan,
                    doc_id=doc_id,
                    source_id=source_id,
                    run_id=run_id,
                    reason="diff_guided_extraction_failed",
                    error=result.error or result.error_type,
                )
            except Exception as e:
                await self._record_document_update_strategy_fallback(
                    plan=update_plan,
                    doc_id=doc_id,
                    source_id=source_id,
                    run_id=run_id,
                    reason="diff_guided_extraction_failed",
                    error=str(e),
                )
                logger.warning(
                    "Diff-guided extraction failed for %s; falling back to full extraction: %s",
                    doc_id,
                    e,
                )

        if hasattr(self.memory_extractor, "extract_unit_memories"):
            result = await self._extract_full_document_units(
                markdown_body=markdown_body,
                source_type=source_type,
                doc_type=doc_type,
                doc_id=doc_id,
                source_id=source_id,
                document_title=document_title,
                document_url=document_url,
            )
            unit_metadata = result.metadata or {}
            await self._record_memory_extraction_result(
                mode="full_document",
                plan=update_plan,
                doc_id=doc_id,
                source_id=source_id,
                run_id=run_id,
                result=result,
                extraction_metadata={
                    key: value
                    for key, value in {
                        "unitized": True,
                        "unit_count": unit_metadata.get("unit_count", 0),
                        "failed_unit_count": unit_metadata.get("failed_unit_count", 0),
                        "partial_error_type": unit_metadata.get("partial_error_type"),
                        "segmentation_version": unit_metadata.get("segmentation_version"),
                        "partition_strategy": unit_metadata.get("partition_strategy"),
                        "max_unit_input_tokens": unit_metadata.get("max_unit_input_tokens"),
                    }.items()
                    if value is not None
                },
            )
            return result

        async with self._heavy_work_slot(source_id):
            result = await self.memory_extractor.extract_memories(
                content=markdown_body,
                source_type=source_type,
                doc_type=doc_type,
            )
        await self._record_memory_extraction_result(
            mode="full_document",
            plan=update_plan,
            doc_id=doc_id,
            source_id=source_id,
            run_id=run_id,
            result=result,
        )
        return result

    def _enforce_diff_guided_evidence_boundary(
        self,
        *,
        result: MemoryExtractionResult,
        updated_document: str,
        plan: DocumentUpdatePlan,
        source_id: str,
        doc_id: str,
    ) -> MemoryExtractionResult:
        """Keep only candidates whose exact evidence intersects the current diff."""

        kept = []
        rejected = 0
        for memory in result.memories:
            quote = (memory.evidence_quote or memory.extraction_context or "").strip()
            if not quote_overlaps_current_changes(
                updated_document,
                quote,
                plan.current_changed_ranges,
            ):
                rejected += 1
                continue
            memory.evidence_quote = quote
            kept.append(memory)
        if rejected:
            logger.warning(
                "Rejected %d diff-guided memory candidate(s) outside changed ranges for %s/%s",
                rejected,
                source_id,
                doc_id,
            )
        return MemoryExtractionResult(
            memories=kept,
            metadata={
                **result.metadata,
                "rejected_outside_changed_range_count": rejected,
            },
        )

    async def _extract_projection_batches(
        self,
        *,
        projection_batches,
        source_type: str,
        doc_type: str,
        source_id: str,
    ) -> MemoryExtractionResult:
        """Execute all transient Observation batches as one extraction outcome."""

        batch_semaphore = asyncio.Semaphore(self._source_parallelism_limit())

        async def extract_one(batch):
            async with batch_semaphore:
                async with self._heavy_work_slot(source_id):
                    return await self.memory_extractor.extract_projection_batch_memories(
                        batch,
                        source_type=source_type,
                        doc_type=doc_type,
                    )

        results = await asyncio.gather(*(extract_one(batch) for batch in projection_batches))
        llm_metrics = _aggregate_extraction_metrics(results)
        memories = []
        failures = [result for result in results if result.error_type]
        for result in results:
            if not result.error_type:
                memories.extend(result.memories)
        if failures:
            first = failures[0]
            return MemoryExtractionResult(
                error_type="projection_batch_failure",
                error=first.error or first.error_type,
                metadata={
                    **llm_metrics,
                    "batch_count": len(projection_batches),
                    "failed_batch_count": len(failures),
                    "extracted_count_before_failure": len(memories),
                },
            )
        return MemoryExtractionResult(
            memories=memories,
            metadata={
                **llm_metrics,
                "batch_count": len(projection_batches),
                "failed_batch_count": 0,
            },
        )

    async def _extract_full_document_units(
        self,
        *,
        markdown_body: str,
        source_type: str,
        doc_type: str,
        doc_id: str,
        source_id: str,
        document_title: str,
        document_url: str,
    ) -> MemoryExtractionResult:
        """Extract a full document by deterministic structural units."""
        unitization_policy = UnitizationPolicy()
        units = unitize_markdown(markdown_body, doc_id=doc_id, policy=unitization_policy)
        packer = ExtractionContextPacker()
        unit_semaphore = asyncio.Semaphore(self._source_parallelism_limit())

        async def extract_one(unit) -> MemoryExtractionResult:
            context = packer.pack(
                document_title=document_title,
                document_url=document_url,
                source_type=source_type,
                unit=unit,
                all_units=units,
                entities=(),
            )
            async with unit_semaphore:
                async with self._heavy_work_slot(source_id):
                    return await self.memory_extractor.extract_unit_memories(
                        context,
                        doc_type=doc_type,
                    )

        results = await asyncio.gather(*(extract_one(unit) for unit in units))
        llm_metrics = _aggregate_extraction_metrics(results)

        all_memories = []
        first_error: MemoryExtractionResult | None = None
        failed_unit_count = 0
        for result in results:
            if result.error_type:
                failed_unit_count += 1
                first_error = first_error or result
                continue
            all_memories.extend(result.memories)

        if first_error:
            return MemoryExtractionResult(
                error_type="partial_unit_failure",
                error=first_error.error or first_error.error_type,
                metadata={
                    **llm_metrics,
                    "unit_count": len(units),
                    "failed_unit_count": failed_unit_count,
                    "extracted_count_before_failure": len(all_memories),
                    "segmentation_version": units[0].segmentation_version if units else "v1",
                    "partition_strategy": "recursive_fit_first",
                    "max_unit_input_tokens": unitization_policy.max_unit_input_tokens,
                },
            )
        return MemoryExtractionResult(
            memories=all_memories,
            metadata={
                **llm_metrics,
                "unit_count": len(units),
                "failed_unit_count": failed_unit_count,
                "partial_error_type": first_error.error_type if first_error else None,
                "segmentation_version": units[0].segmentation_version if units else "v1",
                "partition_strategy": "recursive_fit_first",
                "max_unit_input_tokens": unitization_policy.max_unit_input_tokens,
            },
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

    def _read_previous_normalized_content(self, existing_doc: DocumentRecord | None) -> str | None:
        """Read the previous normalized markdown before the current sync overwrites it."""
        if not existing_doc or not existing_doc.normalized_content_uri:
            return None

        uri = existing_doc.normalized_content_uri
        if self.doc_store and hasattr(self.doc_store, "read_normalized"):
            try:
                content = self.doc_store.read_normalized(uri)
                if content is not None:
                    return content
            except Exception as e:
                logger.warning("Failed to read previous normalized content via document store: %s", e)

        return None

    async def _get_existing_document_memories(self, doc_id: str) -> list[Memory]:
        """Get active memories extracted from the same source document."""
        try:
            memories = await self.db.get_memories_by_source_doc(doc_id)
        except Exception as e:
            logger.warning("Failed to fetch same-document memories for %s: %s", doc_id, e)
            return []
        return [memory for memory in memories if memory.status == "active"][:50]

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
        if not self.memory_store or not hasattr(self.memory_store, "record_audit_event"):
            return

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

    async def _record_document_update_strategy_fallback(
        self,
        *,
        plan: DocumentUpdatePlan,
        doc_id: str,
        source_id: str,
        run_id: str | None,
        reason: str,
        error: str,
    ) -> None:
        """Record a runtime fallback from diff-guided to full extraction."""
        if not self.memory_store or not hasattr(self.memory_store, "record_audit_event"):
            return

        context = self._memory_store_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )
        await self.memory_store.record_audit_event(
            "document_update_strategy_fallback",
            "committed",
            context=context,
            doc_id=doc_id,
            source_id=source_id,
            decision="full_document",
            reason=reason,
            thresholds=plan.thresholds,
            payload={
                "fallback_from": plan.mode,
                "diff_line_count": plan.diff_line_count,
                "added_lines": plan.added_lines,
                "removed_lines": plan.removed_lines,
                "changed_ratio": plan.changed_ratio,
            },
            error=error,
        )

    async def _record_memory_extraction_result(
        self,
        *,
        mode: str,
        plan: DocumentUpdatePlan | None,
        doc_id: str,
        source_id: str,
        run_id: str | None,
        result: MemoryExtractionResult,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record whether a memory extraction call produced usable candidates."""
        if not self.memory_store or not hasattr(self.memory_store, "record_audit_event"):
            return

        context = self._memory_store_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )
        is_change_extraction = mode == "diff_guided"
        if result.error_type:
            event_type = "memory_change_extraction_failed" if is_change_extraction else "memory_extraction_failed"
            status = "failed"
            reason = result.error_type
        else:
            event_type = "memory_change_extraction_completed" if is_change_extraction else "memory_extraction_completed"
            status = "committed"
            reason = plan.reason if plan else "full_document"

        payload: dict[str, Any] = {
            "extraction_mode": mode,
            "extracted_count": len(result.memories),
        }
        if result.metadata:
            payload.update(result.metadata)
        if extraction_metadata:
            payload.update(extraction_metadata)
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
    ) -> None:
        """Emit one content-free LLM aggregate for a completed Source Unit scope."""

        summary = collector.summary(
            source_unit_elapsed_ms=source_unit_elapsed_ms,
        )
        payload = {
            "event": "source_unit_llm_summary",
            "source_id": source_id,
            "source_unit_id": source_unit_id,
            "run_id": run_id,
            "doc_id": doc_id,
            "ok": ok,
            "error_class": error_class,
            **summary.to_payload(),
        }
        logger.info(
            "source_unit_llm_summary %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        if not self.memory_store or not hasattr(
            self.memory_store,
            "record_audit_event",
        ):
            return
        await self.memory_store.record_audit_event(
            "source_unit_llm_summary",
            "committed" if ok else "failed",
            context=self._memory_store_context(
                run_id=run_id,
                source_id=source_id,
                doc_id=doc_id,
            ),
            doc_id=doc_id,
            source_id=source_id,
            reason=("source_unit_lifecycle_completed" if ok else "source_unit_lifecycle_failed"),
            payload={
                "source_unit_id": source_unit_id,
                "error_class": error_class,
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
        if self.memory_store and hasattr(self.memory_store, "operation_context"):
            return self.memory_store.operation_context(
                run_id=run_id,
                source_id=source_id,
                doc_id=doc_id,
            )
        return None

    async def _finalize_projected_document_moves(
        self,
        *,
        predecessor_docs: list[DocumentRecord],
        current_doc_id: str,
        source_unit_id: str,
    ) -> None:
        """Atomically rebind provenance, then remove obsolete document aliases."""
        if not predecessor_docs:
            return
        for predecessor in predecessor_docs:
            if predecessor.doc_id == current_doc_id:
                continue
            await self.db.rebind_projected_document_support(
                predecessor.doc_id,
                current_doc_id,
            )
            deletion_context = {
                "deletion_kind": "source_unit_move",
                "reason": "stable Source Unit moved to a new document locator",
                "source_unit_id": source_unit_id,
                "successor_doc_id": current_doc_id,
            }
            if self.memory_store is not None:
                await self.memory_store.delete_projected_document(
                    predecessor.doc_id,
                    deletion_context=deletion_context,
                )
            else:
                await self.db.delete_projected_document(predecessor.doc_id)

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
        indexed_doc_ids: set[str],
        crawled_doc_ids: set[str],
        source_filter_summary: str | None,
        allow_legacy_orphan_cleanup: bool = False,
        expected_source_activity_epoch: int | None = None,
    ) -> tuple[int, list[FailedDoc]]:
        """Detect and handle documents deleted from the source.

        For each deleted document, persist an explicit Source Unit tombstone,
        apply a gate-checked complete lifecycle ledger, and only then remove
        document storage when no review is required.

        Returns the count of successfully handled deletions plus failures.
        """
        deleted_ids = indexed_doc_ids - crawled_doc_ids
        if not deleted_ids:
            return 0, []

        logger.info(
            "Detected %d deletions for source %s",
            len(deleted_ids),
            source_id,
        )

        deleted_count = 0
        failed_deletions: list[FailedDoc] = []
        for doc_id in deleted_ids:
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
                    if allow_legacy_orphan_cleanup:
                        deletion_context = {
                            "deletion_kind": "rebaseline_legacy_absence",
                            "reason": "not_returned_by_complete_rebaseline_replay",
                        }
                        if self.memory_store is not None:
                            await self.memory_store.delete_projected_document(
                                doc_id,
                                deletion_context=deletion_context,
                            )
                        else:
                            await self.db.delete_projected_document(doc_id)
                        logger.info(
                            "Removed legacy document %s after complete rebaseline replay proved absence",
                            doc_id,
                        )
                        deleted_count += 1
                        continue
                    raise RuntimeError("source absence cannot be reconciled without persisted Source Unit lineage")
                lineage_document_ids = await self.db.list_source_unit_document_ids(source_unit.id)
                current_document_id = lineage_document_ids[0] if lineage_document_ids else doc_id
                if current_document_id != doc_id:
                    current_document = await self.db.get_document(current_document_id)
                    if current_document is not None:
                        await self.db.rebind_projected_document_support(
                            doc_id,
                            current_document_id,
                        )
                        await self.memory_store.delete_projected_document(
                            doc_id,
                            deletion_context={
                                "deletion_kind": "source_unit_move",
                                "reason": "historical document locator is no longer current",
                                "source_unit_id": source_unit.id,
                                "successor_doc_id": current_document_id,
                            },
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
                tombstone = project_source_unit_tombstone(
                    source_type=source_type,
                    run_id=(f"{run_id}:tombstone:{uuid.uuid5(uuid.NAMESPACE_URL, f'{source_id}:{doc_id}').hex}"),
                    source_unit=source_unit,
                    prior_unit_revision=prior_unit_revision,
                    prior_observation_revisions=prior_observation_revisions,
                    reason="not_returned_by_authoritative_snapshot",
                )
                target_revision = tombstone.source_unit_revisions[0]
                lifecycle_result = await self.memory_engine.apply_projected_tombstone(
                    projection=tombstone,
                    doc_id=doc_id,
                    reason=AUTHORITATIVE_SOURCE_UNIT_REMOVAL_REASON,
                    lifecycle_cycle_id=lifecycle_cycle_id,
                    expected_source_activity_epoch=expected_source_activity_epoch,
                )
                await self.db.record_source_projection(
                    tombstone,
                    expected_source_activity_epoch=expected_source_activity_epoch,
                )
                if lifecycle_result["can_delete_document"]:
                    await self.memory_store.delete_projected_document(
                        doc_id,
                        deletion_context={
                            "deletion_kind": "source_absence",
                            "reason": "not_returned_by_latest_successful_crawl",
                            "source_filter_summary": source_filter_summary,
                            "source_unit_id": source_unit.id,
                            "target_unit_revision_id": target_revision.id,
                        },
                    )
                else:
                    logger.warning(
                        "Retained removed document %s while %d lifecycle review(s) are pending",
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
                    FailedDoc(
                        doc_id=doc_id,
                        title=title,
                        error=str(e),
                    )
                )

        return deleted_count, failed_deletions

    # ==================================================================
    # Private: helpers
    # ==================================================================

    async def _get_indexed_doc_ids(self, source_id: str) -> set[str]:
        """Get all doc_ids currently indexed for a source."""
        doc_ids: set[str] = set()
        try:
            async with self.db.db.execute(
                "SELECT doc_id FROM documents WHERE source = ?",
                (source_id,),
            ) as cursor:
                async for row in cursor:
                    doc_ids.add(row[0])
        except Exception as e:
            logger.error(
                "Failed to fetch indexed doc_ids for %s: %s",
                source_id,
                e,
            )
        return doc_ids

    async def _count_missing_pdf_uris(self, source_id: str) -> int:
        async with self.db.db.execute(
            """SELECT COUNT(*)
               FROM documents d
               JOIN sources s ON s.id = d.source
               WHERE d.source = ?
                 AND s.type = 'confluence'
                 AND d.normalized_content_uri IS NOT NULL
                 AND d.normalized_content_uri <> ''
                 AND (d.pdf_content_uri IS NULL OR d.pdf_content_uri = '')""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return int(row[0] if row else 0)

    async def _restore_document_processing_snapshot(
        self,
        *,
        doc_id: str,
        existing_doc: DocumentRecord | None,
    ) -> None:
        async with self._db_lock:
            if existing_doc:
                await self.db.restore_document_snapshot(
                    existing_doc,
                    require_configured_source=True,
                )
            else:
                await self.db.delete_document(doc_id)

    async def _insert_changelog(self, entry: ChangelogEntry) -> None:
        """Insert a changelog entry into the database.

        The DB class does not expose an add_changelog method, so we write
        directly to the changelog table.
        """
        try:
            await self.db.db.execute(
                """INSERT INTO changelog (
                    doc_id, change_type, previous_version, current_version,
                    content_diff, ai_change_summary, detected_at, title, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.doc_id,
                    entry.change_type,
                    entry.previous_version,
                    entry.current_version,
                    entry.content_diff,
                    entry.ai_change_summary,
                    entry.detected_at.isoformat(),
                    entry.title,
                    entry.source,
                ),
            )
            await self.db.db.commit()
        except Exception as e:
            logger.error(
                "Failed to insert changelog for %s: %s",
                entry.doc_id,
                e,
            )
