"""Gene Sync Orchestrator — the heart of MemForge's data pipeline.

Coordinates the full lifecycle of syncing data from a gene (external source)
into the memory layer:

    authenticate -> discover -> fetch -> normalize -> store ->
    enrich (Call 1) -> extract memories (Call 2) -> persist -> detect deletions

Concurrency is managed via an asyncio.Semaphore (for LLM/embedding calls)
and an asyncio.Lock (for SQLite writes). Each content item is processed
independently with retry logic and per-item error isolation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import tiktoken

from memforge.models import (
    ChangelogEntry,
    DocumentMetadata,
    DocumentRecord,
    Entity,
    EnrichmentResult,
    FailedDoc,
    MemoryExtractionResult,
    SyncState,
    content_hash as compute_content_hash,
)
from memforge.retrieval.document_index import DocumentVectorIndex
from memforge.pipeline.sync_memory import SyncMemoryObserver

from memforge.pipeline.document_units import ExtractionContextPacker, UnitizationPolicy, unitize_markdown
from memforge.pipeline.document_update import DocumentUpdatePlan, plan_document_update
from memforge.memory.index_payloads import (
    document_embedding_text,
    embedding_text_hash,
)
from memforge.llm.providers import is_litellm_provider_model
from memforge.memory.project_resolver import resolve_project_key
from memforge.memory.visibility_policy import default_visibility

if TYPE_CHECKING:
    from memforge.genes.base import Gene
    from memforge.memory.engine import MemoryEngine
    from memforge.memory.store import MemoryStore
    from memforge.models import ContentItem, Memory
    from memforge.pipeline.enricher import Enricher
    from memforge.pipeline.memory_extractor import MemoryExtractor
    from memforge.pipeline.source_support_detector import SourceSupportDetector
    from memforge.storage.database import Database
    from memforge.storage.document_store import DocumentStore

logger = logging.getLogger(__name__)

__all__ = [
    "DocumentLifecycleAdmission",
    "ExtractionWorkPool",
    "GeneSyncOrchestrator",
    "SyncMemoryObserver",
    "get_process_document_lifecycle_admission",
]

DEFAULT_INCREMENTAL_SYNC_OVERLAP = timedelta(minutes=10)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    # Keep this list aligned with admin-ui/src/components/admin/syncFailureDetails.ts.
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

        orchestrator = GeneSyncOrchestrator(db, doc_store, enricher, ...)
        state = await orchestrator.sync_gene(gene, source_name, source_id)

    Each content item discovered by the gene flows through:
        1. fetch -> normalize -> store on disk
        2. content hash comparison (skip if unchanged)
        3. Call 1: enricher.enrich_document() -> entity resolution
        4. Call 2: memory_extractor.extract_memories() -> dedup + persist
        5. Document record + embedding upsert
        6. Changelog entry

    Concurrency is bounded by a semaphore (for LLM/embedding API calls)
    and a lock (for SQLite write serialization).
    """

    def __init__(
        self,
        db: Database,
        doc_store: DocumentStore,
        enricher: Enricher,
        memory_extractor: MemoryExtractor,
        memory_engine: MemoryEngine,
        memory_store: MemoryStore,
        vector_store: Any | None = None,
        embed_cfg: dict | None = None,
        source_support_detector: SourceSupportDetector | None = None,
        max_concurrent: int = 3,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
        memory_observer: SyncMemoryObserver | None = None,
    ) -> None:
        self.db = db
        self.doc_store = doc_store
        self.enricher = enricher
        self.memory_extractor = memory_extractor
        self.memory_engine = memory_engine
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.document_index = DocumentVectorIndex(vector_store)
        self.embed_cfg = embed_cfg
        self.source_support_detector = source_support_detector
        self.max_concurrent = max(1, max_concurrent)
        self.extraction_pool = extraction_pool
        self.document_lifecycle_admission = document_lifecycle_admission
        self.memory_observer = memory_observer

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

        Returns
        -------
        SyncState
            Final sync result with counts and error details.
        """
        run_id = uuid.uuid4().hex[:12]
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

        try:
            # ----------------------------------------------------------
            # Step 0: Authenticate
            # ----------------------------------------------------------
            await gene.authenticate()
            logger.info("Gene %s authenticated successfully", source_name)

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
            last_sync_time = None if force_full_sync else (existing_state.last_sync_at if existing_state else None)
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

            async for item in gene.discover(since=last_sync_time):
                items.append(item)
                crawled_doc_ids.add(item.item_id)

            logger.info(
                "Discovered %d content items from %s (since=%s)",
                len(items),
                source_name,
                last_sync_time.isoformat() if last_sync_time else "full sync",
            )
            self._memory_sample(
                "after_discovery",
                source_id=source_id,
                run_id=run_id,
                item_count=len(items),
                indexed_doc_count=len(indexed_doc_ids),
                full_sync=last_sync_time is None,
            )

            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "current": 0,
                        "total": len(items),
                        "title": None,
                    }
                )

            # ----------------------------------------------------------
            # Step 4: Process items concurrently (with retry + error isolation)
            # ----------------------------------------------------------
            progress_counter = 0
            docs_updated_counter = 0
            memories_extracted_counter = 0
            item_semaphore = asyncio.Semaphore(self._source_parallelism_limit())

            async def _process_one(item: ContentItem) -> dict:
                """Process a single item with retry logic and error isolation."""
                nonlocal docs_updated_counter, memories_extracted_counter, progress_counter
                stats = {
                    "processed": False,
                    "updated": False,
                    "memories_extracted": 0,
                    "memories_corroborated": 0,
                    "failed": False,
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
                                    "total": len(items),
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
                                force_reprocess=force_full_sync,
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
                                await asyncio.sleep(delay)
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
                                "total": len(items),
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
                            "total": len(items),
                            "title": item.title,
                            "docs_updated": docs_updated_counter,
                            "memories_extracted": memories_extracted_counter,
                        }
                    )

                return stats

            tasks = [_process_one(item) for item in items]
            results = await asyncio.gather(*tasks)

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

            # ----------------------------------------------------------
            # Step 5: Detect deletions (only on full sync, not incremental)
            # ----------------------------------------------------------
            # When since= is set, the gene only returns CHANGED pages.
            # Pages not returned aren't deleted — they're just unchanged.
            # Only run deletion detection on full syncs (since=None).
            deleted_count = 0
            is_full_sync = last_sync_time is None

            if is_full_sync:
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "detecting_deletions",
                            "current": len(items),
                            "total": len(items),
                            "title": None,
                        }
                    )

                deleted_count, deletion_failures = await self._detect_deletions(
                    source_id=source_id,
                    source_name=source_name,
                    indexed_doc_ids=indexed_doc_ids,
                    crawled_doc_ids=crawled_doc_ids,
                    source_filter_summary=_source_filter_summary(gene, last_sync_time),
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
                    "Skipping deletion detection for incremental sync: returned %d changed docs from %d indexed docs",
                    len(crawled_doc_ids),
                    len(indexed_doc_ids),
                )

            # ----------------------------------------------------------
            # Step 6: Update source doc_count
            # ----------------------------------------------------------
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

        try:
            await self.db.upsert_sync_state(sync_state)
        except Exception as e:
            logger.error("Failed to upsert sync state: %s", e)

        # Record in sync_history for audit trail
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
                    "current": len(items) if "items" in dir() else 0,
                    "total": len(items) if "items" in dir() else 0,
                    "title": None,
                }
            )

        item_count = len(items) if "items" in locals() else 0
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
            item_count=item_count,
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
    ) -> dict:
        doc_id = item.item_id
        self._memory_sample("document_wait_start", source_id=source_id, run_id=run_id, doc_id=doc_id)
        lifecycle_error: Exception | None = None
        lifecycle_ok = False
        async with self._document_lifecycle_slot(source_id, doc_id):
            self._memory_sample("document_lifecycle_enter", source_id=source_id, run_id=run_id, doc_id=doc_id)
            try:
                result = await self._process_item_admitted(
                    gene=gene,
                    item=item,
                    source_name=source_name,
                    source_id=source_id,
                    run_id=run_id,
                    progress_callback=progress_callback,
                    force_reprocess=force_reprocess,
                )
                lifecycle_ok = True
                return result
            except Exception as exc:
                lifecycle_error = exc
                raise
            finally:
                self._memory_sample(
                    "document_lifecycle_exit",
                    source_id=source_id,
                    run_id=run_id,
                    doc_id=doc_id,
                    ok=lifecycle_ok,
                    error=lifecycle_error,
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
    ) -> dict:
        """Process a single content item through the full pipeline.

        Steps:
            1. Fetch raw content
            2. Normalize to markdown
            3. Store raw + normalized on disk
            4. Content hash comparison (skip if unchanged)
            5. Count tokens
            6. Call 1: Enrich document -> entity resolution
            7. Call 2: Extract memories -> dedup + persist
            8. Upsert document record + metadata in DB
            9. Upsert document embedding in vector store
            10. Record changelog

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

        if not markdown_body or not markdown_body.strip():
            logger.warning("Empty normalized content for %s, skipping", doc_id)
            return stats

        # ------------------------------------------------------------------
        # 4. Content hash
        # ------------------------------------------------------------------
        new_hash = compute_content_hash(markdown_body)

        async with self._db_lock:
            existing_hash = await self.db.get_content_hash(doc_id)
            existing_doc = await self.db.get_document(doc_id)
            existing_metadata = await self.db.get_metadata(doc_id) if existing_hash == new_hash else None

        previous_markdown = (
            self._read_previous_normalized_content(existing_doc)
            if existing_hash is not None and existing_hash != new_hash
            else None
        )

        # ------------------------------------------------------------------
        # 3. Store raw + normalized on disk
        # ------------------------------------------------------------------
        raw_uri = self.doc_store.store_raw(
            source_name=source_name,
            title=item.title,
            content=raw.body,
            content_type=raw.content_type,
        )
        norm_uri = self.doc_store.store_normalized(
            source_name=source_name,
            title=item.title,
            markdown=markdown_body,
        )
        self._memory_sample(
            "after_raw_store",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_bytes=len(raw.body),
            content_chars=len(markdown_body),
        )

        source_type = gene.metadata().name
        source_shape = gene.metadata().data_shape
        requires_pdf_uri = gene.requires_pdf_artifact(
            item=item,
            existing_doc=existing_doc,
            existing_hash=existing_hash,
            new_hash=new_hash,
        )

        # ------------------------------------------------------------------
        # 3b. Export PDF (if gene supports it)
        # ------------------------------------------------------------------
        pdf_uri: str | None = None
        if hasattr(gene, "fetch_pdf"):
            try:
                pdf_bytes = await gene.fetch_pdf(item)
            except Exception as e:
                if requires_pdf_uri:
                    raise RuntimeError(f"Confluence PDF export failed for {item.title}: {e}") from e
                logger.warning("PDF export failed for %s: %s", item.title, e)
                pdf_bytes = None
            if pdf_bytes and len(pdf_bytes) > 100:
                pdf_uri = self.doc_store.store_raw(
                    source_name=source_name,
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

        vector_current = await self._document_vector_is_current(
            doc_id=doc_id,
            content_hash=new_hash,
            version=item.version,
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
            raw_content_type=raw.content_type,
            normalized_content_uri=norm_uri,
            pdf_content_uri=pdf_uri,
            last_synced=now,
            client=normalized.source_semantics.get("client") or None,
        )

        if existing_hash == new_hash and existing_metadata and not force_reprocess:
            if existing_doc and not doc_record.pdf_content_uri:
                doc_record.pdf_content_uri = existing_doc.pdf_content_uri

            if vector_current:
                async with self._db_lock:
                    await self.db.upsert_document(doc_record)
                logger.debug("Skipping %s (content unchanged)", doc_id)
                return stats

            if self.document_index.enabled and not self._has_embedding_config():
                raise RuntimeError(f"Cannot repair stale document vector for {doc_id}: embedding config is missing")

            async with self._db_lock:
                await self.db.upsert_document(doc_record)

            document_vector_snapshot = await self._document_vector_snapshot(doc_id)
            try:
                await self._upsert_document_embedding(
                    doc_id=doc_id,
                    metadata=existing_metadata,
                    source_id=source_id,
                    source_type=source_type,
                    space_or_project=item.space_or_project,
                    token_count=token_count,
                    content_hash=new_hash,
                    version=item.version,
                )
            except Exception:
                await self._restore_document_vector_snapshot(doc_id, document_vector_snapshot)
                async with self._db_lock:
                    if existing_doc:
                        await self.db.restore_document_snapshot(existing_doc)
                    else:
                        await self.db.delete_document(doc_id)
                raise

            if progress_callback:
                progress_callback(
                    {
                        "phase": "processing",
                        "event": "document_processed",
                        "title": item.title,
                    }
                )
            logger.info("Repaired stale document vector for unchanged %s (%s)", item.title, doc_id)
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
        document_vector_snapshot = await self._document_vector_snapshot(doc_id)
        async with self._db_lock:
            await self.db.upsert_document(doc_record)

        # ------------------------------------------------------------------
        # 6. Call 1: Enrich document (under semaphore)
        # ------------------------------------------------------------------
        enrichment: EnrichmentResult

        async with self._heavy_work_slot(source_id):
            enrichment = await self.enricher.enrich_document(
                doc_id=doc_id,
                content=markdown_body,
                source_type=source_type,
            )

        logger.debug(
            "Enrichment for %s: %d entities, %d tags, doc_type=%s",
            doc_id,
            len(enrichment.entities),
            len(enrichment.tags),
            enrichment.doc_type,
        )
        self._memory_sample(
            "after_enrich",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            entity_count=len(enrichment.entities),
            tag_count=len(enrichment.tags),
            content_chars=len(markdown_body),
        )

        # ------------------------------------------------------------------
        # 6b. Upsert document embedding before memory mutations
        # ------------------------------------------------------------------
        index_metadata = self._document_metadata_from_enrichment(
            doc_id=doc_id,
            enrichment=enrichment,
            enriched_at=now,
        )
        if self.document_index.enabled and not self._has_embedding_config():
            raise RuntimeError(f"Cannot index document vector for {doc_id}: embedding config is missing")

        if self.document_index.enabled:
            try:
                await self._upsert_document_embedding(
                    doc_id=doc_id,
                    metadata=index_metadata,
                    source_id=source_id,
                    source_type=source_type,
                    space_or_project=item.space_or_project,
                    token_count=token_count,
                    content_hash=new_hash,
                    version=item.version,
                )
            except Exception:
                await self._restore_document_vector_snapshot(doc_id, document_vector_snapshot)
                async with self._db_lock:
                    if existing_doc:
                        await self.db.restore_document_snapshot(existing_doc)
                    else:
                        await self.db.delete_document(doc_id)
                raise

        # ------------------------------------------------------------------
        # 6c. Process enrichment: resolve entities, insert aliases
        # ------------------------------------------------------------------
        entity_ids = await self.memory_engine.process_enrichment(
            doc_id=doc_id,
            enrichment=enrichment,
        )

        # ------------------------------------------------------------------
        # 6d. Get canonical entity names for the resolved IDs
        # ------------------------------------------------------------------
        entity_names: list[str] = []
        for eid in entity_ids:
            name = await self._get_entity_canonical_name(eid)
            if name:
                entity_names.append(name)

        # ------------------------------------------------------------------
        # 6e. Get existing memories for those entities (context for Call 2)
        # ------------------------------------------------------------------
        existing_memories = []
        seen_memory_ids: set[str] = set()
        for eid in entity_ids:
            try:
                entity_memories = await self.db.get_memories_by_entity(eid)
                for mem in entity_memories:
                    if mem.id not in seen_memory_ids and mem.status == "active":
                        existing_memories.append(mem)
                        seen_memory_ids.add(mem.id)
            except Exception as e:
                logger.warning(
                    "Failed to fetch memories for entity %d: %s",
                    eid,
                    e,
                )

        # Cap existing memories to avoid excessive token usage in Call 2
        existing_memories = existing_memories[:50]

        # ------------------------------------------------------------------
        # 7. Call 2: Extract memories (under semaphore)
        # ------------------------------------------------------------------
        extraction_result = await self._extract_for_document_update(
            update_plan=update_plan,
            markdown_body=markdown_body,
            source_type=source_type,
            doc_type=enrichment.doc_type,
            entity_names=entity_names,
            existing_memories=existing_memories,
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
                document_vector_snapshot=document_vector_snapshot,
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
            existing_memory_count=len(existing_memories),
            entity_count=len(entity_ids),
            content_chars=len(markdown_body),
        )

        # ------------------------------------------------------------------
        # 7b. Build metadata once enrichment has resolved canonical entities
        # ------------------------------------------------------------------
        meta_entities = [
            Entity(
                id=eid,
                canonical_name=name,
                tags=[],
                display_name=name,
            )
            for eid, name in zip(entity_ids, entity_names)
        ]
        doc_metadata = DocumentMetadata(
            doc_id=doc_id,
            summary=enrichment.summary,
            tags=enrichment.tags,
            entities=meta_entities,
            doc_type=enrichment.doc_type,
            complexity=enrichment.complexity,
            enriched_at=now,
        )

        # ------------------------------------------------------------------
        # 8. Process memories: reconcile (updates) or insert (new docs)
        # ------------------------------------------------------------------
        memory_context = self._memory_store_context(
            run_id=run_id,
            source_id=source_id,
            doc_id=doc_id,
        )
        # The gene exposes the uploader as a write-time hint on source_semantics.
        # The sync pipeline forwards it to the memory engine so the new memory
        # carries the uploader's owner_user_id at persistence time. This stays a
        # write-time signal only: read-time visibility is decided by the access
        # predicate, never by this hint.
        uploader_user_id = normalized.source_semantics.get("uploader_user_id")
        repo_identifier = normalized.source_semantics.get("repo_identifier")
        source_updated_at = _source_updated_at_for_item(item, normalized.source_semantics)
        if change_type == "updated":
            memory_stats = await self.memory_engine.reconcile_and_persist(
                doc_id=doc_id,
                raw_memories=raw_memories,
                source_type=source_type,
                doc_type=enrichment.doc_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_ids=entity_ids,
                document_content=markdown_body,
                update_mode=update_plan.mode if update_plan else "full_document",
                changed_hunks=update_plan.changed_hunks if update_plan else None,
                update_plan_stats=self._document_update_plan_stats(update_plan),
                audit_context=memory_context,
                user_id=uploader_user_id,
                source_updated_at=source_updated_at,
            )
            stats["memories_extracted"] = memory_stats.get("added", 0)
            stats["memories_corroborated"] = memory_stats.get("updated", 0)
        else:
            memory_stats = await self.memory_engine.process_memories(
                doc_id=doc_id,
                raw_memories=raw_memories,
                source_type=source_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_ids=entity_ids,
                audit_context=memory_context,
                user_id=uploader_user_id,
                source_updated_at=source_updated_at,
            )
            stats["memories_extracted"] = memory_stats.get("inserted", 0)
            stats["memories_corroborated"] = memory_stats.get("corroborated", 0)

        self._memory_sample(
            "after_memory_engine",
            source_id=source_id,
            run_id=run_id,
            doc_id=doc_id,
            raw_memory_count=len(raw_memories),
            memories_extracted=stats["memories_extracted"],
            memories_corroborated=stats["memories_corroborated"],
            entity_count=len(entity_ids),
        )

        if not extraction_result.error_type and self.source_support_detector:
            writer_visibility, writer_owner_user_id = default_visibility(
                source_type,
                user_id=uploader_user_id,
            )
            async with self._heavy_work_slot(source_id):
                support_stats = await self.source_support_detector.detect_and_persist(
                    doc_id=doc_id,
                    source_type=source_type,
                    document=markdown_body,
                    entity_ids=entity_ids,
                    project_key=project_key,
                    db=self.db,
                    memory_store=self.memory_store,
                    writer_visibility=writer_visibility,
                    writer_owner_user_id=writer_owner_user_id,
                    writer_project_key=project_key,
                    source_updated_at=source_updated_at,
                )
            stats["memory_supports_added"] = support_stats.get("added", 0)
            stats["memory_supports_updated"] = support_stats.get("updated", 0)
            stats["memory_supports_removed"] = support_stats.get("removed_stale", 0)
            self._memory_sample(
                "after_source_support",
                source_id=source_id,
                run_id=run_id,
                doc_id=doc_id,
                memory_supports_added=stats["memory_supports_added"],
                memory_supports_updated=stats["memory_supports_updated"],
                memory_supports_removed=stats["memory_supports_removed"],
                entity_count=len(entity_ids),
            )

        async with self._db_lock:
            await self.db.upsert_metadata(doc_metadata)

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
                else f"Updated: {item.title} - {enrichment.summary[:200]}"
            ),
            detected_at=now,
            title=item.title,
            source=source_id,
        )

        async with self._db_lock:
            await self._insert_changelog(changelog_entry)

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
        update_plan: DocumentUpdatePlan | None,
        markdown_body: str,
        source_type: str,
        doc_type: str,
        entity_names: list[str],
        existing_memories: list[Memory],
        doc_id: str,
        source_id: str,
        run_id: str | None,
        document_title: str,
        document_url: str,
    ) -> MemoryExtractionResult:
        """Run full extraction or diff-guided extraction for a document."""
        if (
            update_plan
            and update_plan.mode == "diff_guided"
            and hasattr(self.memory_extractor, "extract_memory_changes")
        ):
            try:
                same_document_memories = await self._get_existing_document_memories(doc_id)
                async with self._heavy_work_slot(source_id):
                    result = await self.memory_extractor.extract_memory_changes(
                        changed_hunks=update_plan.changed_hunks or "",
                        updated_document=markdown_body,
                        source_type=source_type,
                        doc_type=doc_type,
                        entities=entity_names,
                        existing_memories=same_document_memories,
                    )
                await self._record_memory_extraction_result(
                    mode=update_plan.mode,
                    plan=update_plan,
                    doc_id=doc_id,
                    source_id=source_id,
                    run_id=run_id,
                    result=result,
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
                entity_names=entity_names,
                existing_memories=existing_memories,
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
                entities=entity_names,
                existing_memories=existing_memories,
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

    async def _extract_full_document_units(
        self,
        *,
        markdown_body: str,
        source_type: str,
        doc_type: str,
        entity_names: list[str],
        existing_memories: list[Memory],
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
                entities=entity_names,
            )
            async with unit_semaphore:
                async with self._heavy_work_slot(source_id):
                    return await self.memory_extractor.extract_unit_memories(
                        context,
                        doc_type=doc_type,
                        existing_memories=existing_memories,
                    )

        results = await asyncio.gather(*(extract_one(unit) for unit in units))

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

    # ==================================================================
    # Private: deletion detection
    # ==================================================================

    async def _detect_deletions(
        self,
        source_id: str,
        source_name: str,
        indexed_doc_ids: set[str],
        crawled_doc_ids: set[str],
        source_filter_summary: str | None,
    ) -> tuple[int, list[FailedDoc]]:
        """Detect and handle documents deleted from the source.

        For each deleted document:
        - Retire memories that were sourced only from that document
        - Record a changelog entry
        - Delete the document record and its files

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

                # Clean up stored files
                if existing_doc:
                    self.doc_store.delete_document_files(
                        source_name=source_name,
                        title=existing_doc.title,
                    )

                # Delete the document record and hide memories left without support.
                await self.memory_store.delete_document(
                    doc_id,
                    deletion_context={
                        "deletion_kind": "source_absence",
                        "reason": "not_returned_by_latest_successful_crawl",
                        "source_filter_summary": source_filter_summary,
                    },
                )

                logger.info("Deleted document %s (%s)", doc_id, title)
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

    async def _retire_orphaned_memories(
        self,
        doc_id: str,
        source_type: str,
    ) -> int:
        """Retire memories that are sourced only from the given document.

        Memories with other source documents remain active. Memories with
        no remaining sources are marked as retired.

        Returns the count of retired memories.
        """
        retired_count = 0

        try:
            # Get all memories linked to this document, including corroborated support.
            memories = await self.db.get_memories_by_source_doc(doc_id, support_kind=None)

            for memory in memories:
                # Check if this memory has sources beyond the deleted document
                sources = await self.db.get_memory_sources(memory.id)
                other_sources = [s for s in sources if s.doc_id != doc_id]

                if not other_sources:
                    await self.memory_store.retire_memory(memory.id, reason="source_deleted")
                    retired_count += 1
                    logger.debug(
                        "Retired memory %s (sole source %s deleted)",
                        memory.id,
                        doc_id,
                    )

        except Exception as e:
            logger.error(
                "Error retiring memories for doc %s: %s",
                doc_id,
                e,
            )

        return retired_count

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

    async def _get_entity_canonical_name(self, entity_id: int) -> str | None:
        """Look up the canonical name for an entity by its ID."""
        try:
            async with self.db.db.execute(
                "SELECT canonical_name FROM entities WHERE id = ?",
                (entity_id,),
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(
                "Failed to get canonical name for entity %d: %s",
                entity_id,
                e,
            )
            return None

    async def _upsert_document_embedding(
        self,
        doc_id: str,
        metadata: DocumentMetadata,
        source_id: str,
        source_type: str,
        space_or_project: str,
        token_count: int,
        content_hash: str,
        version: str | None,
    ) -> None:
        """Compute and upsert a document-level embedding in the vector store."""
        if not self.document_index.enabled:
            return
        if not self._has_embedding_config():
            raise RuntimeError(f"Cannot index document vector for {doc_id}: embedding config is missing")

        try:
            from memforge.retrieval.embeddings import embed_texts

            # Keep document vectors independent of entity resolution so indexing can
            # happen before any durable memory-layer mutations.
            embedding_text = document_embedding_text(metadata)

            async with self._heavy_work_slot(source_id):
                try:
                    vectors = await asyncio.to_thread(
                        embed_texts,
                        [embedding_text],
                        self.embed_cfg["base_url"],
                        self.embed_cfg["api_key"],
                        self.embed_cfg["model"],
                    )
                except Exception as e:
                    if _is_provider_unreachable(str(e)):
                        raise RuntimeError(f"Embedding provider unreachable: {e}") from e
                    raise

            await asyncio.to_thread(
                self.document_index.upsert,
                doc_id=doc_id,
                embedding=vectors[0],
                document=embedding_text,
                metadata={
                    "source": source_id,
                    "source_type": source_type,
                    "doc_type": metadata.doc_type,
                    "space": space_or_project,
                    "token_count": token_count,
                    "content_hash": content_hash,
                    "version": version or "",
                    "embedding_text_hash": embedding_text_hash(embedding_text),
                },
            )
            logger.debug("Upserted document embedding for %s", doc_id)

        except Exception as e:
            logger.error(
                "Vector indexing failed for %s: %s",
                doc_id,
                e,
            )
            raise

    async def _document_vector_is_current(
        self,
        *,
        doc_id: str,
        content_hash: str,
        version: str | None,
    ) -> bool:
        if not self.document_index.enabled:
            return True
        try:
            return await asyncio.to_thread(
                self.document_index.is_current,
                doc_id,
                content_hash=content_hash,
                version=version,
            )
        except Exception:
            logger.warning("Document vector freshness check failed for %s", doc_id)
            return False

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

    async def _document_vector_snapshot(self, doc_id: str) -> dict | None:
        if not self.document_index.enabled:
            return None
        return await asyncio.to_thread(self.document_index.snapshot, doc_id)

    async def _restore_document_vector_snapshot(
        self,
        doc_id: str,
        snapshot: dict | None,
    ) -> None:
        if not self.document_index.enabled:
            return
        await asyncio.to_thread(self.document_index.restore, doc_id, snapshot)

    async def _restore_document_processing_snapshot(
        self,
        *,
        doc_id: str,
        existing_doc: DocumentRecord | None,
        document_vector_snapshot: dict | None,
    ) -> None:
        await self._restore_document_vector_snapshot(doc_id, document_vector_snapshot)
        async with self._db_lock:
            if existing_doc:
                await self.db.restore_document_snapshot(existing_doc)
            else:
                await self.db.delete_document(doc_id)

    def _has_embedding_config(self) -> bool:
        if not self.embed_cfg:
            return False
        model = str(self.embed_cfg.get("model") or "").strip()
        if not model:
            return False
        if is_litellm_provider_model(model):
            return True
        return all(str(self.embed_cfg.get(key) or "").strip() for key in ("base_url", "api_key", "model"))

    def _document_metadata_from_enrichment(
        self,
        *,
        doc_id: str,
        enrichment: EnrichmentResult,
        enriched_at: datetime,
    ) -> DocumentMetadata:
        return DocumentMetadata(
            doc_id=doc_id,
            summary=enrichment.summary,
            tags=enrichment.tags,
            entities=[
                Entity(
                    id=0,
                    canonical_name=entity.name,
                    tags=entity.tags or ([entity.type] if entity.type != "unknown" else []),
                    display_name=entity.name,
                )
                for entity in enrichment.entities
            ],
            doc_type=enrichment.doc_type,
            complexity=enrichment.complexity,
            enriched_at=enriched_at,
        )

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
