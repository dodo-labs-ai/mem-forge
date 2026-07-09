"""Shared runtime wiring for sync startup paths."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from memforge.config import AppConfig
from memforge.auth import browser_session
from memforge.genes import GENE_REGISTRY, create_gene
from memforge.llm.providers import is_litellm_provider_model
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.health import MemoryIndexHealthChecker, MemoryIndexHealthReport
from memforge.memory.store import MemoryStore
from memforge.models import SyncState
from memforge.pipeline.enricher import Enricher
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.source_support_detector import SourceSupportDetector
from memforge.pipeline.sync_memory import SyncMemoryObserver
from memforge.pipeline.sync import (
    DocumentLifecycleAdmission,
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    get_process_document_lifecycle_admission,
)
from memforge.retrieval.document_index import DocumentVectorIndex
from memforge.retrieval.embeddings import get_chroma_collection
from memforge.source_secrets import decrypt_source_config_for_runtime, source_secret_fields
from memforge.storage.document_store import LocalDocumentStore
from memforge.storage.adapters.sqlite import build_sqlite_adapters

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)


class SyncAlreadyRunningError(RuntimeError):
    """Raised when a source already has an active sync task."""


class SourcePausedError(RuntimeError):
    """Raised when sync is requested for a paused source."""


@dataclass
class RuntimeHealthComponent:
    status: str
    detail: str | None = None
    payload: dict[str, Any] | None = None


@dataclass
class RuntimeHealthReport:
    status: str
    database: RuntimeHealthComponent
    vector_store: RuntimeHealthComponent
    index_consistency: RuntimeHealthComponent | None = None
    audit_failures: RuntimeHealthComponent | None = None
    genes: dict[str, RuntimeHealthComponent] = field(default_factory=dict)


@dataclass
class EffectiveLlmConfig:
    enrichment_model: str
    enrichment_base_url: str
    enrichment_api_key: str
    request_timeout_s: float
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str


@dataclass
class SyncRuntime:
    db: "Database"
    config: AppConfig
    doc_store: LocalDocumentStore
    enricher: Enricher
    memory_extractor: MemoryExtractor
    memory_store: MemoryStore
    memory_engine: MemoryEngine
    vector_store: Any
    embed_cfg: dict[str, str]
    structured_llm_client: LiteLlmStructuredClient | None
    llm_model: str
    source_support_detector: SourceSupportDetector | None
    extraction_pool: ExtractionWorkPool | None = None
    document_lifecycle_admission: DocumentLifecycleAdmission | None = None
    memory_observer: SyncMemoryObserver | None = None
    orchestrator_factory: Callable[["SyncRuntime"], GeneSyncOrchestrator] | None = None

    def orchestrator(self) -> GeneSyncOrchestrator:
        if self.orchestrator_factory is not None:
            return self.orchestrator_factory(self)
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=self.doc_store,
            enricher=self.enricher,
            memory_extractor=self.memory_extractor,
            memory_engine=self.memory_engine,
            memory_store=self.memory_store,
            vector_store=self.vector_store,
            embed_cfg=self.embed_cfg,
            source_support_detector=self.source_support_detector,
            max_concurrent=self.config.llm.enrichment_max_concurrent,
            extraction_pool=self.extraction_pool,
            document_lifecycle_admission=self.document_lifecycle_admission,
            memory_observer=self.memory_observer,
        )


class RuntimeProvider(Protocol):
    """Runtime construction seam for admin apps with non-SQLite stores."""

    def build_adapters(
        self,
        db: "Database",
        memory_collection: Any,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any: ...

    async def build_search_engine(
        self,
        db: "Database",
        config: AppConfig,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any: ...

    async def check_health(self, db: "Database", config: AppConfig) -> RuntimeHealthReport: ...

    async def build_sync_runtime(
        self,
        db: "Database",
        config: AppConfig,
        *,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
    ) -> SyncRuntime: ...

    async def run_source_sync(
        self,
        db: "Database",
        config: AppConfig,
        source: dict,
        runtime: SyncRuntime | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        force_full_sync: bool = False,
    ) -> SyncState: ...


class DefaultRuntimeProvider:
    """Default SQLite-backed runtime provider."""

    def build_adapters(
        self,
        db: "Database",
        memory_collection: Any,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any:
        return build_sqlite_adapters(db, memory_collection, audit_logger=audit_logger)

    async def build_search_engine(
        self,
        db: "Database",
        config: AppConfig,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any:
        return await build_search_engine(
            db=db,
            config=config,
            audit_logger=audit_logger,
        )

    async def build_sync_runtime(
        self,
        db: "Database",
        config: AppConfig,
        *,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
    ) -> SyncRuntime:
        return await build_sync_runtime(
            db=db,
            config=config,
            extraction_pool=extraction_pool,
            document_lifecycle_admission=document_lifecycle_admission,
        )

    async def check_health(self, db: "Database", config: AppConfig) -> RuntimeHealthReport:
        return await check_runtime_health(db, config)

    async def run_source_sync(
        self,
        db: "Database",
        config: AppConfig,
        source: dict,
        runtime: SyncRuntime | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        force_full_sync: bool = False,
    ) -> SyncState:
        return await run_source_sync(
            db=db,
            config=config,
            source=source,
            runtime=runtime,
            progress_callback=progress_callback,
            force_full_sync=force_full_sync,
        )


async def get_effective_llm_config(db: "Database", config: AppConfig) -> EffectiveLlmConfig:
    db_llm = await db.get_llm_config()

    def value(key: str, fallback: str) -> str:
        if not db_llm:
            return fallback
        return db_llm.get(key) or fallback

    return EffectiveLlmConfig(
        enrichment_model=value("enrichment_model", config.llm.enrichment_model),
        enrichment_base_url=value("enrichment_base_url", config.llm.enrichment_base_url),
        enrichment_api_key=value("enrichment_api_key", config.llm.enrichment_api_key),
        request_timeout_s=config.llm.request_timeout_s,
        embedding_model=value("embedding_model", config.llm.embedding_model),
        embedding_base_url=value("embedding_base_url", config.llm.embedding_base_url),
        embedding_api_key=value("embedding_api_key", config.llm.embedding_api_key),
    )


AUDIT_HEALTH_FAILURE_EVENTS = (
    "source_support_verification_failed",
    "contradiction_detection_failed",
    "reconciliation_failed",
    "reconciliation_action_failed",
    "index_operation_failed",
    "review_resolution_failed",
)
AUDIT_HEALTH_WINDOW_HOURS = 24


async def check_runtime_health(db: "Database", config: AppConfig) -> RuntimeHealthReport:
    """Run the default SQLite/Chroma health checks for the OSS runtime."""
    overall = "healthy"

    database = RuntimeHealthComponent(status="ok")
    try:
        memory_count = await db.count_memories()
        database.detail = f"{memory_count} memories"
    except Exception as exc:
        database = RuntimeHealthComponent(status="error", detail=str(exc))
        overall = "degraded"

    vector_store = RuntimeHealthComponent(status="ok")
    try:
        import chromadb

        chroma_path = config.storage.chroma_path
        if Path(chroma_path).exists():
            client = chromadb.PersistentClient(path=chroma_path)
            collections = client.list_collections()
            vector_store.detail = f"{len(collections)} collection(s)"
        else:
            vector_store = RuntimeHealthComponent(
                status="not_configured",
                detail="ChromaDB path does not exist",
            )
    except ImportError:
        vector_store = RuntimeHealthComponent(
            status="not_available",
            detail="chromadb not installed",
        )
    except Exception as exc:
        vector_store = RuntimeHealthComponent(status="error", detail=str(exc))
        overall = "degraded"

    index_consistency: RuntimeHealthComponent | None = None
    try:
        if Path(config.storage.chroma_path).exists():
            memory_collection = get_chroma_collection(
                config.storage.chroma_path,
                name="memories",
            )
            document_collection = get_chroma_collection(
                config.storage.chroma_path,
                name="documents",
            )
            report = await MemoryIndexHealthChecker(
                db=db,
                memory_collection=memory_collection,
                document_collection=document_collection,
            ).check()
            if report.ok:
                index_consistency = RuntimeHealthComponent(
                    status="ok",
                    detail="No index consistency issues",
                )
            else:
                overall = "degraded"
                index_consistency = RuntimeHealthComponent(
                    status="error",
                    detail=f"{len(report.issues)} consistency issue(s)",
                )
        else:
            index_consistency = RuntimeHealthComponent(
                status="not_configured",
                detail="ChromaDB path does not exist",
            )
    except Exception as exc:
        overall = "degraded"
        index_consistency = RuntimeHealthComponent(status="error", detail=str(exc))

    audit_failures: RuntimeHealthComponent | None = None
    try:
        audit_failures = await _recent_audit_failure_health(db)
    except Exception as exc:
        audit_failures = RuntimeHealthComponent(status="warning", detail=str(exc))

    genes: dict[str, RuntimeHealthComponent] = {}
    try:
        sources = await db.list_sources()
        for src in sources:
            source_id = src["id"]
            source_name = src.get("name", source_id)
            sync_state = await db.get_sync_state(source_id)
            if sync_state and sync_state.last_sync_status:
                genes[source_name] = RuntimeHealthComponent(
                    status=sync_state.last_sync_status,
                    detail=_dt_iso(sync_state.last_sync_at),
                )
            else:
                genes[source_name] = RuntimeHealthComponent(status="never_synced")
    except Exception as exc:
        logger.warning("Failed to check gene connectivity: %s", exc)

    return RuntimeHealthReport(
        status=overall,
        database=database,
        vector_store=vector_store,
        index_consistency=index_consistency,
        audit_failures=audit_failures,
        genes=genes,
    )


async def _recent_audit_failure_health(db: "Database") -> RuntimeHealthComponent:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=AUDIT_HEALTH_WINDOW_HOURS)
    since = cutoff.isoformat()
    placeholders = ", ".join("?" for _ in AUDIT_HEALTH_FAILURE_EVENTS)
    params: list[Any] = [*AUDIT_HEALTH_FAILURE_EVENTS, since]
    async with db.db.execute(
        f"""SELECT event_type, COUNT(*), MAX(occurred_at)
            FROM memory_audit_events
            WHERE event_type IN ({placeholders})
              AND occurred_at >= ?
              AND (status = 'failed' OR error IS NOT NULL)
            GROUP BY event_type
            ORDER BY event_type""",
        params,
    ) as cursor:
        rows = await cursor.fetchall()

    counts_by_event_type = {str(row[0]): int(row[1]) for row in rows}
    payload = {
        "window_hours": AUDIT_HEALTH_WINDOW_HOURS,
        "since": since,
        "counts_by_event_type": counts_by_event_type,
        "total": sum(counts_by_event_type.values()),
        "last_seen_at": max((row[2] for row in rows if row[2]), default=None),
    }
    if not rows:
        return RuntimeHealthComponent(
            status="ok",
            detail=f"No audit failures in the last {AUDIT_HEALTH_WINDOW_HOURS}h",
            payload=payload,
        )
    summary = ", ".join(f"{row[0]}={row[1]}" for row in rows)
    return RuntimeHealthComponent(
        status="warning",
        detail=f"Recent audit failures in the last {AUDIT_HEALTH_WINDOW_HOURS}h: {summary}",
        payload=payload,
    )


def _dt_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _has_structured_llm_credentials(llm: EffectiveLlmConfig) -> bool:
    return bool(llm.enrichment_api_key) or is_litellm_provider_model(
        llm.enrichment_model
    )


def _retrieval_config_for_llm(config: AppConfig, llm: EffectiveLlmConfig):
    """Use the effective enrichment model for optional LLM retrieval assists."""
    return replace(
        config.retrieval,
        entity_model=llm.enrichment_model,
        rerank_model=llm.enrichment_model,
    )


async def build_search_engine(
    db: "Database",
    config: AppConfig,
    *,
    audit_logger: MemoryAuditLogger | None = None,
) -> Any:
    """Build the service-owned retrieval engine used by HTTP and agent-proxy clients.

    The optional audit logger is threaded into the relational adapter so any
    promote-to-workspace path reachable through the engine records its attempts
    on the same audit channel as the rest of the runtime.
    """
    from memforge.retrieval.search import SearchEngine

    memory_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="memories",
    )
    llm = await get_effective_llm_config(db, config)
    embed_cfg = {
        "base_url": llm.embedding_base_url,
        "api_key": llm.embedding_api_key,
        "model": llm.embedding_model,
    }
    structured_llm_client = None
    if _has_structured_llm_credentials(llm):
        structured_llm_client = LiteLlmStructuredClient(
            StructuredLlmConfig(
                model=llm.enrichment_model,
                base_url=llm.enrichment_base_url or None,
                api_key=llm.enrichment_api_key or None,
                timeout_s=llm.request_timeout_s,
            )
        )
    adapters = build_sqlite_adapters(db, memory_collection, audit_logger=audit_logger)
    return SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg=embed_cfg,
        config=_retrieval_config_for_llm(config, llm),
        structured_llm_client=structured_llm_client,
    )


async def build_sync_runtime(
    db: "Database",
    config: AppConfig,
    *,
    extraction_pool: ExtractionWorkPool | None = None,
    document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
) -> SyncRuntime:
    llm = await get_effective_llm_config(db, config)
    structured_llm_client = None
    if _has_structured_llm_credentials(llm):
        structured_llm_client = LiteLlmStructuredClient(
            StructuredLlmConfig(
                model=llm.enrichment_model,
                base_url=llm.enrichment_base_url or None,
                api_key=llm.enrichment_api_key or None,
                timeout_s=llm.request_timeout_s,
            )
        )
    doc_store = LocalDocumentStore(config.storage.docs_path)

    enricher = Enricher(
        model=llm.enrichment_model,
        base_url=llm.enrichment_base_url or None,
        api_key=llm.enrichment_api_key or None,
        max_tokens=config.llm.enrichment_max_tokens,
        request_timeout_s=llm.request_timeout_s,
        structured_llm_client=structured_llm_client,
    )
    memory_extractor = MemoryExtractor(
        model=llm.enrichment_model,
        base_url=llm.enrichment_base_url or None,
        api_key=llm.enrichment_api_key or None,
        max_tokens=config.llm.enrichment_max_tokens,
        request_timeout_s=llm.request_timeout_s,
        structured_llm_client=structured_llm_client,
    )

    doc_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="documents",
    )
    memory_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="memories",
    )
    embed_cfg = {
        "base_url": llm.embedding_base_url,
        "api_key": llm.embedding_api_key,
        "model": llm.embedding_model,
    }
    adapters = build_sqlite_adapters(
        db,
        memory_collection,
        audit_logger=MemoryAuditLogger(
            db, default_context=AuditContext(actor_type="sync")
        ),
    )
    memory_store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg=embed_cfg,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
        document_index=DocumentVectorIndex(doc_collection),
    )
    memory_engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=memory_store,
        embed_cfg=embed_cfg,
        structured_llm_client=structured_llm_client,
        llm_model=llm.enrichment_model,
    )
    source_support_kwargs = {
        "structured_llm_client": structured_llm_client,
        "llm_model": llm.enrichment_model,
    }
    source_support_detector = SourceSupportDetector(**source_support_kwargs)

    return SyncRuntime(
        db=db,
        config=config,
        doc_store=doc_store,
        enricher=enricher,
        memory_extractor=memory_extractor,
        memory_store=memory_store,
        memory_engine=memory_engine,
        vector_store=doc_collection,
        embed_cfg=embed_cfg,
        structured_llm_client=structured_llm_client,
        llm_model=llm.enrichment_model,
        source_support_detector=source_support_detector,
        extraction_pool=extraction_pool,
        document_lifecycle_admission=document_lifecycle_admission,
        memory_observer=SyncMemoryObserver(),
    )


async def run_source_sync(
    db: "Database",
    config: AppConfig,
    source: dict,
    runtime: SyncRuntime | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    force_full_sync: bool = False,
) -> SyncState:
    runtime = runtime or await build_sync_runtime(db, config)
    secret_fields = source_secret_fields(source["type"], GENE_REGISTRY)
    source_config = decrypt_source_config_for_runtime(source["config"], secret_fields=secret_fields)
    await browser_session.inject_cookie_for_source(db, source["type"], source_config)
    gene = create_gene(
        name=source["type"],
        config=source_config,
        source_id=source["id"],
    )
    gene.bind_document_store(runtime.doc_store)
    return await runtime.orchestrator().sync_gene(
        gene=gene,
        source_name=source["name"],
        source_id=source["id"],
        progress_callback=progress_callback,
        force_full_sync=force_full_sync,
    )


class SyncService:
    """App-scoped sync runner with task tracking and shutdown cleanup."""

    def __init__(
        self,
        db: "Database",
        config: AppConfig,
        runtime_provider: RuntimeProvider | None = None,
    ) -> None:
        self.db = db
        self.config = config
        self.runtime_provider = runtime_provider or DefaultRuntimeProvider()
        self.tasks: dict[str, asyncio.Task] = {}
        self.queued_tasks: dict[str, asyncio.Task] = {}
        self.progress: dict[str, dict] = {}
        max_active_sources = max(0, int(config.sync.max_active_sources))
        self._source_slots = (
            asyncio.Semaphore(max_active_sources) if max_active_sources else None
        )
        max_extraction_workers = max(0, int(config.sync.max_extraction_workers))
        self._extraction_pool = (
            ExtractionWorkPool(max_extraction_workers)
            if max_extraction_workers
            else None
        )
        max_document_lifecycles = max(0, int(config.sync.max_document_lifecycles))
        self._document_lifecycle_admission = get_process_document_lifecycle_admission(max_document_lifecycles)

    def is_running(self, source_id: str) -> bool:
        task = self.tasks.get(source_id)
        return bool(task and not task.done())

    def running_source_ids(self) -> set[str]:
        return {
            source_id
            for source_id, task in self.tasks.items()
            if task and not task.done()
        }

    async def _ensure_source_can_sync(self, source_id: str) -> dict:
        source = await self.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source not found: {source_id}")
        if source.get("status") == "paused":
            raise SourcePausedError(f"Source is paused: {source_id}")
        return source

    async def start_source(self, source_id: str, *, force_full_sync: bool = False) -> asyncio.Task:
        await self._ensure_source_can_sync(source_id)
        if self.is_running(source_id):
            raise SyncAlreadyRunningError(f"Sync already running for {source_id}")
        task = asyncio.create_task(
            self._run_source_task_with_slot(
                source_id,
                force_full_sync=force_full_sync,
            )
        )
        self.tasks[source_id] = task
        return task

    async def _run_source_task_with_slot(
        self,
        source_id: str,
        *,
        force_full_sync: bool = False,
    ) -> SyncState | None:
        if self._source_slots is None:
            return await self._run_source_task(
                source_id,
                force_full_sync=force_full_sync,
            )

        started_at = datetime.now(timezone.utc).isoformat()
        self.progress[source_id] = {
            "started_at": started_at,
            "phase": "queued",
            "docs_processed": 0,
            "docs_total": 0,
            "docs_updated": 0,
            "docs_failed": 0,
            "memories_extracted": 0,
            "title": None,
        }
        entered_slot = False
        try:
            async with self._source_slots:
                entered_slot = True
                return await self._run_source_task(
                    source_id,
                    force_full_sync=force_full_sync,
                )
        finally:
            if not entered_slot:
                self.tasks.pop(source_id, None)
                self.progress.pop(source_id, None)

    async def request_source_sync(self, source_id: str, *, delay_seconds: float = 1.0) -> bool:
        """Queue one service-owned sync pass for a source, coalescing duplicates."""
        try:
            await self._ensure_source_can_sync(source_id)
        except SourcePausedError:
            return False
        queued = self.queued_tasks.get(source_id)
        if queued and not queued.done():
            return False
        task = asyncio.create_task(self._run_queued_source_sync(source_id, delay_seconds=delay_seconds))
        self.queued_tasks[source_id] = task
        return True

    async def _run_queued_source_sync(self, source_id: str, *, delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            active = self.tasks.get(source_id)
            if active and not active.done():
                await active
            try:
                await self.start_source(source_id)
            except (SourcePausedError, SyncAlreadyRunningError):
                return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Queued sync failed for source %s", source_id)
        finally:
            self.queued_tasks.pop(source_id, None)

    async def run_all_active_sources(self) -> None:
        sources = await self.db.list_sources()
        for source in sources:
            if source.get("status") != "active":
                continue
            if self.is_running(source["id"]):
                continue
            task = await self.start_source(source["id"])
            await task

    async def retire_expired_memories(self) -> int:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        return await runtime.memory_store.retire_expired_memories()

    async def check_memory_index_health(self) -> MemoryIndexHealthReport:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        checker = MemoryIndexHealthChecker(
            db=self.db,
            memory_collection=runtime.memory_store.collection,
            document_collection=runtime.vector_store,
        )
        return await checker.check()

    async def cancel_source(self, source_id: str) -> None:
        task = self.tasks.pop(source_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.progress.pop(source_id, None)

    async def shutdown(self) -> None:
        queued_tasks = list(self.queued_tasks.values())
        self.queued_tasks.clear()
        for task in queued_tasks:
            if task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        source_ids = list(self.tasks)
        for source_id in source_ids:
            await self.cancel_source(source_id)

    async def _run_source_task(self, source_id: str, *, force_full_sync: bool = False) -> SyncState | None:
        started_at = datetime.now(timezone.utc).isoformat()
        source: dict | None = None
        self.progress[source_id] = {
            "started_at": started_at,
            "phase": "discovering",
            "docs_processed": 0,
            "docs_total": 0,
            "docs_updated": 0,
            "docs_failed": 0,
            "memories_extracted": 0,
            "title": None,
        }
        try:
            source = await self.db.get_source(source_id)
            if not source:
                raise ValueError(f"Source not found: {source_id}")

            runtime = await self.runtime_provider.build_sync_runtime(
                self.db,
                self.config,
                extraction_pool=self._extraction_pool,
                document_lifecycle_admission=self._document_lifecycle_admission,
            )

            def on_progress(progress: dict) -> None:
                current = progress.get("current", 0)
                total = progress.get("total", 0)
                self.progress[source_id]["phase"] = progress.get("phase")
                self.progress[source_id]["docs_processed"] = current
                self.progress[source_id]["docs_total"] = total
                self.progress[source_id]["docs_updated"] = progress.get(
                    "docs_updated",
                    self.progress[source_id].get("docs_updated", 0),
                )
                self.progress[source_id]["docs_failed"] = progress.get(
                    "docs_failed",
                    self.progress[source_id].get("docs_failed", 0),
                )
                self.progress[source_id]["memories_extracted"] = progress.get(
                    "memories_extracted",
                    self.progress[source_id].get("memories_extracted", 0),
                )
                self.progress[source_id]["title"] = progress.get("title")

            return await self.runtime_provider.run_source_sync(
                db=self.db,
                config=self.config,
                source=source,
                runtime=runtime,
                progress_callback=on_progress,
                force_full_sync=force_full_sync,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Sync failed for source %s", source_id)
            if source and "browser session" in str(e).lower():
                await browser_session.mark_expired_for_source(
                    self.db,
                    source.get("type", ""),
                    str(source.get("config", {}).get("base_url") or ""),
                    str(e),
                )
            error_state = SyncState(
                source=source_id,
                last_sync_at=datetime.now(timezone.utc),
                last_sync_status="failed",
                error_message=str(e),
            )
            await self.db.upsert_sync_state(error_state)
            await self.db.insert_sync_history(
                source=source_id,
                status="failed",
                docs_processed=0,
                docs_updated=0,
                docs_failed=0,
                memories_extracted=0,
                error_message=str(e),
                failed_docs=None,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return None
        finally:
            self.tasks.pop(source_id, None)
            self.progress.pop(source_id, None)
