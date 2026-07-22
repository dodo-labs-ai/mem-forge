"""Shared runtime wiring for sync startup paths."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from memforge.config import AppConfig
from memforge.auth import browser_session
from memforge.genes import GENE_REGISTRY, create_gene, source_type_supports_sync
from memforge.llm.providers import is_litellm_provider_model
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.local_agent.source_contract import (
    local_agent_collection_is_authoritative,
    local_agent_source_config_revision,
    local_agent_sync_operation,
    source_with_sync_inputs,
)
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.health import MemoryIndexHealthChecker, MemoryIndexHealthReport
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.store import MemoryStore
from memforge.models import SourceSyncRun, SyncState
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.source_support_detector import SourceSupportDetector
from memforge.pipeline.sync_memory import SyncMemoryObserver
from memforge.pipeline.sync import (
    DocumentLifecycleAdmission,
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    SourceSyncMode,
    get_process_document_lifecycle_admission,
)
from memforge.retrieval.embeddings import get_chroma_collection
from memforge.source_secrets import decrypt_source_config_for_runtime, source_secret_fields
from memforge.source_activity import (
    SourceActivityConflict,
    SourceActivityKind,
)
from memforge.storage.document_store import LocalDocumentStore
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.sync_progress import SourceSyncProgressAccumulator, source_sync_progress_from_pipeline

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)


class SyncAlreadyRunningError(RuntimeError):
    """Raised when a source already has an active sync task."""


class SourcePausedError(RuntimeError):
    """Raised when sync is requested for a paused source."""


class SourceNotActiveError(RuntimeError):
    """Raised when sync is requested while a source lifecycle is not active."""


class SourceSyncLeaseLost(RuntimeError):
    """Raised when a worker no longer owns the leased source-sync run."""


class SourceSyncBoundaryError(RuntimeError):
    """Raised when a durable run no longer matches its captured input boundary."""


class SourceSyncUnsupportedError(SourceSyncBoundaryError):
    """Raised when a source type has no ordinary sync execution kind."""


class SourceLifecycleMaintenanceError(SourceSyncBoundaryError, SourceActivityConflict):
    """Raised when an ordinary sync overlaps durable lifecycle maintenance."""


async def authorize_source_sync_maintenance(
    db: Any,
    source_id: str,
    *,
    lifecycle_job_id: str | None = None,
) -> None:
    """Require the exact active lifecycle job token, or no active maintenance."""
    active = await db.get_active_lifecycle_backfill_job(source_id)
    if active is None:
        if lifecycle_job_id is not None:
            raise SourceLifecycleMaintenanceError(
                f"source lifecycle maintenance token is not active: {lifecycle_job_id}"
            )
        return
    if lifecycle_job_id != active.id:
        raise SourceLifecycleMaintenanceError(f"source lifecycle maintenance active: {active.id}")


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
    memory_extractor: MemoryExtractor
    memory_store: MemoryStore
    memory_engine: MemoryEngine
    structured_llm_client: LiteLlmStructuredClient | None
    llm_model: str
    source_support_detector: SourceSupportDetector | None
    relation_discovery: Any | None = None
    extraction_pool: ExtractionWorkPool | None = None
    document_lifecycle_admission: DocumentLifecycleAdmission | None = None
    memory_observer: SyncMemoryObserver | None = None
    orchestrator_factory: Callable[["SyncRuntime"], GeneSyncOrchestrator] | None = None

    def __post_init__(self) -> None:
        if self.document_lifecycle_admission is None:
            self.document_lifecycle_admission = get_process_document_lifecycle_admission(
                max(0, int(self.config.sync.max_document_lifecycles))
            )

    def orchestrator(self) -> GeneSyncOrchestrator:
        if self.orchestrator_factory is not None:
            return self.orchestrator_factory(self)
        return GeneSyncOrchestrator(
            db=self.db,
            doc_store=self.doc_store,
            memory_extractor=self.memory_extractor,
            memory_engine=self.memory_engine,
            memory_store=self.memory_store,
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
        authoritative_snapshot: bool = False,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        lifecycle_job_id: str | None = None,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
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
        authoritative_snapshot: bool = False,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        lifecycle_job_id: str | None = None,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
    ) -> SyncState:
        return await run_source_sync(
            db=db,
            config=config,
            source=source,
            runtime=runtime,
            progress_callback=progress_callback,
            force_full_sync=force_full_sync,
            authoritative_snapshot=authoritative_snapshot,
            reprocess_doc_ids=reprocess_doc_ids,
            execution_mode=execution_mode,
            lifecycle_job_id=lifecycle_job_id,
            lifecycle_cycle_id=lifecycle_cycle_id,
            scope_transition_run_id=scope_transition_run_id,
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
            report = await MemoryIndexHealthChecker(
                db=db,
                memory_collection=memory_collection,
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
    return bool(llm.enrichment_api_key) or is_litellm_provider_model(llm.enrichment_model)


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

    memory_extractor = MemoryExtractor(
        model=llm.enrichment_model,
        base_url=llm.enrichment_base_url or None,
        api_key=llm.enrichment_api_key or None,
        max_tokens=config.llm.memory_extraction_max_tokens,
        request_timeout_s=llm.request_timeout_s,
        structured_llm_client=structured_llm_client,
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
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
    )
    memory_store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg=embed_cfg,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
    )
    memory_engine = MemoryEngine(
        cross_document_candidates=CrossDocumentCandidateRetriever(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
        ),
        db=db,
        memory_store=memory_store,
        embed_cfg=embed_cfg,
        structured_llm_client=structured_llm_client,
        llm_model=llm.enrichment_model,
    )
    relation_discovery = None
    if memory_engine.pair_classifier is not None:
        from memforge.memory.relation_discovery import RelationDiscovery

        relation_discovery = RelationDiscovery(
            store=adapters.relational,
            candidate_retriever=memory_engine.cross_document_candidates,
            pair_classifier=memory_engine.pair_classifier,
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
        memory_extractor=memory_extractor,
        memory_store=memory_store,
        memory_engine=memory_engine,
        structured_llm_client=structured_llm_client,
        llm_model=llm.enrichment_model,
        source_support_detector=source_support_detector,
        relation_discovery=relation_discovery,
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
    authoritative_snapshot: bool = False,
    reprocess_doc_ids: frozenset[str] | None = None,
    execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
    lifecycle_job_id: str | None = None,
    lifecycle_cycle_id: str | None = None,
    scope_transition_run_id: str | None = None,
) -> SyncState:
    await authorize_source_sync_maintenance(
        db,
        str(source["id"]),
        lifecycle_job_id=lifecycle_job_id,
    )
    activity_id = None
    source_activity_epoch: int | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    if lifecycle_job_id is None:
        activity_id = f"source-sync-{uuid.uuid4().hex}"
        try:
            activity_lease = await db.acquire_source_activity(
                activity_id=activity_id,
                source_id=str(source["id"]),
                kind=SourceActivityKind.SYNC,
                lease_seconds=300,
            )
            source_activity_epoch = activity_lease.epoch
        except SourceActivityConflict as exc:
            raise SourceLifecycleMaintenanceError(str(exc)) from exc

        async def heartbeat_activity() -> None:
            while True:
                await asyncio.sleep(60)
                await db.renew_source_activity(
                    activity_id=activity_id,
                    lease_seconds=300,
                )

        heartbeat_task = asyncio.create_task(heartbeat_activity())
    else:
        source_activity_epoch = await db.get_source_activity_epoch(str(source["id"]))
    lifecycle_cycle_id = lifecycle_cycle_id or lifecycle_job_id or activity_id
    if lifecycle_cycle_id is None:
        lifecycle_cycle_id = f"source-sync-cycle-{uuid.uuid4().hex}"
    try:
        runtime = runtime or await build_sync_runtime(db, config)
        secret_fields = source_secret_fields(source["type"], GENE_REGISTRY)
        source_config = decrypt_source_config_for_runtime(source["config"], secret_fields=secret_fields)
        await browser_session.inject_cookie_for_source(db, source["type"], source_config)
        gene = create_gene(
            name=source["type"],
            config=source_config,
            source_id=source["id"],
        )
        sync_kwargs: dict[str, Any] = {
            "gene": gene,
            "source_name": source["name"],
            "source_id": source["id"],
            "progress_callback": progress_callback,
            "force_full_sync": force_full_sync,
            "authoritative_snapshot": authoritative_snapshot,
            "reprocess_doc_ids": reprocess_doc_ids,
            "source_activity_epoch": source_activity_epoch,
            "lifecycle_cycle_id": lifecycle_cycle_id,
            "scope_transition_run_id": scope_transition_run_id,
        }
        if execution_mode is not SourceSyncMode.NORMAL:
            sync_kwargs["execution_mode"] = execution_mode
        sync_operation = runtime.orchestrator().sync_gene(**sync_kwargs)
        if heartbeat_task is None:
            return await sync_operation
        sync_task = asyncio.create_task(sync_operation)
        done, _ = await asyncio.wait(
            {sync_task, heartbeat_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            sync_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sync_task
            await heartbeat_task
            raise SourceLifecycleMaintenanceError(f"source activity heartbeat stopped: {activity_id}")
        return await sync_task
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        if activity_id is not None:
            await db.release_source_activity(activity_id=activity_id)


class SourceSyncWorker:
    """Durable source-sync worker that leases and executes one run at a time."""

    def __init__(
        self,
        db: "Database",
        config: AppConfig,
        runtime_provider: RuntimeProvider | None = None,
        *,
        worker_id: str = "source-sync-worker",
        workspace_id: str | None = None,
        lease_seconds: int = 300,
        heartbeat_seconds: float | None = None,
        progress_flush_seconds: float = 1.0,
    ) -> None:
        self.db = db
        self.config = config
        self.runtime_provider = runtime_provider or DefaultRuntimeProvider()
        self.worker_id = worker_id
        self.workspace_id = workspace_id
        self.lease_seconds = lease_seconds
        self.heartbeat_seconds = (
            max(0.001, float(heartbeat_seconds))
            if heartbeat_seconds is not None
            else max(1.0, min(30.0, lease_seconds / 3))
        )
        self.progress_flush_seconds = max(0.001, float(progress_flush_seconds))
        max_extraction_workers = max(0, int(config.sync.max_extraction_workers))
        self._extraction_pool = ExtractionWorkPool(max_extraction_workers) if max_extraction_workers else None
        max_document_lifecycles = max(0, int(config.sync.max_document_lifecycles))
        self._document_lifecycle_admission = get_process_document_lifecycle_admission(max_document_lifecycles)
        self._relation_runtime: SyncRuntime | None = None

    async def _process_relation_discovery_once(self) -> bool:
        try:
            if self._relation_runtime is None:
                from memforge.memory.relation_discovery import DEFAULT_RELATION_DISCOVERY_BUDGET

                if not await self.db.has_ready_relation_discovery_work(
                    max_attempts=DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts,
                ):
                    return False
                self._relation_runtime = await self.runtime_provider.build_sync_runtime(
                    self.db,
                    self.config,
                    extraction_pool=self._extraction_pool,
                    document_lifecycle_admission=self._document_lifecycle_admission,
                )
            processor = self._relation_runtime.relation_discovery
            if processor is None:
                return False
            result = await processor.process_slice(
                worker_id=f"{self.worker_id}:relation",
            )
            if result.attempted_work:
                logger.info(
                    "Relation discovery slice attempted=%d completed=%d failed=%d "
                    "obsolete=%d pairs=%d llm_calls=%d elapsed_ms=%d",
                    result.attempted_work,
                    result.completed_work,
                    result.failed_work,
                    result.obsolete_work,
                    result.checked_candidate_pairs,
                    result.llm_calls,
                    result.elapsed_ms,
                )
            return bool(result.attempted_work)
        except Exception:
            logger.exception("Relation discovery worker slice failed")
            return False

    async def _heartbeat_until_stopped(
        self,
        run: SourceSyncRun,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                try:
                    renewed = await self.db.heartbeat_source_sync_run(
                        run.run_id,
                        worker_id=self.worker_id,
                        lease_attempt_count=run.lease_attempt_count,
                        lease_seconds=self.lease_seconds,
                    )
                except Exception:
                    logger.exception("Source sync worker heartbeat failed for run %s", run.run_id)
                    lease_lost.set()
                    return
                if not renewed:
                    logger.warning("Source sync worker lost lease for run %s", run.run_id)
                    lease_lost.set()
                    return

    async def _progress_until_stopped(
        self,
        run: SourceSyncRun,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
        latest: dict[str, Any],
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.progress_flush_seconds)
                return
            except asyncio.TimeoutError:
                revision = int(latest.get("revision") or 0)
                snapshot = latest.get("snapshot")
                if revision <= int(latest.get("flushed_revision") or 0) or not isinstance(snapshot, dict):
                    continue
                try:
                    stored = await self.db.report_source_sync_run_progress(
                        run.run_id,
                        worker_id=self.worker_id,
                        lease_attempt_count=run.lease_attempt_count,
                        progress=snapshot,
                    )
                except Exception:
                    logger.exception("Source sync progress update failed for run %s", run.run_id)
                    continue
                if not stored:
                    lease_lost.set()
                    return
                latest["flushed_revision"] = revision

    async def _run_source_sync_with_heartbeat(self, run: SourceSyncRun, **kwargs: Any) -> SyncState | None:
        stop = asyncio.Event()
        lease_lost = asyncio.Event()
        source_type = str(kwargs.get("source", {}).get("type") or "")
        previous_attempt_progress = run.progress if run.lease_attempt_count > 1 else None
        progress_accumulator = SourceSyncProgressAccumulator(previous_attempt_progress)
        latest_progress: dict[str, Any] = {"revision": 0, "flushed_revision": 0, "snapshot": None}

        def report_progress(value: dict[str, Any]) -> None:
            snapshot = source_sync_progress_from_pipeline(value, source_type=source_type)
            if snapshot is None:
                return
            snapshot = progress_accumulator.update(snapshot)
            latest_progress["revision"] = int(latest_progress["revision"]) + 1
            latest_progress["snapshot"] = snapshot

        kwargs["progress_callback"] = report_progress
        heartbeat_task = asyncio.create_task(self._heartbeat_until_stopped(run, stop, lease_lost))
        progress_task = asyncio.create_task(self._progress_until_stopped(run, stop, lease_lost, latest_progress))
        sync_task = asyncio.create_task(self.runtime_provider.run_source_sync(**kwargs))
        try:
            done, _ = await asyncio.wait(
                {sync_task, heartbeat_task, progress_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if lease_lost.is_set() and (heartbeat_task in done or progress_task in done):
                sync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sync_task
                raise SourceSyncLeaseLost(f"source sync lease lost for run {run.run_id}")
            result = await sync_task
            if lease_lost.is_set():
                raise SourceSyncLeaseLost(f"source sync lease lost for run {run.run_id}")
            revision = int(latest_progress["revision"])
            if revision > int(latest_progress["flushed_revision"]):
                snapshot = latest_progress.get("snapshot")
                if isinstance(snapshot, dict):
                    try:
                        stored = await self.db.report_source_sync_run_progress(
                            run.run_id,
                            worker_id=self.worker_id,
                            lease_attempt_count=run.lease_attempt_count,
                            progress=snapshot,
                        )
                    except Exception:
                        logger.exception("Final source sync progress update failed for run %s", run.run_id)
                    else:
                        if not stored:
                            raise SourceSyncLeaseLost(f"source sync lease lost for run {run.run_id}")
                        latest_progress["flushed_revision"] = revision
            return result
        finally:
            stop.set()
            if not sync_task.done():
                sync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sync_task
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

    def _next_retry_at(self, run: SourceSyncRun, failed_at: datetime) -> datetime | None:
        if run.lease_attempt_count >= self.config.sync.worker_max_attempts:
            return None
        exponent = max(0, run.lease_attempt_count - 1)
        delay = min(
            self.config.sync.worker_retry_max_seconds,
            self.config.sync.worker_retry_base_seconds * (2**exponent),
        )
        return failed_at + timedelta(seconds=delay)

    async def run_once(self) -> SourceSyncRun | None:
        run = await self.db.lease_next_source_sync_run(
            worker_id=self.worker_id,
            workspace_id=self.workspace_id,
            lease_seconds=self.lease_seconds,
        )
        if run is None:
            await self._process_relation_discovery_once()
            return None

        source: dict | None = None
        try:
            source = await self.db.get_source(run.source_id)
            if source is None:
                failed = await self.db.fail_source_sync_run(
                    run.run_id,
                    worker_id=self.worker_id,
                    lease_attempt_count=run.lease_attempt_count,
                    error_message=f"Source not found: {run.source_id}",
                    retryable=False,
                )
                if not failed:
                    raise SourceSyncLeaseLost(
                        f"source sync lease lost before missing-source update for run {run.run_id}"
                    )
                return run
            if source.get("status") == "paused":
                failed = await self.db.fail_source_sync_run(
                    run.run_id,
                    worker_id=self.worker_id,
                    lease_attempt_count=run.lease_attempt_count,
                    error_message=f"Source is paused: {run.source_id}",
                    retryable=False,
                )
                if not failed:
                    raise SourceSyncLeaseLost(
                        f"source sync lease lost before paused-source update for run {run.run_id}"
                    )
                return run

            if not source_type_supports_sync(str(source.get("type") or "")):
                raise SourceSyncUnsupportedError(f"Source type {source.get('type')!r} does not support ordinary sync")

            if (
                run.source_config_revision is not None
                and local_agent_source_config_revision(source) != run.source_config_revision
            ):
                raise SourceSyncBoundaryError("source config revision changed before sync execution")

            inputs = await self.db.list_source_sync_inputs(
                source_id=run.source_id,
                workspace_id=run.workspace_id,
                input_snapshot_id=run.input_snapshot_id,
            )
            local_operation = local_agent_sync_operation(source["type"], source.get("config"))
            if run.input_snapshot_id is None:
                if local_operation is not None and run.input_generation_watermark is None:
                    raise SourceSyncBoundaryError("local-agent sync run is missing its input generation boundary")
                if run.input_generation_watermark is not None:
                    inputs = [
                        source_input
                        for source_input in inputs
                        if source_input.input_generation <= run.input_generation_watermark
                    ]
            authoritative_collection = local_agent_collection_is_authoritative(source["type"])
            source = source_with_sync_inputs(
                source,
                inputs,
                authoritative_snapshot=authoritative_collection,
            )

            runtime = await self.runtime_provider.build_sync_runtime(
                self.db,
                self.config,
                extraction_pool=self._extraction_pool,
                document_lifecycle_admission=self._document_lifecycle_admission,
            )
            self._relation_runtime = runtime
            final_state = await self._run_source_sync_with_heartbeat(
                run,
                db=self.db,
                config=self.config,
                source=source,
                runtime=runtime,
                progress_callback=None,
                force_full_sync=run.force_full_sync,
                authoritative_snapshot=authoritative_collection,
                lifecycle_cycle_id=(f"{run.run_id}:attempt:{run.lease_attempt_count}"),
                scope_transition_run_id=run.run_id,
            )
            if final_state is None:
                final_state = SyncState(
                    source=run.source_id,
                    last_sync_at=datetime.now(timezone.utc),
                    last_sync_status="failed",
                    error_message="sync completed without final state",
                )
            if final_state.last_sync_status in {"failed", "partial"}:
                error_message = final_state.error_message or "source sync failed"
                failed_at = datetime.now(timezone.utc)
                next_attempt_at = self._next_retry_at(run, failed_at)
                failed = await self.db.fail_source_sync_run(
                    run.run_id,
                    worker_id=self.worker_id,
                    lease_attempt_count=run.lease_attempt_count,
                    error_message=error_message,
                    final_state=final_state,
                    retryable=next_attempt_at is not None,
                    failed_at=failed_at,
                    next_attempt_at=next_attempt_at,
                )
                if not failed:
                    raise SourceSyncLeaseLost(f"source sync lease lost before failure update for run {run.run_id}")
                return run
            try:
                await runtime.memory_store.attempt_lifecycle_vector_delivery(source_id=run.source_id)
            except Exception:
                # The relational lifecycle graph is already authoritative.
                # Delivery stays durable in the outbox and must never turn a
                # successful source run into a failed extraction transaction.
                logger.exception(
                    "Source sync completed with lifecycle vector delivery still pending for source %s",
                    run.source_id,
                )
            completed = await self.db.complete_source_sync_run(
                run.run_id,
                worker_id=self.worker_id,
                lease_attempt_count=run.lease_attempt_count,
                final_state=final_state,
            )
            if not completed:
                raise SourceSyncLeaseLost(f"source sync lease lost before completion for run {run.run_id}")
            return run
        except asyncio.CancelledError:
            raise
        except SourceSyncLeaseLost:
            logger.warning("Source sync worker stopped terminal update after losing lease for run %s", run.run_id)
            return run
        except Exception as exc:
            logger.exception("Source sync worker failed run %s", run.run_id)
            if source and "browser session" in str(exc).lower():
                await browser_session.mark_expired_for_source(
                    self.db,
                    source.get("type", ""),
                    str(source.get("config", {}).get("base_url") or ""),
                    str(exc),
                )
            failed_at = datetime.now(timezone.utc)
            next_attempt_at = self._next_retry_at(run, failed_at)
            retryable = (
                not isinstance(
                    exc,
                    (SourcePausedError, SourceNotActiveError, SourceSyncBoundaryError),
                )
                and next_attempt_at is not None
            )
            failed = await self.db.fail_source_sync_run(
                run.run_id,
                worker_id=self.worker_id,
                lease_attempt_count=run.lease_attempt_count,
                error_message=str(exc),
                retryable=retryable,
                failed_at=failed_at,
                next_attempt_at=next_attempt_at,
            )
            if not failed:
                logger.warning("Source sync worker stopped failure update after losing lease for run %s", run.run_id)
            return run

    async def run_forever(self, *, poll_seconds: float | None = None) -> None:
        interval = max(0.1, poll_seconds if poll_seconds is not None else self.config.sync.worker_poll_seconds)
        while True:
            run = await self.run_once()
            if run is not None:
                # Fairly interleave one bounded non-destructive slice even while
                # source-sync work remains continuously available.
                await self._process_relation_discovery_once()
            await asyncio.sleep(0 if run is not None else interval)


class SyncService:
    """App-scoped sync runner with task tracking and shutdown cleanup."""

    def __init__(
        self,
        db: "Database",
        config: AppConfig,
        runtime_provider: RuntimeProvider | None = None,
        *,
        workspace_id: str = "default",
    ) -> None:
        self.db = db
        self.config = config
        self.runtime_provider = runtime_provider or DefaultRuntimeProvider()
        self.workspace_id = workspace_id
        self.tasks: dict[str, asyncio.Task] = {}
        self.progress: dict[str, dict] = {}
        max_active_sources = max(0, int(config.sync.max_active_sources))
        self._source_slots = asyncio.Semaphore(max_active_sources) if max_active_sources else None
        max_extraction_workers = max(0, int(config.sync.max_extraction_workers))
        self._extraction_pool = ExtractionWorkPool(max_extraction_workers) if max_extraction_workers else None
        max_document_lifecycles = max(0, int(config.sync.max_document_lifecycles))
        self._document_lifecycle_admission = get_process_document_lifecycle_admission(max_document_lifecycles)

    def is_running(self, source_id: str) -> bool:
        task = self.tasks.get(source_id)
        return bool(task and not task.done())

    async def _ensure_source_can_sync(self, source_id: str) -> dict:
        source = await self.db.get_source(source_id)
        if source is None:
            raise ValueError(f"Source not found: {source_id}")
        if source.get("status") == "paused":
            raise SourcePausedError(f"Source is paused: {source_id}")
        if source.get("status") != "active":
            raise SourceNotActiveError(f"Source is not active: {source_id} ({source.get('status')})")
        if not source_type_supports_sync(str(source.get("type") or "")):
            raise SourceSyncUnsupportedError(f"Source type {source.get('type')!r} does not support ordinary sync")
        await authorize_source_sync_maintenance(self.db, source_id)
        return source

    async def enqueue_source(
        self,
        source_id: str,
        *,
        trigger: str = "manual",
        force_full_sync: bool = False,
        input_snapshot_id: str | None = None,
        source_config_revision: str | None = None,
        predecessor_activity_id: str | None = None,
    ) -> SourceSyncRun:
        source = await self._ensure_source_can_sync(source_id)
        current_config_revision = (
            local_agent_source_config_revision(source)
            if local_agent_sync_operation(source["type"], source.get("config")) is not None
            else None
        )
        if source_config_revision is not None and source_config_revision != current_config_revision:
            raise ValueError("source config revision changed before sync enqueue")
        effective_config_revision = source_config_revision or current_config_revision
        return await self.db.enqueue_source_sync_run(
            source_id=source_id,
            workspace_id=self.workspace_id,
            trigger=trigger,
            force_full_sync=force_full_sync,
            input_snapshot_id=input_snapshot_id,
            source_config_revision=effective_config_revision,
            predecessor_activity_id=predecessor_activity_id,
        )

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
        """Queue one durable sync pass for a source, coalescing duplicates."""
        del delay_seconds
        try:
            run = await self.enqueue_source(source_id, trigger="request")
        except (SourcePausedError, SourceNotActiveError, SourceSyncUnsupportedError):
            return False
        return not run.coalesced

    async def run_all_active_sources(self) -> None:
        sources = await self.db.list_sources()
        for source in sources:
            if source.get("status") != "active":
                continue
            try:
                await self.enqueue_source(source["id"], trigger="schedule_all")
            except SourceSyncUnsupportedError:
                continue

    async def retire_expired_memories(self) -> int:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        return await runtime.memory_store.retire_expired_memories()

    async def check_memory_index_health(self) -> MemoryIndexHealthReport:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        checker = MemoryIndexHealthChecker(
            db=self.db,
            memory_collection=runtime.memory_store.collection,
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
        await self.db.cancel_pending_source_sync_run(
            source_id=source_id,
            error_message="cancelled_by_source_change",
        )

    async def shutdown(self) -> None:
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
