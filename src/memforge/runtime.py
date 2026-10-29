"""Shared runtime wiring for sync startup paths."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from memforge.config import AppConfig
from memforge.evals.agent_evaluation import (
    assessment_sink_for_runtime_sink,
    publish_agent_assessments,
    publish_runtime_events,
)
from memforge.auth import browser_session
from memforge.genes import GENE_REGISTRY, create_gene, source_type_supports_sync
from memforge.llm.bindings import embedding_identity, resolve_active_binding, usable_invocation
from memforge.llm.embedding import (
    EmbeddingIndexMismatch,
    EmbeddingRuntime,
    EmbeddingSettings,
    ProfileObserver,
)
from memforge.llm.embedding_profile import EmbeddingIdentity, EmbeddingProfile
from memforge.llm.providers import ModelConfigError, ResolvedInvocation, provider_descriptor
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.local_agent.source_contract import (
    local_agent_source_config_revision,
    local_agent_sync_operation,
    snapshot_document_ids,
    source_with_sync_inputs,
)
from memforge.source_projection_config import (
    excluded_scope_config,
    projection_access_fingerprint,
    source_access_context,
)
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.health import MemoryIndexHealthChecker, MemoryIndexHealthReport
from memforge.memory.relation_candidate_retrieval import RelationCandidateRetriever
from memforge.memory.store import MemoryStore
from memforge.models import SourceSyncRun, SyncState
from memforge.genes.base import SourceConfigurationError
from memforge.genes.conversation_gene import CollectionAttempt
from memforge.genes.document_gene import DocumentGene
from memforge.genes.family import CollectionFamilyKind
from memforge.pipeline.collection_family import ExcludedScopeGeneFactory, SnapshotBinding
from memforge.source_types import source_type_descriptor
from memforge.conversations.collections import StoredCollections
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.sync_memory import SyncMemoryObserver
from memforge.pipeline.sync import (
    DocumentLifecycleAdmission,
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    SourceSyncMode,
    get_process_document_lifecycle_admission,
    get_process_extraction_work_pool,
)
from memforge.retrieval import vector_index
from memforge.release import current_release
from memforge.source_item_failures import REMOVAL_STAGE, FailedItemRetry
from memforge.source_secrets import decrypt_source_config_for_runtime, source_secret_fields
from memforge.source_activity import (
    SOURCE_SYNC_CANCELLED_BY_USER,
    SourceActivityConflict,
    SourceActivityKind,
    SourceActivityLease,
    SourceSyncHeartbeat,
)
from memforge.storage.document_store import LocalDocumentStore
from memforge.storage.adapters.sqlite import SqliteAdapters, SqliteVectorStore, build_sqlite_adapters
from memforge.sync_progress import SourceSyncProgressAccumulator, source_sync_progress_from_pipeline

if TYPE_CHECKING:
    from memforge.evals.agent_evaluation import RuntimeEventTraceSink
    from memforge.memory.reindex import MemoryVectorIndexes
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

# The trigger of the sync a new release queues for a Source with failed
# Documents of an earlier release and no sync cursor.
RELEASE_RECOVERY_TRIGGER = "release_recovery"


def failed_item_retry_for_run(run: SourceSyncRun) -> FailedItemRetry:
    """Which rows of the failure ledger a sync run retries.

    A run a person requested, or that a request coalesced into, retries every
    row; any other run retries only the rows that may pass (see
    :mod:`memforge.source_item_failures`).
    """

    return FailedItemRetry.ALL if run.retry_failed_items else FailedItemRetry.RETRYABLE


class SyncAlreadyRunningError(RuntimeError):
    """Raised when a source already has an active sync task."""


class SourcePausedError(RuntimeError):
    """Raised when sync is requested for a paused source."""


class SourceNotActiveError(RuntimeError):
    """Raised when sync is requested while a source lifecycle is not active."""


class SourceSyncLeaseLost(RuntimeError):
    """Raised when a worker no longer owns the leased source-sync run."""


class SourceSyncStopRequested(RuntimeError):
    """Raised when a person asked to stop the source-sync run a worker holds."""


class SourceSyncBoundaryError(RuntimeError):
    """Raised when a durable run no longer matches its captured input boundary."""


class SourceSyncUnsupportedError(SourceSyncBoundaryError):
    """Raised when a source type has no ordinary sync execution kind."""


class SourceSyncActivityConflict(SourceSyncBoundaryError, SourceActivityConflict):
    """Raised when an ordinary sync overlaps another Source activity lease."""


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
    # The Documents of every Source whose processing failed, as the failure ledger records them.
    failed_items: RuntimeHealthComponent | None = None


@dataclass(frozen=True)
class ResolvedModels:
    """The models one runtime calls, resolved for one database.

    Each invocation is the active binding of its kind compiled with its
    credential; ``None`` means no model of that kind can be called.
    """

    generation: ResolvedInvocation | None
    embedding: ResolvedInvocation | None
    # The settings of the embedding model that decide its vector space.
    embedding_identity: EmbeddingIdentity | None = None
    request_timeout_s: float = 300.0
    max_input_tokens: int | None = None
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    input_budget_fraction: float = 0.8
    # The route of the model that answers registered Decision tasks. It is
    # reached through the generation model's endpoint and credential; ``None``
    # runs every task on the generation model.
    decision_route: str | None = None

    @property
    def generation_route(self) -> str:
        """The route of the generation model, or an empty string without one."""
        return self.generation.route if self.generation is not None else ""


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
    relation_discovery: Any | None = None
    extraction_pool: ExtractionWorkPool | None = None
    document_lifecycle_admission: DocumentLifecycleAdmission | None = None
    memory_observer: SyncMemoryObserver | None = None
    runtime_event_trace_sink: "RuntimeEventTraceSink | None" = None
    orchestrator_factory: Callable[["SyncRuntime"], GeneSyncOrchestrator] | None = None

    def __post_init__(self) -> None:
        if self.extraction_pool is None:
            self.extraction_pool = get_process_extraction_work_pool(
                max(
                    1,
                    int(self.config.sync.max_extraction_workers or self.config.llm.enrichment_max_concurrent),
                )
            )
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
            max_concurrent=self.config.llm.enrichment_max_concurrent,
            extraction_pool=self.extraction_pool,
            document_lifecycle_admission=self.document_lifecycle_admission,
            memory_observer=self.memory_observer,
            structured_llm_client=self.structured_llm_client,
            runtime_event_trace_sink=self.runtime_event_trace_sink,
        )


def _publish_terminal_runtime_bundles(
    runtime: SyncRuntime,
    state: SyncState,
) -> None:
    sink = runtime.runtime_event_trace_sink
    if sink is None:
        return
    assessment_sink = assessment_sink_for_runtime_sink(sink)
    for bundle in state.runtime_bundles:
        publish_runtime_events(sink, bundle.events)
        publish_agent_assessments(
            assessment_sink,
            bundle.assessments,
            bundle.events,
        )


class RuntimeProvider(Protocol):
    """Runtime construction seam for admin apps with non-SQLite stores."""

    def build_structured_llm_client(
        self,
        models: ResolvedModels,
        *,
        max_concurrent: int,
    ) -> LiteLlmStructuredClient | None: ...

    def build_adapters(
        self,
        db: "Database",
        config: AppConfig,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any:
        """Return the relational, keyword, and vector adapters bound to ``db``.

        The provider owns its vector backend, so callers never open a vector
        collection themselves.
        """
        ...

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
        input_snapshot: SnapshotBinding | None = None,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
        reusable_projection_doc_ids: frozenset[str] = frozenset(),
        record_terminal_result: bool = True,
        source_activity: SourceActivityLease | None = None,
        failed_item_retry: FailedItemRetry = FailedItemRetry.ALL,
        collection_attempt: CollectionAttempt | None = None,
    ) -> SyncState: ...


@runtime_checkable
class VectorIndexProvider(Protocol):
    """A runtime provider whose Memory vector index is settled and rebuilt by this service.

    ``memory_vector_indexes`` returns the indexes of one database, or ``None``
    for a database whose index the provider keeps by its own means. A provider
    without the method keeps every index that way: no index is recorded, no
    rebuild is offered and nothing is guarded for one.
    """

    def memory_vector_indexes(self, db: "Database", config: AppConfig) -> "MemoryVectorIndexes | None": ...


def memory_vector_indexes(
    provider: object,
    db: "Database",
    config: AppConfig,
) -> "MemoryVectorIndexes | None":
    """Return the vector indexes ``provider`` lets this service manage for ``db``, if it does."""
    if isinstance(provider, VectorIndexProvider):
        return provider.memory_vector_indexes(db, config)
    return None


def first_index_settler(provider: object, db: "Database", config: AppConfig) -> ProfileObserver | None:
    """Return what an Embedding Runtime tells its profile to, so that ``db`` gets its first index.

    An installation with no vector index in use gets one the first time its
    embedding model answers: the answer shows the vector length, and with it
    the Embedding Profile the index is recorded for.
    """
    indexes = memory_vector_indexes(provider, db, config)
    if indexes is None:
        return None

    async def settle(profile: EmbeddingProfile) -> None:
        if await indexes.in_use() is None:
            await indexes.settle(profile)

    return settle


class DefaultRuntimeProvider:
    """Default SQLite-backed runtime provider."""

    def memory_vector_indexes(self, db: "Database", config: AppConfig) -> "MemoryVectorIndexes | None":
        return _default_vector_indexes(db, config)

    def build_structured_llm_client(
        self,
        models: ResolvedModels,
        *,
        max_concurrent: int,
    ) -> LiteLlmStructuredClient | None:
        if models.generation is None:
            return None
        return LiteLlmStructuredClient(
            StructuredLlmConfig.for_invocation(
                models.generation,
                decision_model=models.decision_route,
                timeout_s=models.request_timeout_s,
                max_concurrent=max_concurrent,
                max_input_tokens=models.max_input_tokens,
                context_window_tokens=models.context_window_tokens,
                max_output_tokens=models.max_output_tokens,
                input_budget_fraction=models.input_budget_fraction,
            )
        )

    def build_adapters(
        self,
        db: "Database",
        config: AppConfig,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> SqliteAdapters:
        return _build_default_adapters(db, config, audit_logger=audit_logger)

    async def build_search_engine(
        self,
        db: "Database",
        config: AppConfig,
        *,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> Any:
        models = await resolve_models(db, config)
        structured_llm_client = self.build_structured_llm_client(
            models,
            max_concurrent=config.llm.enrichment_max_concurrent,
        )
        return _build_default_search_engine(
            db=db,
            config=config,
            models=models,
            structured_llm_client=structured_llm_client,
            audit_logger=audit_logger,
            first_index=first_index_settler(self, db, config),
        )

    async def build_sync_runtime(
        self,
        db: "Database",
        config: AppConfig,
        *,
        extraction_pool: ExtractionWorkPool | None = None,
        document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
    ) -> SyncRuntime:
        models = await resolve_models(db, config)
        structured_llm_client = self.build_structured_llm_client(
            models,
            max_concurrent=config.llm.enrichment_max_concurrent,
        )
        return _build_default_sync_runtime(
            db=db,
            config=config,
            models=models,
            structured_llm_client=structured_llm_client,
            extraction_pool=extraction_pool,
            document_lifecycle_admission=document_lifecycle_admission,
            first_index=first_index_settler(self, db, config),
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
        input_snapshot: SnapshotBinding | None = None,
        reprocess_doc_ids: frozenset[str] | None = None,
        execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
        lifecycle_cycle_id: str | None = None,
        scope_transition_run_id: str | None = None,
        reusable_projection_doc_ids: frozenset[str] = frozenset(),
        record_terminal_result: bool = True,
        source_activity: SourceActivityLease | None = None,
        failed_item_retry: FailedItemRetry = FailedItemRetry.ALL,
        collection_attempt: CollectionAttempt | None = None,
    ) -> SyncState:
        return await run_source_sync(
            db=db,
            config=config,
            source=source,
            runtime=runtime,
            progress_callback=progress_callback,
            force_full_sync=force_full_sync,
            input_snapshot=input_snapshot,
            reprocess_doc_ids=reprocess_doc_ids,
            execution_mode=execution_mode,
            lifecycle_cycle_id=lifecycle_cycle_id,
            scope_transition_run_id=scope_transition_run_id,
            reusable_projection_doc_ids=reusable_projection_doc_ids,
            record_terminal_result=record_terminal_result,
            source_activity=source_activity,
            failed_item_retry=failed_item_retry,
            collection_attempt=collection_attempt,
        )


async def resolve_models(db: "Database", config: AppConfig) -> ResolvedModels:
    """Resolve the generation and embedding models a runtime calls with ``db``.

    Each is the active binding of its kind for this database, compiled with its
    credential. The decision model comes from ``config``, and so do the request
    limits the generation binding does not state itself.
    """
    generation_binding = await resolve_active_binding(db, config, "generation")
    generation = await usable_invocation(db, config, generation_binding)
    decision_route: str | None = None
    if generation_binding is not None and generation is not None:
        decision_route = _decision_route(generation_binding.provider, config.llm.decision_model)
    embedding_binding = await resolve_active_binding(db, config, "embedding")
    embedding = await usable_invocation(db, config, embedding_binding)
    return ResolvedModels(
        generation=generation,
        embedding=embedding,
        embedding_identity=(
            embedding_identity(embedding_binding)
            if embedding_binding is not None and embedding is not None
            else None
        ),
        request_timeout_s=config.llm.request_timeout_s,
        **generation_token_limits(
            config.llm,
            {} if generation_binding is None else generation_binding.token_limits,
        ),
        input_budget_fraction=config.llm.input_budget_fraction,
        decision_route=decision_route,
    )


def generation_token_limits(llm: Any, binding_limits: Mapping[str, int]) -> dict[str, int | None]:
    """The token limits requests to a generation model are sized with.

    A limit the binding states is the one used. The ``MEMFORGE_LLM_*`` limits
    apply to whatever the binding leaves open, and a model with a stated
    context window takes that as the most input it reads.
    """
    context_window = binding_limits.get("context_window_tokens", llm.context_window_tokens)
    return {
        "max_input_tokens": llm.max_input_tokens or binding_limits.get("context_window_tokens"),
        "context_window_tokens": context_window,
        "max_output_tokens": binding_limits.get("max_output_tokens", llm.max_output_tokens),
    }


def _decision_route(provider: str, decision_model: str) -> str | None:
    """Compile the decision model as a model of the generation model's provider.

    The setting is a model id of that provider, or the full route of one: a
    name that starts with the provider's own route prefix is taken as the route
    it spells.
    """
    name = decision_model.strip()
    if not name:
        return None
    try:
        descriptor = provider_descriptor(provider)
        prefix = descriptor.route_prefixes.get("generation")
        if prefix and len(name) > len(prefix) + 1 and name.startswith(f"{prefix}/"):
            return name
        return descriptor.compile_route("generation", name)
    except ModelConfigError as exc:
        logger.warning(
            "MEMFORGE_DECISION_MODEL is not a model of the generation provider (%s); "
            "Decision tasks run on the generation model",
            exc.code,
        )
        return None


AUDIT_HEALTH_FAILURE_EVENTS = (
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
        index = (
            await vector_index.open_memory_vector_index(db, config.storage.chroma_path)
            if Path(config.storage.chroma_path).exists()
            else None
        )
        if index is not None:
            report = await MemoryIndexHealthChecker(
                db=db,
                memory_collection=index.collection,
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
                detail="No Memory vector index is in use",
            )
    except Exception as exc:
        overall = "degraded"
        index_consistency = RuntimeHealthComponent(status="error", detail=str(exc))

    audit_failures: RuntimeHealthComponent | None = None
    try:
        audit_failures = await recent_audit_failure_health(db)
    except Exception as exc:
        audit_failures = RuntimeHealthComponent(status="warning", detail=str(exc))

    failed_items: RuntimeHealthComponent | None = None
    try:
        failed_items = await failed_item_health(db)
    except Exception as exc:
        failed_items = RuntimeHealthComponent(status="warning", detail=str(exc))

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
        failed_items=failed_items,
    )


async def failed_item_health(store: Any) -> RuntimeHealthComponent:
    """Count the failed Documents of every Source, and those a new release is retrying.

    A failed Document is per-item work the failure ledger retries, so it never
    makes the runtime unhealthy; it is a warning while any is left.
    """

    release = current_release()
    failures = await store.list_source_item_failures()
    retrying = sum(failure.recorded_by_another_release(release) for failure in failures)
    payload = {
        "failed_items": len(failures),
        "retrying_after_upgrade": retrying,
        "sources": len({failure.source_id for failure in failures}),
        "release": release,
    }
    if not failures:
        return RuntimeHealthComponent(status="ok", detail="No failed items", payload=payload)
    return RuntimeHealthComponent(
        status="warning",
        detail=f"{len(failures)} failed item(s) in {payload['sources']} source(s)",
        payload=payload,
    )


async def recent_audit_failure_health(store: Any) -> RuntimeHealthComponent:
    """Count the failed lifecycle and index audit events of the last day; any is a warning."""

    since = (datetime.now(timezone.utc) - timedelta(hours=AUDIT_HEALTH_WINDOW_HOURS)).isoformat()
    counts = await store.count_failed_memory_audit_events(AUDIT_HEALTH_FAILURE_EVENTS, since=since)
    counts_by_event_type = {count.event_type: count.count for count in counts}
    payload = {
        "window_hours": AUDIT_HEALTH_WINDOW_HOURS,
        "since": since,
        "counts_by_event_type": counts_by_event_type,
        "total": sum(counts_by_event_type.values()),
        "last_seen_at": max((count.last_seen_at for count in counts if count.last_seen_at), default=None),
    }
    if not counts:
        return RuntimeHealthComponent(
            status="ok",
            detail=f"No audit failures in the last {AUDIT_HEALTH_WINDOW_HOURS}h",
            payload=payload,
        )
    summary = ", ".join(f"{count.event_type}={count.count}" for count in counts)
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


def build_embedding_runtime(
    invocation: ResolvedInvocation | None,
    config: AppConfig,
    *,
    identity: EmbeddingIdentity | None = None,
    profile_observer: ProfileObserver | None = None,
) -> EmbeddingRuntime:
    """Build the Embedding Runtime that calls ``invocation``.

    Without an invocation the runtime reports that no embedding model is
    configured and every embed call fails. ``identity`` names the vector space
    of the model, so that its vectors reach only an index built with it.
    """
    return EmbeddingRuntime(
        EmbeddingSettings(
            invocation=invocation,
            identity=identity,
            cache_size=config.retrieval.embedding_cache_size,
        ),
        profile_observer=profile_observer,
    )


async def build_active_embedding_runtime(
    db: "Database",
    config: AppConfig,
    provider: object,
) -> EmbeddingRuntime:
    """Build the Embedding Runtime of the active embedding binding of ``db``."""
    binding = await resolve_active_binding(db, config, "embedding")
    invocation = await usable_invocation(db, config, binding)
    return build_embedding_runtime(
        invocation,
        config,
        identity=None if binding is None or invocation is None else embedding_identity(binding),
        profile_observer=first_index_settler(provider, db, config),
    )


def _retrieval_config_for_models(config: AppConfig, models: ResolvedModels):
    """Use the generation model for the optional retrieval rerank."""
    return replace(config.retrieval, rerank_model=models.generation_route)


def _build_default_adapters(
    db: "Database",
    config: AppConfig,
    *,
    audit_logger: MemoryAuditLogger | None,
) -> SqliteAdapters:
    """Bind the SQLite stores to the Memory vector index in use at ``config.storage.chroma_path``."""
    chroma_path = config.storage.chroma_path

    async def open_index() -> vector_index.MemoryVectorIndex | None:
        return await vector_index.open_memory_vector_index(db, chroma_path)

    return build_sqlite_adapters(
        db,
        SqliteVectorStore.of_index_in_use(open_index),
        audit_logger=audit_logger,
    )


def _default_vector_indexes(db: "Database", config: AppConfig) -> "MemoryVectorIndexes | None":
    """The Chroma indexes recorded in a SQLite database; any other store has none here."""
    from memforge.memory.reindex import MemoryVectorIndexes
    from memforge.storage.database import Database

    if not isinstance(db, Database):
        return None
    return MemoryVectorIndexes(db, config.storage.chroma_path)


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
    return await DefaultRuntimeProvider().build_search_engine(
        db,
        config,
        audit_logger=audit_logger,
    )


def _build_default_search_engine(
    *,
    db: "Database",
    config: AppConfig,
    models: ResolvedModels,
    structured_llm_client: LiteLlmStructuredClient | None,
    audit_logger: MemoryAuditLogger | None,
    first_index: ProfileObserver | None = None,
) -> Any:
    from memforge.retrieval.search import SearchEngine

    adapters = _build_default_adapters(db, config, audit_logger=audit_logger)
    return SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embedding=build_embedding_runtime(
            models.embedding,
            config,
            identity=models.embedding_identity,
            profile_observer=first_index,
        ),
        config=_retrieval_config_for_models(config, models),
        structured_llm_client=structured_llm_client,
    )


async def build_sync_runtime(
    db: "Database",
    config: AppConfig,
    *,
    extraction_pool: ExtractionWorkPool | None = None,
    document_lifecycle_admission: DocumentLifecycleAdmission | None = None,
) -> SyncRuntime:
    return await DefaultRuntimeProvider().build_sync_runtime(
        db,
        config,
        extraction_pool=extraction_pool,
        document_lifecycle_admission=document_lifecycle_admission,
    )


def _build_default_sync_runtime(
    *,
    db: "Database",
    config: AppConfig,
    models: ResolvedModels,
    structured_llm_client: LiteLlmStructuredClient | None,
    extraction_pool: ExtractionWorkPool | None,
    document_lifecycle_admission: DocumentLifecycleAdmission | None,
    first_index: ProfileObserver | None = None,
) -> SyncRuntime:
    from memforge.evals.agent_evaluation import runtime_event_trace_sink_from_env
    doc_store = LocalDocumentStore(config.storage.docs_path)
    runtime_event_trace_sink = runtime_event_trace_sink_from_env()

    memory_extractor = MemoryExtractor(
        model=models.generation_route,
        request_timeout_s=models.request_timeout_s,
        structured_llm_client=structured_llm_client,
    )

    embedding = build_embedding_runtime(
        models.embedding,
        config,
        identity=models.embedding_identity,
        profile_observer=first_index,
    )
    adapters = _build_default_adapters(
        db,
        config,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
    )
    memory_store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embedding=embedding,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
    )
    memory_engine = MemoryEngine(
        relation_candidates=RelationCandidateRetriever(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
        ),
        db=db,
        document_store=doc_store,
        memory_store=memory_store,
        embedding=embedding,
        structured_llm_client=structured_llm_client,
        llm_model=models.generation_route,
        runtime_event_trace_sink=runtime_event_trace_sink,
    )
    relation_discovery = None
    if memory_engine.pair_classifier is not None:
        from memforge.memory.relation_discovery import RelationDiscovery

        relation_discovery = RelationDiscovery(
            store=adapters.relational,
            candidate_retriever=memory_engine.relation_candidates,
            pair_classifier=memory_engine.pair_classifier,
        )

    return SyncRuntime(
        db=db,
        config=config,
        doc_store=doc_store,
        memory_extractor=memory_extractor,
        memory_store=memory_store,
        memory_engine=memory_engine,
        structured_llm_client=structured_llm_client,
        llm_model=models.generation_route,
        relation_discovery=relation_discovery,
        extraction_pool=extraction_pool,
        document_lifecycle_admission=document_lifecycle_admission,
        memory_observer=SyncMemoryObserver(),
        runtime_event_trace_sink=runtime_event_trace_sink,
    )


def excluded_scope_gene_factory(source: Mapping[str, Any], source_config: Mapping[str, Any]) -> ExcludedScopeGeneFactory:
    """Authenticated Genes that list the scopes a Source's scope change excluded.

    ``source_config`` is the Source's configuration as the run uses it:
    decrypted, with any browser session cookie injected. Each Gene reads it
    with the excluded scope's fields in place of the current ones, so it
    reuses the current credential and never sees a current scope field.

    Each Gene is open only inside its ``async with`` block and closed when
    the block exits, including when ``authenticate`` fails.

    An excluded scope whose configuration or Gene cannot be built, before
    anything reaches the provider, raises ``SourceConfigurationError``: no
    later run could build it either. ``authenticate`` failures stay as they
    are, because a credential or a network can recover.
    """

    source_type = str(source["type"])

    @contextlib.asynccontextmanager
    async def excluded_scope_gene(scope: Mapping[str, object]) -> AsyncIterator[DocumentGene]:
        try:
            gene = cast(
                DocumentGene,
                create_gene(
                    name=source_type,
                    config=excluded_scope_config(source_type, source_config, scope),
                    source_id=str(source["id"]),
                ),
            )
        except ValueError as exc:
            raise SourceConfigurationError(f"The excluded scope {dict(scope)} cannot be configured: {exc}") from exc
        try:
            await gene.authenticate()
            yield gene
        finally:
            await gene.close()

    return excluded_scope_gene


async def run_source_sync(
    db: "Database",
    config: AppConfig,
    source: dict,
    runtime: SyncRuntime | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    force_full_sync: bool = False,
    input_snapshot: SnapshotBinding | None = None,
    reprocess_doc_ids: frozenset[str] | None = None,
    execution_mode: SourceSyncMode = SourceSyncMode.NORMAL,
    lifecycle_cycle_id: str | None = None,
    scope_transition_run_id: str | None = None,
    reusable_projection_doc_ids: frozenset[str] = frozenset(),
    record_terminal_result: bool = True,
    source_activity: SourceActivityLease | None = None,
    failed_item_retry: FailedItemRetry = FailedItemRetry.ALL,
    collection_attempt: CollectionAttempt | None = None,
) -> SyncState:
    """Run one Source sync under a Source activity lease.

    ``collection_attempt`` names the collection attempt a conversation Source
    syncs within; a document Source ignores it. ``input_snapshot`` binds a
    document Source's run to the complete snapshot its device submitted.
    """
    activity_id = source_activity.id if source_activity is not None else None
    owns_activity = False
    heartbeat_task: asyncio.Task[None] | None = None
    activity_lease = source_activity
    if source_activity is not None:
        if source_activity.source_id != str(source["id"]):
            raise ValueError("durable Source activity does not belong to the synced Source")
        if source_activity.kind is not SourceActivityKind.SYNC:
            raise ValueError("durable Source activity must be a sync authority")
    else:
        activity_id = f"source-sync-{uuid.uuid4().hex}"
        try:
            activity_lease = await db.acquire_source_activity(
                activity_id=activity_id,
                source_id=str(source["id"]),
                kind=SourceActivityKind.SYNC,
                lease_seconds=300,
            )
            owns_activity = True
        except SourceActivityConflict as exc:
            raise SourceSyncActivityConflict(str(exc)) from exc

        async def heartbeat_activity() -> None:
            while True:
                await asyncio.sleep(60)
                await db.renew_source_activity(
                    activity_id=activity_id,
                    lease_seconds=300,
                )

        heartbeat_task = asyncio.create_task(heartbeat_activity())
    lifecycle_cycle_id = lifecycle_cycle_id or activity_id
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
        sync_task: asyncio.Task[Any] | None = None
        try:
            sync_kwargs: dict[str, Any] = {
                "gene": gene,
                "source_name": source["name"],
                "source_id": source["id"],
                "progress_callback": progress_callback,
                "force_full_sync": force_full_sync,
                "input_snapshot": input_snapshot,
                "reprocess_doc_ids": reprocess_doc_ids,
                "execution_mode": execution_mode,
                "source_activity": activity_lease,
                "lifecycle_cycle_id": lifecycle_cycle_id,
                "scope_transition_run_id": scope_transition_run_id,
                "reusable_projection_doc_ids": reusable_projection_doc_ids,
                "record_terminal_result": record_terminal_result,
                "failed_item_retry": failed_item_retry,
                "collection_attempt": collection_attempt,
                "excluded_scope_gene": excluded_scope_gene_factory(source, source_config),
            }
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
                raise SourceSyncActivityConflict(f"source activity heartbeat stopped: {activity_id}")
            return await sync_task
        finally:
            try:
                # A cancelled caller leaves the sync task running; stop it before its client closes.
                if sync_task is not None and not sync_task.done():
                    sync_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sync_task
            finally:
                try:
                    await gene.close()
                except Exception:
                    # The run's own result or error is what callers act on; a failed release must not replace it.
                    logger.warning(
                        "Closing the %s Gene for source %s failed", source["type"], source["id"], exc_info=True
                    )
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        if owns_activity and activity_id is not None:
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
        max_extraction_workers = max(
            1,
            int(config.sync.max_extraction_workers or config.llm.enrichment_max_concurrent),
        )
        self._extraction_pool = get_process_extraction_work_pool(max_extraction_workers)
        max_document_lifecycles = max(0, int(config.sync.max_document_lifecycles))
        self._document_lifecycle_admission = get_process_document_lifecycle_admission(max_document_lifecycles)
        # The runtimes idle work reuses between slices. Each is built for the
        # models in use when it was built (see models_changed).
        self._relation_runtime: SyncRuntime | None = None
        self._vector_runtime: SyncRuntime | None = None
        self._models_version = 0

    def models_changed(self) -> None:
        """Discard the runtimes built for the models in use until now.

        The server calls this when a model binding is replaced, so that the next
        slice of idle work builds its runtime for the model, endpoint and key
        now in use. A Source sync that is running finishes with the runtime it
        started with.
        """
        self._models_version += 1
        self._relation_runtime = None
        self._vector_runtime = None

    async def _build_runtime(self) -> SyncRuntime:
        """Build a sync runtime for the models in use once it is built."""
        while True:
            models_version = self._models_version
            runtime = await self.runtime_provider.build_sync_runtime(
                self.db,
                self.config,
                extraction_pool=self._extraction_pool,
                document_lifecycle_admission=self._document_lifecycle_admission,
            )
            if models_version == self._models_version:
                return runtime

    async def process_lifecycle_vector_delivery_once(self) -> bool:
        """Process one bounded durable vector-outbox slice while the worker is idle."""

        try:
            from memforge.memory.store import LIFECYCLE_VECTOR_DELIVERY_MAX_ATTEMPTS

            pending = await self.db.list_ready_lifecycle_vector_tasks(
                limit=1,
                max_attempts=LIFECYCLE_VECTOR_DELIVERY_MAX_ATTEMPTS,
            )
            if not pending:
                return False
            if self._vector_runtime is None:
                self._vector_runtime = await self._build_runtime()
            result = await self._vector_runtime.memory_store.attempt_lifecycle_vector_delivery()
            self._discard_runtime_of_another_index(result)
            if result.attempted_tasks:
                logger.info(
                    "Lifecycle vector slice attempted=%d delivered=%d failed=%d pending=%s",
                    result.attempted_tasks,
                    result.delivered_tasks,
                    result.failed_tasks,
                    result.pending,
                )
            return bool(result.attempted_tasks)
        except Exception:
            logger.exception("Lifecycle vector worker slice failed")
            return False

    async def process_review_vector_delivery_once(self) -> bool:
        """Process one bounded Review-owned vector-outbox slice."""

        try:
            from memforge.memory.store import LIFECYCLE_VECTOR_DELIVERY_MAX_ATTEMPTS

            pending = await self.db.list_ready_review_vector_tasks(
                limit=1,
                max_attempts=LIFECYCLE_VECTOR_DELIVERY_MAX_ATTEMPTS,
            )
            if not pending:
                return False
            if self._vector_runtime is None:
                self._vector_runtime = await self._build_runtime()
            result = await self._vector_runtime.memory_store.attempt_review_vector_delivery()
            self._discard_runtime_of_another_index(result)
            if result.attempted_tasks:
                logger.info(
                    "Review vector slice attempted=%d delivered=%d failed=%d pending=%s",
                    result.attempted_tasks,
                    result.delivered_tasks,
                    result.failed_tasks,
                    result.pending,
                )
            return bool(result.attempted_tasks)
        except Exception:
            logger.exception("Review vector worker slice failed")
            return False

    def _discard_runtime_of_another_index(self, result: Any) -> None:
        """Build the vector runtime again once its model is not the index's.

        The index may have been rebuilt for a model that became the one in use
        after this runtime was built. The next slice then works with a runtime
        of that model; while the index still waits for its rebuild, the work
        stays pending.
        """
        if EmbeddingIndexMismatch.__name__ in result.error_types:
            self._vector_runtime = None

    async def _process_relation_discovery_once(self) -> bool:
        try:
            if self._relation_runtime is None:
                from memforge.memory.relation_discovery import DEFAULT_RELATION_DISCOVERY_BUDGET

                if not await self.db.has_ready_relation_discovery_work(
                    max_attempts=DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts,
                ):
                    return False
                self._relation_runtime = await self._build_runtime()
            processor = self._relation_runtime.relation_discovery
            if processor is None:
                return False
            result = await processor.process_slice(
                worker_id=f"{self.worker_id}:relation",
            )
            if result.attempted_work:
                logger.info(
                    "Relation discovery slice attempted=%d completed=%d failed=%d "
                    "obsolete=%d pairs=%d llm_calls=%d elapsed_ms=%d exhausted_total=%d",
                    result.attempted_work,
                    result.completed_work,
                    result.failed_work,
                    result.obsolete_work,
                    result.checked_candidate_pairs,
                    result.llm_calls,
                    result.elapsed_ms,
                    result.exhausted_work,
                )
            return bool(result.attempted_work)
        except Exception:
            logger.exception("Relation discovery worker slice failed")
            return False

    async def process_bounded_maintenance_once(self) -> None:
        """Interleave one bounded slice of each non-source lifecycle queue.

        The queues wait while the Memory vector index is rebuilt: their work
        writes vectors and Memories, which a rebuild needs to stand still.
        """

        if await self._vector_reindex_running():
            return
        await self.process_lifecycle_vector_delivery_once()
        await self.process_review_vector_delivery_once()
        await self._process_relation_discovery_once()

    async def _vector_reindex_running(self) -> bool:
        if memory_vector_indexes(self.runtime_provider, self.db, self.config) is None:
            return False
        try:
            return await self.db.vector_reindex_running()
        except Exception:
            logger.exception("Could not tell whether the Memory vector index is being rebuilt")
            return True

    async def _heartbeat_until_stopped(
        self,
        run: SourceSyncRun,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
        stop_requested: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.heartbeat_seconds)
                return
            except asyncio.TimeoutError:
                try:
                    heartbeat = await self.db.heartbeat_source_sync_run(
                        run.run_id,
                        worker_id=self.worker_id,
                        lease_attempt_count=run.lease_attempt_count,
                        lease_seconds=self.lease_seconds,
                    )
                except Exception:
                    logger.exception("Source sync worker heartbeat failed for run %s", run.run_id)
                    lease_lost.set()
                    return
                if heartbeat is SourceSyncHeartbeat.LOST:
                    logger.warning("Source sync worker lost lease for run %s", run.run_id)
                    lease_lost.set()
                    return
                if heartbeat is SourceSyncHeartbeat.STOP_REQUESTED:
                    logger.info("Source sync worker stops run %s on request", run.run_id)
                    stop_requested.set()
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
        stop_requested = asyncio.Event()
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
        heartbeat_task = asyncio.create_task(self._heartbeat_until_stopped(run, stop, lease_lost, stop_requested))
        progress_task = asyncio.create_task(self._progress_until_stopped(run, stop, lease_lost, latest_progress))
        sync_task = asyncio.create_task(self.runtime_provider.run_source_sync(**kwargs))
        try:
            done, _ = await asyncio.wait(
                {sync_task, heartbeat_task, progress_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            # A stop request ends the sync the way a lost lease does: the sync
            # task is cancelled, so it commits nothing further, removes nothing
            # and records no result. A sync that has already finished keeps its
            # result.
            if (lease_lost.is_set() or (stop_requested.is_set() and sync_task not in done)) and (
                heartbeat_task in done or progress_task in done
            ):
                sync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sync_task
                if lease_lost.is_set():
                    raise SourceSyncLeaseLost(f"source sync lease lost for run {run.run_id}")
                raise SourceSyncStopRequested(f"source sync run {run.run_id} was asked to stop")
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
            await self.process_bounded_maintenance_once()
            return None

        if run.stop_requested_at is not None:
            # The worker that held the run stopped before it could end it.
            return await self._end_stopped_run(run, await self.db.get_source(run.source_id))

        if run.lease_attempt_count > self.config.sync.worker_max_attempts:
            failed = await self.db.fail_source_sync_run(
                run.run_id,
                worker_id=self.worker_id,
                lease_attempt_count=run.lease_attempt_count,
                error_message="source sync execution attempt budget exhausted",
                retryable=False,
            )
            if not failed:
                raise SourceSyncLeaseLost(
                    f"source sync lease lost before exhausted-attempt update for run {run.run_id}"
                )
            return run

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

            if run.progress is None:
                initial_progress = source_sync_progress_from_pipeline(
                    {"phase": "discovering", "current": 0, "total": 0},
                    source_type=str(source.get("type") or ""),
                )
                if initial_progress is not None:
                    try:
                        stored = await self.db.report_source_sync_run_progress(
                            run.run_id,
                            worker_id=self.worker_id,
                            lease_attempt_count=run.lease_attempt_count,
                            progress=initial_progress,
                        )
                    except Exception:
                        logger.exception(
                            "Initial source sync progress update failed for run %s",
                            run.run_id,
                        )
                    else:
                        if not stored:
                            raise SourceSyncLeaseLost(
                                f"source sync lease lost before initial progress update for run {run.run_id}"
                            )

            # An operator reprocess reads the provider or stored Documents, not collected input.
            reprocessing = bool(run.reprocess_document_ids)
            input_snapshot: SnapshotBinding | None = None
            reusable_projection_doc_ids: frozenset[str] = frozenset()
            if not reprocessing:
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
                descriptor = source_type_descriptor(str(source.get("type") or ""))
                if (
                    run.input_snapshot_id is not None
                    and descriptor is not None
                    and descriptor.family is CollectionFamilyKind.DOCUMENT
                ):
                    input_snapshot = SnapshotBinding(
                        snapshot_id=run.input_snapshot_id,
                        doc_ids=snapshot_document_ids(inputs),
                        confirmations=await self.db.list_source_sync_snapshot_confirmations(
                            source_id=run.source_id,
                            workspace_id=run.workspace_id,
                            snapshot_id=run.input_snapshot_id,
                        ),
                        excluded_listings=await self.db.list_source_sync_snapshot_excluded_listings(
                            source_id=run.source_id,
                            workspace_id=run.workspace_id,
                            snapshot_id=run.input_snapshot_id,
                        ),
                    )
                if run.input_snapshot_id is not None and not run.force_full_sync:
                    reusable_projection_doc_ids = await self.db.find_reusable_source_projection_memberships(
                        source_id=run.source_id,
                        workspace_id=run.workspace_id,
                        snapshot_id=run.input_snapshot_id,
                        expected_access_hash=projection_access_fingerprint(source_access_context(source)),
                    )
                source = source_with_sync_inputs(
                    source,
                    inputs,
                    input_snapshot_supplied=run.input_snapshot_id is not None,
                )

            runtime = await self._build_runtime()
            self._relation_runtime = runtime
            if run.source_activity is None:
                raise SourceSyncLeaseLost(
                    f"source sync claim is missing Source authority for run {run.run_id}"
                )
            final_state = await self._run_source_sync_with_heartbeat(
                run,
                db=self.db,
                config=self.config,
                source=source,
                runtime=runtime,
                progress_callback=None,
                force_full_sync=run.force_full_sync and not reprocessing,
                input_snapshot=input_snapshot,
                reprocess_doc_ids=frozenset(run.reprocess_document_ids) or None,
                execution_mode=(SourceSyncMode.REPROCESS if reprocessing else SourceSyncMode.NORMAL),
                lifecycle_cycle_id=(f"{run.run_id}:attempt:{run.lease_attempt_count}"),
                scope_transition_run_id=run.run_id,
                reusable_projection_doc_ids=reusable_projection_doc_ids,
                record_terminal_result=False,
                source_activity=run.source_activity,
                failed_item_retry=failed_item_retry_for_run(run),
                collection_attempt=CollectionAttempt(
                    workspace_id=run.workspace_id,
                    # A device collected into the run's input snapshot; a
                    # reprocess collects nothing and plans no attempt.
                    collection_attempt_id=run.input_snapshot_id or f"{run.run_id}:attempt:{run.lease_attempt_count}",
                ),
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
                # A partial result has already exhausted the document-level
                # retry budget and committed its successful documents. Replaying
                # the complete durable run would duplicate successful provider
                # and lifecycle work without improving the failure boundary.
                # An operator reprocess is not retried; the operator requests it again.
                next_attempt_at = (
                    self._next_retry_at(run, failed_at)
                    if final_state.last_sync_status == "failed"
                    and final_state.failure_retryable
                    and not reprocessing
                    else None
                )
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
                # A failure that would be retried ends a run someone asked to
                # stop, and the store then keeps none of its final state.
                ended = await self.db.get_source_sync_run(run.run_id)
                stopped = next_attempt_at is not None and ended is not None and ended.status != "pending"
                if not stopped:
                    _publish_terminal_runtime_bundles(runtime, final_state)
                if next_attempt_at is None or stopped:
                    await self._release_collection(run, source)
                return run
            completed = await self.db.complete_source_sync_run(
                run.run_id,
                worker_id=self.worker_id,
                lease_attempt_count=run.lease_attempt_count,
                final_state=final_state,
            )
            if not completed:
                raise SourceSyncLeaseLost(f"source sync lease lost before completion for run {run.run_id}")
            _publish_terminal_runtime_bundles(runtime, final_state)
            await self._release_collection(run, source)
            return run
        except asyncio.CancelledError:
            raise
        except SourceSyncLeaseLost:
            logger.warning("Source sync worker stopped terminal update after losing lease for run %s", run.run_id)
            return run
        except SourceSyncStopRequested:
            return await self._end_stopped_run(run, source)
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
                    (SourcePausedError, SourceNotActiveError, SourceSyncBoundaryError, SourceConfigurationError),
                )
                and next_attempt_at is not None
                and not run.reprocess_document_ids
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

    async def _end_stopped_run(self, run: SourceSyncRun, source: Mapping[str, Any] | None) -> SourceSyncRun:
        """Fail a run a person stopped, not retryable, and release its Source.

        The run ends like one whose lease was lost: what it committed stays,
        and its sync cursor and failure ledger are left as they were.
        """

        failed = await self.db.fail_source_sync_run(
            run.run_id,
            worker_id=self.worker_id,
            lease_attempt_count=run.lease_attempt_count,
            error_message=SOURCE_SYNC_CANCELLED_BY_USER,
            retryable=False,
        )
        if not failed:
            logger.warning("Source sync worker lost lease before ending stopped run %s", run.run_id)
            return run
        logger.info("Source sync run %s stopped on request", run.run_id)
        if source is not None:
            await self._release_collection(run, source)
        return run

    async def _release_collection(self, run: SourceSyncRun, source: Mapping[str, Any]) -> None:
        """Drop the collected bodies a conversation run has finished with, and every older collection.

        A run that will be retried keeps them: its retry plans the same
        collection attempt again. The run has ended by now, so a failure here
        leaves its outcome alone; the next run that ends prunes everything
        older than itself.
        """

        descriptor = source_type_descriptor(str(source.get("type") or ""))
        if (
            run.input_snapshot_id is None
            or descriptor is None
            or descriptor.family is not CollectionFamilyKind.CONVERSATION
        ):
            return
        try:
            await StoredCollections(self.db, workspace_id=run.workspace_id).prune(
                run.source_id, run.input_snapshot_id
            )
        except Exception:
            logger.exception("Could not drop the collected bodies of run %s", run.run_id)

    async def run_forever(self, *, poll_seconds: float | None = None) -> None:
        interval = max(0.1, poll_seconds if poll_seconds is not None else self.config.sync.worker_poll_seconds)
        while True:
            run = await self.run_once()
            if run is not None:
                # Fairly interleave one bounded non-destructive slice even while
                # source-sync work remains continuously available.
                await self.process_bounded_maintenance_once()
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
        max_extraction_workers = max(
            1,
            int(config.sync.max_extraction_workers or config.llm.enrichment_max_concurrent),
        )
        self._extraction_pool = get_process_extraction_work_pool(max_extraction_workers)
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
        retry_run_id: str | None = None,
        retry_failed_items: bool = False,
    ) -> SourceSyncRun:
        """Queue a sync of the Source, or coalesce into its active run.

        A ``manual`` or ``force`` sync, or one with ``retry_failed_items``,
        retries every failed Document of the Source; a local agent's sync
        carries that intent from the job a person requested.
        """
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
            retry_run_id=retry_run_id,
            retry_failed_items=retry_failed_items,
        )

    async def enqueue_reprocess(self, source_id: str, document_ids: tuple[str, ...]) -> SourceSyncRun:
        """Queue an operator reprocess of the Source Units of these Documents.

        It keeps the sync cursor, so the Source must have one; it is refused
        while another run of the Source is pending or running.
        """
        source = await self._ensure_source_can_sync(source_id)
        if not document_ids:
            raise ValueError("reprocess needs at least one Document")
        if not await self._has_sync_cursor(source_id):
            raise SourceSyncBoundaryError(f"Source {source_id} has not completed a sync to reprocess")
        return await self._enqueue_reprocess_run(source, document_ids)

    async def enqueue_release_recovery(self) -> int:
        """Queue one run per Source to retry the failed Documents another release recorded.

        Each such Document is retried once with this release's code, without
        waiting for a scheduled sync, so a Source without a schedule recovers
        too. A Source with a sync cursor reprocesses those Documents. A Source
        that has never completed a sync has no cursor for a reprocess to keep,
        so it syncs: its discovery is complete, and the run retries every
        failed Document of an earlier release. A Source that is paused,
        inactive or cannot sync keeps its rows; one whose run is already
        pending or running retries them in that run. Returns the number of
        runs queued.
        """
        release = current_release()
        failed_by_source: dict[str, list[str]] = {}
        for failure in await self.db.list_source_item_failures():
            if failure.recorded_by_another_release(release) and failure.stage != REMOVAL_STAGE:
                failed_by_source.setdefault(failure.source_id, []).append(failure.document_id)
        queued = 0
        for source_id, document_ids in sorted(failed_by_source.items()):
            try:
                source = await self._ensure_source_can_sync(source_id)
                if await self._has_sync_cursor(source_id):
                    await self._enqueue_reprocess_run(source, tuple(document_ids))
                else:
                    await self.enqueue_source(source_id, trigger=RELEASE_RECOVERY_TRIGGER)
            except SourceActivityConflict:
                logger.info(
                    "Release %s retries %d failed Document(s) of %s in its active sync run",
                    release,
                    len(document_ids),
                    source_id,
                )
                continue
            except (ValueError, SourcePausedError, SourceNotActiveError, SourceSyncUnsupportedError) as exc:
                logger.info("Release %s cannot retry the failed Documents of %s yet: %s", release, source_id, exc)
                continue
            queued += 1
            logger.info(
                "Release %s queued a retry of %d failed Document(s) of %s recorded by another release",
                release,
                len(document_ids),
                source_id,
            )
        return queued

    async def _has_sync_cursor(self, source_id: str) -> bool:
        state = await self.db.get_sync_state(source_id)
        return state is not None and state.last_sync_at is not None

    async def _enqueue_reprocess_run(self, source: dict, document_ids: tuple[str, ...]) -> SourceSyncRun:
        return await self.db.enqueue_source_sync_run(
            source_id=str(source["id"]),
            workspace_id=self.workspace_id,
            trigger="reprocess",
            source_config_revision=(
                local_agent_source_config_revision(source)
                if local_agent_sync_operation(source["type"], source.get("config")) is not None
                else None
            ),
            reprocess_document_ids=tuple(document_ids),
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

    async def retire_expired_memories(self) -> int:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        return await runtime.memory_store.retire_expired_memories()

    async def check_memory_index_health(self) -> MemoryIndexHealthReport:
        runtime = await self.runtime_provider.build_sync_runtime(self.db, self.config)
        checker = MemoryIndexHealthChecker(
            db=self.db,
            memory_collection=await runtime.memory_store.index_collection(),
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

    async def stop_source(self, source_id: str) -> SourceSyncRun | None:
        """Stop the Source's pending or running durable sync run, and return it as it now is.

        A pending run fails at once; a running run is stopped by its worker at
        the worker's next heartbeat (``Database.request_source_sync_stop``).
        """
        return await self.db.request_source_sync_stop(source_id=source_id)

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
            await self.db.record_source_sync_result(
                error_state,
                started_at=datetime.fromisoformat(started_at),
                finished_at=datetime.now(timezone.utc),
                run_id=f"sync-{uuid.uuid4().hex}",
            )
            return None
        finally:
            self.tasks.pop(source_id, None)
            self.progress.pop(source_id, None)
