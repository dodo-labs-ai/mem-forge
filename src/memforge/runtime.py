"""Shared runtime wiring for sync startup paths."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from memforge.config import AppConfig
from memforge.auth.jira_auth import JiraAuthSessionService, effective_jira_auth_mode
from memforge.genes import GENE_REGISTRY, create_gene
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.health import MemoryIndexHealthChecker, MemoryIndexHealthReport
from memforge.memory.store import MemoryStore
from memforge.models import SyncState
from memforge.pipeline.enricher import Enricher
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.source_support_detector import SourceSupportDetector
from memforge.pipeline.sync import GeneSyncOrchestrator
from memforge.retrieval.embeddings import get_chroma_collection
from memforge.source_secrets import decrypt_source_config_for_runtime, source_secret_fields
from memforge.storage.document_store import LocalDocumentStore

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)


class SyncAlreadyRunningError(RuntimeError):
    """Raised when a source already has an active sync task."""


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

    def orchestrator(self) -> GeneSyncOrchestrator:
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


async def build_sync_runtime(db: "Database", config: AppConfig) -> SyncRuntime:
    llm = await get_effective_llm_config(db, config)
    structured_llm_client = None
    if llm.enrichment_api_key:
        structured_llm_client = LiteLlmStructuredClient(
            StructuredLlmConfig(
                model=llm.enrichment_model,
                base_url=llm.enrichment_base_url or None,
                api_key=llm.enrichment_api_key,
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
    memory_store = MemoryStore(
        db=db,
        memory_collection=memory_collection,
        embed_cfg=embed_cfg,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="sync")),
        document_collection=doc_collection,
    )
    memory_engine = MemoryEngine(
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
    if (
        source["type"] == "jira"
        and effective_jira_auth_mode(source_config) == "browser_cookie"
    ):
        source_config["jira_cookie"] = await JiraAuthSessionService(db).cookie_header_for_sync(
            str(source_config.get("base_url") or ""),
            tls_config=source_config,
            allow_browser_refresh=False,
        )
    gene = create_gene(
        name=source["type"],
        config=source_config,
        source_id=source["id"],
    )
    return await runtime.orchestrator().sync_gene(
        gene=gene,
        source_name=source["name"],
        source_id=source["id"],
        progress_callback=progress_callback,
        force_full_sync=force_full_sync,
    )


class SyncService:
    """App-scoped sync runner with task tracking and shutdown cleanup."""

    def __init__(self, db: "Database", config: AppConfig) -> None:
        self.db = db
        self.config = config
        self.tasks: dict[str, asyncio.Task] = {}
        self.queued_tasks: dict[str, asyncio.Task] = {}
        self.progress: dict[str, dict] = {}

    def is_running(self, source_id: str) -> bool:
        task = self.tasks.get(source_id)
        return bool(task and not task.done())

    def start_source(self, source_id: str, *, force_full_sync: bool = False) -> asyncio.Task:
        if self.is_running(source_id):
            raise SyncAlreadyRunningError(f"Sync already running for {source_id}")
        task = asyncio.create_task(self._run_source_task(source_id, force_full_sync=force_full_sync))
        self.tasks[source_id] = task
        return task

    def request_source_sync(self, source_id: str, *, delay_seconds: float = 1.0) -> bool:
        """Queue one service-owned sync pass for a source, coalescing duplicates."""
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
                self.start_source(source_id)
            except SyncAlreadyRunningError:
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
            await self.start_source(source["id"])

    async def retire_expired_memories(self) -> int:
        runtime = await build_sync_runtime(self.db, self.config)
        return await runtime.memory_store.retire_expired_memories()

    async def check_memory_index_health(self) -> MemoryIndexHealthReport:
        runtime = await build_sync_runtime(self.db, self.config)
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

            runtime = await build_sync_runtime(self.db, self.config)

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

            return await run_source_sync(
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
            if (
                source
                and source.get("type") == "jira"
                and "browser session" in str(e).lower()
            ):
                await JiraAuthSessionService(self.db).mark_expired(
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
