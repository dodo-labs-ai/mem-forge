from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.store import MemoryStore
from memforge.models import (
    ContentItem,
    DocumentMetadata,
    Entity,
    EnrichmentResult,
    GeneMetadata,
    MemoryExtractionResult,
    NormalizedContent,
    RawEntityRef,
    RawContent,
    RawMemory,
    SyncState,
    FailedDoc,
    content_hash,
)
from memforge.pipeline.sync import (
    ExtractionWorkPool,
    GeneSyncOrchestrator,
    summarize_failed_documents,
)
from memforge.runtime import SyncService
from memforge.config import AppConfig, SyncConfig
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "sync-bookkeeping.db"))
    await database.connect()
    yield database
    await database.close()


class EmptyGene:
    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        if False:
            yield ContentItem(item_id="never", title="never", updated_at=datetime.now(timezone.utc))


def test_failed_document_summary_identifies_embedding_provider_outage():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="Embedding provider unreachable: [Errno 111] Connection refused",
            ),
        ],
    )

    assert message == (
        "1 document could not be synced. Embedding provider was unreachable for 1 document. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


def test_failed_document_summary_identifies_llm_provider_outage():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="litellm.InternalServerError: AnthropicException - Cannot connect to host provider.example:443",
            ),
        ],
    )

    assert message == (
        "1 document could not be synced. LLM provider was unreachable for 1 document. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


@pytest.mark.asyncio
async def test_extraction_work_pool_allows_one_source_to_use_all_workers():
    pool = ExtractionWorkPool(max_workers=6)
    entered = 0
    release = asyncio.Event()

    async def hold_slot() -> None:
        nonlocal entered
        async with pool.slot("src-a"):
            entered += 1
            await release.wait()

    tasks = [asyncio.create_task(hold_slot()) for _ in range(6)]
    for _ in range(20):
        if entered == 6:
            break
        await asyncio.sleep(0.01)

    assert entered == 6

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_extraction_work_pool_favors_waiting_source_over_extra_borrowed_work():
    pool = ExtractionWorkPool(max_workers=6)
    release_a = asyncio.Event()
    entered: list[str] = []

    async def hold_source_a() -> None:
        async with pool.slot("src-a"):
            entered.append("a")
            await release_a.wait()

    source_a_tasks = [asyncio.create_task(hold_source_a()) for _ in range(6)]
    for _ in range(20):
        if entered.count("a") == 6:
            break
        await asyncio.sleep(0.01)
    assert entered.count("a") == 6

    source_b_entered = asyncio.Event()
    release_b = asyncio.Event()
    extra_a_entered = asyncio.Event()

    async def wait_source_b() -> None:
        async with pool.slot("src-b"):
            entered.append("b")
            source_b_entered.set()
            await release_b.wait()

    async def wait_extra_source_a() -> None:
        async with pool.slot("src-a"):
            entered.append("extra-a")
            extra_a_entered.set()

    source_b_task = asyncio.create_task(wait_source_b())
    extra_a_task = asyncio.create_task(wait_extra_source_a())
    await asyncio.sleep(0)

    source_a_tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await source_a_tasks[0]

    await asyncio.wait_for(source_b_entered.wait(), timeout=1)
    assert not extra_a_entered.is_set()

    release_b.set()
    release_a.set()
    await asyncio.gather(*source_a_tasks[1:], source_b_task, extra_a_task)


@pytest.mark.asyncio
async def test_shared_extraction_pool_caps_orchestrator_work_across_sources(db: Database):
    for source_id in ("src-pool-a", "src-pool-b"):
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
        )

    pool = ExtractionWorkPool(max_workers=4)
    release_enrichment = asyncio.Event()
    release_fetch = asyncio.Event()
    release_fetch.set()
    enricher = BlockingEnricher(release=release_enrichment, target_entries=4)

    def make_orchestrator() -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=db,
            doc_store=StubDocumentStore(),
            enricher=enricher,
            memory_extractor=NoopMemoryExtractor(),
            memory_engine=NoopMemoryEngine(),
            memory_store=None,
            max_concurrent=4,
            extraction_pool=pool,
        )

    task_a = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=BlockingFetchGene(item_count=4, release=release_fetch),
            source_name="Source A",
            source_id="src-pool-a",
        )
    )
    task_b = asyncio.create_task(
        make_orchestrator().sync_gene(
            gene=BlockingFetchGene(item_count=4, release=release_fetch),
            source_name="Source B",
            source_id="src-pool-b",
        )
    )

    await asyncio.wait_for(enricher.target_reached.wait(), timeout=2)
    await asyncio.sleep(0.05)

    assert enricher.max_active == 4

    release_enrichment.set()
    states = await asyncio.gather(task_a, task_b)
    assert [state.last_sync_status for state in states] == ["success", "success"]


def test_failed_document_summary_keeps_rate_limit_precedence_over_llm_timeout_text():
    message = summarize_failed_documents(
        1,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="litellm.RateLimitError: 429 rate limit after request timeout",
            ),
        ],
    )

    assert message == (
        "1 Confluence document could not be imported. Confluence rate limited 1 document. "
        "Wait a few minutes, then retry the sync."
    )


def test_failed_document_summary_preserves_mixed_failure_guidance():
    message = summarize_failed_documents(
        3,
        [
            FailedDoc(
                doc_id="doc-1",
                title="Doc 1",
                error="Embedding provider unreachable: [Errno 111] Connection refused",
            ),
            FailedDoc(doc_id="doc-2", title="Doc 2", error="Confluence rate limit 429"),
            FailedDoc(doc_id="doc-3", title="Doc 3", error="PDF export did not produce a PDF"),
        ],
    )

    assert message == (
        "3 documents could not be synced. Embedding provider was unreachable for 1 document; "
        "PDF export was unavailable for 1 document; Confluence rate limited 1 document. "
        "Wait a few minutes, then retry the sync. "
        "Check the provider endpoint, network access, and service status, then retry the sync."
    )


class SinceRecordingEmptyGene(EmptyGene):
    def __init__(self) -> None:
        self.seen_since = None

    async def discover(self, since=None):
        self.seen_since = since
        if False:
            yield ContentItem(item_id="never", title="never", updated_at=datetime.now(timezone.utc))


class IncrementalNewDocumentGene:
    def __init__(self) -> None:
        self.seen_since = None

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="agent_session",
            display_name="Agent Session",
            description="",
            default_sync_interval_minutes=0,
            auth_method="local_file",
            data_shape="message",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        self.seen_since = since
        yield ContentItem(
            item_id="doc-new",
            title="New Session",
            source_url="agent-session://new",
            last_modified=datetime.now(timezone.utc),
            content_type="application/json",
            space_or_project="sessions",
            version="new-version",
        )

    async def fetch(self, item):
        return RawContent(item=item, body=b'{"summary":"new"}', content_type="application/json")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body="# New Session\n\nSummary")


class FailingAuthGene:
    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=60,
            auth_method="pat",
            data_shape="ticket",
        )

    async def authenticate(self) -> None:
        raise RuntimeError("auth failed")


class StubDocumentStore:
    def store_raw(self, *, source_name, title, content, content_type, extension=None):
        suffix = extension or ".raw"
        return f"file:///tmp/{source_name}/{title}{suffix}"

    def store_normalized(self, *, source_name, title, markdown):
        return f"file:///tmp/{source_name}/{title}.md"

    def delete_document_files(self, *, source_name, title):
        return None


class FailingPdfDocumentStore(StubDocumentStore):
    def store_raw(self, *, source_name, title, content, content_type, extension=None):
        if content_type == "application/pdf":
            raise RuntimeError("disk full while storing PDF")
        return super().store_raw(
            source_name=source_name,
            title=title,
            content=content,
            content_type=content_type,
            extension=extension,
        )


class NoopMemoryEngine:
    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        return []

    async def process_memories(self, **kwargs):
        return {"inserted": 0, "corroborated": 0, "skipped": 0}

    async def reconcile_and_persist(self, **kwargs):
        return {"added": 0, "updated": 0, "superseded": 0, "deleted": 0, "noop": 0}


class FailingDocumentDeleteMemoryStore:
    async def delete_document(self, doc_id: str, **kwargs):
        raise RuntimeError("delete document failed")


class CountingMemoryEngine(NoopMemoryEngine):
    def __init__(self, inserted: int):
        self.inserted = inserted
        self.enrichment_calls = 0
        self.process_calls = 0

    async def process_enrichment(self, *, doc_id, enrichment, doc_context=None):
        self.enrichment_calls += 1
        return []

    async def process_memories(self, **kwargs):
        self.process_calls += 1
        return {"inserted": self.inserted, "corroborated": 0, "skipped": 0}


class RecordingMemoryEngine(NoopMemoryEngine):
    def __init__(self) -> None:
        self.reconcile_calls: list[dict] = []

    async def reconcile_and_persist(self, **kwargs):
        self.reconcile_calls.append(kwargs)
        return await super().reconcile_and_persist(**kwargs)


class FailingVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}
        self.deleted: list[str] = []

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or []) if record_id in self.upserted]
        return {
            "ids": selected,
            "metadatas": [self.upserted[record_id].get("metadata", {}) for record_id in selected],
            "embeddings": [self.upserted[record_id].get("embedding") for record_id in selected],
            "documents": [self.upserted[record_id].get("document") for record_id in selected],
        }

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = {
                "metadata": metadatas[index] if metadatas else {},
                "embedding": embeddings[index] if embeddings else None,
                "document": documents[index] if documents else None,
            }
        raise RuntimeError("document vector failed after mutation")

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for record_id in ids:
            self.upserted.pop(record_id, None)


class FalseyVectorStore:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}

    def __bool__(self) -> bool:
        return False

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or []) if record_id in self.upserted]
        return {
            "ids": selected,
            "metadatas": [self.upserted[record_id].get("metadata", {}) for record_id in selected],
            "embeddings": [self.upserted[record_id].get("embedding") for record_id in selected],
            "documents": [self.upserted[record_id].get("document") for record_id in selected],
        }

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = {
                "metadata": metadatas[index] if metadatas else {},
                "embedding": embeddings[index] if embeddings else None,
                "document": documents[index] if documents else None,
            }

    def delete(self, *, ids):
        for record_id in ids:
            self.upserted.pop(record_id, None)


class FlakyFalseyVectorStore(FalseyVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise RuntimeError("transient vector failure")
        super().upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


class NoopMemoryExtractor:
    async def extract_memories(self, **kwargs):
        return MemoryExtractionResult(memories=[])


class RecordingMemoryExtractor(NoopMemoryExtractor):
    def __init__(self) -> None:
        self.full_calls: list[dict] = []
        self.change_calls: list[dict] = []
        self.unit_calls: list[dict] = []

    async def extract_memories(self, **kwargs):
        self.full_calls.append(kwargs)
        return MemoryExtractionResult(memories=[])

    async def extract_memory_changes(self, **kwargs):
        self.change_calls.append(kwargs)
        return MemoryExtractionResult(memories=[])

    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        return MemoryExtractionResult(memories=[])


class FailingMemoryExtractor(NoopMemoryExtractor):
    async def extract_memories(self, **kwargs):
        return MemoryExtractionResult(
            memories=[],
            error_type="json_parse_error",
            error="Unterminated string starting at line 393 column 16",
        )


class PartiallyFailingUnitMemoryExtractor(RecordingMemoryExtractor):
    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        if context.unit.heading_path[-1] == "Section 2":
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error="unit failed",
            )
        return MemoryExtractionResult(
            memories=[
                RawMemory(
                    content=f"{context.unit.heading_path[-1]} contains durable design guidance.",
                    memory_type="fact",
                    extraction_context="durable design guidance",
                )
            ]
        )


class BlockingUnitMemoryExtractor(RecordingMemoryExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.started_two = asyncio.Event()
        self.active = 0
        self.max_active = 0

    async def extract_unit_memories(self, context, **kwargs):
        self.unit_calls.append({"context": context, **kwargs})
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.max_active >= 2:
            self.started_two.set()
        try:
            await self.release.wait()
            return MemoryExtractionResult(memories=[])
        finally:
            self.active -= 1


class BlockingFetchGene:
    def __init__(self, item_count: int, release: asyncio.Event):
        self.item_count = item_count
        self.release = release
        self.active_fetches = 0
        self.max_active_fetches = 0

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=60,
            auth_method="pat",
            data_shape="ticket",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        for idx in range(self.item_count):
            yield ContentItem(
                item_id=f"jira-{idx}",
                title=f"Jira {idx}",
                source_url=f"https://jira.example/browse/{idx}",
                last_modified=datetime.now(timezone.utc),
                content_type="application/json",
                space_or_project="PAY",
                version=str(idx),
            )

    async def fetch(self, item):
        self.active_fetches += 1
        self.max_active_fetches = max(self.max_active_fetches, self.active_fetches)
        try:
            await self.release.wait()
            return RawContent(item=item, body=b'{"summary":"test"}', content_type="application/json")
        finally:
            self.active_fetches -= 1

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\nBody")


class PdfBackfillGene(BlockingFetchGene):
    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="confluence",
            display_name="Confluence",
            description="",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    async def fetch_pdf(self, item):
        return b"%PDF-1.4\n" + (b"x" * 128)


class MissingPdfGene(PdfBackfillGene):
    async def fetch_pdf(self, item):
        return None


class UpdatingDocumentGene:
    def __init__(self, markdown: str, version: str = "2") -> None:
        self.markdown = markdown
        self.version = version

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="docs",
            display_name="Documents",
            description="",
            default_sync_interval_minutes=1440,
            auth_method="pat",
            data_shape="document",
        )

    async def authenticate(self) -> None:
        return None

    async def discover(self, since=None):
        yield ContentItem(
            item_id="doc-1",
            title="Design Doc",
            source_url="https://docs.example/doc-1",
            last_modified=datetime.now(timezone.utc),
            content_type="text/markdown",
            space_or_project="ARCH",
            version=self.version,
        )

    async def fetch(self, item):
        return RawContent(item=item, body=self.markdown.encode("utf-8"), content_type="text/markdown")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=self.markdown)


class UpdatingTicketGene(UpdatingDocumentGene):
    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="",
            default_sync_interval_minutes=360,
            auth_method="browser_cookie",
            data_shape="ticket",
        )


class DocumentVisibleEnricher:
    def __init__(self, db: Database, source_id: str):
        self.db = db
        self.source_id = source_id

    async def enrich_document(self, *, doc_id, content, source_type):
        async with self.db.db.execute(
            "SELECT COUNT(*) FROM documents WHERE source = ? AND doc_id = ?",
            (self.source_id, doc_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1
        return EnrichmentResult(
            summary="Summary",
            tags=[],
            entities=[],
            relationships=[],
            doc_type="jira_issue",
            complexity="low",
        )


class EntityMentioningEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        return EnrichmentResult(
            summary="Summary",
            tags=["tag-one"],
            entities=[
                RawEntityRef(
                    name="Raw Extracted Entity",
                    type="service",
                    tags=["service"],
                    aliases=["Raw Alias"],
                )
            ],
            relationships=[],
            doc_type="jira_issue",
            complexity="low",
        )


class BlockingEnricher:
    def __init__(self, release: asyncio.Event, target_entries: int):
        self.release = release
        self.target_entries = target_entries
        self.entered = 0
        self.active = 0
        self.max_active = 0
        self.target_reached = asyncio.Event()

    async def enrich_document(self, *, doc_id, content, source_type):
        del doc_id, content, source_type
        self.entered += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.entered >= self.target_entries:
            self.target_reached.set()
        try:
            await self.release.wait()
            return EnrichmentResult(
                summary="Summary",
                tags=[],
                entities=[],
                relationships=[],
                doc_type="jira_issue",
                complexity="low",
            )
        finally:
            self.active -= 1


class ExplodingEnricher:
    async def enrich_document(self, *, doc_id, content, source_type):
        raise AssertionError("unchanged document should not be enriched")


async def _insert_source_and_doc(db: Database, source_id: str) -> None:
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Architecture",
        config_json="{}",
    )
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-1", source_id, "http://example/doc-1", "Doc 1", "ARCH", now, "1", "hash-1", now),
    )
    await db.update_source_doc_count(source_id, 1)


async def _insert_source_with_docs(db: Database, source_id: str, doc_ids: list[str]) -> None:
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Agent Session Summaries",
        config_json="{}",
    )
    now = datetime.now(timezone.utc).isoformat()
    for doc_id in doc_ids:
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, source_id, f"agent-session://{doc_id}", doc_id, "sessions", now, "1", f"hash-{doc_id}", now),
        )
    await db.update_source_doc_count(source_id, len(doc_ids))


async def _insert_document_with_metadata(
    db: Database,
    *,
    source_id: str,
    doc_id: str,
    title: str,
    markdown: str,
    version: str,
    normalized_content_uri: str | None = None,
) -> None:
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    now = datetime.now(timezone.utc)
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version,
            content_hash, normalized_content_uri, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            doc_id,
            source_id,
            f"http://example/{doc_id}",
            title,
            "ARCH",
            now.isoformat(),
            version,
            content_hash(markdown),
            normalized_content_uri,
            now.isoformat(),
        ),
    )
    await db.upsert_metadata(
        DocumentMetadata(
            doc_id=doc_id,
            summary="Existing summary",
            tags=["existing"],
            entities=[
                Entity(
                    id=1,
                    canonical_name="Existing Entity",
                    tags=[],
                    display_name="Existing Entity",
                )
            ],
            doc_type="jira_issue",
            complexity="low",
            enriched_at=now,
        )
    )
    await db.update_source_doc_count(source_id, 1)


def _audited_memory_store(db: Database) -> MemoryStore:
    adapters = build_sqlite_adapters(db, object())
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(
            db,
            default_context=AuditContext(actor_type="test", run_id="run-sync-bookkeeping"),
        ),
    )


@pytest.mark.asyncio
async def test_successful_zero_change_sync_advances_last_sync_and_keeps_doc_count(db: Database):
    source_id = "src-sync-bookkeeping"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime.now(timezone.utc) - timedelta(days=1)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
    )

    source = await db.get_source(source_id)
    assert state.last_sync_status == "success"
    assert state.docs_processed == 0
    assert state.last_sync_at is not None
    assert state.last_sync_at > previous_sync
    assert source["last_sync"] == state.last_sync_at.isoformat()
    assert source["doc_count"] == 1


@pytest.mark.asyncio
async def test_incremental_sync_uses_overlap_window_for_discovery(db: Database):
    source_id = "src-sync-overlap"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime(2026, 5, 26, 14, 55, 33, tzinfo=timezone.utc)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    gene = SinceRecordingEmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert gene.seen_since == previous_sync - timedelta(minutes=10)


@pytest.mark.asyncio
async def test_incremental_sync_does_not_delete_unchanged_documents_from_small_source(db: Database):
    source_id = "src-agent-sessions-incremental"
    await _insert_source_with_docs(db, source_id, ["doc-old-a", "doc-old-b"])
    previous_sync = datetime.now(timezone.utc) - timedelta(hours=1)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=2,
            docs_updated=2,
        ),
    )
    gene = IncrementalNewDocumentGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Agent Session Summaries",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert gene.seen_since == previous_sync - timedelta(minutes=10)
    assert await db.count_documents(source=source_id) == 3
    assert await db.get_document("doc-old-a") is not None
    assert await db.get_document("doc-old-b") is not None
    audit_rows = await db.list_memory_audit_events(event_type="document_delete_committed")
    assert audit_rows == []


@pytest.mark.asyncio
async def test_force_full_sync_ignores_incremental_cursor(db: Database):
    source_id = "src-force-full-overlap"
    await _insert_source_and_doc(db, source_id)
    previous_sync = datetime(2026, 5, 26, 14, 55, 33, tzinfo=timezone.utc)
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=previous_sync,
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    gene = SinceRecordingEmptyGene()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    await orchestrator.sync_gene(
        gene=gene,
        source_name="Architecture",
        source_id=source_id,
        force_full_sync=True,
    )

    assert gene.seen_since is None


@pytest.mark.asyncio
async def test_force_full_sync_reprocesses_unchanged_document(db: Database, tmp_path):
    source_id = "src-force-reprocess"
    markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    previous_path = tmp_path / "design-doc.md"
    previous_path.write_text(markdown, encoding="utf-8")
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=markdown,
        version="2",
        normalized_content_uri=str(previous_path),
    )
    extractor = RecordingMemoryExtractor()
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="2"),
        source_name="Documents",
        source_id=source_id,
        force_full_sync=True,
    )

    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_updated == 1
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) == 1
    assert extractor.change_calls == []
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "full_document"


@pytest.mark.asyncio
async def test_deletion_failure_marks_sync_failed(db: Database):
    source_id = "src-deletion-failure"
    await _insert_source_and_doc(db, source_id)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=FailingDocumentDeleteMemoryStore(),
    )

    state = await orchestrator.sync_gene(
        gene=EmptyGene(),
        source_name="Architecture",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "delete document failed" in state.failed_docs[0].error
    assert history[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_auth_failure_records_failed_sync_state_without_secondary_error(db: Database):
    source_id = "src-auth-fail"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Auth Failure Source",
        config_json="{}",
    )
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=None,
        enricher=None,
        memory_extractor=None,
        memory_engine=None,
        memory_store=None,
    )

    state = await orchestrator.sync_gene(
        gene=FailingAuthGene(),
        source_name="Auth Failure Source",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    stored_state = await db.get_sync_state(source_id)
    assert state.last_sync_status == "failed"
    assert state.error_message == "auth failed"
    assert stored_state.last_sync_status == "failed"
    assert stored_state.error_message == "auth failed"
    assert history[0]["status"] == "failed"
    assert history[0]["error_message"] == "auth failed"


@pytest.mark.asyncio
async def test_scheduled_sync_uses_tracked_source_tasks(db: Database, monkeypatch):
    source_id = "src-scheduled-tracked"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Scheduled Source",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())
    release = asyncio.Event()
    running_observed = False

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        nonlocal running_observed
        assert force_full_sync is False
        running_observed = service.is_running(running_source_id)
        await release.wait()
        service.tasks.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    run_task = asyncio.create_task(service.run_all_active_sources())
    for _ in range(20):
        if running_observed:
            break
        await asyncio.sleep(0.01)

    assert running_observed is True
    assert source_id in service.tasks
    release.set()
    await run_task
    assert source_id not in service.tasks


@pytest.mark.asyncio
async def test_sync_service_passes_force_full_sync_to_source_task(db: Database, monkeypatch):
    source_id = "src-force-service"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Force Service",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())
    captured: dict[str, object] = {}

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        captured["source_id"] = running_source_id
        captured["force_full_sync"] = force_full_sync

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task = await service.start_source(source_id, force_full_sync=True)
    await task

    assert captured == {"source_id": source_id, "force_full_sync": True}


@pytest.mark.asyncio
async def test_sync_service_limits_active_sources_without_rejecting_queued_sources(
    db: Database,
    monkeypatch,
):
    await db.upsert_source(
        id="src-a",
        type="jira",
        name="Source A",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-b",
        type="jira",
        name="Source B",
        config_json="{}",
    )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=1)),
    )
    release = asyncio.Event()
    started: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        started.append(running_source_id)
        try:
            await release.wait()
        finally:
            service.tasks.pop(running_source_id, None)
            service.progress.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task_a = await service.start_source("src-a")
    task_b = await service.start_source("src-b")
    await asyncio.sleep(0)

    assert service.is_running("src-a")
    assert service.is_running("src-b")
    assert started == ["src-a"]
    assert service.progress["src-b"]["phase"] == "queued"

    release.set()
    await asyncio.gather(task_a, task_b)
    assert started == ["src-a", "src-b"]


@pytest.mark.asyncio
async def test_sync_service_queues_ten_requested_sources_with_two_active(
    db: Database,
    monkeypatch,
):
    source_ids = [f"src-{idx}" for idx in range(10)]
    for source_id in source_ids:
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=f"Source {source_id}",
            config_json="{}",
        )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=2)),
    )
    release = asyncio.Event()
    started: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        started.append(running_source_id)
        try:
            await release.wait()
        finally:
            service.tasks.pop(running_source_id, None)
            service.progress.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    tasks = [await service.start_source(source_id) for source_id in source_ids]
    await asyncio.sleep(0)

    assert started == source_ids[:2]
    assert all(service.progress[source_id]["phase"] == "queued" for source_id in source_ids[2:])

    release.set()
    await asyncio.gather(*tasks)
    assert started == source_ids


@pytest.mark.asyncio
async def test_cancel_queued_source_clears_progress(db: Database, monkeypatch):
    await db.upsert_source(
        id="src-active",
        type="jira",
        name="Active Source",
        config_json="{}",
    )
    await db.upsert_source(
        id="src-queued",
        type="jira",
        name="Queued Source",
        config_json="{}",
    )
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_active_sources=1)),
    )
    release = asyncio.Event()

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        del force_full_sync
        if running_source_id == "src-active":
            await release.wait()

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    task_active = await service.start_source("src-active")
    await service.start_source("src-queued")
    await asyncio.sleep(0)

    assert service.progress["src-queued"]["phase"] == "queued"

    await service.cancel_source("src-queued")

    assert "src-queued" not in service.tasks
    assert "src-queued" not in service.progress

    release.set()
    await task_active


def test_sync_max_active_sources_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_MAX_ACTIVE_SOURCES", "2")

    assert AppConfig().sync.max_active_sources == 2


def test_sync_max_extraction_workers_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SYNC_MAX_EXTRACTION_WORKERS", "6")

    assert AppConfig().sync.max_extraction_workers == 6


@pytest.mark.asyncio
async def test_sync_service_passes_shared_extraction_pool_to_runtime_provider(db: Database):
    source_id = "src-shared-pool"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Shared Pool Source",
        config_json="{}",
    )

    class CapturingRuntimeProvider:
        def __init__(self) -> None:
            self.extraction_pools: list[ExtractionWorkPool | None] = []

        async def build_sync_runtime(
            self,
            db: Database,
            config: AppConfig,
            *,
            extraction_pool: ExtractionWorkPool | None = None,
        ):
            del db, config
            self.extraction_pools.append(extraction_pool)
            return object()

        async def run_source_sync(self, **kwargs):
            del kwargs
            return None

    provider = CapturingRuntimeProvider()
    service = SyncService(
        db,
        AppConfig(sync=SyncConfig(max_extraction_workers=6)),
        runtime_provider=provider,
    )

    task = await service.start_source(source_id)
    await task

    assert provider.extraction_pools == [service._extraction_pool]


@pytest.mark.asyncio
async def test_requested_sync_runs_after_active_source_sync_finishes(db: Database, monkeypatch):
    source_id = "src-queued-after-active"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Queued Source",
        config_json="{}",
    )
    service = SyncService(db, AppConfig())
    first_release = asyncio.Event()
    followup_started = asyncio.Event()
    calls: list[str] = []

    async def fake_run_source_task(running_source_id: str, *, force_full_sync: bool = False):
        calls.append(running_source_id)
        try:
            if len(calls) == 1:
                await first_release.wait()
            else:
                followup_started.set()
        finally:
            service.tasks.pop(running_source_id, None)

    monkeypatch.setattr(service, "_run_source_task", fake_run_source_task)

    first_task = await service.start_source(source_id)
    await asyncio.sleep(0)

    assert await service.request_source_sync(source_id, delay_seconds=0) is True
    await asyncio.sleep(0)
    assert calls == [source_id]

    first_release.set()
    await asyncio.wait_for(followup_started.wait(), timeout=1)
    await first_task
    await service.shutdown()

    assert calls == [source_id, source_id]


@pytest.mark.asyncio
async def test_upsert_sync_state_updates_source_last_sync(db: Database):
    source_id = "src-state-bookkeeping"
    await db.upsert_source(
        id=source_id,
        type="teams",
        name="Team Chat",
        config_json="{}",
    )
    sync_at = datetime.now(timezone.utc)

    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=sync_at,
            last_sync_status="success",
            docs_processed=0,
            docs_updated=0,
        ),
    )

    source = await db.get_source(source_id)
    assert source["last_sync"] == sync_at.isoformat()


@pytest.mark.asyncio
async def test_document_is_indexed_before_enrichment(db: Database):
    source_id = "src-early-document"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    gene = BlockingFetchGene(item_count=1, release=release)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=gene,
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert await db.count_documents(source=source_id) == 1


@pytest.mark.asyncio
async def test_full_document_extraction_failure_is_audited(db: Database):
    source_id = "src-full-extraction-failure"
    await db.upsert_source(
        id=source_id,
        type="docs",
        name="Docs",
        config_json="{}",
    )
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=FailingMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene("# Design Doc\n\nDurable content."),
        source_name="Docs",
        source_id=source_id,
    )

    rows = await db.list_memory_audit_events(event_type="memory_extraction_failed")
    assert state.last_sync_status == "failed"
    assert state.docs_updated == 0
    assert state.docs_failed == 1
    assert state.failed_docs
    assert "json_parse_error" in state.failed_docs[0].error
    assert await db.count_documents(source=source_id) == 0
    assert len(rows) == 3
    assert rows[0].doc_id == "doc-1"
    assert rows[0].source_id == source_id
    assert rows[0].reason == "json_parse_error"
    assert rows[0].error == "Unterminated string starting at line 393 column 16"
    assert rows[0].payload["extracted_count"] == 0


@pytest.mark.asyncio
async def test_document_update_uses_diff_guided_extraction_and_audits_strategy(
    db: Database,
    tmp_path,
):
    source_id = "src-diff-guided-update"
    old_markdown = "# Design Doc\n\nThe service uses PostgreSQL 14."
    new_markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    previous_path = tmp_path / "design-doc.md"
    previous_path.write_text(old_markdown, encoding="utf-8")
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=old_markdown,
        version="1",
        normalized_content_uri=str(previous_path),
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    extraction_rows = await db.list_memory_audit_events(
        event_type="memory_change_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert len(extractor.change_calls) == 1
    assert extractor.full_calls == []
    assert "PostgreSQL 14" in extractor.change_calls[0]["changed_hunks"]
    assert "PostgreSQL 15" in extractor.change_calls[0]["changed_hunks"]
    assert extractor.change_calls[0]["updated_document"] == new_markdown
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "diff_guided"
    assert "PostgreSQL 15" in memory_engine.reconcile_calls[0]["changed_hunks"]
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["data_shape"] == "document"
    assert len(audit_rows) == 1
    assert audit_rows[0].doc_id == "doc-1"
    assert audit_rows[0].source_id == source_id
    assert audit_rows[0].decision == "diff_guided"
    assert audit_rows[0].reason == "small_diff"
    assert audit_rows[0].payload["data_shape"] == "document"
    assert audit_rows[0].payload["previous_version"] == "1"
    assert audit_rows[0].payload["current_version"] == "2"
    assert audit_rows[0].payload["diff_line_count"] > 0
    assert audit_rows[0].thresholds["max_diff_lines"] > 0
    assert len(extraction_rows) == 1
    assert extraction_rows[0].doc_id == "doc-1"
    assert extraction_rows[0].decision == "diff_guided"
    assert extraction_rows[0].payload["extracted_count"] == 0
    assert extraction_rows[0].payload["diff_line_count"] > 0


@pytest.mark.asyncio
async def test_structured_source_update_uses_diff_guided_extraction_and_audits_strategy(
    db: Database,
    tmp_path,
):
    source_id = "src-jira-diff-guided-update"
    old_markdown = "# [Story] PAY-123: Cutoff flow\n\n## Source Metadata\n- Status: In Progress"
    new_markdown = "# [Story] PAY-123: Cutoff flow\n\n## Source Metadata\n- Status: Done"
    previous_path = tmp_path / "jira-ticket.md"
    previous_path.write_text(old_markdown, encoding="utf-8")
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="PAY-123",
        markdown=old_markdown,
        version="1",
        normalized_content_uri=str(previous_path),
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    memory_engine = RecordingMemoryEngine()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingTicketGene(new_markdown),
        source_name="Jira Board",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    extraction_rows = await db.list_memory_audit_events(
        event_type="memory_change_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert len(extractor.change_calls) == 1
    assert extractor.full_calls == []
    assert "Status: In Progress" in extractor.change_calls[0]["changed_hunks"]
    assert "Status: Done" in extractor.change_calls[0]["changed_hunks"]
    assert extractor.change_calls[0]["source_type"] == "jira"
    assert len(memory_engine.reconcile_calls) == 1
    assert memory_engine.reconcile_calls[0]["update_mode"] == "diff_guided"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["reason"] == "small_diff"
    assert memory_engine.reconcile_calls[0]["update_plan_stats"]["data_shape"] == "ticket"
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "diff_guided"
    assert audit_rows[0].reason == "small_diff"
    assert audit_rows[0].payload["data_shape"] == "ticket"
    assert len(extraction_rows) == 1
    assert extraction_rows[0].decision == "diff_guided"


@pytest.mark.asyncio
async def test_document_update_falls_back_to_full_extraction_when_previous_content_missing(
    db: Database,
):
    source_id = "src-full-update-fallback"
    old_markdown = "# Design Doc\n\nThe service uses PostgreSQL 14."
    new_markdown = "# Design Doc\n\nThe service uses PostgreSQL 15."
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=old_markdown,
        version="1",
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(new_markdown),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="document_update_strategy_selected",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert extractor.change_calls == []
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) == 1
    assert extractor.unit_calls[0]["context"].unit.unit_markdown == new_markdown
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "full_document"
    assert audit_rows[0].reason == "previous_content_missing"
    assert audit_rows[0].payload["fallback_from"] == "diff_guided"


@pytest.mark.asyncio
async def test_large_full_document_uses_deterministic_units(db: Database):
    source_id = "src-large-doc-full"
    markdown = "# Design Doc\n\nIntro.\n\n" + "\n\n".join(
        f"## Section {index}\n\n" + ("Durable design detail. " * 900)
        for index in range(8)
    )
    extractor = RecordingMemoryExtractor()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="1"),
        source_name="Documents",
        source_id=source_id,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="memory_extraction_completed",
    )
    assert state.last_sync_status == "success"
    assert state.docs_updated == 1
    assert extractor.full_calls == []
    assert len(extractor.unit_calls) > 1
    assert all(call["context"].unit.unit_id for call in extractor.unit_calls)
    assert len(audit_rows) == 1
    assert audit_rows[0].decision == "full_document"
    assert audit_rows[0].payload["unitized"] is True
    assert audit_rows[0].payload["unit_count"] == len(extractor.unit_calls)
    assert audit_rows[0].payload["segmentation_version"] == "v2"


@pytest.mark.asyncio
async def test_full_document_unit_extraction_honors_orchestrator_concurrency(db: Database):
    markdown = "# Design Doc\n\nIntro.\n\n" + "\n\n".join(
        f"## Section {index}\n\n" + ("Durable design detail. " * 900)
        for index in range(8)
    )
    extractor = BlockingUnitMemoryExtractor()
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, "src-large-doc-full"),
        memory_extractor=extractor,
        memory_engine=NoopMemoryEngine(),
        memory_store=_audited_memory_store(db),
        max_concurrent=1,
    )

    task = asyncio.create_task(
        orchestrator._extract_full_document_units(
            markdown_body=markdown,
            source_type="github_pages",
            doc_type="reference",
            entity_names=[],
            existing_memories=[],
            doc_id="doc-large",
            source_id="src-large-doc-full",
            document_title="Design Doc",
            document_url="https://example.test/design",
        )
    )

    try:
        await asyncio.sleep(0.2)
        assert not extractor.started_two.is_set()
        assert extractor.max_active == 1
    finally:
        extractor.release.set()
        await task


@pytest.mark.asyncio
async def test_partial_unit_extraction_failure_skips_reconciliation(db: Database, tmp_path):
    source_id = "src-partial-unit-failure"
    markdown = "\n\n".join(
        [
            "# Design Doc",
            "Intro.",
            "## Section 1",
            " ".join(["section one durable design guidance"] * 2500),
            "## Section 2",
            " ".join(["section two durable design guidance"] * 2500),
        ]
    )
    previous_path = tmp_path / "design-doc.md"
    previous_path.write_text(markdown, encoding="utf-8")
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="doc-1",
        title="Design Doc",
        markdown=markdown,
        version="1",
        normalized_content_uri=str(previous_path),
    )
    extractor = PartiallyFailingUnitMemoryExtractor()
    memory_engine = RecordingMemoryEngine()
    memory_store = _audited_memory_store(db)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=extractor,
        memory_engine=memory_engine,
        memory_store=memory_store,
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=UpdatingDocumentGene(markdown, version="1"),
        source_name="Documents",
        source_id=source_id,
        force_full_sync=True,
    )

    audit_rows = await db.list_memory_audit_events(
        event_type="memory_extraction_failed",
    )
    assert state.last_sync_status == "failed"
    assert state.docs_updated == 0
    assert state.docs_failed == 1
    assert state.failed_docs
    assert "partial_unit_failure" in state.failed_docs[0].error
    assert len(memory_engine.reconcile_calls) == 0
    assert len(audit_rows) == 3
    assert audit_rows[0].reason == "partial_unit_failure"
    assert audit_rows[0].payload["failed_unit_count"] == 1
    assert audit_rows[0].payload["extracted_count"] == 0


@pytest.mark.asyncio
async def test_item_processing_is_bounded_by_max_concurrent(db: Database):
    source_id = "src-bounded-sync"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    gene = BlockingFetchGene(item_count=5, release=release)
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        max_concurrent=2,
    )

    sync_task = asyncio.create_task(
        orchestrator.sync_gene(
            gene=gene,
            source_name="Jira Board",
            source_id=source_id,
        )
    )
    await asyncio.sleep(0.05)
    release.set()
    await sync_task

    assert gene.max_active_fetches <= 2


@pytest.mark.asyncio
async def test_running_progress_reports_extracted_memories(db: Database):
    source_id = "src-running-memory-progress"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    progress_events: list[dict] = []
    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=CountingMemoryEngine(inserted=3),
        memory_store=None,
        max_concurrent=1,
    )

    await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
        progress_callback=progress_events.append,
    )

    assert any(event.get("memories_extracted") == 3 for event in progress_events)


@pytest.mark.asyncio
async def test_document_vector_failure_happens_before_memory_mutations(db: Database, monkeypatch):
    source_id = "src-vector-before-memory"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    memory_engine = CountingMemoryEngine(inserted=3)
    vector_store = FailingVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=memory_engine,
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert memory_engine.enrichment_calls == 0
    assert memory_engine.process_calls == 0
    assert await db.get_document("jira-0") is None
    assert "jira-0" not in vector_store.upserted


@pytest.mark.asyncio
async def test_falsey_document_collection_still_receives_vector_upsert(db: Database, monkeypatch):
    source_id = "src-falsey-vector"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, source_id),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"]
    assert vector_store.upserted["jira-0"]["metadata"]["version"] == "0"


@pytest.mark.asyncio
async def test_document_vector_text_is_independent_of_extracted_entity_names(db: Database, monkeypatch):
    source_id = "src-vector-text"
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Jira Board",
        config_json="{}",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=EntityMentioningEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    document_text = vector_store.upserted["jira-0"]["document"]
    assert state.last_sync_status == "success"
    assert "Raw Extracted Entity" not in document_text
    assert "Raw Alias" not in document_text
    assert document_text == "Summary\ntag-one\njira_issue\nlow"


@pytest.mark.asyncio
async def test_unchanged_document_repairs_stale_vector_without_llm_reprocessing(
    db: Database,
    monkeypatch,
):
    source_id = "src-stale-vector-repair"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    history = await db.get_sync_history(source=source_id, limit=1)
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert history[0]["docs_updated"] == 0
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"] == content_hash(markdown)
    assert vector_store.upserted["jira-0"]["metadata"]["version"] == "0"


@pytest.mark.asyncio
async def test_unchanged_document_backfills_pdf_uri_without_llm_reprocessing(db: Database):
    source_id = "src-unchanged-pdf-backfill"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == "file:///tmp/Architecture/Jira 0.pdf"


@pytest.mark.asyncio
async def test_missing_pdf_uri_forces_full_sync_without_llm_reprocessing(db: Database):
    source_id = "src-missing-pdf-full-sync"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
        )
    )
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_processed == 1
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == "file:///tmp/Architecture/Jira 0.pdf"


@pytest.mark.asyncio
async def test_missing_required_confluence_pdf_fails_sync_without_hiding_gap(db: Database):
    source_id = "src-required-pdf-failure"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=MissingPdfGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_processed == 0
    assert state.docs_failed == 1
    assert state.error_message == (
        "1 Confluence document could not be imported. PDF export was unavailable for 1 document."
    )
    assert "Confluence PDF export did not produce a PDF" in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_confluence_pdf_storage_failure_is_not_reported_as_export_failure(db: Database):
    source_id = "src-pdf-storage-failure"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=FailingPdfDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=PdfBackfillGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "disk full while storing PDF" in state.failed_docs[0].error
    assert "Confluence PDF export failed" not in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_existing_confluence_pdf_uri_is_preserved_when_unchanged_export_is_unavailable(db: Database):
    source_id = "src-existing-pdf-preserved"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    await db.db.execute("UPDATE sources SET type = ? WHERE id = ?", ("confluence", source_id))
    await db.db.execute(
        "UPDATE documents SET pdf_content_uri = ? WHERE doc_id = ?",
        ("file:///tmp/Architecture/existing.pdf", "jira-0"),
    )
    await db.db.commit()
    release = asyncio.Event()
    release.set()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=None,
        embed_cfg={},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=MissingPdfGene(item_count=1, release=release),
        source_name="Architecture",
        source_id=source_id,
    )

    document = await db.get_document("jira-0")
    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert document is not None
    assert document.pdf_content_uri == "file:///tmp/Architecture/existing.pdf"


@pytest.mark.asyncio
async def test_unchanged_stale_vector_fails_when_embedding_config_is_incomplete(
    db: Database,
):
    source_id = "src-stale-vector-no-embed"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FalseyVectorStore()

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "failed"
    assert "embedding config is missing" in state.failed_docs[0].error


@pytest.mark.asyncio
async def test_embedding_connection_failure_is_reported_as_provider_unreachable(
    db: Database,
    monkeypatch,
):
    release = asyncio.Event()
    release.set()

    def fake_embed_texts(texts, *args, **kwargs):
        raise OSError("[Errno 111] Connection refused")

    async def no_retry_delay(delay):
        return None

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)
    monkeypatch.setattr("memforge.pipeline.sync.asyncio.sleep", no_retry_delay)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=DocumentVisibleEnricher(db, "src-embedding-refused"),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=FalseyVectorStore(),
        embed_cfg={"base_url": "https://embedding.example", "api_key": "test-key", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id="src-embedding-refused",
    )

    assert state.last_sync_status == "failed"
    assert state.docs_failed == 1
    assert "Embedding provider unreachable" in state.failed_docs[0].error
    assert state.error_message is not None
    assert "Embedding provider was unreachable for 1 document" in state.error_message


@pytest.mark.asyncio
async def test_unchanged_document_retries_stale_vector_repair_without_reprocessing(
    db: Database,
    monkeypatch,
):
    source_id = "src-stale-vector-retry"
    markdown = "# Jira 0\n\nBody"
    await _insert_document_with_metadata(
        db,
        source_id=source_id,
        doc_id="jira-0",
        title="Jira 0",
        markdown=markdown,
        version="0",
    )
    release = asyncio.Event()
    release.set()
    vector_store = FlakyFalseyVectorStore()

    def fake_embed_texts(texts, *args, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr("memforge.retrieval.embeddings.embed_texts", fake_embed_texts)

    orchestrator = GeneSyncOrchestrator(
        db=db,
        doc_store=StubDocumentStore(),
        enricher=ExplodingEnricher(),
        memory_extractor=NoopMemoryExtractor(),
        memory_engine=NoopMemoryEngine(),
        memory_store=None,
        vector_store=vector_store,
        embed_cfg={"base_url": "http://embedding", "api_key": "test", "model": "test"},
        max_concurrent=1,
    )

    state = await orchestrator.sync_gene(
        gene=BlockingFetchGene(item_count=1, release=release),
        source_name="Jira Board",
        source_id=source_id,
    )

    assert state.last_sync_status == "success"
    assert state.docs_updated == 0
    assert vector_store.upserted["jira-0"]["metadata"]["content_hash"] == content_hash(markdown)
