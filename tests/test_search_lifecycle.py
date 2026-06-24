"""Search behavior for lifecycle states."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from memforge.config import AppConfig, RetrievalConfig
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.retrieval.search import SearchEngine
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.document_store import StoredDocumentArtifact


class FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [[0.01 for _ in self.ids]]}


class FakeVectorStore:
    def __init__(self, ids: list[str]) -> None:
        self.collection = FakeCollection(ids)

    def similarity(self, distance: float) -> float:
        return max(1.0 - distance, 0.0)


class MemoryBackedDocumentStore:
    def __init__(self, artifacts: dict[str, bytes]) -> None:
        self._artifacts = artifacts

    def get_artifact(
        self,
        uri: str | None,
        media_type: str,
    ) -> StoredDocumentArtifact | None:
        if uri is None or uri not in self._artifacts:
            return None
        return StoredDocumentArtifact(
            uri=uri,
            filename=uri.rsplit("/", 1)[-1],
            media_type=media_type,
            size_bytes=len(self._artifacts[uri]),
        )

    def read_artifact(self, uri: str) -> bytes:
        return self._artifacts[uri]


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "search.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str, status: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
    )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(base_dir=tmp_path / "memforge")


async def _document(db: Database, tmp_path: Path, doc_id: str) -> DocumentRecord:
    config = _config(tmp_path)
    docs_dir = Path(config.storage.docs_path)
    docs_dir.mkdir(parents=True)
    source_md = docs_dir / f"{doc_id}.md"
    source_pdf = docs_dir / f"{doc_id}.pdf"
    source_md.write_text("# Source\n\nDurable search evidence.", encoding="utf-8")
    source_pdf.write_bytes(b"%PDF-1.4\n%search\n")
    now = datetime.now(timezone.utc)
    doc = DocumentRecord(
        doc_id=doc_id,
        source="src-confluence",
        source_url=f"https://confluence.example/{doc_id}",
        title="Search Source",
        space_or_project="PAY",
        author="Sun, Youpeng",
        last_modified=now,
        labels=[],
        version="1",
        content_hash=f"hash-{doc_id}",
        token_count=100,
        raw_content_uri=None,
        raw_content_type="text/html",
        normalized_content_uri=str(source_md),
        pdf_content_uri=str(source_pdf),
        last_synced=now,
    )
    await db.upsert_document(doc)
    return doc


@pytest.mark.asyncio
async def test_default_search_returns_only_active_memories(db, monkeypatch):
    active = _memory("mem-active1", "Active PostgreSQL memory", "active")
    retired = _memory("mem-retired", "Retired PostgreSQL memory", "retired")
    pending = _memory("mem-pending", "Pending PostgreSQL memory", "pending_review")
    superseded = _memory("mem-supers", "Superseded PostgreSQL memory", "superseded")
    for mem in [active, retired, pending, superseded]:
        await db.insert_memory(mem)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([retired.id, pending.id, superseded.id, active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=10)

    assert [r.memory_id for r in result["results"]] == [active.id]


@pytest.mark.asyncio
async def test_search_results_expose_service_artifact_urls_without_storage_uris(
    db,
    tmp_path,
    monkeypatch,
):
    active = _memory("mem-active-artifact", "Active PostgreSQL memory", "active")
    await db.insert_memory(active)
    doc = await _document(db, tmp_path, "doc-search-artifact")
    await db.add_memory_source(active.id, doc.doc_id, "confluence", excerpt="source excerpt", source_observed_at=None)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    config = _config(tmp_path)
    adapters = build_sqlite_adapters(db, FakeCollection([active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=config.retrieval,
        artifact_config=config,
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=1)
    search_result = result["results"][0]

    assert search_result.content_url == "/api/documents/doc-search-artifact/content"
    assert search_result.pdf_url == "/api/documents/doc-search-artifact/pdf"
    assert not hasattr(search_result, "file_uri")
    assert not hasattr(search_result, "pdf_uri")


@pytest.mark.asyncio
async def test_search_result_suggests_detail_for_procedure_memory(
    db,
    tmp_path,
    monkeypatch,
):
    procedure = _memory(
        "mem-procedure-follow-up",
        "Run the deploy script, bootstrap the admin user, then smoke test.",
        "active",
    )
    procedure.memory_type = "procedure"
    await db.insert_memory(procedure)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([procedure.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=_config(tmp_path).retrieval,
        artifact_config=_config(tmp_path),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("deploy runbook", top_k=1)
    search_result = result["results"][0]

    assert search_result.follow_up == {
        "suggested_tool": "get_memory",
        "reason": "summary_may_omit_operational_steps",
    }


@pytest.mark.asyncio
async def test_search_result_omits_follow_up_for_simple_fact_memory(
    db,
    tmp_path,
    monkeypatch,
):
    fact = _memory("mem-fact-no-follow-up", "Service uses HANA.", "active")
    await db.insert_memory(fact)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([fact.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=_config(tmp_path).retrieval,
        artifact_config=_config(tmp_path),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("hana", top_k=1)
    search_result = result["results"][0]

    assert search_result.follow_up is None


@pytest.mark.asyncio
async def test_search_results_resolve_artifacts_through_configured_store(
    db,
    tmp_path,
    monkeypatch,
):
    active = _memory("mem-active-object-artifact", "Active HANA memory", "active")
    await db.insert_memory(active)

    now = datetime.now(timezone.utc)
    doc = DocumentRecord(
        doc_id="doc-object-search-artifact",
        source="src-jira",
        source_url="https://jira.example/browse/PAY-1",
        title="Jira Source",
        space_or_project="PAY",
        author="Sun, Youpeng",
        last_modified=now,
        labels=[],
        version="1",
        content_hash="hash-doc-object-search-artifact",
        token_count=100,
        raw_content_uri=None,
        raw_content_type="application/json",
        normalized_content_uri="object://workspace/doc-object-search-artifact.md",
        pdf_content_uri=None,
        last_synced=now,
    )
    await db.upsert_document(doc)
    await db.add_memory_source(active.id, doc.doc_id, "jira", excerpt="source excerpt", source_observed_at=None)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    config = _config(tmp_path)
    adapters = build_sqlite_adapters(db, FakeCollection([active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=config.retrieval,
        artifact_config=config,
        artifact_store=MemoryBackedDocumentStore(
            {"object://workspace/doc-object-search-artifact.md": b"# Jira Source"}
        ),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("HANA", top_k=1)
    search_result = result["results"][0]

    assert search_result.source_doc_id == doc.doc_id
    assert search_result.content_url == "/api/documents/doc-object-search-artifact/content"
    assert search_result.pdf_url is None


@pytest.mark.asyncio
async def test_document_fallback_suggests_resource_when_artifact_is_available(
    db,
    tmp_path,
    monkeypatch,
):
    doc = await _document(db, tmp_path, "doc-fallback-artifact")

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=_config(tmp_path).retrieval,
        artifact_config=_config(tmp_path),
        document_vector=FakeVectorStore([doc.doc_id]),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("deployment", top_k=1)
    search_result = result["results"][0]

    assert search_result.memory_id is None
    assert search_result.is_document_result is True
    assert search_result.follow_up == {
        "suggested_tool": "get_resource",
        "reason": "document_result_needs_source_artifact",
    }


@pytest.mark.asyncio
async def test_include_superseded_includes_history_but_not_retired_or_pending(db, monkeypatch):
    active = _memory("mem-active1", "Active PostgreSQL memory", "active")
    retired = _memory("mem-retired", "Retired PostgreSQL memory", "retired")
    pending = _memory("mem-pending", "Pending PostgreSQL memory", "pending_review")
    superseded = _memory("mem-supers", "Superseded PostgreSQL memory", "superseded")
    for mem in [active, retired, pending, superseded]:
        await db.insert_memory(mem)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([retired.id, pending.id, superseded.id, active.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", include_superseded=True, top_k=10)

    assert {r.memory_id for r in result["results"]} == {active.id, superseded.id}
