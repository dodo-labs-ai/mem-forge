"""SearchEngine accepts adapters handles and routes channels through them."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.config import RetrievalConfig
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.search import SearchEngine
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [[0.01 for _ in self.ids]]}

    def upsert(self, **kwargs):
        pass

    def delete(self, **kwargs):
        pass

    def get(self, **kwargs):
        return {"ids": []}


def _memory(mem_id: str, content: str, status: str = "active") -> Memory:
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


async def _document(db: Database, doc_id: str, source: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(DocumentRecord(
        doc_id=doc_id, source=source, source_url=f"https://x/{doc_id}", title="t",
        space_or_project="PAY", author="a", last_modified=now, labels=[],
        version="1", content_hash=f"h-{doc_id}", token_count=1, raw_content_uri=None,
        raw_content_type="text/html", normalized_content_uri=None,
        pdf_content_uri=None, last_synced=now,
    ))


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "search-adapters.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_search_routes_vector_and_bm25_through_the_adapters(db, monkeypatch):
    active = _memory("m-active", "PostgreSQL pooling memory")
    await db.insert_memory(active)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection([active.id]))
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
async def test_source_filter_applies_to_vector_hits(db, monkeypatch):
    # Both memories are surfaced by the vector channel (and BM25, since both
    # match the FTS query); only m-backed is supported by a document from
    # source "wiki". The fused-set source filter must drop m-unbacked, so a
    # hit cannot bypass the filter by riding the vector channel.
    backed = _memory("m-backed", "PostgreSQL pooling from the wiki")
    unbacked = _memory("m-unbacked", "PostgreSQL pooling from elsewhere")
    await db.insert_memory(backed)
    await db.insert_memory(unbacked)
    await _document(db, "doc-wiki", "wiki")
    await _document(db, "doc-other", "other")
    await db.add_memory_source("m-backed", "doc-wiki", "wiki", None)
    await db.add_memory_source("m-unbacked", "doc-other", "other", None)

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr("memforge.retrieval.search.analyze_query", fake_analyze_query)

    adapters = build_sqlite_adapters(db, FakeCollection(["m-backed", "m-unbacked"]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", sources=["wiki"], top_k=10)
    assert [r.memory_id for r in result["results"]] == ["m-backed"]
