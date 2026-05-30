"""Search behavior for lifecycle states."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.config import RetrievalConfig
from memforge.models import Memory, content_hash
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.retrieval.search import SearchEngine
from memforge.storage.database import Database


class FakeCollection:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [[0.01 for _ in self.ids]]}


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

    engine = SearchEngine(
        db=db,
        memory_collection=FakeCollection([retired.id, pending.id, superseded.id, active.id]),
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=10)

    assert [r.memory_id for r in result["results"]] == [active.id]


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

    engine = SearchEngine(
        db=db,
        memory_collection=FakeCollection([retired.id, pending.id, superseded.id, active.id]),
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", include_superseded=True, top_k=10)

    assert {r.memory_id for r in result["results"]} == {active.id, superseded.id}
