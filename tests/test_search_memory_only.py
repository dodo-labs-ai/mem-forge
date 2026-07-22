"""Search returns memory results only.

Document storage, health, repair, and artifact retrieval still exist. The search
API and vector index contain Memories only; source artifacts remain available
through Evidence-backed resource access.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from memforge import runtime
from memforge.config import AppConfig, RetrievalConfig
from memforge.models import Memory, content_hash
from memforge.retrieval.search import SearchEngine
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


class FakeCollection:
    def __init__(self, ids: list[str] | None = None) -> None:
        self.ids = ids or []

    def query(self, **kwargs):
        return {"ids": [self.ids], "distances": [[0.0 for _ in self.ids]]}

    def upsert(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return None


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "doc_fallback.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.embedding_base_url = "http://localhost:6655/openai/v1"
    cfg.llm.embedding_api_key = "test-key"
    cfg.server.jwt_secret = "test-secret"
    return cfg


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
    )


@pytest.mark.asyncio
async def test_build_search_engine_has_no_document_vector(db, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "get_chroma_collection", lambda **kwargs: FakeCollection())

    cfg = _config(tmp_path)
    engine = await runtime.build_search_engine(db, cfg)

    assert not hasattr(engine, "_document_vector")


def test_runtime_build_search_engine_does_not_wire_document_fallback_kwargs():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    marker = "async def build_search_engine"
    start = source.index(marker)
    end = source.find("\nasync def ", start + len(marker))
    body = source[start : end if end != -1 else len(source)]

    assert "document_vector" not in body
    assert "document_collection" not in body
    assert "artifact_config" not in body
    assert "artifact_store" not in body


@pytest.mark.asyncio
async def test_search_does_not_append_document_results_when_memory_results_are_short(
    db,
    tmp_path,
    monkeypatch,
):
    memory = _memory("mem-only-result", "PostgreSQL memory")
    await db.insert_memory(memory)

    adapters = build_sqlite_adapters(db, FakeCollection([memory.id]))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1]

    result = await engine.search("PostgreSQL", top_k=5)

    assert [r.memory_id for r in result["results"]] == [memory.id]
    assert all(r.memory_id is not None for r in result["results"])
    assert all(not hasattr(r, "is_document_result") for r in result["results"])
