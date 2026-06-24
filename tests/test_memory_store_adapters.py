"""MemoryStore accepts adapters handles and routes vector/FTS through them."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, content_hash
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class RecordingCollection:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}
        self.deleted: list[str] = []

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = dict(metadatas[index] if metadatas else {})

    def delete(self, *, ids):
        self.deleted.extend(ids)
        for record_id in ids:
            self.upserted.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        selected = [r for r in (ids or list(self.upserted)) if r in self.upserted]
        out = {"ids": selected}
        if include and "metadatas" in include:
            out["metadatas"] = [self.upserted[r] for r in selected]
        if include and "embeddings" in include:
            out["embeddings"] = [[0.1] for _ in selected]
        if include and "documents" in include:
            out["documents"] = [None for _ in selected]
        return out


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


async def _document(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source="src-x",
            source_url="https://x/1",
            title="t",
            space_or_project="PAY",
            author="a",
            last_modified=now,
            labels=[],
            version="1",
            content_hash="h",
            token_count=1,
            raw_content_uri=None,
            raw_content_type="text/html",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "store-adapters.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_collection_property_points_at_the_vector_handle(db):
    collection = RecordingCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )
    assert store.collection is collection


@pytest.mark.asyncio
async def test_insert_then_purge_routes_through_adapters(db, monkeypatch):
    collection = RecordingCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )

    async def fake_embed(text: str):
        return [0.1]

    monkeypatch.setattr(store, "_embed", fake_embed)
    await _document(db, "doc1")

    result = await store.deduplicate_and_insert(
        _memory("m1", "deploy via ArgoCD"),
        "doc1",
        "manual",
        source_observed_at=None,
    )
    assert result == "inserted"
    assert "m1" in collection.upserted

    assert await store.purge_memory("m1") is True
    assert "m1" in collection.deleted
    assert await db.get_memory("m1") is None
