"""MemoryEngine accepts adapters handles and drives the adapters-bound store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, RawMemory
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class RecordingCollection:
    def __init__(self) -> None:
        self.upserted: dict[str, dict] = {}

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None):
        for index, record_id in enumerate(ids):
            self.upserted[record_id] = dict(metadatas[index] if metadatas else {})

    def delete(self, *, ids):
        for record_id in ids:
            self.upserted.pop(record_id, None)

    def get(self, *, ids=None, include=None):
        selected = [r for r in (ids or list(self.upserted)) if r in self.upserted]
        out = {"ids": selected}
        if include and "metadatas" in include:
            out["metadatas"] = [self.upserted[r] for r in selected]
        if include and "embeddings" in include:
            out["embeddings"] = [[0.1] for _ in selected]
        return out


async def _document(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(DocumentRecord(
        doc_id=doc_id, source="src-x", source_url="https://x/1", title="t",
        space_or_project="PAY", author="a", last_modified=now, labels=[],
        version="1", content_hash="h", token_count=1, raw_content_uri=None,
        raw_content_type="text/html", normalized_content_uri=None,
        pdf_content_uri=None, last_synced=now,
    ))


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "engine-adapters.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_process_memories_inserts_through_the_adapters(db, monkeypatch):
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

    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
    )

    stats = await engine.process_memories(
        doc_id="doc1",
        raw_memories=[RawMemory(content="deploy via ArgoCD", memory_type="fact")],
        source_type="manual",
    )
    assert stats["inserted"] == 1
