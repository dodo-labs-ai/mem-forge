"""Repair routines for the supported Memory search projections."""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from typing import Any

import pytest

from memforge.memory.health import MemoryIndexHealthChecker
from memforge.memory.repair import MemoryIndexRepairer
from memforge.models import Memory, content_hash
from memforge.storage.database import Database


class RepairableCollection:
    def __init__(self, records: dict[str, dict[str, Any]] | None = None) -> None:
        self.records = records or {}
        self.embeddings: dict[str, list[float]] = {
            record_id: record.get("embedding", [0.1, 0.2, 0.3])
            for record_id, record in self.records.items()
        }
        self.documents: dict[str, str] = {
            record_id: record.get("document", "")
            for record_id, record in self.records.items()
            if "document" in record
        }

    def get(self, *, ids=None, include=None):
        selected_ids = [record_id for record_id in (ids or list(self.records)) if record_id in self.records]
        include = include or ["metadatas"]
        result: dict[str, Any] = {"ids": selected_ids}
        if "metadatas" in include:
            result["metadatas"] = [
                {k: v for k, v in self.records[record_id].items() if k not in {"embedding", "document"}}
                for record_id in selected_ids
            ]
        if "embeddings" in include:
            result["embeddings"] = [self.embeddings.get(record_id) for record_id in selected_ids]
        if "documents" in include:
            result["documents"] = [self.documents.get(record_id) for record_id in selected_ids]
        return result

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            metadata = dict(metadatas[index] if metadatas else {})
            self.records[record_id] = metadata
            if embeddings:
                self.embeddings[record_id] = embeddings[index]
                self.records[record_id]["embedding"] = embeddings[index]
            if documents:
                self.documents[record_id] = documents[index]
                self.records[record_id]["document"] = documents[index]

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)
            self.embeddings.pop(record_id, None)
            self.documents.pop(record_id, None)


class Float32Collection(RepairableCollection):
    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        stored = (
            [[struct.unpack("f", struct.pack("f", float(value)))[0] for value in vector] for vector in embeddings]
            if embeddings
            else embeddings
        )
        super().upsert(ids=ids, embeddings=stored, metadatas=metadatas, documents=documents)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "repair.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str, status: str = "active") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        project_key="TEST",
        created_at=now,
        updated_at=now,
        status=status,
    )


@pytest.mark.asyncio
async def test_repair_restores_memory_fts_and_vector_consistency(db: Database):
    active = _memory("mem-active", "Active fact")
    retired = _memory("mem-retired", "Retired fact", status="retired")
    await db.insert_memory(active)
    await db.insert_memory(retired)
    entity_id = await db.upsert_entity("canonical entity", display_name="Canonical Entity")
    await db.link_memory_entity(active.id, entity_id)
    await db.db.execute("UPDATE memories_fts SET entities_text = ? WHERE memory_id = ?", ("stale", active.id))
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text) VALUES (?, ?, ?)",
        ("mem-orphan", "Orphan fact", ""),
    )
    await db.db.commit()

    collection = RepairableCollection(
        {
            active.id: {"status": "active", "embedding": [0.1, 0.2, 0.3]},
            retired.id: {"status": "active", "embedding": [0.4, 0.5, 0.6]},
        }
    )
    repairer = MemoryIndexRepairer(
        db=db,
        memory_collection=collection,
        embed_cfg={"base_url": "http://embed.test", "api_key": "key", "model": "model"},
    )
    repairer._embed_memory = lambda memory: _async_value([0.9, 0.8, 0.7])  # type: ignore[method-assign]

    result = await repairer.repair()

    assert (await MemoryIndexHealthChecker(db=db, memory_collection=collection).check()).ok
    assert result.fts_rows_deleted == 1
    assert result.fts_rows_rebuilt >= 1
    assert result.memory_vectors_repaired == 1
    assert result.memory_vectors_deleted == 1
    assert collection.embeddings[active.id] == [0.9, 0.8, 0.7]


async def _async_value(value):
    return value


@pytest.mark.asyncio
async def test_repair_stamps_hash_from_persisted_float32_memory_vector(db: Database):
    active = _memory("mem-f32", "Memory vector round trip")
    await db.insert_memory(active)
    collection = Float32Collection({})
    repairer = MemoryIndexRepairer(
        db=db,
        memory_collection=collection,
        embed_cfg={"base_url": "http://embed.test", "api_key": "key", "model": "model"},
    )
    repairer._embed_memory = lambda memory: _async_value([0.123456789, 0.987654321, 0.555555555])  # type: ignore[method-assign]

    await repairer.repair()

    report = await MemoryIndexHealthChecker(db=db, memory_collection=collection).check()
    assert [issue for issue in report.issues if issue.kind == "chroma_embedding_vector_hash_mismatch"] == []
