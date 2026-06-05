"""Repair routines for deterministic memory index health."""

from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from memforge.memory.health import MemoryIndexHealthChecker
from memforge.memory.repair import MemoryIndexRepairer
from memforge.models import DocumentMetadata, Memory, content_hash
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
        self.deleted: list[str] = []

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
        self.deleted.extend(ids)
        for record_id in ids:
            self.records.pop(record_id, None)
            self.embeddings.pop(record_id, None)
            self.documents.pop(record_id, None)


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
        tags=["tag"],
        project_key="TEST",
        created_at=now,
        updated_at=now,
        status=status,
    )


async def _insert_doc(
    db: Database,
    doc_id: str,
    tmp_path: Path,
    *,
    content: str = "Document body",
) -> Path:
    now = datetime.now(timezone.utc).isoformat()
    path = tmp_path / f"{doc_id}.md"
    path.write_text(content)
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version,
            content_hash, normalized_content_uri, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "src", f"http://test/{doc_id}", doc_id, "TEST", now, "v1", content_hash(content), str(path), now),
    )
    await db.db.commit()
    return path


@pytest.mark.asyncio
async def test_repair_restores_memory_and_document_index_consistency(db: Database, tmp_path: Path):
    active = _memory("mem-active", "Active fact")
    active.entity_refs = ["stale"]
    retired = _memory("mem-retired", "Retired fact", status="retired")
    await db.insert_memory(active)
    await db.insert_memory(retired)
    entity_id = await db.upsert_entity("canonical entity", display_name="Canonical Entity", tags=["team"])
    await db.link_memory_entity(active.id, entity_id)
    await db.db.execute(
        "UPDATE memories_fts SET entities_text = ? WHERE memory_id = ?",
        ("stale", active.id),
    )
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
        ("mem-orphan", "Orphan fact", "", ""),
    )
    await db.db.commit()
    await _insert_doc(db, "doc-current", tmp_path, content="Document body")
    await db.upsert_metadata(DocumentMetadata(
        doc_id="doc-current",
        summary="Current document summary",
        tags=["current"],
        entities=[],
        doc_type="requirement",
        complexity="medium",
        enriched_at=datetime.now(timezone.utc),
    ))

    memory_collection = RepairableCollection({
        active.id: {"status": "active", "embedding": [0.1, 0.2, 0.3]},
        retired.id: {"status": "active", "embedding": [0.4, 0.5, 0.6]},
    })
    document_collection = RepairableCollection({
        "doc-current": {"embedding": [0.7, 0.8, 0.9], "document": "Document body"},
    })
    expected_memory_vector = [0.9, 0.8, 0.7]
    expected_document_vector = [0.6, 0.5, 0.4]

    before = await MemoryIndexHealthChecker(
        db=db,
        memory_collection=memory_collection,
        document_collection=document_collection,
    ).check()
    assert not before.ok

    repairer = MemoryIndexRepairer(
        db=db,
        memory_collection=memory_collection,
        document_collection=document_collection,
        embed_cfg={"base_url": "http://embed.test", "api_key": "key", "model": "model"},
    )

    async def fake_embed_memory(memory: Memory) -> list[float]:
        return expected_memory_vector

    async def fake_embed_document(text: str) -> list[float]:
        return expected_document_vector

    repairer._embed_memory = fake_embed_memory  # type: ignore[assignment]
    repairer._embed_document = fake_embed_document  # type: ignore[assignment]

    result = await repairer.repair()

    after = await MemoryIndexHealthChecker(
        db=db,
        memory_collection=memory_collection,
        document_collection=document_collection,
    ).check()
    assert after.ok
    assert result.fts_rows_deleted == 1
    assert result.fts_rows_rebuilt >= 1
    assert result.memory_vectors_repaired == 1
    assert result.memory_vectors_deleted == 1
    assert result.document_vectors_repaired == 1
    assert memory_collection.embeddings[active.id] == expected_memory_vector
    assert document_collection.embeddings["doc-current"] == expected_document_vector


@pytest.mark.asyncio
async def test_repair_recreates_document_vector_from_metadata_embedding_text(db: Database, tmp_path: Path):
    await _insert_doc(db, "doc-missing", tmp_path, content="Large document body that should not be embedded")
    await db.upsert_metadata(DocumentMetadata(
        doc_id="doc-missing",
        summary="Short semantic summary",
        tags=["payroll"],
        entities=[],
        doc_type="requirement",
        complexity="medium",
        enriched_at=datetime.now(timezone.utc),
    ))
    document_collection = RepairableCollection({})
    captured: dict[str, str] = {}
    repairer = MemoryIndexRepairer(
        db=db,
        memory_collection=RepairableCollection({}),
        document_collection=document_collection,
        embed_cfg={"base_url": "http://embed.test", "api_key": "key", "model": "model"},
    )

    async def fake_embed_document(text: str) -> list[float]:
        captured["text"] = text
        return [0.3, 0.2, 0.1]

    repairer._embed_document = fake_embed_document  # type: ignore[assignment]

    result = await repairer.repair()

    assert result.document_vectors_created == 1
    assert captured["text"] == "Short semantic summary\npayroll\nrequirement\nmedium"
    assert document_collection.documents["doc-missing"] == captured["text"]


def _as_float32(values: list[float]) -> list[float]:
    """Mimic Chroma persisting embeddings as 32-bit floats, so the vector a
    later get() returns is not bit-identical to the Python list upserted."""
    return [struct.unpack("f", struct.pack("f", float(value)))[0] for value in values]


class Float32Collection(RepairableCollection):
    """A repairable fake that persists embeddings as float32, like Chroma.

    Exact-list fakes hide the round trip, so the stored vector hash must be
    derived from what the collection persists, not from the pre-upsert list.
    """

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        stored = [_as_float32(vector) for vector in embeddings] if embeddings else embeddings
        super().upsert(ids=ids, embeddings=stored, metadatas=metadatas, documents=documents)


@pytest.mark.asyncio
async def test_repair_stamps_vector_hash_from_persisted_float32_payload(db: Database, tmp_path: Path):
    # Repairing a document vector must leave embedding_vector_hash describing the
    # vector the collection actually persisted (float32), not the pre-upsert list.
    await _insert_doc(db, "doc-f32", tmp_path, content="body")
    await db.upsert_metadata(DocumentMetadata(
        doc_id="doc-f32",
        summary="Short semantic summary",
        tags=["payroll"],
        entities=[],
        doc_type="requirement",
        complexity="medium",
        enriched_at=datetime.now(timezone.utc),
    ))
    document_collection = Float32Collection({})
    repairer = MemoryIndexRepairer(
        db=db,
        memory_collection=Float32Collection({}),
        document_collection=document_collection,
        embed_cfg={"base_url": "http://embed.test", "api_key": "key", "model": "model"},
    )

    async def fake_embed_document(text: str) -> list[float]:
        return [0.123456789, 0.987654321, 0.555555555]

    repairer._embed_document = fake_embed_document  # type: ignore[assignment]

    await repairer.repair()

    report = await MemoryIndexHealthChecker(
        db=db,
        memory_collection=repairer.memory_collection,
        document_collection=document_collection,
    ).check()
    vector_hash_mismatches = [
        issue for issue in report.issues
        if issue.kind == "document_chroma_embedding_vector_hash_mismatch"
    ]
    assert vector_hash_mismatches == []
