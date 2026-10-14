"""Repair routines for the supported Memory search projections."""

from __future__ import annotations

import asyncio
import struct
from datetime import datetime, timezone
from typing import Any

import pytest
from click.testing import CliRunner

from memforge.memory.health import MemoryIndexHealthChecker
from memforge.memory.repair import MemoryIndexRepairer
from memforge.models import Memory, content_hash
from memforge.storage.database import Database
from tests.embedding_fixture import embedding_from_texts, embedding_response, fixed_vector_embedding
from tests.vector_index_fixture import use_collection_as_index


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
        embedding=fixed_vector_embedding([0.0, 0.0, 0.0]),
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
        embedding=fixed_vector_embedding([0.0, 0.0, 0.0]),
    )
    repairer._embed_memory = lambda memory: _async_value([0.123456789, 0.987654321, 0.555555555])  # type: ignore[method-assign]

    await repairer.repair()

    report = await MemoryIndexHealthChecker(db=db, memory_collection=collection).check()
    assert [issue for issue in report.issues if issue.kind == "chroma_embedding_vector_hash_mismatch"] == []


@pytest.mark.asyncio
async def test_repair_embeds_a_missing_memory_vector_through_the_embedding_runtime(db: Database):
    active = _memory("mem-missing-vector", "Missing vector fact")
    await db.insert_memory(active)
    embedded: list[str] = []

    def embed(texts: list[str]) -> list[list[float]]:
        embedded.extend(texts)
        return [[0.25, 0.5, 0.75] for _ in texts]

    collection = RepairableCollection({})

    result = await MemoryIndexRepairer(
        db=db,
        memory_collection=collection,
        embedding=embedding_from_texts(embed),
    ).repair()

    assert result.memory_vectors_repaired == 1
    assert result.unrepaired_memories == []
    assert embedded == ["Fact: Missing vector fact\nEntities: "]
    assert collection.embeddings[active.id] == [0.25, 0.5, 0.75]
    assert (await MemoryIndexHealthChecker(db=db, memory_collection=collection).check()).ok


def test_repair_indexes_command_embeds_at_the_configured_endpoint(tmp_path, monkeypatch):
    from memforge.config import load_config
    from memforge.main import cli

    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "repair-embedding-model")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", "https://embed.repair.example.test/v1")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_API_KEY", "repair-embedding-key")
    requests: list[dict[str, Any]] = []

    async def provider(**request: Any) -> dict[str, Any]:
        requests.append(request)
        return embedding_response([[0.25, 0.5, 0.75] for _ in request["input"]])

    monkeypatch.setattr("litellm.aembedding", provider)
    collection = RepairableCollection({})
    use_collection_as_index(monkeypatch, collection)

    async def seed() -> None:
        database = Database(load_config().storage.db_path)
        await database.connect()
        await database.insert_memory(_memory("mem-cli-repair", "Command line repair fact"))
        await database.close()

    asyncio.run(seed())

    result = CliRunner().invoke(cli, ["maintenance", "repair-indexes"])

    assert result.exit_code == 0, result.output
    assert "Index health is clean." in result.output
    assert [request["input"] for request in requests] == [["Fact: Command line repair fact\nEntities: "]]
    assert requests[0]["model"] == "openai/repair-embedding-model"
    assert requests[0]["api_base"] == "https://embed.repair.example.test/v1"
    assert requests[0]["api_key"] == "repair-embedding-key"
    assert collection.embeddings["mem-cli-repair"] == [0.25, 0.5, 0.75]


def _seed_memory_without_vector(monkeypatch, tmp_path, memory_id: str) -> RepairableCollection:
    """Give the command line a database with one Memory that has no vector."""
    from memforge.config import load_config

    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    collection = RepairableCollection({})
    use_collection_as_index(monkeypatch, collection)

    async def seed() -> None:
        database = Database(load_config().storage.db_path)
        await database.connect()
        await database.insert_memory(_memory(memory_id, "Command line repair fact"))
        await database.close()

    asyncio.run(seed())
    return collection


def test_repair_indexes_command_lists_the_memories_it_could_not_repair(tmp_path, monkeypatch):
    from memforge.main import cli

    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "")
    _seed_memory_without_vector(monkeypatch, tmp_path, "mem-cli-unrepaired")

    result = CliRunner().invoke(cli, ["maintenance", "repair-indexes"])

    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "consistency issue(s) remain." in result.output
    assert "Unrepaired memory: mem-cli-unrepaired" in result.output
    assert "Index health is clean." not in result.output


def test_repair_indexes_command_repairs_through_a_keyless_endpoint(tmp_path, monkeypatch):
    from memforge.main import cli

    monkeypatch.setenv("MEMFORGE_EMBEDDING_MODEL", "local-embedding")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    requests: list[dict[str, Any]] = []

    async def provider(**request: Any) -> dict[str, Any]:
        requests.append(request)
        return embedding_response([[0.25, 0.5, 0.75] for _ in request["input"]])

    monkeypatch.setattr("litellm.aembedding", provider)
    collection = _seed_memory_without_vector(monkeypatch, tmp_path, "mem-cli-keyless")

    result = CliRunner().invoke(cli, ["maintenance", "repair-indexes"])

    assert result.exit_code == 0, result.output
    assert "Index health is clean." in result.output
    assert requests[0]["model"] == "openai/local-embedding"
    assert requests[0]["api_base"] == "http://localhost:11434/v1"
    assert requests[0]["api_key"] == "no-api-key"
    assert collection.embeddings["mem-cli-keyless"] == [0.25, 0.5, 0.75]
