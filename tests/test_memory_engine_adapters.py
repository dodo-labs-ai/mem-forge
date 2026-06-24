"""MemoryEngine accepts adapters handles and drives the adapters-bound store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.memory.evidence import LifecycleAction, RelationType
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, RawMemory, content_hash
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
        repo_identifier="github.com/shno-labs/mem-forge",
        source_observed_at=None,
    )
    assert stats["inserted"] == 1
    rows = await db.get_memories_by_source_doc("doc1")
    assert rows[0].repo_identifier == "github.com/shno-labs/mem-forge"
    assert collection.upserted[rows[0].id]["repo_identifier"] == ("github.com/shno-labs/mem-forge")

    async with db.db.execute("SELECT * FROM relation_runs") as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["evidence_unit_id"].startswith("eu-doc-")
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.CREATE_MEMORY.value
    evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.source_type == "manual"
    assert evidence_unit.doc_id == "doc1"
    assert evidence_unit.repo_identifier == "github.com/shno-labs/mem-forge"
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (rows[0].id, RelationType.SUPPORTS)
    ]


@pytest.mark.asyncio
async def test_process_memories_persists_explicit_source_observed_at(db, monkeypatch):
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
    await _document(db, "doc-observed")

    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
    )
    observed_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    stats = await engine.process_memories(
        doc_id="doc-observed",
        raw_memories=[RawMemory(content="historical session produced a durable finding", memory_type="fact")],
        source_type="agent_session",
        source_observed_at=observed_at,
    )

    assert stats["inserted"] == 1
    [memory] = await db.get_memories_by_source_doc("doc-observed")
    [source] = await db.get_memory_sources(memory.id)
    assert source.source_observed_at == observed_at


@pytest.mark.asyncio
async def test_process_memories_does_not_rematerialize_superseded_evidence_unit(db, monkeypatch):
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
    raw = RawMemory(content="deploy via ArgoCD", memory_type="fact")
    first = await engine.process_memories(
        doc_id="doc1", raw_memories=[raw], source_type="manual", source_observed_at=None
    )
    [old_memory] = await db.get_memories_by_source_doc("doc1")
    now = datetime.now(timezone.utc)
    replacement = Memory(
        id="mem-replacement",
        memory_type="fact",
        content="deploy via ArgoCD with promotion gates",
        content_hash=content_hash("deploy via ArgoCD with promotion gates"),
        tags=[],
        confidence=0.9,
        created_at=now,
        updated_at=now,
    )
    await db.supersede_memory(
        old_memory.id,
        replacement,
        replacement_reason="newer source",
        replacement_kind="revision",
    )

    second = await engine.process_memories(
        doc_id="doc1", raw_memories=[raw], source_type="manual", source_observed_at=None
    )
    async with db.db.execute("SELECT evidence_unit_id FROM relation_runs ORDER BY id LIMIT 1") as cursor:
        row = await cursor.fetchone()

    assert first["inserted"] == 1
    assert second["skipped"] == 1
    assert row is not None
    assert await db.has_materialized_evidence_unit(row["evidence_unit_id"]) is True
