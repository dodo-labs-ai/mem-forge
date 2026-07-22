"""Visibility and owner flow from intake through the write pipeline to the API."""

from __future__ import annotations

import pytest

from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.store import MemoryStore
from memforge.models import RawMemory
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, ids, embeddings=None, metadatas=None):
        pass

    def delete(self, ids):
        pass


@pytest.fixture
async def engine_fixture(tmp_path):
    database = Database(str(tmp_path / "write_threading.db"))
    await database.connect()
    adapters = build_sqlite_adapters(database, FakeCollection())
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(database),
    )
    engine = MemoryEngine(
        cross_document_candidates=CrossDocumentCandidateRetriever(
            relational=adapters.relational,
            keyword=adapters.keyword,
            vector=adapters.vector,
        ),
        db=database,
        memory_store=store,
        structured_llm_client=object(),
    )
    yield engine
    await database.close()


@pytest.mark.asyncio
async def test_build_memory_stamps_resolved_source_access(engine_fixture):
    from memforge.models import Visibility

    raw = RawMemory(memory_type="fact", content="x", entity_refs=[], confidence=0.8)
    memory = engine_fixture._build_memory(
        raw,
        project_key="ACME",
        visibility=Visibility.WORKSPACE.value,
        owner_user_id=None,
    )
    assert memory.visibility == Visibility.WORKSPACE.value
    assert memory.owner_user_id is None
    assert memory.project_key == "ACME"
    assert not hasattr(memory, "scope")


def test_memory_response_exposes_visibility_not_scope():
    from memforge.models import Memory, Visibility, content_hash
    from memforge.server.admin_api import _memory_to_response

    memory = Memory(
        id="mem-api",
        memory_type="fact",
        content="x",
        content_hash=content_hash("x"),
    )
    body = _memory_to_response(memory).model_dump()
    assert body["visibility"] == Visibility.WORKSPACE.value
    assert body["owner_user_id"] is None
    assert "scope" not in body
