"""Two uploaders submit agent-session content; each row stamps the right owner.

Each uploader's PERSONALIZED search must surface only their own private rows.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.memory.audit import MemoryAuditLogger
from memforge.memory.engine import MemoryEngine
from memforge.memory.store import MemoryStore
from memforge.models import (
    DocumentRecord,
    RawMemory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
)
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, ids=None, embeddings=None, metadatas=None, **kwargs):
        pass

    def delete(self, ids=None, **kwargs):
        pass


async def _document(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(DocumentRecord(
        doc_id=doc_id, source="src-agent", source_url=f"agent://{doc_id}",
        title="t", space_or_project="PROJ", author="a", last_modified=now,
        labels=[], version="1", content_hash=f"h-{doc_id}", token_count=1,
        raw_content_uri=None, raw_content_type="text/markdown",
        normalized_content_uri=None, pdf_content_uri=None, last_synced=now,
    ))


@pytest.fixture
async def engine_fixture(tmp_path, monkeypatch):
    database = Database(str(tmp_path / "two_uploaders.db"))
    await database.connect()
    adapters = build_sqlite_adapters(database, FakeCollection())
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(database),
    )

    async def _fake_embed(text: str) -> list[float]:
        return [0.0]

    monkeypatch.setattr(store, "_embed", _fake_embed)
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=database,
        memory_store=store,
        structured_llm_client=None,
    )
    yield engine, database, adapters
    await database.close()


def _personalized_scope(user_id: str) -> AccessScope:
    return AccessScope(
        user_id=user_id,
        open_projects=frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=True,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.mark.asyncio
async def test_two_uploaders_each_owns_their_private_session_memory(engine_fixture):
    engine, database, adapters = engine_fixture
    await _document(database, "doc-u1")
    await _document(database, "doc-u2")

    u1_raw = [RawMemory(memory_type="fact", content="u1 deploys via argo",
                         entity_refs=[], tags=[], confidence=0.9)]
    u2_raw = [RawMemory(memory_type="fact", content="u2 deploys via flux",
                         entity_refs=[], tags=[], confidence=0.9)]

    await engine.process_memories(
        doc_id="doc-u1",
        raw_memories=u1_raw,
        source_type="agent_session",
        user_id="u-1",
    )
    await engine.process_memories(
        doc_id="doc-u2",
        raw_memories=u2_raw,
        source_type="agent_session",
        user_id="u-2",
    )

    rows = await database.list_memories()
    by_content = {row.content: row for row in rows}
    u1_row = by_content["u1 deploys via argo"]
    u2_row = by_content["u2 deploys via flux"]
    assert u1_row.visibility == Visibility.PRIVATE.value
    assert u1_row.owner_user_id == "u-1"
    assert u2_row.visibility == Visibility.PRIVATE.value
    assert u2_row.owner_user_id == "u-2"

    # PERSONALIZED keyword search by U1: U1's own private memory is visible,
    # U2's private memory stays hidden even on a personalized search.
    u1_hits = await adapters.keyword.search(
        "deploys", _personalized_scope("u-1"), memory_types=None, limit=10,
    )
    u1_ids = {mid for mid, _ in u1_hits}
    assert u1_row.id in u1_ids
    assert u2_row.id not in u1_ids

    u2_hits = await adapters.keyword.search(
        "deploys", _personalized_scope("u-2"), memory_types=None, limit=10,
    )
    u2_ids = {mid for mid, _ in u2_hits}
    assert u2_row.id in u2_ids
    assert u1_row.id not in u2_ids
