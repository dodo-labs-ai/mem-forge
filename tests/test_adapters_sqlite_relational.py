"""SqliteRelationalStore: source-of-truth row reads and writes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import (
    DocumentRecord,
    Memory,
    content_hash,
)
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import RelationalStore
from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore


def _scope(statuses=("active",)) -> AccessScope:
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=statuses,
        active_project=None,
        scope_mode="project-first",
    )


def _memory(mem_id: str, status: str = "active") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=f"content for {mem_id}",
        content_hash=content_hash(f"content for {mem_id}"),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
    )


async def _document(db: Database, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source="src-confluence",
            source_url=f"https://example/{doc_id}",
            title="Doc",
            space_or_project="PAY",
            author="A",
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=10,
            raw_content_uri=None,
            raw_content_type="text/html",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relational.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_satisfies_relational_store_protocol(db):
    assert isinstance(SqliteRelationalStore(db), RelationalStore)


@pytest.mark.asyncio
async def test_insert_then_get_round_trips(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    fetched = await store.get_memory("m1")
    assert fetched is not None
    assert fetched.id == "m1"
    assert await store.get_memory("missing") is None


@pytest.mark.asyncio
async def test_filter_visible_ids_keeps_only_allowed_statuses(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("active1", status="active"))
    await store.insert_memory(_memory("retired1", status="retired"))
    visible = await store.filter_visible_ids(["active1", "retired1"], _scope())
    assert visible == {"active1"}


@pytest.mark.asyncio
async def test_filter_visible_ids_honors_superseded_when_allowed(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("active1", status="active"))
    await store.insert_memory(_memory("supers1", status="superseded"))
    visible = await store.filter_visible_ids(["active1", "supers1"], _scope(statuses=("active", "superseded")))
    assert visible == {"active1", "supers1"}


@pytest.mark.asyncio
async def test_fetch_ranking_metadata_returns_updated_at(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    rows = await store.fetch_ranking_metadata(["m1", "missing"])
    assert isinstance(rows["m1"]["updated_at"], datetime)
    assert "missing" not in rows


@pytest.mark.asyncio
async def test_graph_search_scores_direct_entity_links(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await store.insert_memory(_memory("m2"))
    entity_id = await db.upsert_entity("argocd", "ArgoCD", ["tool"])
    await db.link_memory_entity("m1", entity_id)
    hits = await store.graph_search([entity_id], _scope(), None, limit=10)
    assert [mid for mid, _ in hits] == ["m1"]


@pytest.mark.asyncio
async def test_graph_search_filters_retired_by_scope(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1", status="retired"))
    entity_id = await db.upsert_entity("argocd", "ArgoCD", ["tool"])
    await db.link_memory_entity("m1", entity_id)
    assert await store.graph_search([entity_id], _scope(), None, limit=10) == []


@pytest.mark.asyncio
async def test_temporal_search_returns_rows_in_window(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    after = datetime(2000, 1, 1, tzinfo=timezone.utc)
    before = datetime(2100, 1, 1, tzinfo=timezone.utc)
    hits = await store.temporal_search(after, before, _scope(), None, limit=10)
    assert [mid for mid, _ in hits] == ["m1"]


@pytest.mark.asyncio
async def test_filter_ids_supported_by_sources_uses_the_join(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await store.insert_memory(_memory("m2"))
    await _document(db, "doc1")
    await store.add_memory_source("m1", "doc1", "confluence", "an excerpt", source_observed_at=None)
    # doc1 belongs to source "src-confluence" (see _document); only m1 is
    # supported by a document from that source.
    kept = await store.filter_ids_supported_by_sources(["m1", "m2"], ["src-confluence"])
    assert kept == {"m1"}
    assert await store.filter_ids_supported_by_sources(["m1"], ["other-source"]) == set()


@pytest.mark.asyncio
async def test_add_memory_source_links_provenance(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await _document(db, "doc1")
    await store.add_memory_source("m1", "doc1", "confluence", "an excerpt", source_observed_at=None)
    sources = await db.get_memory_sources("m1")
    assert [s.doc_id for s in sources] == ["doc1"]
