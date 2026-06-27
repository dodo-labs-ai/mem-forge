"""SqliteRelationalStore: source-of-truth row reads and writes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import (
    DocumentRecord,
    Memory,
    MemorySource,
    content_hash,
)
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
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


async def _document(db: Database, doc_id: str, *, source: str = "src-confluence") -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source,
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
async def test_filter_ids_by_source_and_time_uses_the_source_join(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await store.insert_memory(_memory("m2"))
    await _document(db, "doc1")
    await store.add_memory_source("m1", "doc1", "confluence", "an excerpt", source_updated_at=None)
    # doc1 belongs to source "src-confluence" (see _document); only m1 is
    # supported by a document from that source.
    kept = await store.filter_ids_by_source_and_time(
        ["m1", "m2"],
        MemorySourceFilter(source_ids=("src-confluence",)),
        None,
    )
    assert kept == {"m1"}
    assert (
        await store.filter_ids_by_source_and_time(
            ["m1"],
            MemorySourceFilter(source_ids=("other-source",)),
            None,
        )
        == set()
    )


@pytest.mark.asyncio
async def test_list_ids_by_source_and_time_excludes_user_disabled_sources(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-disabled"))
    await store.insert_memory(_memory("m-enabled"))
    await db.upsert_source("src-disabled", "jira", "Disabled Jira", "{}")
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}")
    await _document(db, "doc-disabled", source="src-disabled")
    await _document(db, "doc-enabled", source="src-enabled")
    source_updated_at = datetime(2026, 6, 24, tzinfo=timezone.utc)
    await store.add_memory_source(
        "m-disabled",
        "doc-disabled",
        "jira",
        None,
        source_updated_at=source_updated_at,
    )
    await store.add_memory_source(
        "m-enabled",
        "doc-enabled",
        "jira",
        None,
        source_updated_at=source_updated_at,
    )
    await db.set_source_subscription("src-disabled", LOCAL_DEV_USER_ID, False)

    page, total = await store.list_ids_by_source_and_time(
        None,
        MemoryTimeRange(
            after=datetime(2026, 6, 20, tzinfo=timezone.utc),
            before=datetime(2026, 6, 27, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
        _scope(),
        limit=10,
        offset=0,
    )

    assert page == ["m-enabled"]
    assert total == 1


@pytest.mark.asyncio
async def test_count_source_memories_uses_canonical_memory_source_id(db):
    await db.insert_memory(_memory("m1"))
    await db.insert_memory(_memory("m-retired", status="retired"))
    await _document(db, "doc-legacy", source="legacy-document-label")
    await _document(db, "doc-retired", source="src-exact")
    await db.restore_memory_source_snapshot(
        MemorySource(
            memory_id="m1",
            doc_id="doc-legacy",
            source_id="src-exact",
            source_type="jira",
            source_updated_at=None,
        )
    )
    await db.restore_memory_source_snapshot(
        MemorySource(
            memory_id="m-retired",
            doc_id="doc-retired",
            source_id="src-exact",
            source_type="jira",
            source_updated_at=None,
        )
    )

    assert await db.count_source_memories("src-exact") == 1
    assert await db.count_source_memories("legacy-document-label") == 0


@pytest.mark.asyncio
async def test_source_id_backfill_migration_repairs_legacy_memory_sources(db):
    await db.insert_memory(_memory("m1"))
    await _document(db, "doc1", source="src-backfill")
    await db.add_memory_source("m1", "doc1", "confluence", None, source_updated_at=None)
    await db.db.execute("UPDATE memory_sources SET source_id = NULL WHERE memory_id = ?", ("m1",))
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 28")
    await db.db.commit()

    await db._run_migrations()

    assert await db.count_source_memories("src-backfill") == 1


@pytest.mark.asyncio
async def test_source_id_invariant_rejects_unresolved_blank_provenance(db):
    await db.insert_memory(_memory("m1"))
    await _document(db, "doc1", source="src-backfill")
    await db.add_memory_source("m1", "doc1", "confluence", None, source_updated_at=None)
    await db.db.execute("UPDATE memory_sources SET source_id = '' WHERE memory_id = ?", ("m1",))
    await db.db.commit()

    with pytest.raises(RuntimeError, match="without source_id"):
        await db._assert_memory_source_ids_resolved()


@pytest.mark.asyncio
async def test_count_source_memories_returns_zero_when_no_search_visible_statuses(db, monkeypatch):
    monkeypatch.setattr("memforge.storage.database.allowed_search_statuses", lambda _include=False: ())

    assert await db.count_source_memories("src-exact") == 0


@pytest.mark.asyncio
async def test_add_memory_source_links_provenance(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await _document(db, "doc1")
    await store.add_memory_source("m1", "doc1", "confluence", "an excerpt", source_updated_at=None)
    sources = await db.get_memory_sources("m1")
    assert [s.doc_id for s in sources] == ["doc1"]
