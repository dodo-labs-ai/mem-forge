"""SqliteRelationalStore: source-of-truth row reads and writes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import (
    DocumentRecord,
    Memory,
    MemorySource,
    Visibility,
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


def _memory(
    mem_id: str,
    status: str = "active",
    *,
    memory_type: str = "fact",
    visibility: str = Visibility.WORKSPACE.value,
    owner_user_id: str | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type=memory_type,
        content=f"content for {mem_id}",
        content_hash=content_hash(f"content for {mem_id}"),
        visibility=visibility,
        owner_user_id=owner_user_id,
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
async def test_link_query_entities_resolves_explicit_aliases_and_reports_unmatched(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC", "pcc", entity_id, "admin_manual")
    await db.link_memory_entity("m1", entity_id)

    result = await store.link_query_entities(
        "why did the deployment fail",
        scope=_scope(),
        explicit_entities=("PCC", "unknown thing"),
        limit=5,
    )

    assert result.unmatched_explicit_entities == ("unknown thing",)
    assert [(c.entity_id, c.channel, c.matched_alias, c.activates_graph) for c in result.candidates] == [
        (entity_id, "explicit", "PCC", True)
    ]


@pytest.mark.asyncio
async def test_link_query_entities_exact_window_is_visibility_constrained(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("visible"))
    await store.insert_memory(
        _memory(
            "private-other-user",
            visibility=Visibility.PRIVATE.value,
            owner_user_id="other-user",
        )
    )
    visible_entity = await db.upsert_entity("blocker hint", "Blocker Hint", ["feature"])
    hidden_entity = await db.upsert_entity("secret project", "Secret Project", ["feature"])
    await db.link_memory_entity("visible", visible_entity)
    await db.link_memory_entity("private-other-user", hidden_entity)

    result = await store.link_query_entities(
        "create blocker hint for the secret project",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias, c.activates_graph) for c in result.candidates] == [
        (visible_entity, "alias_exact", "blocker hint", True)
    ]


@pytest.mark.asyncio
async def test_link_query_entities_compact_match_does_not_activate_graph(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("blocker hint", "Blocker Hint", ["feature"])
    await db.link_memory_entity("m1", entity_id)

    result = await store.link_query_entities(
        "create blockerhints",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias, c.activates_graph) for c in result.candidates] == [
        (entity_id, "alias_compact", "blocker hint", False)
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_recalls_partial_alias_overlap(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.link_memory_entity("m1", entity_id)

    result = await store.link_query_entities(
        "control center validation failures",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias, c.activates_graph) for c in result.candidates] == [
        (entity_id, "alias_fts", "payroll control center", True)
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_refreshes_after_insert_alias(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC Engine Lifecycle", "pcc engine lifecycle", entity_id, "admin_manual")
    await db.link_memory_entity("m1", entity_id)

    result = await store.link_query_entities(
        "engine lifecycle validation failures",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias, c.activates_graph) for c in result.candidates] == [
        (entity_id, "alias_fts", "pcc engine lifecycle", True)
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_ignores_tag_only_overlap(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    await store.insert_memory(_memory("m2"))
    tagged_entity = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    named_entity = await db.upsert_entity("product release calendar", "Product Release Calendar", ["planning"])
    await db.link_memory_entity("m1", tagged_entity)
    await db.link_memory_entity("m2", named_entity)

    result = await store.link_query_entities(
        "product release status",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (named_entity, "alias_fts", "product release calendar")
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_honors_source_filter(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-wiki"))
    await store.insert_memory(_memory("m-jira"))
    await _document(db, "doc-wiki", source="wiki")
    await _document(db, "doc-jira", source="jira")
    wiki_entity = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    jira_entity = await db.upsert_entity("payroll control process", "Payroll Control Process", ["product"])
    await db.link_memory_entity("m-wiki", wiki_entity)
    await db.link_memory_entity("m-jira", jira_entity)
    await store.add_memory_source("m-wiki", "doc-wiki", "confluence", None, source_updated_at=None)
    await store.add_memory_source("m-jira", "doc-jira", "jira", None, source_updated_at=None)

    result = await store.link_query_entities(
        "control center validation",
        scope=_scope(),
        source_filter=MemorySourceFilter(source_ids=("wiki",)),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (wiki_entity, "alias_fts", "payroll control center")
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_honors_memory_types(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-fact", memory_type="fact"))
    await store.insert_memory(_memory("m-procedure", memory_type="procedure"))
    fact_entity = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    procedure_entity = await db.upsert_entity("payroll control process", "Payroll Control Process", ["product"])
    await db.link_memory_entity("m-fact", fact_entity)
    await db.link_memory_entity("m-procedure", procedure_entity)

    result = await store.link_query_entities(
        "control process validation",
        scope=_scope(),
        memory_types=("procedure",),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (procedure_entity, "alias_fts", "payroll control process")
    ]


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_refreshes_after_remove_alias(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC Engine", "pcc engine", entity_id, "admin_manual")
    await db.link_memory_entity("m1", entity_id)

    assert (
        await store.link_query_entities("pcc engine validation", scope=_scope(), limit=5)
    ).candidates

    await db.remove_entity_alias(entity_id=entity_id, alias_normalized="pcc engine")

    result = await store.link_query_entities("pcc engine validation", scope=_scope(), limit=5)
    assert result.candidates == ()


@pytest.mark.asyncio
async def test_link_query_entities_honors_disabled_sources(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-disabled"))
    await store.insert_memory(_memory("m-enabled"))
    await db.upsert_source("src-disabled", "jira", "Disabled Jira", "{}")
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}")
    await _document(db, "doc-disabled", source="src-disabled")
    await _document(db, "doc-enabled", source="src-enabled")
    disabled_entity = await db.upsert_entity("disabled blocker", "Disabled Blocker", ["feature"])
    enabled_entity = await db.upsert_entity("enabled blocker", "Enabled Blocker", ["feature"])
    await db.link_memory_entity("m-disabled", disabled_entity)
    await db.link_memory_entity("m-enabled", enabled_entity)
    await store.add_memory_source("m-disabled", "doc-disabled", "jira", None, source_updated_at=None)
    await store.add_memory_source("m-enabled", "doc-enabled", "jira", None, source_updated_at=None)
    await db.set_source_subscription("src-disabled", LOCAL_DEV_USER_ID, False)

    result = await store.link_query_entities(
        "disabled blocker and enabled blocker",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (enabled_entity, "alias_exact", "enabled blocker")
    ]


@pytest.mark.asyncio
async def test_link_query_entities_honors_source_filter(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-wiki"))
    await store.insert_memory(_memory("m-jira"))
    await _document(db, "doc-wiki", source="wiki")
    await _document(db, "doc-jira", source="jira")
    wiki_entity = await db.upsert_entity("wiki blocker", "Wiki Blocker", ["feature"])
    jira_entity = await db.upsert_entity("jira blocker", "Jira Blocker", ["feature"])
    await db.link_memory_entity("m-wiki", wiki_entity)
    await db.link_memory_entity("m-jira", jira_entity)
    await store.add_memory_source("m-wiki", "doc-wiki", "confluence", None, source_updated_at=None)
    await store.add_memory_source("m-jira", "doc-jira", "jira", None, source_updated_at=None)

    result = await store.link_query_entities(
        "wiki blocker and jira blocker",
        scope=_scope(),
        source_filter=MemorySourceFilter(source_ids=("wiki",)),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (wiki_entity, "alias_exact", "wiki blocker")
    ]


@pytest.mark.asyncio
async def test_link_query_entities_honors_memory_types(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-fact", memory_type="fact"))
    await store.insert_memory(_memory("m-procedure", memory_type="procedure"))
    fact_entity = await db.upsert_entity("fact blocker", "Fact Blocker", ["feature"])
    procedure_entity = await db.upsert_entity("procedure blocker", "Procedure Blocker", ["feature"])
    await db.link_memory_entity("m-fact", fact_entity)
    await db.link_memory_entity("m-procedure", procedure_entity)

    result = await store.link_query_entities(
        "fact blocker and procedure blocker",
        scope=_scope(),
        memory_types=("procedure",),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (procedure_entity, "alias_exact", "procedure blocker")
    ]


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
async def test_compact_entity_lookup_indexes_migration(db):
    await db.db.execute("DROP INDEX IF EXISTS idx_entity_aliases_compact")
    await db.db.execute("DROP INDEX IF EXISTS idx_entities_canonical_compact")
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 32")
    await db.db.commit()

    await db._run_migrations()

    alias_indexes = await db.db.execute_fetchall("PRAGMA index_list(entity_aliases)")
    entity_indexes = await db.db.execute_fetchall("PRAGMA index_list(entities)")
    assert "idx_entity_aliases_compact" in {row["name"] for row in alias_indexes}
    assert "idx_entities_canonical_compact" in {row["name"] for row in entity_indexes}


@pytest.mark.asyncio
async def test_entity_alias_search_fts_migration_backfills_existing_entities(db):
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC Engine", "pcc engine", entity_id, "admin_manual")
    await db.db.execute("DELETE FROM entity_alias_search_fts")
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 33")
    await db.db.commit()

    await db._run_migrations()

    rows = await db.db.execute_fetchall(
        "SELECT entity_id FROM entity_alias_search_fts WHERE entity_alias_search_fts MATCH ?",
        ('"engine"',),
    )
    assert [row["entity_id"] for row in rows] == [entity_id]


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
