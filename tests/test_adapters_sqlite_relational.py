"""SqliteRelationalStore: source-of-truth row reads and writes."""

from __future__ import annotations

import asyncio
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
from memforge.storage.admin_memory import MemoryAdminListFilters


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


async def _restore_legacy_memory_sources_table(db: Database) -> None:
    """Recreate the pre-v66 projection shape for migration contract tests."""

    await db.db.execute(
        """CREATE TABLE memory_sources_legacy (
            memory_id        TEXT NOT NULL REFERENCES memories(id),
            doc_id           TEXT NOT NULL REFERENCES documents(doc_id),
            source_id        TEXT,
            source_type      TEXT NOT NULL,
            excerpt          TEXT,
            support_kind     TEXT NOT NULL DEFAULT 'extracted',
            added_at         TEXT NOT NULL DEFAULT (datetime('now')),
            source_updated_at TEXT,
            PRIMARY KEY (memory_id, doc_id)
        )"""
    )
    await db.db.execute(
        """INSERT INTO memory_sources_legacy (
               memory_id, doc_id, source_id, source_type, excerpt,
               support_kind, added_at, source_updated_at
           )
           SELECT memory_id, doc_id, source_id, source_type, excerpt,
                  support_kind, added_at, source_updated_at
             FROM memory_sources"""
    )
    await db.db.execute("DROP TABLE memory_sources")
    await db.db.execute("ALTER TABLE memory_sources_legacy RENAME TO memory_sources")
    await db.db.execute("CREATE INDEX idx_memory_sources_doc ON memory_sources(doc_id)")
    await db.db.execute("CREATE INDEX idx_memory_sources_doc_kind ON memory_sources(doc_id, support_kind)")
    await db.db.execute("CREATE INDEX idx_memory_sources_source ON memory_sources(source_id)")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relational.db"))
    await database.connect()
    for source_id in (
        "src-confluence",
        "src-target",
        "src-other",
        "src-enabled",
        "src-disabled",
        "src-payroll",
        "src-exact",
        "src-backfill",
        "wiki",
        "jira",
    ):
        await database.upsert_source(
            source_id,
            "test",
            source_id,
            "{}",
            "workspace",
            LOCAL_DEV_USER_ID,
            created_by_user_id=LOCAL_DEV_USER_ID,
        )
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
async def test_graph_search_honors_source_filter_and_time_range(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-target"))
    await store.insert_memory(_memory("m-other"))
    await _document(db, "doc-target", source="src-target")
    await _document(db, "doc-target-2", source="src-target")
    await _document(db, "doc-other", source="src-other")
    await store.add_memory_source(
        "m-target",
        "doc-target",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
    )
    await store.add_memory_source(
        "m-target",
        "doc-target-2",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
    )
    await store.add_memory_source(
        "m-other",
        "doc-other",
        "jira",
        None,
        source_updated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )
    entity_id = await db.upsert_entity("argocd", "ArgoCD", ["tool"])
    await db.link_memory_entity("m-target", entity_id)
    await db.link_memory_entity("m-other", entity_id)

    hits = await store.graph_search(
        [entity_id],
        _scope(),
        None,
        limit=10,
        source_filter=MemorySourceFilter(source_ids=("src-target",)),
        time_range=MemoryTimeRange(
            after=datetime(2026, 6, 20, tzinfo=timezone.utc),
            before=datetime(2026, 6, 27, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
    )

    assert hits == [("m-target", 1.0)]


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
async def test_link_query_entities_alias_fts_binds_query_before_source_count_params(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-enabled"))
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}", access_policy="workspace", owner_user_id="dev")
    await db.upsert_source(
        "src-disabled", "jira", "Disabled Jira", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _document(db, "doc-enabled", source="src-enabled")
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.link_memory_entity("m-enabled", entity_id)
    await store.add_memory_source("m-enabled", "doc-enabled", "jira", None, source_updated_at=None)
    await db.set_source_subscription("src-disabled", LOCAL_DEV_USER_ID, False)

    result = await store.link_query_entities(
        "control center validation failures",
        scope=_scope(),
        limit=5,
    )

    assert [(c.entity_id, c.channel, c.matched_alias) for c in result.candidates] == [
        (entity_id, "alias_fts", "payroll control center")
    ]
    assert result.candidates[0].visible_source_count == 1


@pytest.mark.asyncio
async def test_link_query_entities_alias_fts_refreshes_after_remove_alias(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m1"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC Engine", "pcc engine", entity_id, "admin_manual")
    await db.link_memory_entity("m1", entity_id)

    assert (await store.link_query_entities("pcc engine validation", scope=_scope(), limit=5)).candidates

    await db.remove_entity_alias(entity_id=entity_id, alias_normalized="pcc engine")

    result = await store.link_query_entities("pcc engine validation", scope=_scope(), limit=5)
    assert result.candidates == ()


@pytest.mark.asyncio
async def test_upsert_entity_rolls_back_after_cancellation(db, monkeypatch):
    refresh_started = asyncio.Event()
    original_refresh = db._refresh_entity_alias_search_unlocked

    async def refresh_then_block(entity_id: int) -> None:
        await original_refresh(entity_id)
        refresh_started.set()
        await asyncio.Future()

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", refresh_then_block)
    upsert = asyncio.create_task(db.upsert_entity("cancelled entity", "Cancelled Entity", ["product"]))
    await refresh_started.wait()
    upsert.cancel()

    with pytest.raises(asyncio.CancelledError):
        await upsert

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", original_refresh)
    assert await db.get_entity_by_canonical("cancelled entity") is None

    entity_id = await db.upsert_entity("committed entity", "Committed Entity", ["product"])
    assert await db.get_entity(entity_id) is not None


@pytest.mark.asyncio
async def test_upsert_entity_rolls_back_after_fts_refresh_failure(db, monkeypatch):
    original_refresh = db._refresh_entity_alias_search_unlocked

    async def refresh_then_fail(entity_id: int) -> None:
        await original_refresh(entity_id)
        raise RuntimeError("entity alias search refresh failed")

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", refresh_then_fail)

    with pytest.raises(RuntimeError, match="entity alias search refresh failed"):
        await db.upsert_entity("failed entity", "Failed Entity", ["product"])

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", original_refresh)
    assert await db.get_entity_by_canonical("failed entity") is None

    entity_id = await db.upsert_entity("recovered entity", "Recovered Entity", ["product"])
    assert await db.get_entity(entity_id) is not None


@pytest.mark.asyncio
async def test_remove_entity_alias_rolls_back_after_cancellation(db, monkeypatch):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-alias-cancelled"))
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.insert_alias("PCC Engine", "pcc engine", entity_id, "admin_manual")
    await db.link_memory_entity("m-alias-cancelled", entity_id)
    refresh_started = asyncio.Event()
    original_refresh = db._refresh_entity_alias_search_unlocked

    async def refresh_then_block(refresh_entity_id: int) -> None:
        await original_refresh(refresh_entity_id)
        refresh_started.set()
        await asyncio.Future()

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", refresh_then_block)
    removal = asyncio.create_task(db.remove_entity_alias(entity_id=entity_id, alias_normalized="pcc engine"))
    await refresh_started.wait()
    removal.cancel()

    with pytest.raises(asyncio.CancelledError):
        await removal

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", original_refresh)
    alias = await db.get_entity_by_alias("pcc engine")
    assert alias is not None
    assert alias.canonical_id == entity_id
    assert (await store.link_query_entities("pcc engine validation", scope=_scope(), limit=5)).candidates

    assert await db.remove_entity_alias(
        entity_id=entity_id,
        alias_normalized="pcc engine",
    )


@pytest.mark.asyncio
async def test_merge_entities_reads_source_and_target_inside_write_lock(db):
    source_id = await db.upsert_entity("old payroll name", "Old Payroll Name", ["product"])
    target_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])

    await db._write_lock.acquire()
    merge = asyncio.create_task(db.merge_entities(source_id=source_id, target_id=target_id))
    try:
        await asyncio.sleep(0)
        assert not merge.done()
        await db.db.execute(
            "UPDATE entities SET canonical_name = ?, display_name = ? WHERE id = ?",
            ("current payroll name", "Current Payroll Name", source_id),
        )
        await db.db.commit()
    finally:
        db._write_lock.release()

    result = await merge

    assert result["source_name"] == "current payroll name"
    merged_alias = await db.get_entity_by_alias("current payroll name")
    assert merged_alias is not None
    assert merged_alias.canonical_id == target_id


@pytest.mark.asyncio
async def test_merge_entities_rolls_back_partial_work_after_cancellation(db, monkeypatch):
    await db.insert_memory(_memory("m-merge-cancelled"))
    source_id = await db.upsert_entity("old payroll name", "Old Payroll Name", ["product"])
    target_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.link_memory_entity("m-merge-cancelled", source_id)
    await db.insert_alias("Old Payroll Alias", "old payroll alias", source_id, "admin_manual")
    refresh_started = asyncio.Event()

    async def block_refresh(_entity_id: int) -> None:
        refresh_started.set()
        await asyncio.Future()

    monkeypatch.setattr(db, "_refresh_entity_alias_search_unlocked", block_refresh)
    merge = asyncio.create_task(db.merge_entities(source_id=source_id, target_id=target_id))
    await refresh_started.wait()
    merge.cancel()

    with pytest.raises(asyncio.CancelledError):
        await merge

    assert await db.get_entity(source_id) is not None
    assert await db.get_memory_entity_ids("m-merge-cancelled") == [source_id]
    alias = await db.get_entity_by_alias("old payroll alias")
    assert alias is not None
    assert alias.canonical_id == source_id


@pytest.mark.asyncio
async def test_merge_entities_rejects_self_merge_without_mutation(db):
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])

    with pytest.raises(ValueError, match="Source and target entities must differ"):
        await db.merge_entities(source_id=entity_id, target_id=entity_id)

    assert await db.get_entity(entity_id) is not None


@pytest.mark.asyncio
async def test_link_query_entities_honors_disabled_sources(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-disabled"))
    await store.insert_memory(_memory("m-enabled"))
    await db.upsert_source(
        "src-disabled", "jira", "Disabled Jira", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}", access_policy="workspace", owner_user_id="dev")
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
async def test_link_query_entities_visible_source_count_excludes_disabled_sources(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-mixed"))
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}", access_policy="workspace", owner_user_id="dev")
    await db.upsert_source(
        "src-disabled", "jira", "Disabled Jira", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _document(db, "doc-enabled", source="src-enabled")
    await _document(db, "doc-disabled", source="src-disabled")
    entity_id = await db.upsert_entity("mixed blocker", "Mixed Blocker", ["feature"])
    await db.link_memory_entity("m-mixed", entity_id)
    await store.add_memory_source("m-mixed", "doc-enabled", "jira", None, source_updated_at=None)
    await store.add_memory_source("m-mixed", "doc-disabled", "jira", None, source_updated_at=None)
    await db.set_source_subscription("src-disabled", LOCAL_DEV_USER_ID, False)

    result = await store.link_query_entities("mixed blocker", scope=_scope(), limit=5)

    assert len(result.candidates) == 1
    assert result.candidates[0].entity_id == entity_id
    assert result.candidates[0].visible_memory_count == 1
    assert result.candidates[0].visible_source_count == 1


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
async def test_link_query_entities_fanout_honors_time_range(db):
    store = SqliteRelationalStore(db)
    await store.insert_memory(_memory("m-fresh"))
    await store.insert_memory(_memory("m-stale"))
    await _document(db, "doc-fresh", source="src-payroll")
    await _document(db, "doc-stale", source="src-payroll")
    entity_id = await db.upsert_entity("cutoff blocker", "Cutoff Blocker", ["feature"])
    await db.link_memory_entity("m-fresh", entity_id)
    await db.link_memory_entity("m-stale", entity_id)
    await store.add_memory_source(
        "m-fresh",
        "doc-fresh",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
    )
    await store.add_memory_source(
        "m-stale",
        "doc-stale",
        "jira",
        None,
        source_updated_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    result = await store.link_query_entities(
        "cutoff blocker",
        scope=_scope(),
        source_filter=MemorySourceFilter(source_ids=("src-payroll",)),
        time_range=MemoryTimeRange(
            after=datetime(2026, 6, 19, tzinfo=timezone.utc),
            before=datetime(2026, 6, 21, tzinfo=timezone.utc),
            date_type="source_updated_at",
        ),
        limit=5,
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.entity_id == entity_id
    assert candidate.visible_memory_count == 1
    assert candidate.visible_source_count == 1
    assert candidate.specificity == pytest.approx(1.0)


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
    await db.upsert_source(
        "src-disabled", "jira", "Disabled Jira", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await db.upsert_source("src-enabled", "jira", "Enabled Jira", "{}", access_policy="workspace", owner_user_id="dev")
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
async def test_source_scoped_reads_use_the_exact_projection_source(db):
    memory = _memory("m-overlapping-source")
    memory.project_key = "PAY"
    await db.insert_memory(memory)
    await _document(db, "doc-overlap", source="src-confluence")
    await db.restore_memory_source_snapshot(
        MemorySource(
            memory_id=memory.id,
            doc_id="doc-overlap",
            source_id="src-other",
            source_type="confluence",
            excerpt="independent support",
            source_updated_at=None,
        )
    )

    assert [item.id for item in await db.list_memories(source="src-other")] == [memory.id]
    for search in (None, "m-overlapping-source"):
        page = await db.query_memory_admin_page(
            scope=_scope(),
            filters=MemoryAdminListFilters(source="src-other", search=search),
            limit=10,
            offset=0,
        )
        assert [item.id for item in page.memories] == [memory.id]
        assert page.total == 1
    assert await db.list_resolved_projects_for_source("src-other") == [("PAY", 1)]


@pytest.mark.asyncio
async def test_source_id_backfill_migration_repairs_legacy_memory_sources(db):
    await db.insert_memory(_memory("m1"))
    await _document(db, "doc1", source="src-backfill")
    await db.add_memory_source("m1", "doc1", "confluence", None, source_updated_at=None)
    await _restore_legacy_memory_sources_table(db)
    await db.db.execute("UPDATE memory_sources SET source_id = NULL WHERE memory_id = ?", ("m1",))
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 28")
    await db.db.commit()

    await db._run_migrations()

    assert await db.count_source_memories("src-backfill") == 1


@pytest.mark.asyncio
async def test_memory_source_identity_migration_preserves_existing_projection(db):
    await db.insert_memory(_memory("m-source-identity"))
    await _document(db, "doc-source-identity", source="src-exact")
    await db.add_memory_source(
        "m-source-identity",
        "doc-source-identity",
        "jira",
        None,
        source_updated_at=None,
    )
    await _restore_legacy_memory_sources_table(db)
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 66")
    await db.db.commit()

    await db._run_migrations()

    primary_key = await db.db.execute_fetchall("PRAGMA table_info(memory_sources)")
    assert [
        row["name"]
        for row in sorted(
            (row for row in primary_key if row["pk"]),
            key=lambda row: row["pk"],
        )
    ] == [
        "memory_id",
        "source_id",
        "doc_id",
    ]
    rows = await db.db.execute_fetchall(
        """SELECT memory_id, source_id, doc_id FROM memory_sources
             WHERE memory_id = ?""",
        ("m-source-identity",),
    )
    assert [tuple(row) for row in rows] == [("m-source-identity", "src-exact", "doc-source-identity")]
    metadata = await db.db.execute_fetchall(
        """SELECT source_id FROM memory_search_metadata_trigram
             WHERE memory_id = ? AND doc_id = ?""",
        ("m-source-identity", "doc-source-identity"),
    )
    assert [row["source_id"] for row in metadata] == ["src-exact"]


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
async def test_entity_alias_search_fts_rebuild_migration_removes_stale_tag_tokens(db):
    entity_id = await db.upsert_entity("payroll control center", "Payroll Control Center", ["product"])
    await db.db.execute("DELETE FROM entity_alias_search_fts")
    await db.db.execute(
        """INSERT INTO entity_alias_search_fts (
               entity_id,
               canonical_name,
               alias_normalized,
               search_text
           ) VALUES (?, ?, ?, ?)""",
        (entity_id, "payroll control center", "payroll control center", "payroll control center product"),
    )
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 34")
    await db.db.commit()

    await db._run_migrations()

    rows = await db.db.execute_fetchall(
        "SELECT entity_id FROM entity_alias_search_fts WHERE entity_alias_search_fts MATCH ?",
        ('"product"',),
    )
    assert rows == []


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
