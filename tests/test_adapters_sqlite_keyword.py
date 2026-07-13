"""SqliteKeywordSearch: the read-path FTS5 query and the standalone delete."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import DocumentRecord, Memory, content_hash
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import KeywordSearch
from memforge.storage.adapters.sqlite.keyword import SqliteKeywordSearch


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
    content: str,
    status: str = "active",
    repo_identifier: str | None = None,
    updated_at: datetime | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=updated_at or now,
        status=status,
        repo_identifier=repo_identifier,
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "keyword.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_satisfies_keyword_search_protocol(db):
    assert isinstance(SqliteKeywordSearch(db), KeywordSearch)


@pytest.mark.asyncio
async def test_search_matches_active_memory_by_content(db):
    await db.insert_memory(_memory("m1", "PostgreSQL connection pooling"))
    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search('"PostgreSQL"', _scope(), None, limit=10)
    assert [mid for mid, _ in hits] == ["m1"]


@pytest.mark.asyncio
async def test_search_filters_by_status_via_scope(db):
    active = _memory("m-active", "Redis cache eviction", status="active")
    retired = _memory("m-retired", "Redis cache eviction", status="retired")
    await db.insert_memory(active)
    await db.insert_memory(retired)
    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search('"Redis"', _scope(), None, limit=10)
    assert [mid for mid, _ in hits] == ["m-active"]


@pytest.mark.asyncio
async def test_search_filters_by_memory_type(db):
    await db.insert_memory(_memory("m1", "deploy via ArgoCD"))
    keyword = SqliteKeywordSearch(db)
    assert await keyword.search('"deploy"', _scope(), ["decision"], limit=10) == []
    assert [mid for mid, _ in await keyword.search('"deploy"', _scope(), ["fact"], limit=10)] == ["m1"]


@pytest.mark.asyncio
async def test_metadata_title_tokens_recall_source_title(db):
    now = datetime.now(timezone.utc).isoformat()
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, labels, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "SFPAY-179397",
            "src-jira",
            "https://jira.example/browse/SFPAY-179397",
            "SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
            "PAY",
            '["lifecycle"]',
            now,
            "1",
            "doc-hash",
            now,
        ),
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    assert keyword.metadata_search_channels == (
        "bm25_metadata_tokens",
        "metadata_alias",
        "metadata_trigram",
    )
    assert keyword.disabled_metadata_search_channels == ()
    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]
    assert hits[0].channel == "bm25_metadata_tokens"
    assert "metadata_any" in hits[0].matched_fields
    assert hits[0].source_refs[0].source_id == "src-jira"
    assert hits[0].source_refs[0].doc_id == "SFPAY-179397"


@pytest.mark.asyncio
async def test_metadata_search_can_return_subchannel_hits_for_ranker(db):
    now = datetime.now(timezone.utc).isoformat()
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, labels, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "SFPAY-179397",
            "src-jira",
            "https://jira.example/browse/SFPAY-179397",
            "SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
            "PAY",
            '["lifecycle"]',
            now,
            "1",
            "doc-hash",
            now,
        ),
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata(
        '"create" "blocker" "hint"',
        _scope(),
        None,
        limit=10,
        include_subchannel_hits=True,
    )

    assert [hit.memory_id for hit in hits] == ["m-blocker", "m-blocker", "m-blocker"]
    assert [hit.channel for hit in hits] == [
        "bm25_metadata_tokens",
        "metadata_alias",
        "metadata_trigram",
    ]


@pytest.mark.asyncio
async def test_metadata_alias_recall_splits_camel_case_source_title(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: PeriodCutOffBlockerHint regression",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"period" "cut" "off" "blocker" "hint"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]
    assert hits[0].channel == "metadata_alias"
    assert "metadata_alias" in hits[0].matched_fields
    assert any("PeriodCutOffBlockerHint" in text for text in hits[0].matched_text)


@pytest.mark.asyncio
async def test_metadata_trigram_recall_handles_compound_no_space_query(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"create" "blockerhints"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]
    assert hits[0].channel == "metadata_trigram"
    assert "metadata_trigram" in hits[0].matched_fields
    assert any("Create Blocker Hint" in text for text in hits[0].matched_text)


@pytest.mark.asyncio
async def test_metadata_index_populates_from_atomic_memory_source_insert(db):
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.insert_memory_with_source_and_relation(
        _memory("m-blocker", "Lifecycle assignment skips person assignment creation"),
        doc_id="SFPAY-179397",
        source_type="jira",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]


@pytest.mark.asyncio
async def test_metadata_search_excludes_user_disabled_source_rows(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )
    await db.set_source_subscription("src-jira", LOCAL_DEV_USER_ID, False)

    keyword = SqliteKeywordSearch(db)

    assert await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10) == []


@pytest.mark.asyncio
async def test_metadata_search_applies_source_filter_to_matching_support_row(db):
    await db.insert_memory(_memory("m-shared", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await db.upsert_source(
        "src-wiki", "confluence", "Payroll Wiki", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _upsert_doc(
        db,
        doc_id="SFPAY-179397",
        source="src-jira",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await _upsert_doc(
        db,
        doc_id="wiki-runbook",
        source="src-wiki",
        title="Payroll lifecycle runbook",
    )
    await db.add_memory_source(
        "m-shared",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )
    await db.add_memory_source(
        "m-shared",
        "wiki-runbook",
        "confluence",
        support_kind="extracted",
        source_updated_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)

    assert (
        await keyword.search_metadata(
            '"create" "blocker" "hint"',
            _scope(),
            None,
            limit=10,
            source_filter=MemorySourceFilter(source_ids=("src-wiki",)),
        )
        == []
    )


@pytest.mark.asyncio
async def test_metadata_search_applies_source_time_range_to_matching_support_row(db):
    await db.insert_memory(_memory("m-shared", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-shared",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)

    assert (
        await keyword.search_metadata(
            '"create" "blocker" "hint"',
            _scope(),
            None,
            limit=10,
            time_range=MemoryTimeRange(
                after=datetime(2026, 6, 19, tzinfo=timezone.utc),
                before=datetime(2026, 6, 21, tzinfo=timezone.utc),
                date_type="source_updated_at",
            ),
        )
        == []
    )


@pytest.mark.asyncio
async def test_metadata_search_applies_client_filter_to_matching_support_row(db):
    await db.insert_memory(_memory("m-codex", "Agent session summary"))
    await db.upsert_source(
        "src-codex", "agent_session", "Codex Sessions", "{}", access_policy="private", owner_user_id="dev"
    )
    await _upsert_doc(
        db,
        doc_id="codex-session-1",
        source="src-codex",
        title="Create Blocker Hint investigation",
        client="codex",
    )
    await db.add_memory_source(
        "m-codex",
        "codex-session-1",
        "agent_session",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)

    hits = await keyword.search_metadata(
        '"create" "blocker" "hint"',
        _scope(),
        None,
        limit=10,
        source_filter=MemorySourceFilter(clients=("codex",)),
    )

    assert [hit.memory_id for hit in hits] == ["m-codex"]
    assert hits[0].source_refs[0].doc_id == "codex-session-1"
    assert (
        await keyword.search_metadata(
            '"create" "blocker" "hint"',
            _scope(),
            None,
            limit=10,
            source_filter=MemorySourceFilter(clients=("claude-code",)),
        )
        == []
    )


@pytest.mark.asyncio
async def test_metadata_search_applies_repo_identifier_filter(db):
    await db.insert_memory(
        _memory(
            "m-repo",
            "Repository investigation summary",
            repo_identifier="github.tools.sap/hcm/memforge-cloud",
        )
    )
    await db.upsert_source(
        "src-codex", "agent_session", "Codex Sessions", "{}", access_policy="private", owner_user_id="dev"
    )
    await _upsert_doc(
        db,
        doc_id="codex-session-1",
        source="src-codex",
        title="Create Blocker Hint investigation",
        client="codex",
    )
    await db.add_memory_source(
        "m-repo",
        "codex-session-1",
        "agent_session",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)

    hits = await keyword.search_metadata(
        '"create" "blocker" "hint"',
        _scope(),
        None,
        limit=10,
        source_filter=MemorySourceFilter(
            repo_identifiers=("github.tools.sap/hcm/memforge-cloud",),
        ),
    )

    assert [hit.memory_id for hit in hits] == ["m-repo"]
    assert (
        await keyword.search_metadata(
            '"create" "blocker" "hint"',
            _scope(),
            None,
            limit=10,
            source_filter=MemorySourceFilter(repo_identifiers=("github.tools.sap/hcm/other",)),
        )
        == []
    )


@pytest.mark.asyncio
async def test_metadata_search_applies_memory_updated_time_range(db):
    await db.insert_memory(
        _memory(
            "m-updated",
            "Lifecycle assignment skips person assignment creation",
            updated_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
        )
    )
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-updated",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)

    hits = await keyword.search_metadata(
        '"create" "blocker" "hint"',
        _scope(),
        None,
        limit=10,
        time_range=MemoryTimeRange(
            after=datetime(2026, 6, 19, tzinfo=timezone.utc),
            before=datetime(2026, 6, 21, tzinfo=timezone.utc),
            date_type="memory_updated_at",
        ),
    )

    assert [hit.memory_id for hit in hits] == ["m-updated"]
    assert (
        await keyword.search_metadata(
            '"create" "blocker" "hint"',
            _scope(),
            None,
            limit=10,
            time_range=MemoryTimeRange(
                after=datetime(2026, 6, 21, tzinfo=timezone.utc),
                before=datetime(2026, 6, 22, tzinfo=timezone.utc),
                date_type="memory_updated_at",
            ),
        )
        == []
    )


async def _upsert_doc(
    db: Database,
    *,
    doc_id: str = "SFPAY-179397",
    title: str,
    source: str = "src-jira",
    client: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source,
            source_url=f"https://jira.example/browse/{doc_id}",
            title=title,
            space_or_project="PAY",
            author="tester",
            last_modified=now,
            labels=["lifecycle"],
            version="1",
            content_hash=f"hash-{doc_id}-{title}",
            token_count=1,
            raw_content_uri=None,
            raw_content_type="text/html",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
            client=client,
        )
    )


@pytest.mark.asyncio
async def test_metadata_index_refreshes_when_document_title_changes(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(db, title="Old Lifecycle Title")
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]


@pytest.mark.asyncio
async def test_metadata_index_refreshes_when_source_name_changes(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(db, title="SFPAY-179397")
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    assert await keyword.search_metadata('"blocker" "hint" "queue"', _scope(), None, limit=10) == []

    await db.upsert_source(
        "src-jira", "jira", "Create Blocker Hint Queue", "{}", access_policy="workspace", owner_user_id="dev"
    )

    hits = await keyword.search_metadata('"blocker" "hint" "queue"', _scope(), None, limit=10)
    assert [hit.memory_id for hit in hits] == ["m-blocker"]


@pytest.mark.asyncio
async def test_rebuild_metadata_index_backfills_existing_sources(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )
    await db.db.execute("DELETE FROM memory_search_metadata_fts")
    await db.db.execute("DELETE FROM memory_search_metadata_alias_fts")
    await db.db.execute("DELETE FROM memory_search_metadata_trigram")
    await db.db.commit()

    keyword = SqliteKeywordSearch(db)
    assert await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10) == []

    await db.rebuild_memory_metadata_fts()

    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)
    assert [hit.memory_id for hit in hits] == ["m-blocker"]


@pytest.mark.asyncio
async def test_metadata_index_drops_deleted_document_support(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )
    keyword = SqliteKeywordSearch(db)
    assert [
        hit.memory_id for hit in await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)
    ] == ["m-blocker"]

    await db.delete_document("SFPAY-179397")

    assert await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10) == []


@pytest.mark.asyncio
async def test_metadata_search_collapses_multiple_matching_sources_per_memory(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    await _upsert_doc(
        db,
        doc_id="SFPAY-179397",
        title="SFPAY-179397: Create Blocker Hint in On Demand Lifecycle Assignment",
    )
    await _upsert_doc(
        db,
        doc_id="SFPAY-179398",
        title="SFPAY-179398: Create Blocker Hint follow-up",
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179397",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )
    await db.add_memory_source(
        "m-blocker",
        "SFPAY-179398",
        "jira",
        support_kind="extracted",
        source_updated_at=datetime.now(timezone.utc),
    )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=10)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]
    assert {ref.doc_id for ref in hits[0].source_refs} == {"SFPAY-179397", "SFPAY-179398"}


@pytest.mark.asyncio
async def test_metadata_search_preserves_source_refs_before_memory_limit(db):
    await db.insert_memory(_memory("m-blocker", "Lifecycle assignment skips person assignment creation"))
    await db.upsert_source("src-jira", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev")
    doc_ids = [f"SFPAY-17939{i}" for i in range(6)]
    for doc_id in doc_ids:
        await _upsert_doc(
            db,
            doc_id=doc_id,
            title=f"{doc_id}: Create Blocker Hint follow-up",
        )
        await db.add_memory_source(
            "m-blocker",
            doc_id,
            "jira",
            support_kind="extracted",
            source_updated_at=datetime.now(timezone.utc),
        )

    keyword = SqliteKeywordSearch(db)
    hits = await keyword.search_metadata('"create" "blocker" "hint"', _scope(), None, limit=1)

    assert [hit.memory_id for hit in hits] == ["m-blocker"]
    assert {ref.doc_id for ref in hits[0].source_refs} == set(doc_ids)


@pytest.mark.asyncio
async def test_remove_deletes_the_fts_row(db):
    await db.insert_memory(_memory("m1", "PostgreSQL connection pooling"))
    keyword = SqliteKeywordSearch(db)
    assert await keyword.search('"PostgreSQL"', _scope(), None, limit=10) != []
    await keyword.remove("m1")
    assert await keyword.search('"PostgreSQL"', _scope(), None, limit=10) == []
