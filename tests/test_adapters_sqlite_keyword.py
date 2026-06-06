"""SqliteKeywordSearch: the read-path FTS5 query and the standalone delete."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import Memory, content_hash
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


def _memory(mem_id: str, content: str, status: str = "active") -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
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
async def test_remove_deletes_the_fts_row(db):
    await db.insert_memory(_memory("m1", "PostgreSQL connection pooling"))
    keyword = SqliteKeywordSearch(db)
    assert await keyword.search('"PostgreSQL"', _scope(), None, limit=10) != []
    await keyword.remove("m1")
    assert await keyword.search('"PostgreSQL"', _scope(), None, limit=10) == []
