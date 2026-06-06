# Cross-datastore isolation is a deployment-time property of the bound
# adapter handle, not of the core predicate; this test pack only exercises
# the predicate.
"""A row with NULL visibility is hidden everywhere.

The write-path invariant blocks NULL visibility on normal inserts, but a
malformed migration or an out-of-band SQL touch could land such a row in
the table. The predicate is default-deny: an unknown visibility value
satisfies neither branch, so every channel must skip it for every scope.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.retrieval.access_predicate import is_visible
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "null.db"))
    await database.connect()
    yield database
    await database.close()


def _scope(*, include_private: bool) -> AccessScope:
    return AccessScope(
        user_id="u-1",
        open_projects=frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=include_private,
        allowed_statuses=("active", "superseded"),
        active_project=None,
        scope_mode="project-first",
    )


async def _insert_null_visibility_row(db: Database, mid: str, content: str) -> None:
    """Insert a row with visibility=NULL via raw SQL.

    The write-path invariant rejects NULL through the normal API AND the
    schema declares visibility NOT NULL with a CHECK on the allowed set, so
    the test rewrites the schema in place via PRAGMA writable_schema to
    relax both clauses just long enough to fabricate a malformed row, then
    the connection is reset so subsequent reads see the rewritten schema.
    """
    now = datetime.now(timezone.utc).isoformat()
    # Drop NOT NULL and the visibility CHECK by editing sqlite_master, then
    # reset the connection so the parser picks up the new schema definition.
    await db.db.execute("PRAGMA writable_schema = 1")
    await db.db.execute(
        "UPDATE sqlite_master SET sql = "
        "REPLACE(sql, 'visibility          TEXT NOT NULL DEFAULT ''workspace''', "
        "             'visibility          TEXT             DEFAULT ''workspace''') "
        "WHERE type = 'table' AND name = 'memories'"
    )
    await db.db.execute(
        "UPDATE sqlite_master SET sql = "
        "REPLACE(sql, 'CHECK (visibility IN (''private'',''workspace''))', "
        "             'CHECK (1 = 1)') "
        "WHERE type = 'table' AND name = 'memories'"
    )
    await db.db.execute(
        "UPDATE sqlite_master SET sql = "
        "REPLACE(sql, 'CHECK ((visibility = ''private'') = (owner_user_id IS NOT NULL))', "
        "             'CHECK (1 = 1)') "
        "WHERE type = 'table' AND name = 'memories'"
    )
    await db.db.execute("PRAGMA writable_schema = 0")
    await db.db.commit()
    # Force the schema parser to reread by closing and reopening the connection.
    db_path = db.db_path
    await db._db.close()
    import aiosqlite
    db._db = await aiosqlite.connect(db_path)
    db._db.row_factory = aiosqlite.Row
    await db.db.execute(
        """INSERT INTO memories (
            id, memory_type, content, content_hash, tags,
            visibility, owner_user_id, project_key, confidence,
            corroboration_count, contradiction_count,
            valid_from, valid_until, superseded_by, status,
            retirement_reason, retired_at, superseded_at,
            replacement_reason, extraction_context,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, '[]',
                  NULL, NULL, ?, 0.5,
                  1, 0,
                  NULL, NULL, NULL, 'active',
                  NULL, NULL, NULL,
                  NULL, NULL,
                  ?, ?)""",
        (mid, "fact", content, content_hash(content + mid),
         SHARED_PROJECT_KEY, now, now),
    )
    # FTS5 row so BM25 can attempt to surface it; the predicate must still hide.
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
        "VALUES (?, ?, '', '')",
        (mid, content),
    )
    await db.db.commit()


async def _insert_workspace_row(db: Database, mid: str, content: str) -> None:
    await db.insert_memory(Memory(
        id=mid, memory_type="fact", content=content,
        content_hash=content_hash(content + mid),
        visibility=WORKSPACE, owner_user_id=None,
        project_key=SHARED_PROJECT_KEY, tags=[],
    ))


def test_is_visible_null_row_default_denied():
    row = {
        "status": "active",
        "visibility": None,
        "owner_user_id": None,
        "project_key": SHARED_PROJECT_KEY,
    }
    assert is_visible(row, _scope(include_private=False)) is False
    assert is_visible(row, _scope(include_private=True)) is False


@pytest.mark.asyncio
async def test_keyword_channel_hides_null_visibility_row(db):
    await _insert_null_visibility_row(db, "n-bm25", "argocd deploys things")
    await _insert_workspace_row(db, "ws-bm25", "argocd deploys things")
    adapters = build_sqlite_adapters(db, memory_collection=None)

    for include_private in (False, True):
        hits = await adapters.keyword.search(
            "argocd", _scope(include_private=include_private),
            memory_types=None, limit=10,
        )
        ids = {mid for mid, _ in hits}
        assert "n-bm25" not in ids
        assert "ws-bm25" in ids


@pytest.mark.asyncio
async def test_temporal_channel_hides_null_visibility_row(db):
    from datetime import timedelta
    await _insert_null_visibility_row(db, "n-temp", "x")
    await _insert_workspace_row(db, "ws-temp", "x")
    adapters = build_sqlite_adapters(db, memory_collection=None)
    now = datetime.now(timezone.utc)

    for include_private in (False, True):
        hits = await adapters.relational.temporal_search(
            after=now - timedelta(days=1), before=None,
            scope=_scope(include_private=include_private),
            memory_types=None, limit=10,
        )
        ids = {mid for mid, _ in hits}
        assert "n-temp" not in ids
        assert "ws-temp" in ids


@pytest.mark.asyncio
async def test_graph_channel_hides_null_visibility_row(db):
    eid = await db.upsert_entity("argocd", "tool")
    await _insert_null_visibility_row(db, "n-graph", "deploys via argocd")
    await db.link_memory_entity("n-graph", eid)
    await _insert_workspace_row(db, "ws-graph", "deploys via argocd")
    await db.link_memory_entity("ws-graph", eid)
    adapters = build_sqlite_adapters(db, memory_collection=None)

    for include_private in (False, True):
        hits = await adapters.relational.graph_search(
            [eid], _scope(include_private=include_private),
            memory_types=None, limit=10,
        )
        ids = {mid for mid, _ in hits}
        assert "n-graph" not in ids
        assert "ws-graph" in ids


@pytest.mark.asyncio
async def test_filter_visible_ids_strips_null_visibility_row(db):
    await _insert_null_visibility_row(db, "n-post", "y")
    await _insert_workspace_row(db, "ws-post", "y")
    adapters = build_sqlite_adapters(db, memory_collection=None)

    for include_private in (False, True):
        survivors = await adapters.relational.filter_visible_ids(
            ["n-post", "ws-post"], _scope(include_private=include_private),
        )
        assert survivors == {"ws-post"}
