"""Alias-expansion produces an FTS5-valid query that still matches.

The keyword channel composes the BM25 query out of two pieces with very
different syntactic contracts:

* the user's typed terms, which arrive untrusted and must be sanitized into a
  flat AND of quoted phrases;
* an alias OR-group built from canonical entity aliases, which is engine-
  authored FTS5 and must reach the matcher with its parens, ``OR`` operator,
  and quoted phrases intact.

The regression covered here is what happens when those two contracts get
mixed: if the alias clause is run through the user-input sanitizer, ``OR``
collapses to a literal phrase and the parens vanish, turning the MATCH into an
implicit AND of every word plus the literal token ``OR``. No row contains
``OR`` as content, so the channel returns nothing.

The composition itself also matters. FTS5 will not parse a phrase list
followed directly by a parenthesized OR-group; the two halves must be tied
together with an explicit operator. The keyword channel ORs them, so an
alias hit broadens recall instead of narrowing it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.config import RetrievalConfig
from memforge.models import Memory, content_hash
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.retrieval.search import SearchEngine
from memforge.storage.adapters.context import (
    LOCAL_DEV_USER_ID,
    AccessScope,
)
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database
from memforge.memory.lifecycle import allowed_search_statuses


class FakeCollection:
    """Vector stub so the SearchEngine can be constructed in isolation."""

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs):
        pass

    def delete(self, **kwargs):
        pass

    def get(self, **kwargs):
        return {"ids": []}


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
    )


def _scope() -> AccessScope:
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=allowed_search_statuses(False),
        active_project=None,
        scope_mode="project-first",
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "alias-expansion.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_bm25_with_alias_expansion_still_matches(db):
    """User terms plus an alias OR-group must still hit the keyword index.

    The memory contains the user's literal phrase (``payroll cutoff``) but
    none of the alias terms; BM25 must surface it because the alias clause
    is an independent OR-group, not an additional AND constraint.
    """
    target = _memory("mem-pay", "Payroll cutoff is the 25th of each month.")
    distractor = _memory("mem-other", "Compliance review schedule for Q4.")
    await db.insert_memory(target)
    await db.insert_memory(distractor)

    cutoff_id = await db.upsert_entity("cutoff", "Cutoff", tags=["concept"])
    for alias in ("reverse cutoff", "cut off", "reverse cut off"):
        await db.insert_alias(alias, alias.lower(), cutoff_id, source="manual")

    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )

    hits = await engine._bm25_search(
        query="payroll cutoff",
        analysis=QueryAnalysis(detected_entity_ids=[cutoff_id]),
        memory_types=None,
        sources=None,
        scope=_scope(),
        limit=10,
    )
    assert [mem_id for mem_id, _ in hits] == ["mem-pay"]


@pytest.mark.asyncio
async def test_build_alias_clause_emits_fts5_or_group(db):
    """The alias clause is the parenthesized OR-group, nothing more.

    Aliases that already appear in the user query are skipped so the clause
    only contributes new terms. The user query itself is composed in by
    ``_bm25_search`` after sanitization, never returned from this helper.
    """
    cutoff_id = await db.upsert_entity("cutoff", "Cutoff", tags=["concept"])
    for alias in ("reverse cutoff", "cut off", "reverse cut off"):
        await db.insert_alias(alias, alias.lower(), cutoff_id, source="manual")

    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )

    clause = await engine._build_alias_clause([cutoff_id], "payroll cutoff")
    assert clause == '("reverse cutoff" OR "cut off" OR "reverse cut off")'


@pytest.mark.asyncio
async def test_build_alias_clause_returns_empty_when_no_new_terms(db):
    """No aliases past the substring filter means an empty clause.

    A returned ``""`` lets ``_bm25_search`` keep the FTS query as the
    sanitized user terms alone, instead of appending a stray space or empty
    parens that would invalidate the MATCH.
    """
    entity_id = await db.upsert_entity("payroll", "Payroll", tags=["concept"])
    await db.insert_alias("payroll", "payroll", entity_id, source="manual")

    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )

    clause = await engine._build_alias_clause([entity_id], "payroll cutoff")
    assert clause == ""
