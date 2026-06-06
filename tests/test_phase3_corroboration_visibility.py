"""Adversarial tests for write-side corroboration boundaries.

`add_source_support` is the mutating gate for support edges; it must reject
any writer-vs-target mismatch on visibility, owner, or project so that a
support edge cannot leak across a boundary the access predicate enforces
on the read side.
"""

import pytest

from memforge.memory.store import MemoryStore
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


@pytest.mark.asyncio
async def test_add_source_support_does_not_cross_visibility(tmp_path):
    db = Database(str(tmp_path / "c.db"))
    await db.connect()
    try:
        priv = Memory(
            id="m-priv", memory_type="fact", content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.PRIVATE.value, owner_user_id="u-alice",
            project_key=SHARED_PROJECT_KEY, tags=[],
        )
        await db.insert_memory(priv)

        adapters = build_sqlite_adapters(db, memory_collection=None)
        store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector,
                            embed_cfg={}, dedup_threshold=0.08)

        outcome = await store.add_source_support(
            memory_id="m-priv",
            doc_id="d-extra",
            source_type="confluence",
            writer_visibility=Visibility.WORKSPACE.value,
            writer_owner_user_id=None,
        )
        assert outcome == "rejected"

        stored = await db.get_memory("m-priv")
        assert stored.corroboration_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_add_source_support_does_not_cross_private_owner(tmp_path):
    db = Database(str(tmp_path / "c_owner.db"))
    await db.connect()
    try:
        alice = Memory(
            id="m-alice", memory_type="fact", content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.PRIVATE.value, owner_user_id="u-alice",
            project_key=SHARED_PROJECT_KEY, tags=[],
        )
        await db.insert_memory(alice)

        adapters = build_sqlite_adapters(db, memory_collection=None)
        store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector,
                            embed_cfg={}, dedup_threshold=0.08)

        outcome = await store.add_source_support(
            memory_id="m-alice",
            doc_id="d-bob",
            source_type="agent_session",
            writer_visibility=Visibility.PRIVATE.value,
            writer_owner_user_id="u-bob",
        )
        assert outcome == "rejected"

        stored = await db.get_memory("m-alice")
        assert stored.corroboration_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_add_source_support_does_not_cross_workspace_project(tmp_path):
    db = Database(str(tmp_path / "ws_proj.db"))
    await db.connect()
    try:
        risk = Memory(
            id="m-risk", memory_type="fact", content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.WORKSPACE.value, owner_user_id=None,
            project_key="RISK", tags=[],
        )
        await db.insert_memory(risk)

        adapters = build_sqlite_adapters(db, memory_collection=None)
        store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector,
                            embed_cfg={}, dedup_threshold=0.08)

        outcome = await store.add_source_support(
            memory_id="m-risk",
            doc_id="d-pay",
            source_type="confluence",
            writer_visibility=Visibility.WORKSPACE.value,
            writer_owner_user_id=None,
            writer_project_key="PAY",
        )
        assert outcome == "rejected"

        stored = await db.get_memory("m-risk")
        assert stored.corroboration_count == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_add_source_support_rejects_workspace_writer_with_none_project_against_real_project_target(tmp_path):
    """A workspace writer with project_key=None gets normalized to UNSORTED on
    persistence, so it must NOT corroborate a target with project_key="PAY".
    Currently the writer-None path skips the check, allowing the cross-project edge."""
    target = Memory(
        id="m-pay", memory_type="fact", content="x", content_hash=content_hash("x"),
        visibility=Visibility.WORKSPACE.value, owner_user_id=None,
        project_key="PAY", tags=[],
    )
    db = Database(str(tmp_path / "c2.db"))
    await db.connect()
    try:
        await db.insert_memory(target)
        adapters = build_sqlite_adapters(db, memory_collection=None)
        store = MemoryStore(adapters.relational, adapters.keyword, adapters.vector,
                            embed_cfg={}, dedup_threshold=0.08)
        outcome = await store.add_source_support(
            memory_id="m-pay",
            doc_id="d-noproj",
            source_type="confluence",
            writer_visibility=Visibility.WORKSPACE.value,
            writer_owner_user_id=None,
            writer_project_key=None,
        )
        assert outcome == "rejected"
        stored = await db.get_memory("m-pay")
        assert stored.corroboration_count == 1
    finally:
        await db.close()
