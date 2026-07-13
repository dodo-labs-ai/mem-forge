import pytest
from memforge.memory.audit import MemoryAuditLogger
from memforge.models import Memory, Visibility, content_hash, SHARED_PROJECT_KEY
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


@pytest.mark.asyncio
async def test_promote_to_workspace_raises_audits_and_does_not_mutate(tmp_path):
    db = Database(str(tmp_path / "promote.db"))
    await db.connect()
    try:
        priv = Memory(
            id="m-priv",
            memory_type="fact",
            content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.PRIVATE.value,
            owner_user_id="u-alice",
            project_key=SHARED_PROJECT_KEY,
            tags=[],
        )
        await db.insert_memory(priv)

        audit = MemoryAuditLogger(db)
        adapters = build_sqlite_adapters(db, memory_collection=None, audit_logger=audit)
        with pytest.raises(NotImplementedError):
            await adapters.relational.promote_to_workspace(
                memory_id="m-priv",
                actor_user_id="u-alice",
                reason="manual promote",
            )

        # Row state is untouched.
        stored = await db.get_memory("m-priv")
        assert stored.visibility == Visibility.PRIVATE.value
        assert stored.owner_user_id == "u-alice"

        # The audit ledger recorded the attempt.
        events = await db.list_memory_audit_events(memory_id="m-priv")
        promoted = [e for e in events if e.event_type == "memory_promoted"]
        assert len(promoted) == 1
        assert promoted[0].status == "failed"
        assert promoted[0].reason == "not_implemented"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_promote_to_workspace_rejects_non_owner(tmp_path):
    db = Database(str(tmp_path / "promote_owner.db"))
    await db.connect()
    try:
        priv = Memory(
            id="m-priv",
            memory_type="fact",
            content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.PRIVATE.value,
            owner_user_id="u-alice",
            project_key=SHARED_PROJECT_KEY,
            tags=[],
        )
        await db.insert_memory(priv)

        # Build adapters WITH an audit logger so the assertion that no
        # memory_promoted row was emitted is meaningful. Without this, the
        # test cannot catch an implementation that audits the hostile attempt
        # before raising PermissionError.
        audit = MemoryAuditLogger(db)
        adapters = build_sqlite_adapters(db, memory_collection=None, audit_logger=audit)
        # Non-owner attempts to promote: PermissionError, not NIE, before any
        # audit emission (a hostile actor never triggers an audit row).
        with pytest.raises(PermissionError):
            await adapters.relational.promote_to_workspace(
                memory_id="m-priv",
                actor_user_id="u-eve",
                reason="hostile",
            )

        # No audit row was written for the hostile attempt.
        events = await db.list_memory_audit_events(memory_id="m-priv")
        assert [e for e in events if e.event_type == "memory_promoted"] == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_promote_to_workspace_missing_memory_raises_lookup(tmp_path):
    db = Database(str(tmp_path / "promote_missing.db"))
    await db.connect()
    try:
        adapters = build_sqlite_adapters(db, memory_collection=None)
        with pytest.raises(LookupError):
            await adapters.relational.promote_to_workspace(
                memory_id="m-nope",
                actor_user_id="u-alice",
                reason="x",
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_promote_to_workspace_rejects_workspace_memory(tmp_path):
    db = Database(str(tmp_path / "promote_ws.db"))
    await db.connect()
    try:
        ws = Memory(
            id="m-ws",
            memory_type="fact",
            content="x",
            content_hash=content_hash("x"),
            visibility=Visibility.WORKSPACE.value,
            owner_user_id=None,
            project_key=SHARED_PROJECT_KEY,
            tags=[],
        )
        await db.insert_memory(ws)
        adapters = build_sqlite_adapters(db, memory_collection=None)
        with pytest.raises(ValueError):
            await adapters.relational.promote_to_workspace(
                memory_id="m-ws",
                actor_user_id="u-alice",
                reason="confused",
            )
    finally:
        await db.close()
