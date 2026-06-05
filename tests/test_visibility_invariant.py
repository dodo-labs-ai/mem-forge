# tests/test_visibility_invariant.py
import pytest
from memforge.models import Memory, Visibility, content_hash
from memforge.storage.database import Database

WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _mem(**kw):
    base = dict(id="mem-x", memory_type="fact", content="c", content_hash=content_hash("c"))
    base.update(kw)
    return Memory(**base)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "inv.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_insert_rejects_private_without_owner(db):
    with pytest.raises(ValueError, match="owner_user_id"):
        await db.insert_memory(_mem(visibility=PRIVATE, owner_user_id=None))


@pytest.mark.asyncio
async def test_insert_rejects_workspace_with_owner(db):
    with pytest.raises(ValueError, match="owner_user_id"):
        await db.insert_memory(_mem(visibility=WORKSPACE, owner_user_id="u-1"))


@pytest.mark.asyncio
async def test_insert_rejects_reserved_org_visibility(db):
    # 'org' is intentionally not a Visibility member: it is the reserved value the
    # write path must reject, so it stays a literal here.
    with pytest.raises(ValueError, match="visibility"):
        await db.insert_memory(_mem(visibility="org", owner_user_id=None))


@pytest.mark.asyncio
async def test_insert_persists_workspace_row(db):
    await db.insert_memory(_mem(id="mem-ok", visibility=WORKSPACE))
    stored = await db.get_memory("mem-ok")
    assert stored.visibility == WORKSPACE
    assert stored.owner_user_id is None


@pytest.mark.asyncio
async def test_fresh_check_rejects_private_without_owner(db):
    # Layer 1: the CHECK on the fresh schema rejects a raw-SQL bypass write.
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        await db.db.execute(
            "INSERT INTO memories (id, memory_type, content, content_hash, visibility, owner_user_id) "
            f"VALUES ('raw-bad','fact','c','h','{PRIVATE}',NULL)"
        )


@pytest.mark.asyncio
async def test_fresh_check_rejects_workspace_with_owner(db):
    # Layer 1: the CHECK rejects a workspace row that carries an owner.
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        await db.db.execute(
            "INSERT INTO memories (id, memory_type, content, content_hash, visibility, owner_user_id) "
            f"VALUES ('raw-ws-owner','fact','c','h','{WORKSPACE}','u-1')"
        )


@pytest.mark.asyncio
async def test_fresh_check_rejects_unknown_visibility(db):
    # Layer 1: the CHECK rejects a visibility value outside the allowed set.
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        await db.db.execute(
            "INSERT INTO memories (id, memory_type, content, content_hash, visibility, owner_user_id) "
            "VALUES ('raw-bad-vis','fact','c','h','org',NULL)"
        )


@pytest.mark.asyncio
async def test_supersede_rejects_invalid_new_memory(db):
    # Layer 2: supersede_memory validates the incoming new memory.
    await db.insert_memory(_mem(id="old-1", visibility=WORKSPACE))
    with pytest.raises(ValueError, match="owner_user_id"):
        await db.supersede_memory(
            "old-1",
            _mem(id="new-1", visibility=PRIVATE, owner_user_id=None),
            replacement_reason="x",
        )


@pytest.mark.asyncio
async def test_restore_snapshot_rejects_invalid_memory(db):
    # Layer 2: restore_memory_snapshot validates before the UPDATE.
    with pytest.raises(ValueError, match="visibility"):
        await db.restore_memory_snapshot(
            _mem(id="r-1", visibility="org"),
            search_visible_statuses={"active"},
        )


@pytest.mark.asyncio
async def test_relational_adapter_delegates_invariant(db):
    # Layer 3: the RelationalStore adapter delegates to the validating Database method.
    from memforge.storage.adapters.sqlite import build_sqlite_adapters
    adapters = build_sqlite_adapters(db, memory_collection=None)  # insert path never touches the vector
    with pytest.raises(ValueError, match="owner_user_id"):
        await adapters.relational.insert_memory(_mem(visibility=PRIVATE, owner_user_id=None))


@pytest.mark.asyncio
async def test_row_to_memory_has_no_scope_attr(db):
    await db.insert_memory(_mem(id="mem-r", visibility=WORKSPACE))
    stored = await db.get_memory("mem-r")
    assert not hasattr(stored, "scope")
    assert stored.visibility == WORKSPACE
