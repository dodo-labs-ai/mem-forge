import pytest
from memforge.models import Memory, Visibility, content_hash, SHARED_PROJECT_KEY
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.adapters.context import AccessScope


@pytest.mark.asyncio
async def test_filter_visible_ids_strips_other_users_private(tmp_path):
    db = Database(str(tmp_path / "f.db"))
    await db.connect()
    try:
        await db.insert_memory(
            Memory(
                id="ws",
                memory_type="fact",
                content="x",
                content_hash=content_hash("x1"),
                visibility=Visibility.WORKSPACE.value,
                owner_user_id=None,
                project_key=SHARED_PROJECT_KEY,
            )
        )
        await db.insert_memory(
            Memory(
                id="priv",
                memory_type="fact",
                content="y",
                content_hash=content_hash("y1"),
                visibility=Visibility.PRIVATE.value,
                owner_user_id="u-2",
                project_key=SHARED_PROJECT_KEY,
            )
        )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        scope = AccessScope(
            user_id="u-1",
            include_private=True,  # PERSONALIZED
            allowed_statuses=("active",),
            active_project=None,
            scope_mode="project-first",
        )
        # Even if a leaky channel returned 'priv', filter_visible_ids must strip it.
        survivors = await adapters.relational.filter_visible_ids(["ws", "priv"], scope)
        assert survivors == {"ws"}
    finally:
        await db.close()
