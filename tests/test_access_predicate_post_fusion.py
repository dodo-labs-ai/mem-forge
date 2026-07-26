from datetime import datetime, timezone

import pytest
from memforge.models import DocumentRecord, Memory, Visibility, content_hash, SHARED_PROJECT_KEY
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


@pytest.mark.asyncio
async def test_filter_visible_ids_admits_virtual_user_provenance_without_widening_dangling_sources(
    tmp_path,
):
    db = Database(str(tmp_path / "virtual-provenance.db"))
    await db.connect()
    try:
        observed_at = datetime.now(timezone.utc)
        for memory_id in ("manual", "dangling"):
            await db.insert_memory(
                Memory(
                    id=memory_id,
                    memory_type="fact",
                    content=f"{memory_id} private memory",
                    content_hash=content_hash(memory_id),
                    visibility=Visibility.PRIVATE.value,
                    owner_user_id="u-1",
                    project_key=SHARED_PROJECT_KEY,
                )
            )
        for doc_id, source in (
            ("doc-manual", "user_memory"),
            ("doc-dangling", "missing-configured-source"),
        ):
            await db.upsert_document(
                DocumentRecord(
                    doc_id=doc_id,
                    source=source,
                    source_url=f"memforge://{doc_id}",
                    title=doc_id,
                    space_or_project=SHARED_PROJECT_KEY,
                    author="u-1",
                    last_modified=observed_at,
                    labels=[],
                    version="1",
                    content_hash=content_hash(doc_id),
                    token_count=1,
                    raw_content_uri=None,
                    raw_content_type=None,
                    normalized_content_uri=None,
                    pdf_content_uri=None,
                    last_synced=observed_at,
                )
            )
        await db.add_memory_source(
            "manual",
            "doc-manual",
            "user_memory",
            source_updated_at=observed_at,
        )
        await db.add_memory_source(
            "dangling",
            "doc-dangling",
            "confluence",
            source_updated_at=observed_at,
        )
        adapters = build_sqlite_adapters(db, memory_collection=None)
        scope = AccessScope(
            user_id="u-1",
            include_private=True,
            allowed_statuses=("active",),
            active_project=None,
            scope_mode="project-first",
        )

        survivors = await adapters.relational.filter_visible_ids(
            ["manual", "dangling"],
            scope,
        )

        assert survivors == {"manual"}
    finally:
        await db.close()
