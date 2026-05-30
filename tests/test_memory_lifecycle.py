"""Tests for the lean memory lifecycle state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from meminception.models import Memory, content_hash
from meminception.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "lifecycle.db"))
    await database.connect()
    yield database
    await database.close()


def _make_memory(mem_id: str, content: str, *, status: str = "active", corroboration_count: int = 1) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        corroboration_count=corroboration_count,
        created_at=now,
        updated_at=now,
        status=status,
    )


async def _insert_doc(db: Database, doc_id: str, source: str = "src-1") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, source, f"http://test/{doc_id}", doc_id, "TEST", now, "1", f"hash-{doc_id}", now),
    )
    await db.db.commit()


class TestStatusNormalization:
    @pytest.mark.asyncio
    async def test_decayed_input_is_stored_as_retired_with_reason(self, db):
        mem = _make_memory("mem-decay001", "This fact should be hidden")
        await db.insert_memory(mem)

        await db.update_memory_status(mem.id, "decayed", reason="admin_hidden")

        stored = await db.get_memory(mem.id)
        assert stored.status == "retired"
        assert stored.retirement_reason == "admin_hidden"
        assert stored.retired_at is not None


class TestSupportAwareRetirement:
    @pytest.mark.asyncio
    async def test_source_deletion_with_remaining_support_keeps_memory_active(self, db):
        await _insert_doc(db, "doc-a")
        await _insert_doc(db, "doc-b")
        mem = _make_memory("mem-support1", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-a", "confluence")
        await db.add_memory_source(mem.id, "doc-b", "confluence")

        await db.remove_memory_source(mem.id, "doc-a", retire_reason="source_deleted")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert [s.doc_id for s in sources] == ["doc-b"]

    @pytest.mark.asyncio
    async def test_source_deletion_with_no_remaining_support_retires_memory(self, db):
        await _insert_doc(db, "doc-a")
        mem = _make_memory("mem-nosupport", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-a", "confluence")

        await db.remove_memory_source(mem.id, "doc-a", retire_reason="source_deleted")

        stored = await db.get_memory(mem.id)
        assert stored.status == "retired"
        assert stored.retirement_reason == "source_deleted"
        assert stored.retired_at is not None

    @pytest.mark.asyncio
    async def test_document_deletion_retires_only_zero_support_memory(self, db):
        await _insert_doc(db, "doc-a", source="src-delete")
        await _insert_doc(db, "doc-b", source="src-keep")
        mem = _make_memory("mem-docdel01", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-a", "confluence")
        await db.add_memory_source(mem.id, "doc-b", "confluence")

        await db.delete_document("doc-a")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert stored.corroboration_count == 1
        assert [s.doc_id for s in sources] == ["doc-b"]

    @pytest.mark.asyncio
    async def test_losing_last_extracted_source_keeps_corroborated_memory_active(self, db):
        await _insert_doc(db, "doc-owner")
        await _insert_doc(db, "doc-support")
        mem = _make_memory("mem-owner01", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted")
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated")

        await db.remove_memory_source(mem.id, "doc-owner", retire_reason="source_deleted")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert stored.retirement_reason is None
        assert stored.corroboration_count == 1
        assert [(source.doc_id, source.support_kind) for source in sources] == [
            ("doc-support", "corroborated"),
        ]

    @pytest.mark.asyncio
    async def test_removing_corroborated_source_keeps_extracted_owner_active(self, db):
        await _insert_doc(db, "doc-owner")
        await _insert_doc(db, "doc-support")
        mem = _make_memory("mem-corrob01", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted")
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated")

        await db.remove_memory_source(mem.id, "doc-support", retire_reason="no_support")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert stored.corroboration_count == 1
        assert [(source.doc_id, source.support_kind) for source in sources] == [
            ("doc-owner", "extracted"),
        ]

    @pytest.mark.asyncio
    async def test_document_deletion_keeps_corroborated_memory_active(self, db):
        await _insert_doc(db, "doc-owner", source="src-delete")
        await _insert_doc(db, "doc-support", source="src-keep")
        mem = _make_memory("mem-docown01", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted")
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated")

        await db.delete_document("doc-owner")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert stored.retirement_reason is None
        assert stored.corroboration_count == 1
        assert [(source.doc_id, source.support_kind) for source in sources] == [
            ("doc-support", "corroborated"),
        ]

    @pytest.mark.asyncio
    async def test_source_cascade_retires_instead_of_deleting_zero_support_memory(self, db):
        await _insert_doc(db, "doc-a", source="src-delete")
        mem = _make_memory("mem-srcdel01", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-a", "confluence")

        await db.delete_source_cascade("src-delete")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored is not None
        assert stored.status == "retired"
        assert stored.retirement_reason == "source_deleted"
        assert sources == []


class TestSupersessionMetadata:
    @pytest.mark.asyncio
    async def test_supersede_sets_valid_until_and_supersession_metadata(self, db):
        old = _make_memory("mem-old0001", "PostgreSQL version is 14")
        new = _make_memory("mem-new0001", "PostgreSQL version is 16")
        await db.insert_memory(old)

        await db.supersede_memory(old.id, new, replacement_reason="same_source_replacement")

        stored_old = await db.get_memory(old.id)
        stored_new = await db.get_memory(new.id)
        assert stored_old.status == "superseded"
        assert stored_old.superseded_by == new.id
        assert stored_old.valid_until is not None
        assert stored_old.superseded_at is not None
        assert stored_old.replacement_reason == "same_source_replacement"
        assert stored_new.status == "active"


class TestExpiryRetirement:
    @pytest.mark.asyncio
    async def test_expired_memory_moves_to_retired_with_reason(self, db):
        expired = _make_memory("mem-expired1", "Temporary rollout flag is enabled")
        expired.valid_until = datetime.now(timezone.utc) - timedelta(days=1)
        await db.insert_memory(expired)

        retired_count = await db.retire_expired_memories()

        stored = await db.get_memory(expired.id)
        assert retired_count == 1
        assert stored.status == "retired"
        assert stored.retirement_reason == "expired"
        assert stored.retired_at is not None


class TestHardPurge:
    @pytest.mark.asyncio
    async def test_hard_purge_removes_memory_content_indexes_and_sensitive_links(self, db):
        await _insert_doc(db, "doc-a")
        entity_id = await db.upsert_entity("postgresql", display_name="PostgreSQL", tags=["technology"])
        memory = _make_memory("mem-purge01", "Private credential detail")
        dependent = _make_memory("mem-dependent", "This memory used to replace the purged one")
        await db.insert_memory(memory)
        await db.insert_memory(dependent)
        await db.add_memory_source(memory.id, "doc-a", "confluence", excerpt="sensitive excerpt")
        await db.db.execute(
            "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (memory.id, entity_id),
        )
        await db.db.execute(
            """INSERT INTO memory_contradictions
               (memory_id_a, memory_id_b, classification, reason)
               VALUES (?, ?, ?, ?)""",
            (memory.id, dependent.id, "contradiction", "test"),
        )
        await db.db.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            (memory.id, dependent.id),
        )
        await db.db.commit()

        purged = await db.purge_memory(memory.id)

        assert purged is True
        assert await db.get_memory(memory.id) is None
        assert await db.get_memory_sources(memory.id) == []
        async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with db.db.execute("SELECT COUNT(*) FROM memory_entities WHERE memory_id = ?", (memory.id,)) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with db.db.execute(
            "SELECT COUNT(*) FROM memory_contradictions WHERE memory_id_a = ? OR memory_id_b = ?",
            (memory.id, memory.id),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0

        stored_dependent = await db.get_memory(dependent.id)
        assert stored_dependent.status == "retired"
        assert stored_dependent.superseded_by is None
        assert stored_dependent.retirement_reason == "privacy_removed"
