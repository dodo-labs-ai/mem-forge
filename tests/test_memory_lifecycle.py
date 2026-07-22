"""Tests for the lean memory lifecycle state machine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memforge.models import DocumentRecord, Memory, SyncState, content_hash
from memforge.storage.database import Database


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
    if await db.get_source(source) is None:
        await db.upsert_source(
            id=source,
            type="confluence",
            name=source,
            config_json="{}",
            access_policy="workspace",
            owner_user_id="owner-1",
        )
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
        await db.add_memory_source(mem.id, "doc-a", "confluence", source_updated_at=None)
        await db.add_memory_source(mem.id, "doc-b", "confluence", source_updated_at=None)

        await db.remove_memory_source(mem.id, "doc-a", source_id="src-1", retire_reason="source_deleted")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored.status == "active"
        assert [s.doc_id for s in sources] == ["doc-b"]

    @pytest.mark.asyncio
    async def test_source_deletion_with_no_remaining_support_retires_memory(self, db):
        await _insert_doc(db, "doc-a")
        mem = _make_memory("mem-nosupport", "Service uses PostgreSQL")
        await db.insert_memory(mem)
        await db.add_memory_source(mem.id, "doc-a", "confluence", source_updated_at=None)

        await db.remove_memory_source(mem.id, "doc-a", source_id="src-1", retire_reason="source_deleted")

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
        await db.add_memory_source(mem.id, "doc-a", "confluence", source_updated_at=None)
        await db.add_memory_source(mem.id, "doc-b", "confluence", source_updated_at=None)

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
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted", source_updated_at=None)
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated", source_updated_at=None)

        await db.remove_memory_source(mem.id, "doc-owner", source_id="src-1", retire_reason="source_deleted")

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
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted", source_updated_at=None)
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated", source_updated_at=None)

        await db.remove_memory_source(mem.id, "doc-support", source_id="src-1", retire_reason="no_support")

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
        await db.add_memory_source(mem.id, "doc-owner", "confluence", support_kind="extracted", source_updated_at=None)
        await db.add_memory_source(mem.id, "doc-support", "jira", support_kind="corroborated", source_updated_at=None)

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
        await db.add_memory_source(mem.id, "doc-a", "confluence", source_updated_at=None)

        await db.delete_source_cascade("src-delete")

        stored = await db.get_memory(mem.id)
        sources = await db.get_memory_sources(mem.id)
        assert stored is not None
        assert stored.status == "retired"
        assert stored.retirement_reason == "source_deleted"
        assert sources == []

    @pytest.mark.asyncio
    async def test_source_cascade_removes_subscription_and_durable_sync_state(self, db):
        source_id = "src-reusable"
        await db.upsert_source(
            source_id, "confluence", "Reusable Source", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.set_source_subscription(source_id, "user-1", False)
        await db.create_source_sync_input(
            source_id=source_id,
            raw_uri="object-store://old-input",
            raw_sha256="old-sha",
            raw_content_type="application/json",
        )
        run = await db.enqueue_source_sync_run(source_id=source_id, trigger="manual")
        leased = await db.lease_next_source_sync_run(worker_id="test-worker")
        assert leased is not None and leased.run_id == run.run_id
        assert await db.complete_source_sync_run(
            run.run_id,
            worker_id="test-worker",
            lease_attempt_count=leased.lease_attempt_count,
            final_state=SyncState(source=source_id, last_sync_status="success"),
        )

        await db.delete_source_cascade(source_id)

        assert await db.get_latest_source_sync_run(source_id=source_id) is None
        assert await db.list_source_sync_inputs(source_id=source_id) == []

        await db.upsert_source(
            source_id, "confluence", "Recreated Source", "{}", access_policy="workspace", owner_user_id="dev"
        )
        assert await db.is_source_enabled_for_user(source_id, "user-1") is True

    @pytest.mark.asyncio
    async def test_source_cascade_durably_records_exact_artifacts_for_cleanup(self, db):
        source_id = "src-artifacts"
        now = datetime.now(timezone.utc)
        await db.upsert_source(
            source_id, "confluence", "Artifact Source", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.upsert_document(
            DocumentRecord(
                doc_id="doc-artifacts",
                source=source_id,
                source_url="https://wiki.example.test/doc-artifacts",
                title="Architecture",
                space_or_project="SFPAY",
                author=None,
                last_modified=now,
                labels=[],
                version="1",
                content_hash="artifact-hash",
                token_count=100,
                raw_content_uri="object-store://workspace/documents/src-artifacts/raw.html",
                raw_content_type="text/html",
                normalized_content_uri="object-store://workspace/documents/src-artifacts/page.md",
                pdf_content_uri="object-store://workspace/documents/src-artifacts/page.pdf",
                last_synced=now,
            )
        )
        await db.create_source_sync_input(
            source_id=source_id,
            raw_uri="object-store://workspace/documents/src-artifacts/package.json",
            raw_sha256="package-input-hash",
            raw_content_type="application/json",
        )

        await db.delete_source_cascade(source_id)

        tasks = await db.list_source_artifact_cleanup_tasks(limit=10)
        assert {(task.source_id, task.artifact_uri) for task in tasks} == {
            (source_id, "object-store://workspace/documents/src-artifacts/raw.html"),
            (source_id, "object-store://workspace/documents/src-artifacts/page.md"),
            (source_id, "object-store://workspace/documents/src-artifacts/page.pdf"),
            (source_id, "object-store://workspace/documents/src-artifacts/package.json"),
        }

    @pytest.mark.asyncio
    async def test_artifact_cleanup_removes_exact_file_and_completes_outbox_task(self, db, tmp_path):
        from memforge.storage.document_store import LocalDocumentStore
        from memforge.storage.source_cleanup import SourceArtifactCleanupService

        source_id = "src-cleanup"
        now = datetime.now(timezone.utc)
        document_store = LocalDocumentStore(str(tmp_path / "documents"))
        artifact_uri = document_store.store_normalized(source_id, "Architecture", "# Architecture")
        await db.upsert_source(
            source_id, "confluence", "Cleanup Source", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.upsert_document(
            DocumentRecord(
                doc_id="doc-cleanup",
                source=source_id,
                source_url="https://wiki.example.test/doc-cleanup",
                title="Architecture",
                space_or_project="SFPAY",
                author=None,
                last_modified=now,
                labels=[],
                version="1",
                content_hash="cleanup-hash",
                token_count=10,
                raw_content_uri=None,
                raw_content_type=None,
                normalized_content_uri=artifact_uri,
                pdf_content_uri=None,
                last_synced=now,
            )
        )
        await db.delete_source_cascade(source_id)

        processed = await SourceArtifactCleanupService(db, document_store).run_pending(limit=10)

        assert processed == 1
        assert document_store.get_artifact(artifact_uri, "text/markdown") is None
        assert await db.list_source_artifact_cleanup_tasks(limit=10) == []

    @pytest.mark.asyncio
    async def test_artifact_cleanup_completes_uri_not_owned_by_current_store(self, db, tmp_path):
        from memforge.storage.document_store import LocalDocumentStore
        from memforge.storage.source_cleanup import SourceArtifactCleanupService

        source_id = "src-legacy-artifact"
        now = datetime.now(timezone.utc)
        stale_uri = "/old-container/.memforge/documents/src-legacy-artifact/page.md"
        await db.upsert_source(
            source_id,
            "confluence",
            "Legacy Artifact Source",
            "{}",
            access_policy="workspace",
            owner_user_id="dev",
        )
        await db.upsert_document(
            DocumentRecord(
                doc_id="doc-legacy-artifact",
                source=source_id,
                source_url="https://wiki.example.test/doc-legacy-artifact",
                title="Legacy Architecture",
                space_or_project="SFPAY",
                author=None,
                last_modified=now,
                labels=[],
                version="1",
                content_hash="legacy-artifact-hash",
                token_count=10,
                raw_content_uri=None,
                raw_content_type=None,
                normalized_content_uri=stale_uri,
                pdf_content_uri=None,
                last_synced=now,
            )
        )
        await db.delete_source_cascade(source_id)

        processed = await SourceArtifactCleanupService(
            db,
            LocalDocumentStore(str(tmp_path / "current-documents")),
        ).run_pending(limit=10)

        assert processed == 1
        assert await db.list_source_artifact_cleanup_tasks(limit=10) == []

    @pytest.mark.asyncio
    async def test_document_deletion_uses_the_same_artifact_cleanup_outbox(self, db):
        source_id = "src-document-cleanup"
        now = datetime.now(timezone.utc)
        await db.upsert_source(
            source_id, "confluence", "Document Cleanup", "{}", access_policy="workspace", owner_user_id="dev"
        )
        await db.upsert_document(
            DocumentRecord(
                doc_id="doc-document-cleanup",
                source=source_id,
                source_url="https://wiki.example.test/doc-document-cleanup",
                title="Architecture",
                space_or_project="SFPAY",
                author=None,
                last_modified=now,
                labels=[],
                version="1",
                content_hash="document-cleanup-hash",
                token_count=10,
                raw_content_uri=None,
                raw_content_type=None,
                normalized_content_uri="object-store://workspace/documents/src-document-cleanup/page.md",
                pdf_content_uri=None,
                last_synced=now,
            )
        )

        await db.delete_document("doc-document-cleanup")

        tasks = await db.list_source_artifact_cleanup_tasks(limit=10)
        assert [(task.source_id, task.artifact_uri) for task in tasks] == [
            (source_id, "object-store://workspace/documents/src-document-cleanup/page.md")
        ]

    @pytest.mark.asyncio
    async def test_source_deletion_fence_rejects_new_document_writes(self, db):
        source_id = "src-fenced"
        now = datetime.now(timezone.utc)
        await db.upsert_source(
            source_id, "confluence", "Fenced Source", "{}", access_policy="workspace", owner_user_id="dev"
        )

        await db.db.execute("UPDATE sources SET status = 'deleting' WHERE id = ?", (source_id,))
        await db.db.commit()
        with pytest.raises(ValueError, match="Source is being deleted"):
            await db.upsert_document(
                DocumentRecord(
                    doc_id="doc-after-fence",
                    source=source_id,
                    source_url="https://wiki.example.test/doc-after-fence",
                    title="Architecture",
                    space_or_project="SFPAY",
                    author=None,
                    last_modified=now,
                    labels=[],
                    version="1",
                    content_hash="fenced-hash",
                    token_count=10,
                    raw_content_uri=None,
                    raw_content_type=None,
                    normalized_content_uri=None,
                    pdf_content_uri=None,
                    last_synced=now,
                )
            )


class TestSupersessionMetadata:
    @pytest.mark.asyncio
    async def test_supersede_sets_valid_until_and_supersession_metadata(self, db):
        old = _make_memory("mem-old0001", "PostgreSQL version is 14")
        new = _make_memory("mem-new0001", "PostgreSQL version is 16")
        await db.insert_memory(old)

        await db.supersede_memory(
            old.id,
            new,
            replacement_reason="same_source_replacement",
            replacement_kind="supersession",
        )

        stored_old = await db.get_memory(old.id)
        stored_new = await db.get_memory(new.id)
        assert stored_old.status == "superseded"
        assert stored_old.superseded_by == new.id
        assert stored_old.valid_until is not None
        assert stored_old.superseded_at is not None
        assert stored_old.replacement_reason == "same_source_replacement"
        assert stored_old.replacement_kind == "supersession"
        assert stored_new.status == "active"

    @pytest.mark.asyncio
    async def test_supersede_preserves_source_provenance_on_old_memory(self, db):
        old = _make_memory("mem-oldsrc1", "PostgreSQL version is 14")
        new = _make_memory("mem-newsrc1", "PostgreSQL version is 16")
        await _insert_doc(db, "doc-postgres-14", source="confluence")
        await db.insert_memory(old)
        await db.corroborate_memory(
            old.id,
            "doc-postgres-14",
            "confluence",
            "PostgreSQL version is 14",
            support_kind="extracted",
            source_updated_at=None,
        )

        await db.supersede_memory(
            old.id,
            new,
            replacement_reason="same_source_replacement",
            replacement_kind="supersession",
        )

        sources = await db.get_memory_sources(old.id)
        assert [(source.doc_id, source.source_type, source.support_kind, source.excerpt) for source in sources] == [
            ("doc-postgres-14", "confluence", "extracted", "PostgreSQL version is 14")
        ]

    @pytest.mark.asyncio
    async def test_resolve_current_memory_id_walks_supersession_chain(self, db):
        first = _make_memory("mem-chain-1", "First version")
        second = _make_memory("mem-chain-2", "Second version")
        third = _make_memory("mem-chain-3", "Third version")
        await db.insert_memory(first)

        await db.supersede_memory(first.id, second, replacement_reason="revision", replacement_kind="revision")
        await db.supersede_memory(second.id, third, replacement_reason="revision", replacement_kind="revision")

        assert await db.resolve_current_memory_id(first.id) == third.id
        assert await db.resolve_current_memory_id(second.id) == third.id
        assert await db.resolve_current_memory_id(third.id) == third.id
        assert await db.resolve_current_memory_id("missing-memory") is None

    @pytest.mark.asyncio
    async def test_resolve_current_memory_id_rejects_cycles(self, db):
        first = _make_memory("mem-cycle-1", "First version")
        second = _make_memory("mem-cycle-2", "Second version")
        await db.insert_memory(first)
        await db.insert_memory(second)
        await db.db.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ? WHERE id = ?",
            (second.id, first.id),
        )
        await db.db.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ? WHERE id = ?",
            (first.id, second.id),
        )
        await db.db.commit()

        with pytest.raises(RuntimeError, match="cycle"):
            await db.resolve_current_memory_id(first.id)

    @pytest.mark.asyncio
    async def test_supersede_rejects_invalid_replacement_kind(self, db):
        old = _make_memory("mem-old-bad-kind", "Old fact")
        new = _make_memory("mem-new-bad-kind", "New fact")
        await db.insert_memory(old)

        with pytest.raises(ValueError, match="replacement kind"):
            await db.supersede_memory(
                old.id,
                new,
                replacement_reason="typo",
                replacement_kind="revison",  # type: ignore[arg-type],
            )


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
        entity_id = await db.upsert_entity("postgresql", display_name="PostgreSQL")
        memory = _make_memory("mem-purge01", "Private credential detail")
        dependent = _make_memory("mem-dependent", "This memory used to replace the purged one")
        await db.insert_memory(memory)
        await db.insert_memory(dependent)
        await db.add_memory_source(
            memory.id, "doc-a", "confluence", excerpt="sensitive excerpt", source_updated_at=None
        )
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
            """UPDATE memories SET
                status = 'superseded',
                superseded_by = ?,
                superseded_at = '2026-07-19T00:00:00+00:00',
                replacement_reason = 'explicit replacement',
                replacement_kind = 'supersession'
               WHERE id = ?""",
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
        assert stored_dependent.superseded_at is None
        assert stored_dependent.replacement_reason is None
        assert stored_dependent.replacement_kind is None
        assert stored_dependent.retirement_reason == "privacy_removed"
