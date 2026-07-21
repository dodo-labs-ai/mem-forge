"""Tests for the memory review workbench: schema, lifecycle, and approve/reject."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.review_service import (
    ReviewAlreadyResolved,
    ReviewError,
    ReviewService,
    ReviewStaleConflict,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    DocumentRecord,
    Memory,
    MemoryReview,
    ReplacementKind,
    ReviewKind,
    ReviewStatus,
    content_hash,
    generate_review_id,
)
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubChromaCollection:
    """In-memory ChromaDB stand-in. Tracks ids -> metadata so tests can assert
    that approve/reject keep the vector index aligned with SQLite.
    """

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.documents: dict[str, str] = {}

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for i, record_id in enumerate(ids):
            metadata = metadatas[i] if metadatas else {}
            self.records[record_id] = dict(metadata)
            if embeddings:
                self.embeddings[record_id] = embeddings[i]
            if documents:
                self.documents[record_id] = documents[i]

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)
            self.embeddings.pop(record_id, None)
            self.documents.pop(record_id, None)

    def query(self, **kwargs):
        return {"ids": [list(self.records.keys())], "distances": [[0.5] * len(self.records)]}

    def get(self, *, ids=None, include=None):
        selected_ids = [record_id for record_id in (ids or list(self.records)) if record_id in self.records]
        include = include or []
        result: dict[str, Any] = {"ids": selected_ids}
        if "metadatas" in include:
            result["metadatas"] = [self.records[record_id] for record_id in selected_ids]
        if "embeddings" in include:
            result["embeddings"] = [self.embeddings.get(record_id) for record_id in selected_ids]
        if "documents" in include:
            result["documents"] = [self.documents.get(record_id) for record_id in selected_ids]
        return result


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "reviews.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
def chroma() -> StubChromaCollection:
    return StubChromaCollection()


@pytest.fixture
def memory_store(db, chroma) -> MemoryStore:
    audit_logger = MemoryAuditLogger(db, default_context=AuditContext(actor_type="test", run_id="run-review"))
    adapters = build_sqlite_adapters(db, chroma)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=audit_logger,
    )

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[assignment]
    return store


@pytest.fixture
def review_service(db, memory_store) -> ReviewService:
    return ReviewService(db=db, memory_store=memory_store)


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _insert_doc(db: Database, doc_id: str, source: str = "src-1") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, source, f"http://test/{doc_id}", doc_id, "TEST", now, "1", f"hash-{doc_id}", now),
    )
    await db.db.commit()


async def _upsert_doc_with_artifacts(
    db: Database,
    tmp_path: Path,
    doc_id: str,
    *,
    normalized_content_uri: str | None,
    pdf_content_uri: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source="src-confluence",
            source_url=f"http://test/{doc_id}",
            title=doc_id,
            space_or_project="TEST",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=100,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=normalized_content_uri,
            pdf_content_uri=pdf_content_uri,
            last_synced=now,
        )
    )


def _memory(mem_id: str, content: str, *, status: str = "active", confidence: float = 0.9) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=confidence,
        created_at=now,
        updated_at=now,
        status=status,
    )


async def _seed_supersede_review(
    db: Database,
    chroma: StubChromaCollection,
    *,
    review_reason: str = "Newer doc updates this fact",
    replacement_kind: ReplacementKind = "supersession",
    suffix: str = "1234",
) -> tuple[Memory, Memory, MemoryReview]:
    """Build the canonical SUPERSEDE review: active incumbent, pending challenger."""
    incumbent = _memory(f"mem-incu{suffix}", "PostgreSQL is version 14")
    await db.insert_memory(incumbent)
    chroma.upsert(ids=[incumbent.id], metadatas=[{"status": "active"}])

    challenger = _memory(
        f"mem-chal{suffix}",
        "PostgreSQL is version 16",
        status="pending_review",
    )
    await db.insert_memory(challenger)

    review = MemoryReview(
        id=generate_review_id(),
        kind=ReviewKind.SUPERSEDE.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason=review_reason,
        expected_incumbent_updated_at=incumbent.updated_at.isoformat(),
        expected_challenger_updated_at=challenger.updated_at.isoformat(),
        replacement_kind=replacement_kind,
        created_at=datetime.now(timezone.utc),
    )
    await db.insert_memory_review(review)

    # Re-fetch to pick up the on-disk timestamps so guards see the actual values.
    incumbent = await db.get_memory(incumbent.id)  # type: ignore[assignment]
    challenger = await db.get_memory(challenger.id)  # type: ignore[assignment]
    review = await db.get_memory_review(review.id)  # type: ignore[assignment]
    # Re-pin expectations to the freshly stored timestamps so the fixture is
    # not pre-stale before the test even starts.
    await db.refresh_memory_review_expectations(
        review.id,
        expected_incumbent_updated_at=incumbent.updated_at.isoformat(),
        expected_challenger_updated_at=challenger.updated_at.isoformat(),
    )
    review = await db.get_memory_review(review.id)  # type: ignore[assignment]
    return incumbent, challenger, review


async def _attach_related_challenger(
    db: Database,
    review: MemoryReview,
    *,
    suffix: str = "rel1",
) -> Memory:
    challenger = _memory(
        f"mem-{suffix}",
        "PostgreSQL version changes should be reviewed as one grouped case",
        status="pending_review",
    )
    await db.insert_memory(challenger)
    await db.add_memory_review_related_challenger(
        review.id,
        challenger.id,
        reason="Same source document produced another challenger",
    )
    stored = await db.get_memory(challenger.id)
    assert stored is not None
    return stored


async def _seed_cross_source_review(
    db: Database,
    chroma: StubChromaCollection,
) -> tuple[Memory, Memory, MemoryReview]:
    incumbent = _memory("mem-cross-inc", "The service uses PostgreSQL 14")
    challenger = _memory("mem-cross-new", "The service uses MySQL 8")
    await db.insert_memory(incumbent)
    await db.insert_memory(challenger)
    chroma.upsert(
        ids=[incumbent.id, challenger.id],
        metadatas=[{"status": "active"}, {"status": "active"}],
    )
    review = MemoryReview(
        id=generate_review_id(),
        kind=ReviewKind.CROSS_SOURCE_CONFLICT.value,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason="contradiction: database versions disagree",
        expected_incumbent_updated_at=(await db.get_memory(incumbent.id)).updated_at.isoformat(),
        expected_challenger_updated_at=(await db.get_memory(challenger.id)).updated_at.isoformat(),
        created_at=datetime.now(timezone.utc),
    )
    await db.insert_memory_review(review)
    return incumbent, challenger, review


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestReviewCrud:
    @pytest.mark.asyncio
    async def test_insert_and_fetch_review_round_trips_all_fields(self, db, chroma):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        loaded = await db.get_memory_review(review.id)

        assert loaded is not None
        assert loaded.kind == "supersede"
        assert loaded.status == "pending"
        assert loaded.incumbent_memory_id == incumbent.id
        assert loaded.challenger_memory_id == challenger.id
        assert loaded.reason == "Newer doc updates this fact"
        assert loaded.expected_incumbent_updated_at == incumbent.updated_at.isoformat()
        assert loaded.expected_challenger_updated_at == challenger.updated_at.isoformat()
        assert loaded.replacement_kind == "supersession"

    @pytest.mark.asyncio
    async def test_review_replacement_kind_round_trips_through_db_and_api(self, db, chroma, tmp_path):
        _, _, review = await _seed_supersede_review(
            db,
            chroma,
            replacement_kind="revision",
            suffix="revk",
        )

        loaded = await db.get_memory_review(review.id)
        listed = await db.list_memory_reviews(status="pending")

        assert loaded is not None
        assert loaded.replacement_kind == "revision"
        assert {item.id: item.replacement_kind for item in listed}[review.id] == "revision"

        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            list_response = client.get("/api/memory-reviews", params={"status": "open"})
            detail_response = client.get(f"/api/memory-reviews/{review.id}")

        assert list_response.status_code == 200
        row = next(item for item in list_response.json()["data"] if item["id"] == review.id)
        assert row["replacement_kind"] == "revision"
        assert detail_response.status_code == 200
        assert detail_response.json()["replacement_kind"] == "revision"

    @pytest.mark.asyncio
    async def test_list_pending_reviews_filters_by_status(self, db, chroma):
        _, _, review = await _seed_supersede_review(db, chroma)
        await db.resolve_memory_review(
            review.id,
            status=ReviewStatus.APPROVED.value,
            reviewer="me",
            review_note=None,
        )

        pending = await db.list_memory_reviews(status="pending")
        approved = await db.list_memory_reviews(status="approved")

        assert pending == []
        assert [r.id for r in approved] == [review.id]

    @pytest.mark.asyncio
    async def test_open_reviews_include_pending_and_stale_but_not_resolved(self, db, chroma):
        _, _, pending_review = await _seed_supersede_review(db, chroma, suffix="pend")
        _, _, stale_review = await _seed_supersede_review(db, chroma, suffix="stale")
        _, _, approved_review = await _seed_supersede_review(db, chroma, suffix="appr")
        await db.resolve_memory_review(
            stale_review.id,
            status=ReviewStatus.STALE.value,
            reviewer=None,
            review_note=None,
        )
        await db.resolve_memory_review(
            approved_review.id,
            status=ReviewStatus.APPROVED.value,
            reviewer="me",
            review_note=None,
        )

        open_reviews = await db.list_memory_reviews(status="open")
        open_count = await db.count_memory_reviews(status="open")

        assert {r.id for r in open_reviews} == {pending_review.id, stale_review.id}
        assert open_count == 2

    @pytest.mark.asyncio
    async def test_review_list_includes_pending_challenger_snapshot(self, db, chroma, tmp_path):
        from memforge.server.admin_api import create_admin_app

        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="queue")

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            response = client.get("/api/memory-reviews", params={"status": "open", "limit": 10})

        assert response.status_code == 200
        rows = response.json()["data"]
        row = next(item for item in rows if item["id"] == review.id)
        assert row["incumbent"]["id"] == incumbent.id
        assert row["incumbent"]["status"] == "active"
        assert row["incumbent"]["content"] == incumbent.content
        assert row["challenger"]["id"] == challenger.id
        assert row["challenger"]["status"] == "pending_review"
        assert row["challenger"]["content"] == challenger.content

    @pytest.mark.asyncio
    async def test_related_challenger_conflict_is_explicit(self, db, chroma):
        _, _, first_review = await _seed_supersede_review(db, chroma, suffix="one")
        _, _, second_review = await _seed_supersede_review(db, chroma, suffix="two")
        related = await _attach_related_challenger(db, first_review)

        await db.add_memory_review_related_challenger(
            first_review.id,
            related.id,
            reason="Repeated insert for the same visible case is idempotent",
        )
        assert len(await db.list_memory_review_related_challengers(first_review.id)) == 1

        with pytest.raises(ValueError, match="already attached"):
            await db.add_memory_review_related_challenger(
                second_review.id,
                related.id,
                reason="A challenger cannot move silently to another review",
            )

    @pytest.mark.asyncio
    async def test_purge_memory_removes_related_challenger_references(self, db, chroma):
        _, _, review = await _seed_supersede_review(db, chroma)
        related = await _attach_related_challenger(db, review)

        purged = await db.purge_memory(related.id)

        assert purged is True
        assert await db.get_memory(related.id) is None
        assert await db.list_memory_review_related_challengers(review.id) == []

    @pytest.mark.asyncio
    async def test_review_detail_uses_service_readable_artifact_urls_only(
        self,
        db,
        chroma,
        tmp_path,
    ):
        from memforge.server.admin_api import create_admin_app

        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="urls")
        docs_dir = Path(_config(tmp_path).storage.docs_path)
        docs_dir.mkdir(parents=True)
        incumbent_md = docs_dir / "incumbent.md"
        incumbent_md.write_text("# Incumbent evidence", encoding="utf-8")
        await _upsert_doc_with_artifacts(
            db,
            tmp_path,
            "doc-review-incumbent",
            normalized_content_uri=str(incumbent_md),
        )
        await _upsert_doc_with_artifacts(
            db,
            tmp_path,
            "doc-review-challenger",
            normalized_content_uri="/tmp/missing-review-source.md",
        )
        await db.add_memory_source(
            incumbent.id,
            "doc-review-incumbent",
            "confluence",
            excerpt="incumbent source",
            source_updated_at=None,
        )
        await db.add_memory_source(
            challenger.id,
            "doc-review-challenger",
            "confluence",
            excerpt="challenger source",
            source_updated_at=None,
        )

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            response = client.get(f"/api/memory-reviews/{review.id}")

        assert response.status_code == 200
        payload = response.json()
        incumbent_source = payload["incumbent"]["sources"][0]
        challenger_source = payload["challenger"]["sources"][0]
        assert incumbent_source["content_url"] == "/api/documents/doc-review-incumbent/content"
        assert challenger_source["content_url"] is None
        assert "file_uri" not in incumbent_source
        assert "pdf_uri" not in incumbent_source

    @pytest.mark.asyncio
    async def test_review_detail_uses_injected_document_store_for_artifact_urls(
        self,
        db,
        chroma,
        tmp_path,
    ):
        from memforge.server.admin_api import create_admin_app
        from memforge.storage.document_store import StoredDocumentArtifact

        class MemoryBackedDocumentStore:
            def __init__(self) -> None:
                self.objects = {"mem://review-incumbent.md": b"# Incumbent object evidence"}

            def get_artifact(self, uri: str | None, media_type: str):
                if uri not in self.objects:
                    return None
                return StoredDocumentArtifact(
                    uri=uri,
                    filename="review-incumbent.md",
                    media_type=media_type,
                    size_bytes=len(self.objects[uri]),
                )

            def read_artifact(self, uri: str) -> bytes:
                return self.objects[uri]

            def read_normalized(self, stored_path: str) -> str | None:
                content = self.objects.get(stored_path)
                return content.decode("utf-8") if content else None

            def store_raw(self, *args, **kwargs) -> str:
                raise AssertionError("not used")

            def store_normalized(self, *args, **kwargs) -> str:
                raise AssertionError("not used")

            def store_pdf(self, *args, **kwargs) -> str:
                raise AssertionError("not used")

        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="objecturls")
        await _upsert_doc_with_artifacts(
            db,
            tmp_path,
            "doc-review-object-incumbent",
            normalized_content_uri="mem://review-incumbent.md",
        )
        await db.add_memory_source(
            incumbent.id,
            "doc-review-object-incumbent",
            "jira",
            excerpt="incumbent source",
            source_updated_at=None,
        )

        app = create_admin_app(
            db=db,
            config=_config(tmp_path),
            document_store=MemoryBackedDocumentStore(),
        )
        with TestClient(app) as client:
            detail = client.get(f"/api/memory-reviews/{review.id}")
            content = client.get("/api/documents/doc-review-object-incumbent/content")

        assert detail.status_code == 200
        incumbent_source = detail.json()["incumbent"]["sources"][0]
        assert incumbent_source["content_url"] == ("/api/documents/doc-review-object-incumbent/content")
        assert content.status_code == 200
        assert content.text == "# Incumbent object evidence"


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_promotes_challenger_and_supersedes_incumbent(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        result = await review_service.approve(review.id, reviewer="alice", note=None)

        stored_review = result.review
        stored_incumbent = await db.get_memory(incumbent.id)
        stored_challenger = await db.get_memory(challenger.id)

        assert stored_review.status == "approved"
        assert stored_review.reviewer == "alice"
        assert stored_review.resolved_at is not None
        assert stored_incumbent.status == "superseded"
        assert stored_incumbent.superseded_by == challenger.id
        assert stored_incumbent.replacement_reason == review.reason
        assert stored_challenger.status == "active"

    @pytest.mark.asyncio
    async def test_approve_uses_review_replacement_kind(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(
            db,
            chroma,
            replacement_kind="revision",
            suffix="apprrev",
        )

        await review_service.approve(review.id, reviewer="alice", note=None)

        stored_incumbent = await db.get_memory(incumbent.id)
        stored_challenger = await db.get_memory(challenger.id)

        assert stored_incumbent.status == "superseded"
        assert stored_incumbent.superseded_by == challenger.id
        assert stored_incumbent.replacement_kind == "revision"
        assert stored_challenger.status == "active"

    @pytest.mark.asyncio
    async def test_approve_keeps_search_indexes_aligned(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        await review_service.approve(review.id, reviewer="alice", note=None)

        async with db.db.execute("SELECT memory_id FROM memories_fts ORDER BY memory_id") as cursor:
            fts_ids = [row[0] async for row in cursor]
        assert challenger.id in fts_ids
        assert incumbent.id not in fts_ids
        assert challenger.id in chroma.records
        assert chroma.records[challenger.id]["status"] == "active"
        assert incumbent.id not in chroma.records

    @pytest.mark.asyncio
    async def test_approve_records_review_and_supersede_audit(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        await review_service.approve(review.id, reviewer="alice", note=None)

        challenger_rows = await db.list_memory_audit_events(memory_id=challenger.id)
        audit_rows = await db.list_memory_audit_events(operation_id=challenger_rows[0].operation_id)
        event_types = {row.event_type for row in audit_rows}
        assert {"review_approved", "memory_supersede_committed"}.issubset(event_types)
        assert {row.review_id for row in audit_rows if row.event_type == "review_approved"} == {review.id}
        assert {row.actor_id for row in audit_rows if row.event_type == "review_approved"} == {"alice"}
        assert {
            (row.memory_id, row.candidate_id) for row in audit_rows if row.event_type == "memory_supersede_committed"
        } == {(incumbent.id, challenger.id)}
        assert len({row.operation_id for row in audit_rows}) == 1

    @pytest.mark.asyncio
    async def test_approve_does_not_emit_review_approved_when_review_resolution_fails(
        self, db, chroma, review_service, monkeypatch
    ):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        original_resolve = db.resolve_memory_review

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "resolve_memory_review", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await review_service.approve(review.id, reviewer="alice", note=None)

        monkeypatch.setattr(db, "resolve_memory_review", original_resolve)
        challenger_rows = await db.list_memory_audit_events(memory_id=challenger.id)
        audit_rows = await db.list_memory_audit_events(operation_id=challenger_rows[0].operation_id)
        stored_review = await db.get_memory_review(review.id)
        stored_incumbent = await db.get_memory(incumbent.id)
        stored_challenger = await db.get_memory(challenger.id)
        assert stored_review.status == "pending"
        assert stored_incumbent.status == "active"
        assert stored_challenger.status == "pending_review"
        assert incumbent.id in chroma.records
        assert challenger.id not in chroma.records
        assert "review_approved" not in {row.event_type for row in audit_rows}

    @pytest.mark.asyncio
    async def test_approve_audits_review_resolution_failure(self, db, chroma, review_service, monkeypatch):
        _, challenger, review = await _seed_supersede_review(db, chroma, suffix="fail")

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "resolve_memory_review", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await review_service.approve(review.id, reviewer="alice", note=None)

        audit_rows = await db.list_memory_audit_events(memory_id=challenger.id)
        failure_rows = [row for row in audit_rows if row.event_type == "review_resolution_failed"]
        assert len(failure_rows) == 1
        assert failure_rows[0].review_id == review.id
        assert failure_rows[0].actor_id == "alice"
        assert failure_rows[0].error == "resolution failed"

    @pytest.mark.asyncio
    async def test_approve_preserves_linked_entity_text_in_fts(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        entity_id = await db.upsert_entity(
            "postgresql",
            display_name="PostgreSQL",
            tags=["technology"],
        )
        await db.link_memory_entity(challenger.id, entity_id)

        await review_service.approve(review.id, reviewer="alice", note=None)

        async with db.db.execute(
            "SELECT entities_text FROM memories_fts WHERE memory_id = ?",
            (challenger.id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert "postgresql" in row[0].lower()

    @pytest.mark.asyncio
    async def test_approve_retires_related_challengers_as_redundant(self, db, chroma, review_service):
        _, _, review = await _seed_supersede_review(db, chroma)
        related = await _attach_related_challenger(db, review)
        chroma.upsert(ids=[related.id], metadatas=[{"status": "pending_review"}])

        await review_service.approve(review.id, reviewer="alice", note=None)

        stored_related = await db.get_memory(related.id)
        assert stored_related.status == "retired"
        assert stored_related.retirement_reason == "review_redundant"
        assert related.id not in chroma.records

    @pytest.mark.asyncio
    async def test_repeated_approve_returns_clear_409_without_partial_mutation(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        await review_service.approve(review.id, reviewer="alice", note=None)

        snapshot_incumbent = await db.get_memory(incumbent.id)
        snapshot_challenger = await db.get_memory(challenger.id)

        with pytest.raises(ReviewAlreadyResolved):
            await review_service.approve(review.id, reviewer="bob", note=None)

        assert (await db.get_memory(incumbent.id)).updated_at == snapshot_incumbent.updated_at
        assert (await db.get_memory(challenger.id)).updated_at == snapshot_challenger.updated_at

    @pytest.mark.asyncio
    async def test_stale_incumbent_blocks_approval_and_marks_review_stale(self, db, chroma, review_service):
        incumbent, _, review = await _seed_supersede_review(db, chroma)

        await db.update_memory_content(
            incumbent.id,
            new_content="PostgreSQL is now version 15",
            new_confidence=None,
            new_tags=None,
        )

        with pytest.raises(ReviewStaleConflict):
            await review_service.approve(review.id, reviewer="alice", note=None)

        stored = await db.get_memory_review(review.id)
        assert stored.status == "stale"


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_retires_challenger_with_reason_and_removes_from_indexes(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        chroma.upsert(ids=[challenger.id], metadatas=[{"status": "pending_review"}])

        result = await review_service.reject(
            review.id,
            reviewer="alice",
            note="Source is unreliable",
        )

        stored_review = result.review
        stored_incumbent = await db.get_memory(incumbent.id)
        stored_challenger = await db.get_memory(challenger.id)

        assert stored_review.status == "rejected"
        assert stored_review.review_note == "Source is unreliable"
        assert stored_incumbent.status == "active"
        assert stored_challenger.status == "retired"
        assert stored_challenger.retirement_reason == "rejected"

        async with db.db.execute(
            "SELECT memory_id FROM memories_fts WHERE memory_id = ?",
            (challenger.id,),
        ) as cursor:
            assert (await cursor.fetchone()) is None
        assert challenger.id not in chroma.records

    @pytest.mark.asyncio
    async def test_reject_requires_a_note(self, db, chroma, review_service):
        _, _, review = await _seed_supersede_review(db, chroma)

        with pytest.raises(ReviewError):
            await review_service.reject(review.id, reviewer="alice", note="   ")

        stored = await db.get_memory_review(review.id)
        assert stored.status == "pending"

    @pytest.mark.asyncio
    async def test_reject_records_review_audit_with_reviewer(self, db, chroma, review_service):
        _, challenger, review = await _seed_supersede_review(db, chroma)

        await review_service.reject(review.id, reviewer="alice", note="bad source")

        audit_rows = await db.list_memory_audit_events(memory_id=challenger.id)
        review_rows = [row for row in audit_rows if row.event_type == "review_rejected"]
        assert len(review_rows) == 1
        assert review_rows[0].review_id == review.id
        assert review_rows[0].actor_id == "alice"

    @pytest.mark.asyncio
    async def test_reject_retires_related_challengers(self, db, chroma, review_service):
        _, _, review = await _seed_supersede_review(db, chroma)
        related = await _attach_related_challenger(db, review)
        chroma.upsert(ids=[related.id], metadatas=[{"status": "pending_review"}])

        await review_service.reject(review.id, reviewer="alice", note="bad source")

        stored_related = await db.get_memory(related.id)
        assert stored_related.status == "retired"
        assert stored_related.retirement_reason == "rejected"
        assert related.id not in chroma.records

    @pytest.mark.asyncio
    async def test_reject_rolls_back_when_review_resolution_fails(self, db, chroma, review_service, monkeypatch):
        _, challenger, review = await _seed_supersede_review(db, chroma)
        original_resolve = db.resolve_memory_review

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "resolve_memory_review", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await review_service.reject(review.id, reviewer="alice", note="bad source")

        monkeypatch.setattr(db, "resolve_memory_review", original_resolve)
        stored_review = await db.get_memory_review(review.id)
        stored_challenger = await db.get_memory(challenger.id)
        audit_rows = await db.list_memory_audit_events(memory_id=challenger.id)
        assert stored_review.status == "pending"
        assert stored_challenger.status == "pending_review"
        assert "review_rejected" not in {row.event_type for row in audit_rows}

    @pytest.mark.asyncio
    async def test_reject_rolls_back_related_challengers_when_review_resolution_fails(
        self, db, chroma, review_service, monkeypatch
    ):
        _, challenger, review = await _seed_supersede_review(db, chroma)
        related = await _attach_related_challenger(db, review)
        chroma.upsert(
            ids=[challenger.id, related.id],
            metadatas=[
                {"status": "pending_review"},
                {"status": "pending_review"},
            ],
        )

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "resolve_memory_review", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await review_service.reject(review.id, reviewer="alice", note="bad source")

        stored_challenger = await db.get_memory(challenger.id)
        stored_related = await db.get_memory(related.id)
        assert stored_challenger.status == "pending_review"
        assert stored_related.status == "pending_review"
        assert challenger.id not in chroma.records
        assert related.id not in chroma.records

    @pytest.mark.asyncio
    async def test_repeated_reject_returns_clear_409(self, db, chroma, review_service):
        _, _, review = await _seed_supersede_review(db, chroma)

        await review_service.reject(review.id, reviewer="alice", note="bad source")

        with pytest.raises(ReviewAlreadyResolved):
            await review_service.reject(review.id, reviewer="bob", note="again")


# ---------------------------------------------------------------------------
# Non-destructive cross-source finding resolution
# ---------------------------------------------------------------------------


class TestCrossSourceReviewResolution:
    @pytest.mark.asyncio
    async def test_approve_acknowledges_finding_without_mutating_memories(
        self, db, chroma, review_service
    ):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)

        result = await review_service.approve(
            review.id,
            reviewer="alice",
            note="confirmed conflict; no authority decision yet",
        )

        assert result.review.status == "approved"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"
        assert set(chroma.records) == {incumbent.id, challenger.id}

    @pytest.mark.asyncio
    async def test_reject_dismisses_finding_without_mutating_memories(
        self, db, chroma, review_service
    ):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)

        result = await review_service.reject(
            review.id,
            reviewer="alice",
            note="claims apply to different deployments",
        )

        assert result.review.status == "rejected"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"
        assert set(chroma.records) == {incumbent.id, challenger.id}


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_repins_expectations_after_drift(self, db, chroma, review_service):
        incumbent, _, review = await _seed_supersede_review(db, chroma)

        await db.update_memory_content(
            incumbent.id,
            new_content="PostgreSQL is now version 15",
            new_confidence=None,
            new_tags=None,
        )

        result = await review_service.refresh(review.id)

        refreshed_incumbent = await db.get_memory(incumbent.id)
        assert result.review.status == "pending"
        assert result.review.expected_incumbent_updated_at == refreshed_incumbent.updated_at.isoformat()

    @pytest.mark.asyncio
    async def test_refresh_clears_stale_attempt_metadata(self, db, chroma, review_service):
        incumbent, _, review = await _seed_supersede_review(db, chroma)

        await db.update_memory_content(
            incumbent.id,
            new_content="PostgreSQL is now version 15",
            new_confidence=None,
            new_tags=None,
        )
        with pytest.raises(ReviewStaleConflict):
            await review_service.approve(review.id, reviewer="alice", note=None)

        stale = await db.get_memory_review(review.id)
        assert stale.status == "stale"

        result = await review_service.refresh(review.id)

        assert result.review.status == "pending"
        assert result.review.reviewer is None
        assert result.review.review_note is None
        assert result.review.resolved_at is None
