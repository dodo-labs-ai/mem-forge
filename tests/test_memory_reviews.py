"""Tests for the memory review workbench: schema, lifecycle, and approve/reject."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.lifecycle_plan import LifecycleReviewStatus
from memforge.memory.lifecycle_planner import lifecycle_memory_version
from memforge.memory.review_decision import memory_review_decision_fingerprint
from memforge.memory.review_service import (
    ResolvedReview,
    ReviewAlreadyResolved,
    ReviewError,
    ReviewService,
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


async def _approve(review_service: ReviewService, review_id: str, **kwargs):
    review = await review_service.db.get_memory_review(review_id)
    assert review is not None
    related = await review_service._load_related_challengers(review)
    return await review_service.approve(
        review_id,
        expected_fingerprint=memory_review_decision_fingerprint(review, related),
        **kwargs,
    )


async def _reject(review_service: ReviewService, review_id: str, **kwargs):
    review = await review_service.db.get_memory_review(review_id)
    assert review is not None
    related = await review_service._load_related_challengers(review)
    return await review_service.reject(
        review_id,
        expected_fingerprint=memory_review_decision_fingerprint(review, related),
        **kwargs,
    )


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
    await db.upsert_source(
        id="src-confluence",
        type="confluence",
        name="Review evidence",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="dev",
    )
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

    # Re-fetch first so the optimistic guards use the exact persisted values.
    incumbent = await db.get_memory(incumbent.id)  # type: ignore[assignment]
    challenger = await db.get_memory(challenger.id)  # type: ignore[assignment]
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
    review = await db.get_memory_review(review.id)  # type: ignore[assignment]
    return incumbent, challenger, review


async def _seed_lifecycle_review(db: Database, *, review_id: str = "review-lifecycle") -> str:
    incumbent = _memory("mem-lifecycle-current", "The service stays with Team Vita.")
    await db.insert_memory(incumbent)
    await db.upsert_source(
        id="src-lifecycle",
        type="jira",
        name="Mount Tai Backlog",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    now = datetime.now(timezone.utc).isoformat()
    payload = {
        "scope": {
            "id": "scope-lifecycle",
            "source_id": "src-lifecycle",
            "source_unit_id": "unit-jira-1",
            "base_unit_revision_id": "unitrev-1",
            "target_unit_revision_id": "unitrev-2",
        },
        "stale_guard": {
            "observation_revision_ids": [],
            "support_set_hashes": {incumbent.id: "support-hash"},
            "memory_versions": {incumbent.id: incumbent.updated_at.isoformat()},
        },
    }
    await db.db.execute(
        """INSERT INTO lifecycle_plans (
               id, reconciliation_scope_id, source_id, source_unit_id,
               target_unit_revision_id, status, payload_json, payload_hash, created_at
           ) VALUES (?, ?, ?, ?, ?, 'applied', ?, ?, ?)""",
        (
            "plan-lifecycle-review",
            "scope-lifecycle",
            "src-lifecycle",
            "unit-jira-1",
            "unitrev-2",
            json.dumps(payload),
            "payload-hash",
            now,
        ),
    )
    staged = {
        "proposed_disposition": "supersede",
        "replacement_memory_id": "mem-lifecycle-proposed",
        "candidate": {
            "content": "The service moves to Team Pfizer.",
            "memory_type": "fact",
            "confidence": 0.91,
        },
        "proposed_mutations": [],
    }
    await db.db.execute(
        """INSERT INTO lifecycle_reviews (
               id, lifecycle_plan_id, incumbent_memory_id, status,
               staged_evidence_json, reason, created_at
           ) VALUES (?, 'plan-lifecycle-review', ?, 'pending', ?, ?, ?)""",
        (
            review_id,
            incumbent.id,
            json.dumps(staged),
            "candidate_supersede_vs_audit_keep",
            now,
        ),
    )
    await db.db.commit()
    return review_id


async def _seed_refreshable_stale_lifecycle_review(db: Database) -> str:
    review_id = await _seed_lifecycle_review(db, review_id="review-lifecycle-stale")
    incumbent = await db.get_memory("mem-lifecycle-current")
    assert incumbent is not None
    support_hash = await db.get_memory_support_set_hash(incumbent.id)
    payload = await db.get_lifecycle_plan_payload("plan-lifecycle-review")
    assert payload is not None
    payload["stale_guard"] = {
        "observation_revision_ids": [],
        "support_set_hashes": {incumbent.id: support_hash},
        "memory_versions": {incumbent.id: lifecycle_memory_version(incumbent)},
    }
    staged = {
        "proposed_disposition": "keep",
        "replacement_memory_id": None,
        "candidate": {
            "content": "The service moves to Team Pfizer.",
            "memory_type": "fact",
            "confidence": 0.91,
        },
        "proposed_mutations": [
            {
                "mutation_type": "refresh_memory_index",
                "memory_id": incumbent.id,
                "source_id": "src-lifecycle",
                "evidence_reference_ids": [],
                "replacement_memory_id": None,
                "payload": {},
            }
        ],
    }
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO source_units (
               id, source_id, unit_type, provider_key, locator_json,
               current_revision_id, updated_at
           ) VALUES ('unit-jira-1', 'src-lifecycle', 'issue', 'PAY-1', '{}', 'unitrev-2', ?)""",
        (now,),
    )
    await db.db.execute(
        "UPDATE lifecycle_plans SET payload_json = ? WHERE id = 'plan-lifecycle-review'",
        (json.dumps(payload),),
    )
    await db.db.execute(
        """UPDATE lifecycle_reviews
              SET status = 'stale', staged_evidence_json = ?, resolved_at = ?
            WHERE id = ?""",
        (json.dumps(staged), now, review_id),
    )
    await db.db.commit()
    return review_id


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
            list_response = client.get("/api/v1/memory-reviews", params={"status": "open"})
            detail_response = client.get(f"/api/v1/memory-reviews/{review.id}")

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
    async def test_open_reviews_include_only_actionable_pending_reviews(self, db, chroma):
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

        assert [r.id for r in open_reviews] == [pending_review.id]
        assert open_count == 1

    @pytest.mark.asyncio
    async def test_review_list_includes_pending_challenger_snapshot(self, db, chroma, tmp_path):
        from memforge.server.admin_api import create_admin_app

        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="queue")

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            response = client.get("/api/v1/memory-reviews", params={"status": "open", "limit": 10})

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
            response = client.get(f"/api/v1/memory-reviews/{review.id}")

        assert response.status_code == 200
        payload = response.json()
        incumbent_document = payload["incumbent"]["evidence"][0]["document"]
        challenger_document = payload["challenger"]["evidence"][0]["document"]
        assert incumbent_document["content_url"] == "/api/v1/documents/doc-review-incumbent/content"
        assert challenger_document["content_url"] is None
        assert "file_uri" not in incumbent_document
        assert "pdf_uri" not in incumbent_document

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
            detail = client.get(f"/api/v1/memory-reviews/{review.id}")
            content = client.get("/api/v1/documents/doc-review-object-incumbent/content")

        assert detail.status_code == 200
        incumbent_document = detail.json()["incumbent"]["evidence"][0]["document"]
        assert incumbent_document["content_url"] == (
            "/api/v1/documents/doc-review-object-incumbent/content"
        )
        assert content.status_code == 200
        assert content.text == "# Incumbent object evidence"


# ---------------------------------------------------------------------------
# Approve
# ---------------------------------------------------------------------------


class TestUnifiedLifecycleReviewApi:
    @pytest.mark.asyncio
    async def test_queue_and_detail_present_lifecycle_review_as_two_user_decisions(self, db, tmp_path):
        review_id = await _seed_lifecycle_review(db)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            queue = client.get("/api/v1/memory-reviews", params={"status": "open"})
            detail = client.get(f"/api/v1/memory-reviews/{review_id}")

        assert queue.status_code == 200
        row = next(item for item in queue.json()["data"] if item["id"] == review_id)
        assert row["review_origin"] == "lifecycle"
        assert row["source_name"] == "Mount Tai Backlog"
        assert row["presentation"]["decision_label"] == "Updated"
        assert row["presentation"]["summary"] == ("Use the proposed source state or keep the current memory?")
        assert [action["key"] for action in row["presentation"]["actions"]] == [
            "use_latest_state",
            "keep_current_state",
        ]
        assert detail.status_code == 200
        assert detail.json()["incumbent"]["content"] == "The service stays with Team Vita."
        assert detail.json()["challenger"]["content"] == "The service moves to Team Pfizer."

    @pytest.mark.asyncio
    async def test_open_queue_excludes_resolved_lifecycle_reviews(self, db, tmp_path):
        review_id = await _seed_lifecycle_review(db)
        await db.resolve_lifecycle_review(
            review_id,
            LifecycleReviewStatus.REJECTED,
            reviewer="owner-1",
            review_note="Keep the current state.",
        )
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            queue = client.get("/api/v1/memory-reviews", params={"status": "open"})

        assert queue.status_code == 200
        assert queue.json()["total"] == 0
        assert queue.json()["data"] == []

    @pytest.mark.asyncio
    async def test_stale_lifecycle_review_refresh_is_guarded_and_idempotent(self, db, tmp_path):
        review_id = await _seed_refreshable_stale_lifecycle_review(db)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            stale_detail = client.get(f"/api/v1/memory-reviews/{review_id}").json()
            rejected = client.post(
                f"/api/v1/memory-reviews/{review_id}/refresh",
                json={"expected_fingerprint": "review-decision-v1:outdated"},
            )
            refreshed = client.post(
                f"/api/v1/memory-reviews/{review_id}/refresh",
                json={"expected_fingerprint": stale_detail["decision_fingerprint"]},
            )
            replayed = client.post(
                f"/api/v1/memory-reviews/{review_id}/refresh",
                json={"expected_fingerprint": stale_detail["decision_fingerprint"]},
            )

        assert rejected.status_code == 409
        assert refreshed.status_code == 200
        assert replayed.status_code == 200
        assert refreshed.json()["id"] == replayed.json()["id"]
        assert refreshed.json()["status"] == "pending"
        assert refreshed.json()["review_origin"] == "lifecycle"
        old_review = await db.get_lifecycle_review(review_id)
        assert old_review is not None and old_review.status is LifecycleReviewStatus.STALE
        new_review = await db.get_lifecycle_review(refreshed.json()["id"])
        assert new_review is not None
        assert new_review.staged_evidence["refreshed_from_review_id"] == review_id

    @pytest.mark.asyncio
    async def test_queue_pages_one_stable_order_across_memory_and_lifecycle_reviews(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, newest = await _seed_supersede_review(db, chroma, suffix="page-new")
        _, _, older = await _seed_supersede_review(db, chroma, suffix="page-old")
        lifecycle_new = await _seed_lifecycle_review(db, review_id="review-lifecycle-page-new")
        await db.db.execute(
            """INSERT INTO lifecycle_reviews (
                   id, lifecycle_plan_id, incumbent_memory_id, status,
                   staged_evidence_json, reason, created_at
               )
               SELECT ?, lifecycle_plan_id, incumbent_memory_id, status,
                      staged_evidence_json, reason, ?
               FROM lifecycle_reviews WHERE id = ?""",
            (
                "review-lifecycle-page-old",
                "2026-01-01T00:00:00+00:00",
                lifecycle_new,
            ),
        )
        await db.db.execute(
            "UPDATE lifecycle_reviews SET created_at = ? WHERE id = ?",
            ("2026-01-03T00:00:00+00:00", lifecycle_new),
        )
        await db.db.execute(
            "UPDATE memory_reviews SET created_at = ? WHERE id = ?",
            ("2026-01-04T00:00:00+00:00", newest.id),
        )
        await db.db.execute(
            "UPDATE memory_reviews SET created_at = ? WHERE id = ?",
            ("2026-01-02T00:00:00+00:00", older.id),
        )
        await db.db.commit()

        lifecycle_page = await db.list_lifecycle_reviews(
            status=LifecycleReviewStatus.PENDING,
            limit=1,
            offset=1,
            newest_first=True,
        )
        assert [review.id for review in lifecycle_page] == ["review-lifecycle-page-old"]
        assert lifecycle_page[0].source_id == "src-lifecycle"
        assert await db.count_lifecycle_reviews(status=LifecycleReviewStatus.PENDING) == 2

        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            first_page = client.get(
                "/api/v1/memory-reviews",
                params={"status": "open", "limit": 2, "offset": 0},
            )
            second_page = client.get(
                "/api/v1/memory-reviews",
                params={"status": "open", "limit": 2, "offset": 2},
            )

        assert first_page.status_code == 200
        assert second_page.status_code == 200
        assert first_page.json()["total"] == 4
        assert second_page.json()["total"] == 4
        assert [row["id"] for row in first_page.json()["data"]] == [
            newest.id,
            lifecycle_new,
        ]
        assert [row["id"] for row in second_page.json()["data"]] == [
            older.id,
            "review-lifecycle-page-old",
        ]

    @pytest.mark.asyncio
    async def test_keep_current_state_requires_and_records_a_note(self, db, tmp_path):
        review_id = await _seed_lifecycle_review(db)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            detail = client.get(f"/api/v1/memory-reviews/{review_id}").json()
            decision = {"expected_fingerprint": detail["decision_fingerprint"]}
            missing = client.post(f"/api/v1/memory-reviews/{review_id}/reject", json=decision)
            kept = client.post(
                f"/api/v1/memory-reviews/{review_id}/reject",
                json={
                    **decision,
                    "note": "The current ownership remains valid.",
                },
            )

        assert missing.status_code == 400
        assert kept.status_code == 200
        assert kept.json()["status"] == "rejected"
        stored = await db.get_lifecycle_review(review_id)
        assert stored is not None
        assert stored.review_note == "The current ownership remains valid."
        assert stored.reviewer == "dev"
        open_rows = await db.list_lifecycle_reviews(
            "src-lifecycle",
            status=LifecycleReviewStatus.PENDING,
        )
        assert open_rows == []

    @pytest.mark.asyncio
    async def test_open_total_excludes_dynamically_stale_rows_before_paging(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, current = await _seed_supersede_review(db, chroma, suffix="current-page")
        stale_incumbent, _, stale = await _seed_supersede_review(db, chroma, suffix="stale-page")
        await db.db.execute(
            "UPDATE memories SET updated_at = ? WHERE id = ?",
            ("2030-01-01T00:00:00+00:00", stale_incumbent.id),
        )
        await db.db.commit()
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            first = client.get(
                "/api/v1/memory-reviews",
                params={"status": "open", "limit": 1, "offset": 0},
            )
            beyond = client.get(
                "/api/v1/memory-reviews",
                params={"status": "open", "limit": 1, "offset": 1},
            )
            stale_page = client.get(
                "/api/v1/memory-reviews",
                params={"status": "stale", "limit": 1, "offset": 0},
            )

        assert first.json()["total"] == 1
        assert [item["id"] for item in first.json()["data"]] == [current.id]
        assert beyond.json()["data"] == []
        assert [item["id"] for item in stale_page.json()["data"]] == [stale.id]
        stored_stale = await db.get_memory_review(stale.id)
        assert stored_stale.status == "pending"
        assert stored_stale.resolved_at is None

    @pytest.mark.asyncio
    async def test_review_reads_hide_another_principals_private_participants(
        self,
        db,
        chroma,
        tmp_path,
    ):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)
        for memory in (incumbent, challenger):
            await db.db.execute(
                "UPDATE memories SET visibility = 'private', owner_user_id = ? WHERE id = ?",
                ("alice", memory.id),
            )
        await db.db.commit()

        from memforge.server.admin_api import create_admin_app

        bob_app = create_admin_app(
            db=db,
            config=_config(tmp_path),
            principal_resolver=lambda _request: "bob",
        )
        with TestClient(bob_app) as client:
            queue = client.get("/api/v1/memory-reviews", params={"status": "open"})
            detail = client.get(f"/api/v1/memory-reviews/{review.id}")

        assert queue.json()["total"] == 0
        assert detail.status_code == 404

        alice_app = create_admin_app(
            db=db,
            config=_config(tmp_path),
            principal_resolver=lambda _request: "alice",
        )
        with TestClient(alice_app) as client:
            assert client.get(f"/api/v1/memory-reviews/{review.id}").status_code == 200

    @pytest.mark.asyncio
    async def test_review_reads_and_decisions_include_related_participant_visibility(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, review = await _seed_supersede_review(db, chroma)
        related = await _attach_related_challenger(db, review)
        await db.db.execute(
            "UPDATE memories SET visibility = 'private', owner_user_id = ? WHERE id = ?",
            ("alice", related.id),
        )
        await db.db.commit()
        from memforge.server.admin_api import create_admin_app

        bob_app = create_admin_app(
            db=db,
            config=_config(tmp_path),
            principal_resolver=lambda _request: "bob",
        )
        with TestClient(bob_app) as client:
            assert client.get(f"/api/v1/memory-reviews/{review.id}").status_code == 404
            assert client.get("/api/v1/memory-reviews", params={"status": "open"}).json()["total"] == 0

    @pytest.mark.asyncio
    async def test_related_membership_change_invalidates_validated_manifest(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, review = await _seed_supersede_review(db, chroma)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            detail = client.get(f"/api/v1/memory-reviews/{review.id}").json()
            decisions = [
                {
                    "review_id": review.id,
                    "decision": "approve",
                    "expected_fingerprint": detail["decision_fingerprint"],
                }
            ]
            validated = client.post(
                "/api/v1/memory-reviews/decisions/validate",
                json={"decisions": decisions},
            ).json()
            await _attach_related_challenger(db, review, suffix="late-related")
            applied = client.post(
                "/api/v1/memory-reviews/decisions/apply",
                json={
                    "decisions": decisions,
                    "validation_receipt": validated["validation_receipt"],
                },
            )

        assert applied.status_code == 200
        assert applied.json()["results"][0]["outcome"] == "stale"
        assert (await db.get_memory_review(review.id)).status == "pending"

    @pytest.mark.asyncio
    async def test_single_decision_requires_current_fingerprint_and_records_principal(
        self,
        db,
        chroma,
        tmp_path,
    ):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            detail = client.get(f"/api/v1/memory-reviews/{review.id}").json()
            spoofed = client.post(
                f"/api/v1/memory-reviews/{review.id}/approve",
                json={
                    "expected_fingerprint": detail["decision_fingerprint"],
                    "reviewer": "mallory",
                },
            )
            stale = client.post(
                f"/api/v1/memory-reviews/{review.id}/approve",
                json={"expected_fingerprint": "review-decision-v1:stale"},
            )
            applied = client.post(
                f"/api/v1/memory-reviews/{review.id}/approve",
                json={"expected_fingerprint": detail["decision_fingerprint"]},
            )
            replayed = client.post(
                f"/api/v1/memory-reviews/{review.id}/approve",
                json={"expected_fingerprint": detail["decision_fingerprint"]},
            )
            unrelated_fingerprint = client.post(
                f"/api/v1/memory-reviews/{review.id}/approve",
                json={"expected_fingerprint": "review-decision-v1:unrelated"},
            )

        assert spoofed.status_code == 422
        assert stale.status_code == 409
        assert applied.status_code == 200
        assert replayed.status_code == 200
        assert unrelated_fingerprint.status_code == 409
        stored = await db.get_memory_review(review.id)
        assert stored is not None and stored.reviewer == "dev"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"

    @pytest.mark.asyncio
    async def test_manifest_validation_is_read_only_and_apply_is_per_review(
        self,
        db,
        chroma,
        tmp_path,
    ):
        incumbent, challenger, cross_review = await _seed_cross_source_review(db, chroma)
        lifecycle_review_id = await _seed_lifecycle_review(db, review_id="review-lifecycle-batch")
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            cross = client.get(f"/api/v1/memory-reviews/{cross_review.id}").json()
            lifecycle = client.get(f"/api/v1/memory-reviews/{lifecycle_review_id}").json()
            decisions = [
                {
                    "review_id": cross_review.id,
                    "decision": "approve",
                    "expected_fingerprint": cross["decision_fingerprint"],
                    "rationale": "The claims concern the same deployment.",
                    "confidence": 0.94,
                    "risk": "low",
                },
                {
                    "review_id": lifecycle_review_id,
                    "decision": "reject",
                    "expected_fingerprint": lifecycle["decision_fingerprint"],
                    "note": "The current source state remains authoritative.",
                    "risk": "high",
                },
                {
                    "review_id": "missing-review",
                    "decision": "approve",
                    "expected_fingerprint": "review-decision-v1:missing",
                },
            ]
            validated = client.post(
                "/api/v1/memory-reviews/decisions/validate",
                json={"decisions": decisions},
            )
            assert (await db.get_memory_review(cross_review.id)).status == "pending"
            assert (await db.get_lifecycle_review(lifecycle_review_id)).status is LifecycleReviewStatus.PENDING
            applied = client.post(
                "/api/v1/memory-reviews/decisions/apply",
                json={
                    "decisions": decisions,
                    "validation_receipt": validated.json()["validation_receipt"],
                },
            )

        assert validated.status_code == 200
        assert [item["outcome"] for item in validated.json()["results"]] == [
            "ready",
            "ready",
            "not_found",
        ]
        assert applied.status_code == 200
        assert [item["outcome"] for item in applied.json()["results"]] == [
            "applied",
            "applied",
            "not_found",
        ]
        assert applied.json()["applied"] == 2
        assert applied.json()["failed"] == 1
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"
        assert (await db.get_memory_review(cross_review.id)).status == "approved"
        assert (await db.get_lifecycle_review(lifecycle_review_id)).status is LifecycleReviewStatus.REJECTED

    @pytest.mark.asyncio
    async def test_manifest_apply_requires_receipt_for_the_exact_principal_and_cohort(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, review = await _seed_cross_source_review(db, chroma)
        from memforge.server.admin_api import create_admin_app

        app = create_admin_app(db=db, config=_config(tmp_path))
        with TestClient(app) as client:
            detail = client.get(f"/api/v1/memory-reviews/{review.id}").json()
            decisions = [
                {
                    "review_id": review.id,
                    "decision": "approve",
                    "expected_fingerprint": detail["decision_fingerprint"],
                }
            ]
            missing = client.post(
                "/api/v1/memory-reviews/decisions/apply",
                json={"decisions": decisions},
            )
            validated = client.post(
                "/api/v1/memory-reviews/decisions/validate",
                json={"decisions": decisions},
            )
            changed = client.post(
                "/api/v1/memory-reviews/decisions/apply",
                json={
                    "decisions": [{**decisions[0], "decision": "reject", "note": "Different context"}],
                    "validation_receipt": validated.json()["validation_receipt"],
                },
            )

        assert missing.status_code == 409
        assert changed.status_code == 409
        assert (await db.get_memory_review(review.id)).status == "pending"

    @pytest.mark.asyncio
    async def test_manifest_receipt_cannot_be_replayed_in_another_workspace(
        self,
        db,
        chroma,
        tmp_path,
    ):
        _, _, review = await _seed_cross_source_review(db, chroma)
        from memforge.server.admin_api import create_admin_app

        config = _config(tmp_path)
        workspace_a = create_admin_app(db=db, config=config, workspace_id="workspace-a")
        workspace_b = create_admin_app(db=db, config=config, workspace_id="workspace-b")
        with TestClient(workspace_a) as client_a:
            detail = client_a.get(f"/api/v1/memory-reviews/{review.id}").json()
            decisions = [
                {
                    "review_id": review.id,
                    "decision": "approve",
                    "expected_fingerprint": detail["decision_fingerprint"],
                }
            ]
            receipt = client_a.post(
                "/api/v1/memory-reviews/decisions/validate",
                json={"decisions": decisions},
            ).json()["validation_receipt"]

        with TestClient(workspace_b) as client_b:
            replayed = client_b.post(
                "/api/v1/memory-reviews/decisions/apply",
                json={"decisions": decisions, "validation_receipt": receipt},
            )

        assert replayed.status_code == 409
        assert (await db.get_memory_review(review.id)).status == "pending"


class TestReviewResolutionConcurrency:
    @pytest.mark.asyncio
    async def test_compare_and_set_allows_only_one_pending_resolution(self, db, chroma):
        _, _, review = await _seed_cross_source_review(db, chroma)

        results = await asyncio.gather(
            db.resolve_memory_review(
                review.id,
                status="approved",
                reviewer="alice",
                review_note=None,
            ),
            db.resolve_memory_review(
                review.id,
                status="rejected",
                reviewer="bob",
                review_note="different context",
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(item, ValueError) for item in results) == 1
        stored = await db.get_memory_review(review.id)
        assert stored is not None and stored.status in {"approved", "rejected"}

    @pytest.mark.asyncio
    async def test_concurrent_destructive_approvals_have_one_atomic_owner(
        self,
        db,
        chroma,
        review_service,
    ):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        results = await asyncio.gather(
            _approve(review_service, review.id, reviewer="alice"),
            _approve(review_service, review.id, reviewer="bob"),
            return_exceptions=True,
        )

        assert sum(isinstance(item, ResolvedReview) for item in results) == 1
        assert sum(isinstance(item, ReviewAlreadyResolved) for item in results) == 1
        assert (await db.get_memory_review(review.id)).status == "approved"
        assert (await db.get_memory(incumbent.id)).status == "superseded"
        assert (await db.get_memory(challenger.id)).status == "active"

    @pytest.mark.asyncio
    async def test_concurrent_destructive_opposite_decisions_never_compensate_winner(
        self,
        db,
        chroma,
        review_service,
    ):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        results = await asyncio.gather(
            _approve(review_service, review.id, reviewer="alice"),
            _reject(review_service, review.id, reviewer="bob", note="Keep current"),
            return_exceptions=True,
        )

        assert sum(isinstance(item, ResolvedReview) for item in results) == 1
        assert sum(isinstance(item, ReviewAlreadyResolved) for item in results) == 1
        stored = await db.get_memory_review(review.id)
        current_incumbent = await db.get_memory(incumbent.id)
        current_challenger = await db.get_memory(challenger.id)
        if stored.status == "approved":
            assert current_incumbent.status == "superseded"
            assert current_challenger.status == "active"
        else:
            assert stored.status == "rejected"
            assert current_incumbent.status == "active"
            assert current_challenger.status == "retired"

    @pytest.mark.asyncio
    async def test_cross_connection_cannot_attach_participant_after_resolution_cohort_lock(
        self,
        db,
        chroma,
        review_service,
        monkeypatch,
    ):
        _, _, review = await _seed_supersede_review(db, chroma, suffix="cohortlock")
        related = _memory("mem-related-cohortlock", "PostgreSQL is version 17", status="pending_review")
        await db.insert_memory(related)
        second = Database(db.db_path)
        await second.connect()
        entered_guard = asyncio.Event()
        release_guard = asyncio.Event()
        original_guard = db._assert_no_active_source_support_unlocked

        async def pause_after_cohort_lock(memory_id: str) -> None:
            entered_guard.set()
            await release_guard.wait()
            await original_guard(memory_id)

        monkeypatch.setattr(db, "_assert_no_active_source_support_unlocked", pause_after_cohort_lock)
        resolution = asyncio.create_task(_approve(review_service, review.id, reviewer="alice"))
        await entered_guard.wait()
        attachment = asyncio.create_task(
            second.add_memory_review_related_challenger(review.id, related.id, reason="late candidate")
        )
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(attachment), timeout=0.05)
        release_guard.set()

        try:
            await resolution
            with pytest.raises(ValueError, match="not pending"):
                await attachment
        finally:
            await second.close()

        assert await db.list_memory_review_related_challengers(review.id) == []
        assert (await db.get_memory_review(review.id)).status == "approved"


class TestApprove:
    @pytest.mark.asyncio
    async def test_approve_promotes_challenger_and_supersedes_incumbent(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        result = await _approve(review_service, review.id, reviewer="alice", note=None)

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

        await _approve(review_service, review.id, reviewer="alice", note=None)

        stored_incumbent = await db.get_memory(incumbent.id)
        stored_challenger = await db.get_memory(challenger.id)

        assert stored_incumbent.status == "superseded"
        assert stored_incumbent.superseded_by == challenger.id
        assert stored_incumbent.replacement_kind == "revision"
        assert stored_challenger.status == "active"

    @pytest.mark.asyncio
    async def test_approve_keeps_search_indexes_aligned(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        await _approve(review_service, review.id, reviewer="alice", note=None)

        async with db.db.execute("SELECT memory_id FROM memories_fts ORDER BY memory_id") as cursor:
            fts_ids = [row[0] async for row in cursor]
        assert challenger.id in fts_ids
        assert incumbent.id not in fts_ids
        assert challenger.id in chroma.records
        assert chroma.records[challenger.id]["status"] == "active"
        assert incumbent.id not in chroma.records
        rows = await db.db.execute_fetchall(
            "SELECT memory_id, operation, status FROM review_vector_outbox "
            "WHERE review_id = ? ORDER BY memory_id",
            (review.id,),
        )
        assert {(row["memory_id"], row["operation"], row["status"]) for row in rows} == {
            (incumbent.id, "delete", "completed"),
            (challenger.id, "upsert", "completed"),
        }

    @pytest.mark.asyncio
    async def test_vector_failure_keeps_durable_retry_without_compensating_review(
        self,
        db,
        chroma,
        review_service,
        memory_store,
        monkeypatch,
    ):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="vectorretry")
        original_reconcile = memory_store._reconcile_lifecycle_vector_target

        async def fail_vector(_memory_id: str) -> None:
            raise RuntimeError("vector provider unavailable")

        monkeypatch.setattr(memory_store, "_reconcile_lifecycle_vector_target", fail_vector)
        result = await _approve(review_service, review.id, reviewer="alice")

        assert result.review.status == "approved"
        assert (await db.get_memory(incumbent.id)).status == "superseded"
        assert (await db.get_memory(challenger.id)).status == "active"
        failed = await db.db.execute_fetchall(
            "SELECT status, attempts, error FROM review_vector_outbox WHERE review_id = ?",
            (review.id,),
        )
        assert len(failed) == 2
        assert {row["status"] for row in failed} == {"failed"}
        assert {row["attempts"] for row in failed} == {1}
        assert all("vector provider unavailable" in row["error"] for row in failed)

        monkeypatch.setattr(memory_store, "_reconcile_lifecycle_vector_target", original_reconcile)
        await db.db.execute(
            "UPDATE review_vector_outbox SET next_attempt_at = NULL WHERE review_id = ?",
            (review.id,),
        )
        await db.db.commit()
        delivery = await memory_store.attempt_review_vector_delivery(review.id)

        assert delivery.pending is False
        completed = await db.db.execute_fetchall(
            "SELECT status, attempts FROM review_vector_outbox WHERE review_id = ?",
            (review.id,),
        )
        assert {row["status"] for row in completed} == {"completed"}
        assert {row["attempts"] for row in completed} == {2}
        assert challenger.id in chroma.records
        assert incumbent.id not in chroma.records

    @pytest.mark.asyncio
    async def test_approve_records_review_and_supersede_audit(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)

        await _approve(review_service, review.id, reviewer="alice", note=None)

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
        original_resolve = db.apply_memory_review_resolution

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "apply_memory_review_resolution", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await _approve(review_service, review.id, reviewer="alice", note=None)

        monkeypatch.setattr(db, "apply_memory_review_resolution", original_resolve)
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

        monkeypatch.setattr(db, "apply_memory_review_resolution", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await _approve(review_service, review.id, reviewer="alice", note=None)

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
        )
        await db.link_memory_entity(challenger.id, entity_id)

        await _approve(review_service, review.id, reviewer="alice", note=None)

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

        await _approve(review_service, review.id, reviewer="alice", note=None)

        stored_related = await db.get_memory(related.id)
        assert stored_related.status == "retired"
        assert stored_related.retirement_reason == "review_redundant"
        assert related.id not in chroma.records

    @pytest.mark.asyncio
    async def test_approve_fails_closed_before_mutation_when_incumbent_has_active_support(
        self,
        db,
        chroma,
        review_service,
        monkeypatch,
    ):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma, suffix="supportguard")
        guarded: list[str] = []

        async def active_support_guard(memory_id: str) -> None:
            guarded.append(memory_id)
            raise ValueError(
                "direct terminal Memory transition rejected while active source support remains; "
                "projected lifecycle required"
            )

        monkeypatch.setattr(db, "_assert_no_active_source_support_unlocked", active_support_guard)

        with pytest.raises(ReviewError, match="active source support"):
            await _approve(review_service, review.id, reviewer="alice")

        assert guarded == [incumbent.id]
        assert (await db.get_memory_review(review.id)).status == "pending"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "pending_review"

    @pytest.mark.asyncio
    async def test_repeated_approve_returns_clear_409_without_partial_mutation(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        await _approve(review_service, review.id, reviewer="alice", note=None)

        snapshot_incumbent = await db.get_memory(incumbent.id)
        snapshot_challenger = await db.get_memory(challenger.id)

        with pytest.raises(ReviewAlreadyResolved):
            await _approve(review_service, review.id, reviewer="bob", note=None)

        assert (await db.get_memory(incumbent.id)).updated_at == snapshot_incumbent.updated_at
        assert (await db.get_memory(challenger.id)).updated_at == snapshot_challenger.updated_at

    @pytest.mark.asyncio
    async def test_memory_drift_expires_review_before_it_can_be_decided(self, db, chroma, review_service):
        incumbent, _, review = await _seed_supersede_review(db, chroma)

        await db.update_memory_content(
            incumbent.id,
            new_content="PostgreSQL is now version 15",
            new_confidence=None,
        )

        with pytest.raises(ReviewAlreadyResolved, match="already stale"):
            await _approve(review_service, review.id, reviewer="alice", note=None)

        stored = await db.get_memory_review(review.id)
        assert stored.status == "stale"
        assert stored.resolved_at is not None
        assert await db.list_memory_reviews(status="open") == []


# ---------------------------------------------------------------------------
# Reject
# ---------------------------------------------------------------------------


class TestReject:
    @pytest.mark.asyncio
    async def test_reject_retires_challenger_with_reason_and_removes_from_indexes(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_supersede_review(db, chroma)
        chroma.upsert(ids=[challenger.id], metadatas=[{"status": "pending_review"}])

        result = await _reject(
            review_service,
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
            await _reject(review_service, review.id, reviewer="alice", note="   ")

        stored = await db.get_memory_review(review.id)
        assert stored.status == "pending"

    @pytest.mark.asyncio
    async def test_reject_records_review_audit_with_reviewer(self, db, chroma, review_service):
        _, challenger, review = await _seed_supersede_review(db, chroma)

        await _reject(review_service, review.id, reviewer="alice", note="bad source")

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

        await _reject(review_service, review.id, reviewer="alice", note="bad source")

        stored_related = await db.get_memory(related.id)
        assert stored_related.status == "retired"
        assert stored_related.retirement_reason == "rejected"
        assert related.id not in chroma.records

    @pytest.mark.asyncio
    async def test_reject_rolls_back_when_review_resolution_fails(self, db, chroma, review_service, monkeypatch):
        _, challenger, review = await _seed_supersede_review(db, chroma)
        original_resolve = db.apply_memory_review_resolution

        async def fail_resolve(*args, **kwargs):
            raise RuntimeError("resolution failed")

        monkeypatch.setattr(db, "apply_memory_review_resolution", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await _reject(review_service, review.id, reviewer="alice", note="bad source")

        monkeypatch.setattr(db, "apply_memory_review_resolution", original_resolve)
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

        monkeypatch.setattr(db, "apply_memory_review_resolution", fail_resolve)

        with pytest.raises(RuntimeError, match="resolution failed"):
            await _reject(review_service, review.id, reviewer="alice", note="bad source")

        stored_challenger = await db.get_memory(challenger.id)
        stored_related = await db.get_memory(related.id)
        assert stored_challenger.status == "pending_review"
        assert stored_related.status == "pending_review"
        assert challenger.id in chroma.records
        assert related.id in chroma.records

    @pytest.mark.asyncio
    async def test_repeated_reject_returns_clear_409(self, db, chroma, review_service):
        _, _, review = await _seed_supersede_review(db, chroma)

        await _reject(review_service, review.id, reviewer="alice", note="bad source")

        with pytest.raises(ReviewAlreadyResolved):
            await _reject(review_service, review.id, reviewer="bob", note="again")


# ---------------------------------------------------------------------------
# Non-destructive cross-source finding resolution
# ---------------------------------------------------------------------------


class TestCrossSourceReviewResolution:
    @pytest.mark.asyncio
    async def test_approve_acknowledges_finding_without_mutating_memories(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)

        result = await _approve(
            review_service,
            review.id,
            reviewer="alice",
            note="confirmed conflict; no authority decision yet",
        )

        assert result.review.status == "approved"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"
        assert set(chroma.records) == {incumbent.id, challenger.id}

    @pytest.mark.asyncio
    async def test_reject_dismisses_finding_without_mutating_memories(self, db, chroma, review_service):
        incumbent, challenger, review = await _seed_cross_source_review(db, chroma)

        result = await _reject(
            review_service,
            review.id,
            reviewer="alice",
            note="claims apply to different deployments",
        )

        assert result.review.status == "rejected"
        assert (await db.get_memory(incumbent.id)).status == "active"
        assert (await db.get_memory(challenger.id)).status == "active"
        assert set(chroma.records) == {incumbent.id, challenger.id}
