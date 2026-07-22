"""Deterministic memory index health checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from memforge.memory.health import MemoryIndexHealthChecker
from memforge.memory.index_payloads import embedding_vector_hash
from memforge.models import Memory, content_hash
from memforge.storage.database import Database


class InspectableCollection:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records

    def get(self, *, include=None, **kwargs):
        ids = list(self.records.keys())
        include = include or ["metadatas"]
        result: dict[str, Any] = {"ids": ids}
        if "metadatas" in include:
            result["metadatas"] = [
                {k: v for k, v in self.records[record_id].items() if k != "embedding"} for record_id in ids
            ]
        if "embeddings" in include:
            result["embeddings"] = [self.records[record_id].get("embedding") for record_id in ids]
        return result


class FailingCollection:
    def get(self, **kwargs):
        raise RuntimeError("collection unavailable")


class FailingFtsDatabase:
    def __init__(self, wrapped: Database) -> None:
        self.wrapped = wrapped

    @property
    def db(self):
        return self

    def execute(self, sql, *args, **kwargs):
        if "memories_fts" in sql:
            raise RuntimeError("fts unavailable")
        return self.wrapped.db.execute(sql, *args, **kwargs)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "health.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str, status: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status=status,
    )


@pytest.mark.asyncio
async def test_health_reports_active_missing_from_fts_and_chroma(db: Database):
    active = _memory("mem-active", "Active fact", "active")
    await db.insert_memory(active)
    await db.db.execute("DELETE FROM memories_fts WHERE memory_id = ?", (active.id,))
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("active_missing_fts", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}
    assert ("active_missing_chroma", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_non_active_present_in_fts_and_chroma(db: Database):
    retired = _memory("mem-retired", "Retired fact", "retired")
    await db.insert_memory(retired)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({retired.id: {"status": "retired"}}),
    )

    report = await checker.check()

    issue_keys = {(issue.kind, issue.memory_id) for issue in report.issues}
    assert ("non_active_present_fts", retired.id) in issue_keys
    assert ("non_active_present_chroma", retired.id) in issue_keys


@pytest.mark.asyncio
async def test_health_reports_fts_unavailable(db: Database):
    checker = MemoryIndexHealthChecker(db=FailingFtsDatabase(db), memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_unavailable", None) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_stale_fts_content(db: Database):
    active = _memory("mem-stale-fts", "Current fact", "active")
    await db.insert_memory(active)
    await db.db.execute(
        "UPDATE memories_fts SET content = ? WHERE memory_id = ?",
        ("Old fact", active.id),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_content_mismatch", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_duplicate_fts_rows(db: Database):
    active = _memory("mem-duplicate-fts", "Current fact", "active")
    await db.insert_memory(active)
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text) VALUES (?, ?, ?)",
        (active.id, active.content, ""),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_duplicate", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_fts_orphan(db: Database):
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text) VALUES (?, ?, ?)",
        ("mem-orphan-fts", "Orphan fact", ""),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_orphan", "mem-orphan-fts") in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_chroma_metadata_status_mismatch(db: Database):
    active = _memory("mem-active", "Active fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({active.id: {"status": "retired"}}),
    )

    report = await checker.check()

    assert ("chroma_status_mismatch", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_content_hash_mismatch(db: Database):
    active = _memory("mem-stale-vector", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection(
            {
                active.id: {"status": "active", "content_hash": "old-hash"},
            }
        ),
    )

    report = await checker.check()

    assert ("chroma_content_hash_mismatch", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_missing_content_hash(db: Database):
    active = _memory("mem-no-hash", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection(
            {
                active.id: {"status": "active"},
            }
        ),
    )

    report = await checker.check()

    assert ("chroma_content_hash_missing", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_missing_status(db: Database):
    active = _memory("mem-no-status", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection(
            {
                active.id: {"content_hash": active.content_hash},
            }
        ),
    )

    report = await checker.check()

    assert ("chroma_status_missing", active.id) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_embedding_text_hash_missing(db: Database):
    active = _memory("mem-no-embedding-hash", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection(
            {
                active.id: {"status": "active", "content_hash": active.content_hash},
            }
        ),
    )

    report = await checker.check()

    assert ("chroma_embedding_text_hash_missing", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_embedding_vector_hash_mismatch(db: Database):
    active = _memory("mem-vector-mismatch", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection(
            {
                active.id: {
                    "status": "active",
                    "content_hash": active.content_hash,
                    "embedding": [0.1, 0.2, 0.3],
                    "embedding_vector_hash": embedding_vector_hash([0.3, 0.2, 0.1]),
                },
            }
        ),
    )

    report = await checker.check()

    assert ("chroma_embedding_vector_hash_mismatch", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_chroma_unavailable(db: Database):
    checker = MemoryIndexHealthChecker(db=db, memory_collection=FailingCollection())

    report = await checker.check()

    assert ("memory_chroma_unavailable", None) in {(issue.kind, issue.memory_id) for issue in report.issues}


@pytest.mark.asyncio
async def test_health_reports_confluence_document_missing_pdf_uri(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.upsert_source(
        "src-conf", "confluence", "Architecture", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version,
            content_hash, normalized_content_uri, pdf_content_uri, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "confluence-123",
            "src-conf",
            "https://wiki.example/doc",
            "Architecture",
            "PAY",
            now,
            "1",
            "hash-doc",
            "/tmp/architecture.md",
            None,
            now,
        ),
    )
    await db.db.commit()

    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("confluence_pdf_uri_missing", "confluence-123") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }
