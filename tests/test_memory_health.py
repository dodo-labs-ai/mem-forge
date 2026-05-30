"""Deterministic memory index health checks."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from meminception.memory.health import MemoryIndexHealthChecker
from meminception.memory.index_payloads import embedding_vector_hash
from meminception.models import Memory, content_hash
from meminception.storage.database import Database


class InspectableCollection:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records

    def get(self, *, include=None, **kwargs):
        ids = list(self.records.keys())
        include = include or ["metadatas"]
        result: dict[str, Any] = {"ids": ids}
        if "metadatas" in include:
            result["metadatas"] = [
                {k: v for k, v in self.records[record_id].items() if k != "embedding"}
                for record_id in ids
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

    assert ("fts_content_mismatch", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_duplicate_fts_rows(db: Database):
    active = _memory("mem-duplicate-fts", "Current fact", "active")
    await db.insert_memory(active)
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
        (active.id, active.content, "", ""),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_duplicate", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_fts_orphan(db: Database):
    await db.db.execute(
        "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
        ("mem-orphan-fts", "Orphan fact", "", ""),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(db=db, memory_collection=InspectableCollection({}))

    report = await checker.check()

    assert ("fts_orphan", "mem-orphan-fts") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


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
        memory_collection=InspectableCollection({
            active.id: {"status": "active", "content_hash": "old-hash"},
        }),
    )

    report = await checker.check()

    assert ("chroma_content_hash_mismatch", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_missing_content_hash(db: Database):
    active = _memory("mem-no-hash", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({
            active.id: {"status": "active"},
        }),
    )

    report = await checker.check()

    assert ("chroma_content_hash_missing", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_missing_status(db: Database):
    active = _memory("mem-no-status", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({
            active.id: {"content_hash": active.content_hash},
        }),
    )

    report = await checker.check()

    assert ("chroma_status_missing", active.id) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_memory_chroma_embedding_text_hash_missing(db: Database):
    active = _memory("mem-no-embedding-hash", "Current fact", "active")
    await db.insert_memory(active)
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({
            active.id: {"status": "active", "content_hash": active.content_hash},
        }),
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
        memory_collection=InspectableCollection({
            active.id: {
                "status": "active",
                "content_hash": active.content_hash,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_vector_hash": embedding_vector_hash([0.3, 0.2, 0.1]),
            },
        }),
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
async def test_health_reports_document_missing_from_chroma(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-active", "src", "http://test/doc-active", "doc-active", "TEST", now, "1", "hash-doc", now),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({}),
    )

    report = await checker.check()

    assert ("document_missing_chroma", "doc-active") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_document_chroma_orphan(db: Database):
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({"doc-stale": {}}),
    )

    report = await checker.check()

    assert ("document_chroma_orphan", "doc-stale") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_document_chroma_content_hash_mismatch(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-current", "src", "http://test/doc-current", "doc-current", "TEST", now, "v2", "hash-current", now),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({
            "doc-current": {"content_hash": "hash-old", "version": "v1"},
        }),
    )

    report = await checker.check()
    issue_keys = {(issue.kind, issue.memory_id) for issue in report.issues}
    assert ("document_chroma_content_hash_mismatch", "doc-current") in issue_keys
    assert ("document_chroma_version_mismatch", "doc-current") in issue_keys


@pytest.mark.asyncio
async def test_health_reports_document_chroma_missing_freshness_metadata(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-no-meta", "src", "http://test/doc-no-meta", "doc-no-meta", "TEST", now, "v2", "hash-current", now),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({"doc-no-meta": {}}),
    )

    report = await checker.check()
    issue_keys = {(issue.kind, issue.memory_id) for issue in report.issues}
    assert ("document_chroma_content_hash_missing", "doc-no-meta") in issue_keys
    assert ("document_chroma_version_missing", "doc-no-meta") in issue_keys


@pytest.mark.asyncio
async def test_health_reports_document_chroma_missing_embedding_text_hash(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-no-embedding-hash", "src", "http://test/doc", "doc", "TEST", now, "v2", "hash-current", now),
    )
    await db.db.execute(
        """INSERT INTO document_metadata
           (doc_id, summary, tags, entities, doc_type, complexity, enriched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("doc-no-embedding-hash", "Summary", '["tag"]', "[]", "design", "medium", now),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({
            "doc-no-embedding-hash": {"content_hash": "hash-current", "version": "v2"},
        }),
    )

    report = await checker.check()

    assert ("document_chroma_embedding_text_hash_missing", "doc-no-embedding-hash") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_document_chroma_embedding_vector_hash_missing(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-no-vector-hash", "src", "http://test/doc", "doc", "TEST", now, "v2", "hash-current", now),
    )
    await db.db.commit()
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=InspectableCollection({
            "doc-no-vector-hash": {
                "content_hash": "hash-current",
                "version": "v2",
                "embedding": [0.1, 0.2, 0.3],
            },
        }),
    )

    report = await checker.check()

    assert ("document_chroma_embedding_vector_hash_missing", "doc-no-vector-hash") in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_document_chroma_unavailable(db: Database):
    checker = MemoryIndexHealthChecker(
        db=db,
        memory_collection=InspectableCollection({}),
        document_collection=FailingCollection(),
    )

    report = await checker.check()

    assert ("document_chroma_unavailable", None) in {
        (issue.kind, issue.memory_id) for issue in report.issues
    }


@pytest.mark.asyncio
async def test_health_reports_confluence_document_missing_pdf_uri(db: Database):
    now = datetime.now(timezone.utc).isoformat()
    await db.upsert_source("src-conf", "confluence", "Architecture", "{}")
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
