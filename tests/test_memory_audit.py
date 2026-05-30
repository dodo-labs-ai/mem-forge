"""Audit ledger tests for memory evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memforge.memory.audit import MemoryAuditEvent, MemoryAuditLogger
from memforge.memory.store import MemoryStore
from memforge.models import Memory, content_hash
from memforge.storage.database import Database


class RecordingCollection:
    def __init__(self) -> None:
        self.records = {}

    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, *, ids, embeddings=None, metadatas=None) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id] = metadatas[index] if metadatas else {}

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "audit.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_insert_memory_audit_event_round_trips_payload(db: Database):
    event = MemoryAuditEvent(
        event_id="evt-1",
        operation_id="op-1",
        event_type="memory_insert_committed",
        status="committed",
        actor_type="sync",
        source_id="src-1",
        doc_id="doc-1",
        memory_id="mem-1",
        reason="new source-grounded memory",
        payload={"content_hash": "hash-1"},
        occurred_at=datetime.now(timezone.utc),
    )

    await db.insert_memory_audit_event(event)

    rows = await db.list_memory_audit_events(operation_id="op-1")
    assert len(rows) == 1
    assert rows[0].event_type == "memory_insert_committed"
    assert rows[0].status == "committed"
    assert rows[0].payload["content_hash"] == "hash-1"


@pytest.mark.asyncio
async def test_insert_memory_audit_event_normalizes_occurred_at_to_utc(db: Database):
    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_id="evt-tz",
            operation_id="op-tz",
            event_type="source_support_verification_failed",
            status="failed",
            occurred_at=datetime(2026, 5, 26, 6, 0, tzinfo=timezone(timedelta(hours=2))),
        )
    )

    rows = await db.list_memory_audit_events(operation_id="op-tz")
    async with db.db.execute(
        "SELECT occurred_at FROM memory_audit_events WHERE event_id = ?",
        ("evt-tz",),
    ) as cursor:
        raw_occurred_at = (await cursor.fetchone())[0]

    assert rows[0].occurred_at == datetime(2026, 5, 26, 4, 0, tzinfo=timezone.utc)
    assert raw_occurred_at == "2026-05-26T04:00:00+00:00"


@pytest.mark.asyncio
async def test_list_memory_audit_events_filters_by_memory_id(db: Database):
    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_id="evt-a",
            operation_id="op-a",
            event_type="memory_retire_committed",
            status="committed",
            memory_id="mem-a",
        )
    )
    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_id="evt-b",
            operation_id="op-b",
            event_type="memory_retire_committed",
            status="committed",
            memory_id="mem-b",
        )
    )

    rows = await db.list_memory_audit_events(memory_id="mem-a")

    assert [row.event_id for row in rows] == ["evt-a"]


@pytest.mark.asyncio
async def test_store_operation_events_share_operation_id(db: Database):
    now = datetime.now(timezone.utc)
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("doc-1", "src-1", "http://test/doc-1", "Doc 1", "TEST", now.isoformat(), "1", "hash-doc-1", now.isoformat()),
    )
    await db.db.commit()
    memory = Memory(
        id="mem-group1",
        memory_type="fact",
        content="Grouped audit event fact",
        content_hash=content_hash("Grouped audit event fact"),
        confidence=0.9,
        created_at=now,
        updated_at=now,
    )
    store = MemoryStore(
        db=db,
        memory_collection=RecordingCollection(),
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db),
    )

    async def fake_embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[assignment]

    await store.deduplicate_and_insert(memory, "doc-1", "confluence")

    rows = await db.list_memory_audit_events(memory_id=memory.id)
    operation_ids = {row.operation_id for row in rows}
    assert len(operation_ids) == 1
    assert {row.event_type for row in rows} >= {
        "fts_upsert_committed",
        "chroma_upsert_attempted",
        "chroma_upsert_committed",
        "memory_insert_committed",
    }
