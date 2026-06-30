"""Tests for user-facing memory lifecycle actions exposed to MCP/API clients."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.lifecycle_service import MemoryLifecycleConflict, MemoryLifecycleService
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, Visibility, content_hash
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


class RecordingCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []

    def upsert(self, *, ids, embeddings=None, metadatas=None, documents=None) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "embedding": embeddings[index] if embeddings else None,
                "metadata": metadatas[index] if metadatas else {},
                "document": documents[index] if documents else None,
            }

    def delete(self, *, ids) -> None:
        for record_id in ids:
            self.deleted.append(record_id)
            self.records.pop(record_id, None)

    def query(self, **_params):
        return {"ids": [[]], "distances": [[]]}

    def get(self, *, ids=None, include=None):
        selected = [record_id for record_id in (ids or self.records) if record_id in self.records]
        include = include or []
        result: dict[str, Any] = {"ids": selected}
        if "metadatas" in include:
            result["metadatas"] = [self.records[record_id]["metadata"] for record_id in selected]
        if "embeddings" in include:
            result["embeddings"] = [self.records[record_id]["embedding"] for record_id in selected]
        if "documents" in include:
            result["documents"] = [self.records[record_id]["document"] for record_id in selected]
        return result


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "lifecycle.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.91,
        tags=["jira"],
        created_at=now,
        updated_at=now,
    )


def _store(db: Database, collection: RecordingCollection) -> MemoryStore:
    audit_logger = MemoryAuditLogger(db, default_context=AuditContext(actor_type="test", run_id="lifecycle-test"))
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=audit_logger,
    )

    async def fake_embed(_text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    store._embed = fake_embed  # type: ignore[method-assign]
    return store


@pytest.mark.asyncio
async def test_create_memory_writes_private_user_memory_with_provenance(db: Database):
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    result = await service.create_memory(
        content="Use Status and FollowUpStepStatus when polling PayrollProcessingTriggerViews.",
        reason="User explicitly asked MemForge to remember this.",
        memory_type="fact",
        tags=["dwc", "polling"],
        owner_user_id="andrew.sun01@sap.com",
        client="codex",
        repo_identifier="github.com/shno-labs/mem-forge",
    )

    stored = await db.get_memory(result.memory_id)
    sources = await db.get_memory_sources(result.memory_id)
    assert result.status == "inserted"
    assert stored is not None
    assert stored.content == "Use Status and FollowUpStepStatus when polling PayrollProcessingTriggerViews."
    assert stored.visibility == Visibility.PRIVATE.value
    assert stored.owner_user_id == "andrew.sun01@sap.com"
    assert stored.project_key == "UNSORTED"
    assert stored.repo_identifier == "github.com/shno-labs/mem-forge"
    assert stored.tags == ["dwc", "polling"]
    assert [(source.doc_id, source.source_type) for source in sources] == [
        (f"user-memory-{result.memory_id}", "user_memory")
    ]
    document = await db.get_document(f"user-memory-{result.memory_id}")
    assert document is not None
    assert document.source == "user_memory"
    assert document.client == "codex"


@pytest.mark.asyncio
async def test_retire_memory_uses_expected_hash_guard(db: Database):
    memory = _memory("mem-retire-tool", "Old fact")
    await db.insert_memory(memory)
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    with pytest.raises(MemoryLifecycleConflict, match="content_hash_mismatch"):
        await service.retire_memory(
            memory.id,
            reason="User says this is stale",
            expected_content_hash="wrong",
        )

    await service.retire_memory(
        memory.id,
        reason="User says this is stale",
        expected_content_hash=memory.content_hash,
    )

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert stored.status == "retired"
    assert stored.retirement_reason == "User says this is stale"


@pytest.mark.asyncio
async def test_replace_document_memory_creates_correction_provenance_without_carrying_old_sources(db: Database):
    old = _memory("mem-replace-tool-old", "Mount Tai defects use queue A")
    await db.insert_memory(old)
    await db.upsert_document(
        DocumentRecord(
            doc_id="doc-jira-old",
            source="jira",
            source_url="https://jira.example/browse/MT-1",
            title="MT-1",
            space_or_project="MT",
            author=None,
            last_modified=datetime(2026, 6, 20, tzinfo=timezone.utc),
            labels=[],
            version="1",
            content_hash="doc-hash",
            token_count=100,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=datetime(2026, 6, 20, tzinfo=timezone.utc),
        )
    )
    await db.add_memory_source(
        old.id,
        "doc-jira-old",
        "jira",
        "Original Jira excerpt",
        source_updated_at=datetime(2026, 6, 20, tzinfo=timezone.utc),
    )
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    result = await service.replace_memory(
        old.id,
        replacement_content="Mount Tai defects use queue B",
        reason="User corrected the queue.",
        expected_content_hash=old.content_hash,
        replacement_kind="revision",
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    new_sources = await db.get_memory_sources(result.replacement_memory_id)
    assert stored_old is not None
    assert stored_old.status == "superseded"
    assert stored_old.superseded_by == result.replacement_memory_id
    assert stored_new is not None
    assert stored_new.status == "active"
    assert stored_new.content == "Mount Tai defects use queue B"
    assert stored_new.visibility == old.visibility
    assert stored_new.project_key == "UNSORTED"
    assert [(source.doc_id, source.source_type) for source in new_sources] == [
        (f"correction-{result.replacement_memory_id}", "user_correction")
    ]


@pytest.mark.asyncio
async def test_replace_agent_claim_memory_updates_claim_lineage(db: Database):
    observed_at = datetime(2026, 6, 28, tzinfo=timezone.utc)
    old = _memory("mem-agent-tool-old", "Use claude-code to invoke Claude Code CLI")
    await db.upsert_document(
        DocumentRecord(
            doc_id="concept-claude-cli",
            source="src-agent-sessions-codex",
            source_url="memforge://agent-session/concept-claude-cli",
            title="Claude Code CLI convention",
            space_or_project="UNSORTED",
            author="andrew.sun01@sap.com",
            last_modified=observed_at,
            labels=["agent_session"],
            version="1",
            content_hash="concept-hash",
            token_count=20,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=observed_at,
        )
    )
    await db.insert_memory_and_upsert_agent_claim(
        old,
        doc_id="concept-claude-cli",
        source_type="agent_session",
        excerpt="Use claude-code to invoke Claude Code CLI",
        relation_outcome=None,
        claim_id="claim-claude-cli",
        concept_id="concept-claude-cli",
        display_anchor="Claude Code CLI convention",
        claim_text="Use claude-code to invoke Claude Code CLI",
        memory_type=old.memory_type,
        tags=list(old.tags),
        confidence=old.confidence,
        observed_at=observed_at,
        source_updated_at=observed_at,
        concept_projection={
            "concept_id": "concept-claude-cli",
            "source_id": "src-agent-sessions-codex",
            "owner_user_id": "andrew.sun01@sap.com",
            "workspace": "/workspace",
            "repo_identifier": "github.com/shno-labs/mem-forge",
            "concept_type": "topic",
            "concept_path": "concepts/claude-cli.md",
            "title": "Claude Code CLI convention",
            "markdown_body": "# Claude Code CLI convention\n",
            "frontmatter": {},
        },
    )
    collection = RecordingCollection()
    store = _store(db, collection)
    service = MemoryLifecycleService(db=db, memory_store=store)

    result = await service.replace_memory(
        old.id,
        replacement_content="Invoke Claude Code with `claude`, not `claude-code`.",
        reason="User corrected the command name.",
        expected_content_hash=old.content_hash,
        replacement_kind="revision",
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(result.replacement_memory_id)
    claim = await db.get_agent_claim("claim-claude-cli")
    new_sources = await db.get_memory_sources(result.replacement_memory_id)
    assert stored_old is not None
    assert stored_old.status == "superseded"
    assert stored_old.superseded_by == result.replacement_memory_id
    assert stored_new is not None
    assert stored_new.status == "active"
    assert stored_new.content == "Invoke Claude Code with `claude`, not `claude-code`."
    assert claim is not None
    assert claim["memory_id"] == result.replacement_memory_id
    assert claim["claim_text"] == "Invoke Claude Code with `claude`, not `claude-code`."
    assert [(source.doc_id, source.source_type) for source in new_sources] == [
        ("concept-claude-cli", "agent_session")
    ]


@pytest.mark.asyncio
async def test_create_memory_route_audits_request_principal_and_client(db: Database, tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    app = create_admin_app(
        db=db,
        config=AppConfig(base_dir=tmp_path / "memforge"),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/memories/create",
            json={
                "content": "Use canonical payroll trigger status fields.",
                "reason": "User confirmed the readable memory preview.",
                "memory_type": "fact",
                "tags": ["payroll", "polling"],
                "client": "codex",
                "repo_identifier": "github.com/shno-labs/mem-forge",
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    stored = await db.get_memory(payload["memory_id"])
    assert payload["status"] == "inserted"
    assert stored is not None
    assert stored.owner_user_id == "andrew.sun01@sap.com"
    assert stored.visibility == "private"
    audit_rows = await db.list_memory_audit_events(
        memory_id=payload["memory_id"],
        event_type="memory_insert_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"
    document = await db.get_document(f"user-memory-{payload['memory_id']}")
    assert document is not None
    assert document.client == "codex"


@pytest.mark.asyncio
async def test_retire_memory_route_returns_conflict_for_stale_content_hash(db: Database, tmp_path):
    from memforge.server.admin_api import create_admin_app

    memory = _memory("mem-retire-route", "Route guarded fact")
    await db.insert_memory(memory)
    app = create_admin_app(db=db, config=AppConfig(base_dir=tmp_path / "memforge"))

    with TestClient(app) as client:
        response = client.post(
            f"/api/memories/{memory.id}/retire",
            json={
                "reason": "User says this is stale",
                "expected_content_hash": "wrong",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "content_hash_mismatch"


@pytest.mark.asyncio
async def test_retire_memory_route_audits_request_principal(db: Database, tmp_path):
    from memforge.server.admin_api import create_admin_app

    memory = _memory("mem-retire-route-audit", "Route audited fact")
    await db.insert_memory(memory)
    app = create_admin_app(
        db=db,
        config=AppConfig(base_dir=tmp_path / "memforge"),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/memories/{memory.id}/retire",
            json={
                "reason": "User says this is stale",
                "expected_content_hash": memory.content_hash,
            },
        )

    assert response.status_code == 200, response.text
    audit_rows = await db.list_memory_audit_events(
        memory_id=memory.id,
        event_type="memory_retire_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"


@pytest.mark.asyncio
async def test_replace_memory_route_audits_request_principal(db: Database, tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    async def fake_embed(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr("memforge.memory.store.MemoryStore._embed", fake_embed)

    memory = _memory("mem-replace-route-audit", "Route replacement audited fact")
    await db.insert_memory(memory)
    app = create_admin_app(
        db=db,
        config=AppConfig(base_dir=tmp_path / "memforge"),
        principal_resolver=lambda _request: "andrew.sun01@sap.com",
    )

    with TestClient(app) as client:
        response = client.post(
            f"/api/memories/{memory.id}/replace",
            json={
                "replacement_content": "Route replacement corrected fact",
                "reason": "User corrected this memory.",
                "expected_content_hash": memory.content_hash,
                "replacement_kind": "supersession",
            },
        )

    assert response.status_code == 200, response.text
    audit_rows = await db.list_memory_audit_events(
        memory_id=memory.id,
        event_type="memory_supersede_committed",
    )
    assert len(audit_rows) == 1
    assert audit_rows[0].actor_type == "user"
    assert audit_rows[0].actor_id == "andrew.sun01@sap.com"
