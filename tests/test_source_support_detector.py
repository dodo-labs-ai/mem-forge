"""Tests for source-support detection and corroborated provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import SourceSupportDecision, SourceSupportResponse, StructuredLlmError
from memforge.memory.audit import AuditContext, MemoryAuditLogger
from memforge.memory.evidence import CandidateBucket, LifecycleAction, RelationType
from memforge.memory.store import MemoryStore
from memforge.models import Memory, content_hash
from memforge.pipeline.source_support_detector import SourceSupportDetector
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "source-support.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(
    mem_id: str,
    content: str,
    *,
    project_key: str = "PAY",
    visibility: str = "workspace",
    owner_user_id: str | None = None,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        project_key=project_key,
        tags=["payroll"],
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
        visibility=visibility,
        owner_user_id=owner_user_id,
    )


async def _insert_doc(db: Database, doc_id: str, *, project: str = "PAY") -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, f"src-{project}", f"http://test/{doc_id}", doc_id, project, now, "1", f"hash-{doc_id}", now),
    )
    await db.db.commit()


async def _seed_memory(
    db: Database,
    memory: Memory,
    *,
    doc_id: str,
    entity_id: int,
    support_kind: str = "extracted",
) -> None:
    await db.insert_memory(memory)
    await db.add_memory_source(
        memory.id,
        doc_id,
        "confluence",
        "original excerpt",
        support_kind=support_kind,
        source_updated_at=None,
    )
    await db.link_memory_entity(memory.id, entity_id)


def _support_response(items: list[dict]) -> SourceSupportResponse:
    return SourceSupportResponse(decisions=[SourceSupportDecision.model_validate(item) for item in items])


class FakeStructuredSupportClient:
    def __init__(
        self,
        responses: list[SourceSupportResponse] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.error = error
        self.prompts: list[str] = []

    async def verify_source_support(self, prompt: str) -> SourceSupportResponse:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return SourceSupportResponse(decisions=[])


class FakeCollection:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, ids):
        self.deleted.extend(ids)


def _memory_store(db: Database, collection: FakeCollection | None = None) -> MemoryStore:
    adapters = build_sqlite_adapters(db, collection or FakeCollection())
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )


def _audited_memory_store(db: Database, collection: FakeCollection | None = None) -> MemoryStore:
    adapters = build_sqlite_adapters(db, collection or FakeCollection())
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="test")),
    )


@dataclass
class RecordingContext:
    operation_id: str


@pytest.mark.asyncio
async def test_support_detection_requires_memory_store(db: Database):
    detector = SourceSupportDetector(structured_llm_client=None)

    with pytest.raises(TypeError):
        await detector.detect_and_persist(
            doc_id="doc-support",
            source_type="jira",
            document="content",
            entity_ids=[],
            project_key="PAY",
            db=db,
            source_updated_at=None,
        )


@pytest.mark.asyncio
async def test_detect_and_persist_routes_corroborated_support_through_store(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-route-support", "Period lifecycle assignment transitions to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, str | None, str, datetime | None]] = []

        def operation_context(self, **fields):
            return None

        async def record_audit_event(self, *args, **kwargs) -> None:
            return None

        async def add_source_support(
            self,
            memory_id: str,
            doc_id: str,
            source_type: str,
            excerpt: str | None = None,
            *,
            support_kind: str = "extracted",
            context=None,
            writer_visibility: str | None = None,
            writer_owner_user_id: str | None = None,
            writer_project_key: str | None = None,
            source_updated_at: datetime | None = None,
            relation_outcome=None,
        ) -> str:
            assert relation_outcome is not None
            await db.record_relation_outcome_bundle(relation_outcome)
            self.calls.append((memory_id, doc_id, source_type, excerpt, support_kind, source_updated_at))
            return "inserted"

        async def remove_source_support(self, memory_id: str, doc_id: str, reason: str = "no_support") -> bool:
            return False

    excerpt = "The assignment transitions to ASSIGNED."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "same rule"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)
    store = RecordingStore()
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=store,  # type: ignore[arg-type]
        source_updated_at=source_updated_at,
    )

    assert result["added"] == 1
    assert store.calls == [(memory.id, "doc-support", "confluence", excerpt, "corroborated", source_updated_at)]

    async with db.db.execute(
        """SELECT rr.*
           FROM relation_runs rr
           JOIN evidence_relations er ON er.relation_run_id = rr.id
           WHERE er.memory_id = ?
           ORDER BY rr.started_at""",
        (memory.id,),
    ) as cursor:
        relation_runs = [dict(row) async for row in cursor]
    assert len(relation_runs) == 1
    assert relation_runs[0]["lifecycle_action"] == LifecycleAction.ATTACH_SUPPORT.value
    assert relation_runs[0]["result_memory_id"] == memory.id
    evidence_unit = await db.get_evidence_unit(relation_runs[0]["evidence_unit_id"])
    assert evidence_unit is not None
    assert evidence_unit.doc_id == "doc-support"
    assert evidence_unit.source_type == "confluence"
    assert evidence_unit.excerpt == excerpt
    relations = await db.get_evidence_relations(evidence_unit.id)
    assert [(relation.memory_id, relation.relation_type) for relation in relations] == [
        (memory.id, RelationType.SUPPORTS)
    ]
    candidates = await db.get_relation_candidates(relation_runs[0]["id"])
    assert [(candidate.memory_id, candidate.bucket, candidate.was_checked) for candidate in candidates] == [
        (memory.id, CandidateBucket.SHARED_ENTITIES, True)
    ]

    run_id = relation_runs[0]["id"]
    assert await db.remove_memory_source(memory.id, "doc-support") is False
    assert await db.get_relation_run(run_id) is None
    assert await db.get_evidence_unit(evidence_unit.id) is None

    retry_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "same rule"},
                ]
            )
        ]
    )
    retry_detector = SourceSupportDetector(structured_llm_client=retry_client)
    retry_store = RecordingStore()

    retry_result = await retry_detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=retry_store,  # type: ignore[arg-type],
        source_updated_at=None,
    )

    assert retry_result["added"] == 1
    assert await db.get_relation_run(run_id) is not None


@pytest.mark.asyncio
async def test_get_corroborated_sources_by_doc_maps_source_updated_at(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    memory = _memory("mem-corroborated-observed", "Supported memory.")
    entity_id = await db.upsert_entity("observed support", display_name="Observed Support", tags=["feature"])
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "support excerpt",
        support_kind="corroborated",
        source_updated_at=source_updated_at,
    )

    sources = await db.get_corroborated_sources_by_doc("doc-support")
    assert len(sources) == 1
    assert sources[0].source_updated_at == source_updated_at


@pytest.mark.asyncio
async def test_sqlite_add_memory_source_overwrites_source_updated_at_with_unknown(db: Database):
    await _insert_doc(db, "doc-origin")
    memory = _memory("mem-source-updated-clear", "Source timestamp can become unknown.")
    entity_id = await db.upsert_entity("observed clear", display_name="Observed Clear", tags=["feature"])
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    await db.add_memory_source(
        memory.id,
        "doc-origin",
        "confluence",
        "observed excerpt",
        source_updated_at=source_updated_at,
    )
    await db.add_memory_source(
        memory.id,
        "doc-origin",
        "confluence",
        "unknown excerpt",
        source_updated_at=None,
    )

    sources = await db.get_memory_sources(memory.id)
    assert len(sources) == 1
    assert sources[0].excerpt == "unknown excerpt"
    assert sources[0].source_updated_at is None


@pytest.mark.asyncio
async def test_sqlite_corroborate_memory_overwrites_source_updated_at_with_unknown(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    memory = _memory("mem-corroborate-observed-clear", "Corroborated timestamp can become unknown.")
    entity_id = await db.upsert_entity("corroborate clear", display_name="Corroborate Clear", tags=["feature"])
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    source_updated_at = datetime(2026, 6, 20, 4, 23, 51, tzinfo=timezone.utc)

    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "observed excerpt",
        support_kind="corroborated",
        source_updated_at=source_updated_at,
    )
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "unknown excerpt",
        support_kind="corroborated",
        source_updated_at=None,
    )

    sources = await db.get_memory_sources(memory.id)
    support_source = next(source for source in sources if source.doc_id == "doc-support")
    assert support_source.source_updated_at is None


@pytest.mark.asyncio
async def test_source_support_candidates_exclude_disabled_private_sources(db: Database):
    await _insert_doc(db, "doc-origin", project="ORIGIN")
    await _insert_doc(db, "doc-support", project="PAY")
    await db.upsert_source(
        "src-ORIGIN",
        "confluence",
        "Origin",
        "{}",
        access_policy="private",
        owner_user_id="alice@example.com",
    )
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory(
        "mem-disabled-source-support",
        "Period lifecycle assignment transitions to ASSIGNED.",
        visibility="private",
        owner_user_id="alice@example.com",
    )
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.set_source_subscription("src-ORIGIN", "alice@example.com", enabled=False)

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {
                        "memory_id": memory.id,
                        "supported": True,
                        "excerpt": "The assignment transitions to ASSIGNED.",
                        "reason": "same rule",
                    },
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="The assignment transitions to ASSIGNED.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        writer_visibility="private",
        writer_owner_user_id="alice@example.com",
        writer_project_key="PAY",
        source_updated_at=None,
    )

    assert result["checked"] == 0
    assert result["added"] == 0
    assert structured_client.prompts == []


@pytest.mark.asyncio
async def test_source_support_workspace_candidates_ignore_personal_disabled_sources(db: Database):
    await _insert_doc(db, "doc-origin", project="PAY")
    await _insert_doc(db, "doc-support", project="PAY")
    await db.db.execute("UPDATE documents SET source = ? WHERE doc_id = ?", ("src-ORIGIN", "doc-origin"))
    await db.db.commit()
    await db.upsert_source(
        "src-ORIGIN",
        "confluence",
        "Origin",
        "{}",
        access_policy="workspace",
        owner_user_id="alice@example.com",
    )
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-workspace-source-support", "Period lifecycle assignment transitions to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.set_source_subscription("src-ORIGIN", "alice@example.com", enabled=False)

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {
                        "memory_id": memory.id,
                        "supported": True,
                        "excerpt": "The assignment transitions to ASSIGNED.",
                        "reason": "same workspace rule",
                    },
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="The assignment transitions to ASSIGNED.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        writer_visibility="workspace",
        writer_owner_user_id="alice@example.com",
        writer_project_key="PAY",
        source_updated_at=None,
    )

    assert result["checked"] == 1
    assert result["added"] == 1
    assert structured_client.prompts


@pytest.mark.asyncio
async def test_rejected_source_support_does_not_record_support_relation(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-rejected-support", "Period lifecycle assignment transitions to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    class RejectingStore:
        def operation_context(self, **fields):
            return None

        async def record_audit_event(self, *args, **kwargs) -> None:
            return None

        async def add_source_support(self, *args, **kwargs) -> str:
            return "rejected"

        async def remove_source_support(self, memory_id: str, doc_id: str, reason: str = "no_support") -> bool:
            return False

    excerpt = "The assignment transitions to ASSIGNED."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "same rule"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=RejectingStore(),  # type: ignore[arg-type],
        source_updated_at=None,
    )

    async with db.db.execute(
        "SELECT COUNT(*) FROM evidence_relations WHERE memory_id = ?",
        (memory.id,),
    ) as cursor:
        relation_count = (await cursor.fetchone())[0]
    assert result["added"] == 0
    assert result["updated"] == 0
    assert result["skipped"] == 1
    assert relation_count == 0


@pytest.mark.asyncio
async def test_existing_support_refresh_routes_through_store(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("refresh support", display_name="Refresh Support", tags=["feature"])
    memory = _memory("mem-refresh-support", "Existing support refresh stays audited.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "old excerpt",
        support_kind="corroborated",
        source_updated_at=None,
    )

    class RecordingStore:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, str | None, str, datetime | None]] = []

        def operation_context(self, **fields):
            return None

        async def record_audit_event(self, *args, **kwargs) -> None:
            return None

        async def add_source_support(
            self,
            memory_id: str,
            doc_id: str,
            source_type: str,
            excerpt: str | None = None,
            *,
            support_kind: str = "extracted",
            context=None,
            writer_visibility: str | None = None,
            writer_owner_user_id: str | None = None,
            writer_project_key: str | None = None,
            source_updated_at: datetime | None = None,
            relation_outcome=None,
        ) -> str:
            assert relation_outcome is not None
            await db.record_relation_outcome_bundle(relation_outcome)
            self.calls.append((memory_id, doc_id, source_type, excerpt, support_kind, source_updated_at))
            return "updated"

        async def remove_source_support(self, memory_id: str, doc_id: str, reason: str = "no_support") -> bool:
            return False

    excerpt = "Existing support refresh stays audited."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "same rule"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)
    store = RecordingStore()
    source_updated_at = datetime(2026, 6, 21, 5, 0, tzinfo=timezone.utc)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=store,  # type: ignore[arg-type]
        source_updated_at=source_updated_at,
    )

    assert result["updated"] == 1
    assert store.calls == [(memory.id, "doc-support", "confluence", excerpt, "corroborated", source_updated_at)]


@pytest.mark.asyncio
async def test_support_detection_adds_corroborated_source_with_validated_excerpt(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-support1", "Period lifecycle assignment must transition to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    excerpt = "When the assignment is confirmed, the period lifecycle assignment transitions to ASSIGNED."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "same rule"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=f"# Support Doc\n\n{excerpt}",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    support = [source for source in sources if source.doc_id == "doc-support"][0]
    assert result["added"] == 1
    assert stored.corroboration_count == 2
    assert support.support_kind == "corroborated"
    assert support.excerpt == excerpt


@pytest.mark.asyncio
async def test_support_detection_audits_invalid_excerpt_rejection(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-support-invalid", "Period lifecycle assignment must transition to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": "not in document", "reason": "same rule"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="The document says something else.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_audited_memory_store(db),
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert result["skipped"] == 1
    assert "source_support_rejected" in {row.event_type for row in audit_rows}
    assert {row.reason for row in audit_rows if row.event_type == "source_support_rejected"} == {"invalid_excerpt"}


@pytest.mark.asyncio
async def test_existing_support_refresh_audits_unsupported_decision(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("refresh support", display_name="Refresh Support", tags=["feature"])
    memory = _memory("mem-refresh-malformed", "Existing support refresh stays audited.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "old excerpt",
        support_kind="corroborated",
        source_updated_at=None,
    )
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {
                        "memory_id": memory.id,
                        "supported": False,
                        "excerpt": "Existing support refresh stays audited.",
                        "reason": "not directly supported",
                    },
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="Existing support refresh stays audited.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_audited_memory_store(db),
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    assert result["removed_stale"] == 1
    assert "source_support_rejected" in {row.event_type for row in audit_rows}
    assert {row.reason for row in audit_rows if row.event_type == "source_support_rejected"} == {"unsupported"}


@pytest.mark.asyncio
async def test_support_detection_audits_verified_support_with_model_and_reason(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("period lifecycle", display_name="Period Lifecycle", tags=["feature"])
    memory = _memory("mem-support-verified", "Period lifecycle assignment must transition to ASSIGNED.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    excerpt = "The period lifecycle assignment must transition to ASSIGNED before release."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "direct statement"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client, llm_model="claude-test")

    await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_audited_memory_store(db),
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(memory_id=memory.id)
    verified = [row for row in audit_rows if row.event_type == "source_support_verified"]
    assert len(verified) == 1
    assert verified[0].model == "claude-test"
    assert verified[0].prompt_hash
    assert verified[0].reason == "direct statement"
    assert verified[0].evidence_refs == [{"excerpt": excerpt}]


@pytest.mark.asyncio
async def test_support_detection_audits_structured_llm_failure(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("llm support", display_name="LLM Support", tags=["feature"])
    memory = _memory("mem-support-llmfail", "LLM failures are auditable.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    structured_client = FakeStructuredSupportClient(error=StructuredLlmError("structured unavailable"))
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="LLM failures are auditable.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_audited_memory_store(db),
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(event_type="source_support_verification_failed")
    sources = await db.get_memory_sources(memory.id)
    assert result["skipped"] == 0
    assert [(row.doc_id, row.reason, row.payload_class, row.payload, row.error) for row in audit_rows] == [
        ("doc-support", "candidate_support", "llm_response_error", {}, "structured unavailable")
    ]
    assert [source.doc_id for source in sources] == ["doc-origin"]


@pytest.mark.asyncio
async def test_support_detection_audits_missing_structured_client_and_writes_no_support(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("structured missing", display_name="Structured Missing", tags=["feature"])
    memory = _memory("mem-support-no-structured", "Structured verifier must be available.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)

    detector = SourceSupportDetector(structured_llm_client=None)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="Structured verifier must be available.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_audited_memory_store(db),
        source_updated_at=None,
    )

    audit_rows = await db.list_memory_audit_events(event_type="source_support_verification_failed")
    sources = await db.get_memory_sources(memory.id)
    assert result["added"] == 0
    assert result["updated"] == 0
    assert [(row.doc_id, row.error) for row in audit_rows] == [
        ("doc-support", "structured source-support LLM unavailable")
    ]
    assert [source.doc_id for source in sources] == ["doc-origin"]


@pytest.mark.asyncio
async def test_support_detection_reprocessing_updates_better_excerpt_without_increment(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("payroll group", display_name="Payroll Group", tags=["feature"])
    memory = _memory("mem-support2", "Off-cycle payroll group creation checks cutoff state first.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "jira",
        "cutoff state checked first",
        support_kind="corroborated",
        source_updated_at=None,
    )

    better_excerpt = "The backend checks cutoff state before name duplication when creating an off-cycle payroll group."
    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": better_excerpt, "reason": "more specific"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="jira",
        document=better_excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    support = [source for source in sources if source.doc_id == "doc-support"][0]
    assert result["updated"] == 1
    assert stored.corroboration_count == 2
    assert support.excerpt == better_excerpt


@pytest.mark.asyncio
async def test_support_detection_removes_stale_corroborated_support_on_document_update(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("adaptive scheduling", display_name="Project Payroll", tags=["feature"])
    memory = _memory("mem-support3", "Project Payroll supports on-demand correction groups.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "on-demand correction groups are supported",
        support_kind="corroborated",
        source_updated_at=None,
    )

    detector = SourceSupportDetector(structured_llm_client=None)
    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="This updated document no longer contains the old supporting excerpt.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    assert result["removed_stale"] == 1
    assert stored.status == "active"
    assert stored.corroboration_count == 1
    assert [source.doc_id for source in sources] == ["doc-origin"]


@pytest.mark.asyncio
async def test_support_detection_cleans_indexes_when_last_support_is_removed(db: Database):
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("last support", display_name="Last Support", tags=["feature"])
    memory = _memory("mem-last-support", "Last support removal should hide the memory from search.")
    await _seed_memory(db, memory, doc_id="doc-support", entity_id=entity_id, support_kind="corroborated")
    collection = FakeCollection()

    detector = SourceSupportDetector(structured_llm_client=None)
    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="jira",
        document="The updated document no longer supports the memory.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db, collection),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    async with db.db.execute(
        "SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?",
        (memory.id,),
    ) as cursor:
        fts_count = (await cursor.fetchone())[0]
    assert result["removed_stale"] == 1
    assert stored.status == "retired"
    assert fts_count == 0
    assert collection.deleted == [memory.id]


@pytest.mark.asyncio
async def test_stale_support_removal_reuses_source_support_operation_context(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("adaptive scheduling", display_name="Project Payroll", tags=["feature"])
    memory = _memory("mem-support-context", "Project Payroll supports correction groups.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "confluence",
        "correction groups are supported",
        support_kind="corroborated",
        source_updated_at=None,
    )

    class RecordingStore:
        def __init__(self) -> None:
            self.context = RecordingContext(operation_id="op-source-support")
            self.removals: list[tuple[str, str, object | None]] = []

        def operation_context(self, **fields):
            return self.context

        async def record_audit_event(self, *args, **kwargs) -> None:
            return None

        async def add_source_support(self, *args, **kwargs) -> str:
            return "unchanged"

        async def remove_source_support(
            self,
            memory_id: str,
            doc_id: str,
            reason: str = "no_support",
            *,
            context=None,
        ) -> bool:
            self.removals.append((memory_id, doc_id, context))
            return True

    store = RecordingStore()
    detector = SourceSupportDetector(structured_llm_client=None)

    await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document="This updated document no longer supports the old excerpt.",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=store,  # type: ignore[arg-type],
        source_updated_at=None,
    )

    assert store.removals == [(memory.id, "doc-support", store.context)]


@pytest.mark.asyncio
async def test_support_detection_removes_existing_support_when_verifier_says_unsupported(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("cutoff", display_name="Cutoff", tags=["feature"])
    memory = _memory("mem-support-false", "Cutoff validation runs before payroll group creation.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    old_excerpt = "Cutoff validation runs before payroll group creation."
    await db.corroborate_memory(
        memory.id,
        "doc-support",
        "jira",
        old_excerpt,
        support_kind="corroborated",
        source_updated_at=None,
    )

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {
                        "memory_id": memory.id,
                        "supported": False,
                        "excerpt": old_excerpt,
                        "reason": "context no longer matches",
                    },
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="jira",
        document=f"The previous text still appears: {old_excerpt}",
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    sources = await db.get_memory_sources(memory.id)
    assert result["removed_stale"] == 1
    assert stored.status == "active"
    assert stored.corroboration_count == 1
    assert [source.doc_id for source in sources] == ["doc-origin"]


@pytest.mark.asyncio
async def test_existing_corroborated_support_is_revalidated_before_new_candidates(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("ranking", display_name="Ranking", tags=["feature"])
    existing = _memory("mem-existing-support", "Ranking keeps existing corroborated rows in the verifier batch.")
    await _seed_memory(db, existing, doc_id="doc-origin", entity_id=entity_id)
    await db.corroborate_memory(
        existing.id,
        "doc-support",
        "confluence",
        "old ranking excerpt",
        support_kind="corroborated",
        source_updated_at=None,
    )
    for i in range(3):
        await _insert_doc(db, f"doc-origin-{i}")
        candidate = _memory(f"mem-new-candidate-{i}", f"Candidate memory {i} shares the same entity.")
        await _seed_memory(db, candidate, doc_id=f"doc-origin-{i}", entity_id=entity_id)

    updated_excerpt = "Ranking keeps existing corroborated rows in the verifier batch."

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {
                        "memory_id": existing.id,
                        "supported": True,
                        "excerpt": updated_excerpt,
                        "reason": "still supported",
                    },
                ]
            ),
            SourceSupportResponse(decisions=[]),
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client, max_candidates=1)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="confluence",
        document=updated_excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    support = [source for source in await db.get_memory_sources(existing.id) if source.doc_id == "doc-support"][0]
    assert result["updated"] == 1
    assert support.excerpt == updated_excerpt
    assert existing.id in structured_client.prompts[0]


@pytest.mark.asyncio
async def test_corroborated_support_does_not_participate_in_same_document_reconciliation(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("validation", display_name="Validation", tags=["feature"])
    extracted = _memory("mem-extracted", "Validation A is mandatory.")
    corroborated = _memory("mem-corroborated", "Validation B is mandatory.")
    await _seed_memory(db, extracted, doc_id="doc-support", entity_id=entity_id, support_kind="extracted")
    await _seed_memory(db, corroborated, doc_id="doc-origin", entity_id=entity_id, support_kind="extracted")
    await db.corroborate_memory(
        corroborated.id,
        "doc-support",
        "confluence",
        "Validation B is mandatory.",
        support_kind="corroborated",
        source_updated_at=None,
    )

    existing = await db.get_memories_by_source_doc("doc-support")

    assert [memory.id for memory in existing] == [extracted.id]


@pytest.mark.asyncio
async def test_support_detection_rejects_link_only_excerpt(db: Database):
    await _insert_doc(db, "doc-origin")
    await _insert_doc(db, "doc-support")
    entity_id = await db.upsert_entity("sonarqube", display_name="SonarQube", tags=["technology"])
    memory = _memory("mem-support4", "SonarQube major issues were resolved.")
    await _seed_memory(db, memory, doc_id="doc-origin", entity_id=entity_id)
    excerpt = "https://sonar.example.test/project/issues?id=payroll-processing"

    structured_client = FakeStructuredSupportClient(
        [
            _support_response(
                [
                    {"memory_id": memory.id, "supported": True, "excerpt": excerpt, "reason": "link"},
                ]
            )
        ]
    )
    detector = SourceSupportDetector(structured_llm_client=structured_client)

    result = await detector.detect_and_persist(
        doc_id="doc-support",
        source_type="jira",
        document=excerpt,
        entity_ids=[entity_id],
        project_key="PAY",
        db=db,
        memory_store=_memory_store(db),
        source_updated_at=None,
    )

    stored = await db.get_memory(memory.id)
    assert result["added"] == 0
    assert stored.corroboration_count == 1
