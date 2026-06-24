"""Tests for pre-persistence memory quality filtering and provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.engine import MemoryEngine
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, RawMemory, ReconcileAction, ReconcileOperation, content_hash
from memforge.storage.database import Database
from memforge.storage.adapters.sqlite import build_sqlite_adapters


METADATA_CONTENT = (
    "The ACD document 'Payroll Processing V2 - Project Payroll' in PAY space was authored by "
    "Sun, Youpeng, has document status 'Greenliving', and was last modified on 2026-05-14."
)
METADATA_CONTEXT = "Author: Sun, Youpeng ... Document Status | Greenliving ... Last modified: 2026-05-14"

LINK_CONTENT = (
    "The ACD 'Payroll Processing V2 - Project Payroll' links to the Payroll Processing concept at: "
    "https://github.example/Payroll%20Processing.md"
)
LINK_CONTEXT = "Link to Concept | https://github.example/Payroll%20Processing.md"

OPEN_QUESTION_CONTENT = (
    "Synchronous checks executed at request time should be considered for full repetition in the "
    "asynchronous processing phase."
)
OPEN_QUESTION_CONTEXT = (
    "we should bear it in mind and discuss whether all the synchronous checks would be fully repeated"
)

CONDITIONAL_RULE_CONTENT = (
    "If an employee's regular pay date is changed via a deviating payroll process and the employee is "
    "assigned to an on-demand AP group, the out-of-sequence validation should be repeated."
)
CONDITIONAL_RULE_CONTEXT = (
    "if the regular pay date ... has been changed via a deviating payroll process, the same validation "
    "should be repeated"
)


class DirectInsertStore:
    """Tiny MemoryStore stand-in that exercises MemoryEngine without embeddings."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def deduplicate_and_insert(
        self,
        memory: Memory,
        doc_id: str,
        source_type: str,
        entity_ids: list[int] | None = None,
        excerpt: str | None = None,
        *,
        source_observed_at: datetime | None,
        relation_outcome=None,
    ) -> str:
        await self.db.insert_memory(memory)
        if relation_outcome is not None:
            await self.db.record_relation_outcome_bundle(relation_outcome)
        return "inserted"


class FailingUpdateAuditStore(DirectInsertStore):
    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.audit_events: list[tuple[str, str, dict]] = []

    def operation_context(self, **fields):
        return None

    async def record_audit_event(self, event_type: str, status: str, **fields) -> None:
        self.audit_events.append((event_type, status, fields))

    async def update_memory(self, *args, **kwargs) -> None:
        raise RuntimeError("update failed")


class FakeCollection:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, ids):
        self.deleted.extend(ids)


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(str(tmp_path / "memory-quality.db"))
    await database.connect()
    yield database
    await database.close()


def _raw(content: str, context: str) -> RawMemory:
    return RawMemory(
        content=content,
        memory_type="fact",
        confidence=0.9,
        entity_refs=[],
        tags=["payroll"],
        extraction_context=context,
    )


def _config(tmp_path: Path) -> AppConfig:
    return AppConfig(base_dir=tmp_path / "memforge")


async def _insert_document(
    db: Database,
    *,
    doc_id: str = "doc-acd",
    raw_content_uri: str | None = "/tmp/source.raw",
    normalized_content_uri: str | None = "/tmp/source.md",
    pdf_content_uri: str | None = None,
) -> DocumentRecord:
    now = datetime.now(timezone.utc)
    doc = DocumentRecord(
        doc_id=doc_id,
        source="src-confluence",
        source_url=f"https://confluence.example/{doc_id}",
        title="Payroll Processing V2",
        space_or_project="PAY",
        author="Sun, Youpeng",
        last_modified=now,
        labels=[],
        version="1",
        content_hash=f"hash-{doc_id}",
        token_count=100,
        raw_content_uri=raw_content_uri,
        raw_content_type="text/html",
        normalized_content_uri=normalized_content_uri,
        pdf_content_uri=pdf_content_uri,
        last_synced=now,
    )
    await db.upsert_document(doc)
    return doc


async def _insert_memory(db: Database, *, mem_id: str, content: str) -> Memory:
    now = datetime.now(timezone.utc)
    memory = Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        tags=["payroll"],
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
    )
    await db.insert_memory(memory)
    return memory


async def _fts_has_memory(db: Database, memory_id: str) -> bool:
    async with db.db.execute(
        "SELECT 1 FROM memories_fts WHERE memory_id = ?",
        (memory_id,),
    ) as cursor:
        return await cursor.fetchone() is not None


def test_classifier_skips_document_metadata_candidate():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(METADATA_CONTENT, METADATA_CONTEXT))

    assert quality.keep is False
    assert quality.skip_reason == "metadata_only"


def test_classifier_skips_reference_only_link_list_candidate():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(LINK_CONTENT, LINK_CONTEXT))

    assert quality.keep is False
    assert quality.skip_reason == "reference_only"


def test_classifier_skips_unresolved_design_question():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(OPEN_QUESTION_CONTENT, OPEN_QUESTION_CONTEXT))

    assert quality.keep is False
    assert quality.skip_reason == "open_question"


def test_classifier_skips_memory_system_narration():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "MemForge memories are loaded at SessionStart and used as warm context for l3-demo.",
            "",
        )
    )

    assert quality.keep is False
    assert quality.skip_reason == "self_referential"


def test_classifier_skips_candidate_citing_internal_memory_id():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "Prefer sum() over manual accumulator loops (project convention, mem-a2229a2c).",
            "",
        )
    )

    assert quality.keep is False
    assert quality.skip_reason == "self_referential"


def test_classifier_keeps_conditional_ap_rule():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(CONDITIONAL_RULE_CONTENT, CONDITIONAL_RULE_CONTEXT))

    assert quality.keep is True
    assert quality.skip_reason is None


def test_classifier_keeps_useful_memory_with_link_list_context():
    from memforge.memory.quality import classify_memory_candidate

    raw = _raw(
        "The on-demand AP group follows the Payroll Processing concept for out-of-sequence validation.",
        (
            "Link to Concept | https://github.example/Payroll%20Processing.md "
            "The on-demand AP group follows the Payroll Processing concept for validation."
        ),
    )

    quality = classify_memory_candidate(raw)

    assert quality.keep is True
    assert quality.skip_reason is None


@pytest.mark.asyncio
async def test_engine_skips_metadata_only_candidate(db: Database):
    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = MemoryEngine(
        relational=adapters.relational, vector=adapters.vector, db=db, memory_store=DirectInsertStore(db)
    )

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(METADATA_CONTENT, METADATA_CONTEXT)],
        source_type="confluence",
        source_observed_at=None,
    )

    assert stats == {"inserted": 0, "corroborated": 0, "skipped": 1}
    assert await db.count_memories() == 0


@pytest.mark.asyncio
async def test_engine_skips_open_question_candidate(db: Database):
    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = MemoryEngine(
        relational=adapters.relational, vector=adapters.vector, db=db, memory_store=DirectInsertStore(db)
    )

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(OPEN_QUESTION_CONTENT, OPEN_QUESTION_CONTEXT)],
        source_type="confluence",
        source_observed_at=None,
    )

    assert stats == {"inserted": 0, "corroborated": 0, "skipped": 1}
    assert await db.count_memories() == 0


@pytest.mark.asyncio
async def test_engine_keeps_conditional_ap_rule(db: Database):
    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = MemoryEngine(
        relational=adapters.relational, vector=adapters.vector, db=db, memory_store=DirectInsertStore(db)
    )

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(CONDITIONAL_RULE_CONTENT, CONDITIONAL_RULE_CONTEXT)],
        source_type="confluence",
        source_observed_at=None,
    )

    memories = await db.list_memories()
    assert stats == {"inserted": 1, "corroborated": 0, "skipped": 0}
    assert len(memories) == 1
    assert memories[0].content == CONDITIONAL_RULE_CONTENT


@pytest.mark.asyncio
async def test_reconciliation_skips_bad_replacement_candidate_instead_of_superseding(db: Database, monkeypatch):
    doc = await _insert_document(db, doc_id="doc-acd")
    old_memory = await _insert_memory(
        db,
        mem_id="mem-oldgood",
        content="Payroll Processing V2 uses the Payroll Processing concept as its reference design.",
    )
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence", source_observed_at=None)
    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=DirectInsertStore(db),
        structured_llm_client=object(),
    )
    good_extraction = _raw("Payroll Processing V2 validates changed regular pay dates.", "accepted rule")
    bad_replacement = _raw(LINK_CONTENT, LINK_CONTEXT)

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.SUPERSEDE,
                memory_id=old_memory.id,
                memory=bad_replacement,
                reason="Bad replacement from a link-list row",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[good_extraction],
        source_type="confluence",
        doc_type="design-doc",
        source_observed_at=None,
    )

    stored_old = await db.get_memory(old_memory.id)
    assert stats["superseded"] == 0
    assert stats["skipped"] == 1
    assert stored_old.status == "active"
    assert await db.count_memories() == 1


@pytest.mark.asyncio
async def test_reconciliation_action_failure_is_audited_without_fallback(db: Database, monkeypatch):
    doc = await _insert_document(db, doc_id="doc-fallback")
    old_memory = await _insert_memory(db, mem_id="mem-fallback-old", content="Old fact")
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence", source_observed_at=None)
    store = FailingUpdateAuditStore(db)
    adapters = build_sqlite_adapters(db, FakeCollection())
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )
    replacement = _raw("New fact", "new excerpt")

    async def fake_reconcile_memories(**kwargs):
        return [
            ReconcileOperation(
                action=ReconcileAction.UPDATE,
                memory_id=old_memory.id,
                memory=replacement,
                reason="refresh",
            )
        ]

    monkeypatch.setattr("memforge.pipeline.reconciler.reconcile_memories", fake_reconcile_memories)

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[replacement],
        source_type="confluence",
        doc_type="design-doc",
        source_observed_at=None,
    )

    assert stats["added"] == 0
    assert stats["skipped"] == 1
    assert await db.count_memories() == 1
    assert [event[0] for event in store.audit_events] == [
        "reconciliation_decision_returned",
        "reconciliation_action_failed",
    ]
    assert store.audit_events[0][2]["memory_id"] == old_memory.id
    assert store.audit_events[1][2]["memory_id"] == old_memory.id


@pytest.mark.asyncio
async def test_reconciliation_all_filtered_update_retires_sole_source_memory(db: Database):
    doc = await _insert_document(db, doc_id="doc-acd")
    old_memory = await _insert_memory(
        db,
        mem_id="mem-sole001",
        content="Payroll Processing V2 repeats AP validation after changed regular pay dates.",
    )
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence", source_observed_at=None)
    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[_raw(LINK_CONTENT, LINK_CONTEXT)],
        source_type="confluence",
        doc_type="design-doc",
        source_observed_at=None,
    )

    stored_old = await db.get_memory(old_memory.id)
    assert stats["skipped"] == 1
    assert stored_old.status == "retired"
    assert stored_old.retirement_reason == "no_support"
    assert await db.get_memory_sources(old_memory.id) == []
    assert await _fts_has_memory(db, old_memory.id) is False
    assert collection.deleted == [old_memory.id]


@pytest.mark.asyncio
async def test_reconciliation_all_filtered_update_removes_one_source_but_keeps_supported_memory_active(db: Database):
    doc = await _insert_document(db, doc_id="doc-acd")
    other_doc = await _insert_document(db, doc_id="doc-runbook")
    old_memory = await _insert_memory(
        db,
        mem_id="mem-supported",
        content="Payroll Processing V2 repeats AP validation after changed regular pay dates.",
    )
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence", source_observed_at=None)
    await db.add_memory_source(old_memory.id, other_doc.doc_id, "confluence", source_observed_at=None)
    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )
    engine = MemoryEngine(
        relational=adapters.relational,
        vector=adapters.vector,
        db=db,
        memory_store=store,
        structured_llm_client=object(),
    )

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[_raw(LINK_CONTENT, LINK_CONTEXT)],
        source_type="confluence",
        doc_type="design-doc",
        source_observed_at=None,
    )

    stored_old = await db.get_memory(old_memory.id)
    remaining_sources = await db.get_memory_sources(old_memory.id)
    assert stats["skipped"] == 1
    assert stored_old.status == "active"
    assert stored_old.corroboration_count == 1
    assert [source.doc_id for source in remaining_sources] == [other_doc.doc_id]
    assert await _fts_has_memory(db, old_memory.id) is True
    assert collection.deleted == []


@pytest.mark.asyncio
async def test_store_document_delete_cleans_indexes_for_last_corroborated_source(db: Database):
    doc = await _insert_document(db, doc_id="doc-support")
    memory = await _insert_memory(
        db,
        mem_id="mem-corrob-last",
        content="A corroborated source can be the last valid source support.",
    )
    await db.add_memory_source(
        memory.id,
        doc.doc_id,
        "jira",
        excerpt="A corroborated source can be the last valid source support.",
        support_kind="corroborated",
        source_observed_at=None,
    )
    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )

    retired_ids = await store.delete_document(doc.doc_id)

    stored = await db.get_memory(memory.id)
    assert retired_ids == [memory.id]
    assert stored.status == "retired"
    assert stored.corroboration_count == 0
    assert await _fts_has_memory(db, memory.id) is False
    assert collection.deleted == [memory.id]


@pytest.mark.asyncio
async def test_store_source_cascade_cleans_indexes_for_retired_memories(db: Database):
    doc = await _insert_document(db, doc_id="doc-source-delete")
    memory = await _insert_memory(
        db,
        mem_id="mem-source-delete",
        content="A source cascade should remove retired memories from search.",
    )
    await db.add_memory_source(
        memory.id,
        doc.doc_id,
        "confluence",
        excerpt="A source cascade should remove retired memories from search.",
        source_observed_at=None,
    )
    collection = FakeCollection()
    adapters = build_sqlite_adapters(db, collection)
    store = MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
    )

    retired_ids = await store.delete_source_cascade(doc.source)

    stored = await db.get_memory(memory.id)
    assert retired_ids == [memory.id]
    assert stored.status == "retired"
    assert await _fts_has_memory(db, memory.id) is False
    assert collection.deleted == [memory.id]


def test_memory_extraction_prompt_rejects_metadata_and_preserves_modality():
    from memforge.pipeline.memory_extractor import MEMORY_EXTRACTION_PROMPT

    prompt = MEMORY_EXTRACTION_PROMPT.lower()

    assert "do not extract document metadata" in prompt
    assert "author" in prompt
    assert "last modified" in prompt
    assert "document status" in prompt
    assert "link list" in prompt
    assert "preserve conditional language" in prompt
    assert "do not turn open questions into decisions" in prompt
    assert "agent_session" in prompt
    assert "validation commands" in prompt
    assert "runtime notes" in prompt
    assert "local paths" in prompt


def test_memory_change_extraction_prompt_rejects_operational_metadata_changes():
    from memforge.pipeline.memory_extractor import MEMORY_CHANGE_EXTRACTION_PROMPT

    prompt = MEMORY_CHANGE_EXTRACTION_PROMPT.lower()

    assert "operational metadata" in prompt
    assert "status" in prompt
    assert "assignee" in prompt
    assert "sprint" in prompt
    assert "timestamps" in prompt
    assert '"memories": []' in prompt
    assert "only removes old durable knowledge" in prompt
    assert "reconciliation will decide whether to retire the old memory" in prompt
    assert "do not create memories about the edit itself" in prompt
    assert "sender name and timestamp prefix" in prompt


def test_memory_extraction_prompt_preserves_weak_reference_relationships():
    from memforge.pipeline.memory_extractor import MEMORY_EXTRACTION_PROMPT

    prompt = MEMORY_EXTRACTION_PROMPT.lower()

    assert "reference/link-only evidence" in prompt
    assert "preserve the weaker relationship exactly as stated" in prompt
    assert "do not infer" in prompt


@pytest.mark.asyncio
async def test_admin_memory_detail_exposes_service_artifact_urls_only(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    docs_dir = tmp_path / "memforge" / "documents"
    docs_dir.mkdir(parents=True)
    source_pdf = docs_dir / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n")

    doc = await _insert_document(
        db,
        doc_id="doc-pdf-uri",
        pdf_content_uri=str(source_pdf),
    )
    memory = await _insert_memory(
        db,
        mem_id="mem-pdfuri1",
        content="Payroll Processing V2 supports adaptive scheduling adjustments.",
    )
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt", source_observed_at=None)

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get(f"/api/memories/{memory.id}")

    assert response.status_code == 200
    source = response.json()["sources"][0]
    assert source["content_url"] is None
    assert source["pdf_url"] == "/api/documents/doc-pdf-uri/pdf"
    assert "file_uri" not in source
    assert "pdf_uri" not in source


@pytest.mark.asyncio
async def test_admin_document_artifact_urls_serve_docker_safe_content(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    docs_dir = tmp_path / "memforge" / "documents"
    docs_dir.mkdir(parents=True)
    source_md = docs_dir / "source.md"
    source_pdf = docs_dir / "source.pdf"
    source_md.write_text("# Source\n\nDurable memory evidence.", encoding="utf-8")
    source_pdf.write_bytes(b"%PDF-1.4\n%memforge\n")

    doc = await _insert_document(
        db,
        doc_id="doc-artifact-url",
        normalized_content_uri=str(source_md),
        pdf_content_uri=str(source_pdf),
    )
    memory = await _insert_memory(
        db,
        mem_id="mem-artifact-url",
        content="Payroll Processing V2 keeps source artifacts available through the service.",
    )
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt", source_observed_at=None)

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        detail = client.get(f"/api/memories/{memory.id}")
        manifest = client.get("/api/documents/doc-artifact-url/artifacts")
        markdown_artifact = client.get("/api/documents/doc-artifact-url/artifacts/normalized_markdown")
        pdf_artifact = client.get("/api/documents/doc-artifact-url/artifacts/pdf")
        pdf_head = client.head("/api/documents/doc-artifact-url/artifacts/pdf")
        missing_artifact = client.get("/api/documents/doc-artifact-url/artifacts/raw_source")
        missing_document = client.get("/api/documents/missing-doc/artifacts")
        content = client.get("/api/documents/doc-artifact-url/content")
        pdf = client.get("/api/documents/doc-artifact-url/pdf")

    assert detail.status_code == 200
    source = detail.json()["sources"][0]
    assert source["content_url"] == "/api/documents/doc-artifact-url/content"
    assert source["pdf_url"] == "/api/documents/doc-artifact-url/pdf"
    assert "file_uri" not in source
    assert "pdf_uri" not in source
    assert manifest.status_code == 200
    artifacts = manifest.json()["artifacts"]
    assert artifacts["normalized_markdown"]["url"] == ("/api/documents/doc-artifact-url/artifacts/normalized_markdown")
    assert artifacts["pdf"]["url"] == "/api/documents/doc-artifact-url/artifacts/pdf"
    assert markdown_artifact.status_code == 200
    assert markdown_artifact.text == "# Source\n\nDurable memory evidence."
    assert pdf_artifact.status_code == 200
    assert pdf_artifact.content == b"%PDF-1.4\n%memforge\n"
    assert pdf_head.status_code == 200
    assert missing_artifact.status_code == 404
    assert missing_document.status_code == 404
    assert content.status_code == 200
    assert content.text == "# Source\n\nDurable memory evidence."
    assert pdf.status_code == 200
    assert pdf.content == b"%PDF-1.4\n%memforge\n"


@pytest.mark.asyncio
async def test_admin_document_content_alias_falls_back_to_raw_source(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    docs_dir = tmp_path / "memforge" / "documents"
    docs_dir.mkdir(parents=True)
    raw_source = docs_dir / "source.html"
    raw_source.write_text("<h1>Raw source</h1>", encoding="utf-8")

    await _insert_document(
        db,
        doc_id="doc-raw-artifact-url",
        raw_content_uri=str(raw_source),
        normalized_content_uri=None,
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        manifest = client.get("/api/documents/doc-raw-artifact-url/artifacts")
        raw_artifact = client.get("/api/documents/doc-raw-artifact-url/artifacts/raw_source")
        content = client.get("/api/documents/doc-raw-artifact-url/content")

    assert manifest.status_code == 200
    artifacts = manifest.json()["artifacts"]
    assert "normalized_markdown" not in artifacts
    assert artifacts["raw_source"]["url"] == "/api/documents/doc-raw-artifact-url/artifacts/raw_source"
    assert raw_artifact.status_code == 200
    assert raw_artifact.text == "<h1>Raw source</h1>"
    assert content.status_code == 200
    assert content.text == "<h1>Raw source</h1>"


@pytest.mark.asyncio
async def test_admin_document_artifacts_can_use_non_filesystem_store(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.document_store import StoredDocumentArtifact

    class MemoryBackedDocumentStore:
        def __init__(self):
            self.objects = {
                "mem://doc.md": (
                    b"# Durable source\n\nEvidence from a durable object store.",
                    "source.md",
                )
            }

        def get_artifact(self, uri: str | None, media_type: str):
            if uri not in self.objects:
                return None
            content, filename = self.objects[uri]
            return StoredDocumentArtifact(
                uri=uri,
                filename=filename,
                media_type=media_type,
                size_bytes=len(content),
            )

        def read_artifact(self, uri: str) -> bytes:
            return self.objects[uri][0]

        def read_normalized(self, stored_path: str) -> str | None:
            content = self.objects.get(stored_path)
            return content[0].decode("utf-8") if content else None

        def store_raw(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

        def store_normalized(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

        def store_pdf(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

        def delete_document_files(self, *, source_name: str, title: str) -> None:
            raise AssertionError("not used")

    await _insert_document(
        db,
        doc_id="doc-object-artifact-url",
        normalized_content_uri="mem://doc.md",
        raw_content_uri=None,
    )
    memory = await _insert_memory(
        db,
        mem_id="mem-object-artifact-url",
        content="A durable object artifact should be exposed through provenance URLs.",
    )
    await db.add_memory_source(
        memory.id, "doc-object-artifact-url", "jira", excerpt="source excerpt", source_observed_at=None
    )

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        document_store=MemoryBackedDocumentStore(),
    )
    with TestClient(app) as client:
        detail = client.get(f"/api/memories/{memory.id}")
        manifest = client.get("/api/documents/doc-object-artifact-url/artifacts")
        content = client.get("/api/documents/doc-object-artifact-url/content")

    assert detail.status_code == 200
    source = detail.json()["sources"][0]
    assert source["content_url"] == "/api/documents/doc-object-artifact-url/content"
    assert manifest.status_code == 200
    assert manifest.json()["artifacts"]["normalized_markdown"]["size_bytes"] == 55
    assert content.status_code == 200
    assert content.text == "# Durable source\n\nEvidence from a durable object store."


@pytest.mark.asyncio
async def test_admin_document_artifacts_reject_local_paths_outside_docs_root(
    db: Database,
    tmp_path: Path,
):
    from memforge.server.admin_api import create_admin_app

    outside = tmp_path / "outside-secret.md"
    outside.write_text("should not be served", encoding="utf-8")
    await _insert_document(
        db,
        doc_id="doc-outside-artifact-root",
        normalized_content_uri=str(outside),
        raw_content_uri=None,
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        manifest = client.get("/api/documents/doc-outside-artifact-root/artifacts")
        content = client.get("/api/documents/doc-outside-artifact-root/content")
        artifact = client.get("/api/documents/doc-outside-artifact-root/artifacts/normalized_markdown")

    assert manifest.status_code == 200
    assert manifest.json()["artifacts"] == {}
    assert content.status_code == 404
    assert artifact.status_code == 404


@pytest.mark.asyncio
async def test_delete_source_uses_injected_document_store(
    db: Database,
    tmp_path: Path,
    monkeypatch,
):
    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class RecordingDocumentStore:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str]] = []

        def delete_document_files(self, *, source_name: str, title: str) -> None:
            self.deleted.append((source_name, title))

        def get_artifact(self, uri, media_type):
            return None

        def read_artifact(self, uri: str) -> bytes:
            raise AssertionError("not used")

        def read_normalized(self, stored_path: str) -> str | None:
            return None

        def store_raw(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

        def store_normalized(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

        def store_pdf(self, *args, **kwargs) -> str:
            raise AssertionError("not used")

    class NoopMemoryStore:
        async def delete_source_cascade(self, source_id: str):
            return []

    async def fake_build_memory_store(*args, **kwargs):
        return NoopMemoryStore()

    monkeypatch.setattr(admin_api, "_build_memory_store", fake_build_memory_store)
    await db.upsert_source("src-confluence", "confluence", "Delete Route Source", "{}")
    await _insert_document(
        db,
        doc_id="doc-delete-route",
        raw_content_uri=None,
        normalized_content_uri="mem://doc.md",
    )

    store = RecordingDocumentStore()
    app = create_admin_app(db=db, config=_config(tmp_path), document_store=store)
    with TestClient(app) as client:
        response = client.delete("/api/sources/src-confluence")

    assert response.status_code == 200, response.text
    assert store.deleted == [("Delete Route Source", "Payroll Processing V2")]


def test_sync_previous_content_read_does_not_bypass_document_store(tmp_path: Path):
    from memforge.pipeline.sync import GeneSyncOrchestrator

    outside = tmp_path / "outside-previous.md"
    outside.write_text("previous content", encoding="utf-8")

    class RejectingDocumentStore:
        def read_normalized(self, stored_path: str) -> str | None:
            assert stored_path == str(outside)
            return None

    orchestrator = GeneSyncOrchestrator(
        db=object(),
        doc_store=RejectingDocumentStore(),
        enricher=object(),
        memory_extractor=object(),
        memory_engine=object(),
        memory_store=object(),
    )
    doc = DocumentRecord(
        doc_id="doc-previous-outside-root",
        source="src-confluence",
        source_url="https://confluence.example/doc-previous-outside-root",
        title="Previous Source",
        space_or_project="PAY",
        author="Sun, Youpeng",
        last_modified=datetime.now(timezone.utc),
        labels=[],
        version="1",
        content_hash="hash-doc-previous-outside-root",
        token_count=100,
        raw_content_uri=None,
        raw_content_type="text/html",
        normalized_content_uri=str(outside),
        pdf_content_uri=None,
        last_synced=datetime.now(timezone.utc),
    )

    assert orchestrator._read_previous_normalized_content(doc) is None


def test_confluence_gene_declares_pdf_artifact_requirement() -> None:
    from memforge.genes.confluence_gene import ConfluenceGene
    from memforge.models import ContentItem

    item = ContentItem(
        item_id="confluence-1",
        title="Source Page",
        source_url="https://confluence.example/1",
        last_modified=datetime.now(timezone.utc),
        version="1",
    )
    existing = DocumentRecord(
        doc_id="confluence-1",
        source="src-confluence",
        source_url="https://confluence.example/1",
        title="Source Page",
        space_or_project="PAY",
        author="Sun, Youpeng",
        last_modified=datetime.now(timezone.utc),
        labels=[],
        version="1",
        content_hash="old-hash",
        token_count=100,
        raw_content_uri=None,
        raw_content_type="text/html",
        normalized_content_uri="mem://doc.md",
        pdf_content_uri="mem://doc.pdf",
        last_synced=datetime.now(timezone.utc),
    )

    assert ConfluenceGene.requires_pdf_artifact(
        object(),
        item=item,
        existing_doc=None,
        existing_hash=None,
        new_hash="new-hash",
    )
    assert not ConfluenceGene.requires_pdf_artifact(
        object(),
        item=item,
        existing_doc=existing,
        existing_hash="old-hash",
        new_hash="old-hash",
    )


@pytest.mark.asyncio
async def test_admin_memory_list_search_accepts_hyphenated_jira_id(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    doc = await _insert_document(db, doc_id="jira-PAY-176425")
    memory = await _insert_memory(
        db,
        mem_id="mem-jira-id-search",
        content="A period switch waits for off-cycle payments to finish.",
    )
    await db.add_memory_source(
        memory.id,
        doc.doc_id,
        "jira",
        excerpt="A period switch can only occur once all off-cycle groups have completed payments.",
        support_kind="corroborated",
        source_observed_at=None,
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/memories", params={"search": "PAY-176425"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == memory.id


@pytest.mark.asyncio
async def test_admin_memory_search_endpoint_uses_service_search_engine(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from memforge.memory.lifecycle import allowed_search_statuses
    from memforge.models import SearchResult
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID

    calls = []

    class FakeSearchEngine:
        async def search(self, **kwargs):
            calls.append(kwargs)
            return {
                "query": kwargs["query"],
                "results": [
                    SearchResult(
                        memory_id="mem-proxy-search",
                        memory_type="fact",
                        summary="Proxy search stays service-owned.",
                        confidence=0.9,
                        relevance_score=1.0,
                        source_doc_id="doc-proxy",
                        content_url="/api/documents/doc-proxy/content",
                        pdf_url="/api/documents/doc-proxy/pdf",
                    )
                ],
            }

    class FakeRuntimeProvider:
        async def build_search_engine(self, _db, _config, *, audit_logger=None):
            return FakeSearchEngine()

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        runtime_provider=FakeRuntimeProvider(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/memories/search",
            json={"query": "proxy search", "top_k": 3},
        )

    assert response.status_code == 200
    payload = response.json()
    expected_scope = AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=allowed_search_statuses(False),
        active_project=None,
        # The request omits active_project, so the project-aware default
        # falls back to flat workspace ranking.
        scope_mode="workspace",
    )
    assert calls == [
        {
            "query": "proxy search",
            "memory_types": None,
            "sources": None,
            "source_filter": None,
            "time_range": None,
            "entities": None,
            "include_superseded": False,
            "top_k": 3,
            "request_scope": expected_scope,
        }
    ]
    assert payload["results"][0]["memory_id"] == "mem-proxy-search"
    assert payload["results"][0]["pdf_url"] == "/api/documents/doc-proxy/pdf"


@pytest.mark.asyncio
async def test_admin_recent_changes_endpoint_returns_memory_updates(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    now = datetime.now(timezone.utc).isoformat()
    await _insert_document(db, doc_id="doc-recent-change")
    memory = await _insert_memory(
        db,
        mem_id="mem-recent-change",
        content="Recent changes are service-owned for MCP proxy clients.",
    )
    async with db.db.execute(
        """INSERT INTO changelog
           (doc_id, change_type, previous_version, current_version, detected_at, title, source)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("doc-recent-change", "updated", "v1", "v2", now, "Recent Change", "confluence"),
    ):
        pass
    await db.db.commit()

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/recent-changes", params={"include_memories": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_changes"] == 1
    assert payload["changelog_entries"][0]["doc_id"] == "doc-recent-change"
    assert {item["id"] for item in payload["recent_memories"]} == {memory.id}


@pytest.mark.asyncio
async def test_admin_memory_list_search_accepts_fts_operator_text(db: Database, tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    await _insert_document(db, doc_id="jira-PAY-176426")
    await _insert_memory(
        db,
        mem_id="mem-operator-search",
        content="The AND gate condition is documented for payroll validation.",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/memories", params={"search": "AND"})

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_memory_delete_cleans_search_indexes(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from memforge.server.admin_api import create_admin_app

    memory = await _insert_memory(
        db,
        mem_id="mem-admin-delete",
        content="Admin delete should hide retired memories from search.",
    )
    collection = FakeCollection()
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.get_chroma_collection",
        lambda **kwargs: collection,
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.delete(f"/api/memories/{memory.id}")

    stored = await db.get_memory(memory.id)
    assert response.status_code == 200
    assert stored.status == "retired"
    assert await _fts_has_memory(db, memory.id) is False
    assert collection.deleted == [memory.id]


@pytest.mark.asyncio
async def test_admin_pending_review_status_cleans_search_indexes(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from memforge.server.admin_api import create_admin_app

    memory = await _insert_memory(
        db,
        mem_id="mem-admin-pending",
        content="Admin pending review should hide quarantined memories from search.",
    )
    collection = FakeCollection()
    monkeypatch.setattr(
        "memforge.retrieval.embeddings.get_chroma_collection",
        lambda **kwargs: collection,
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(f"/api/memories/{memory.id}", json={"status": "pending_review"})

    stored = await db.get_memory(memory.id)
    assert response.status_code == 200
    assert stored.status == "pending_review"
    assert await _fts_has_memory(db, memory.id) is False
    assert collection.deleted == [memory.id]
