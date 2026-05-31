"""Tests for pre-persistence memory quality filtering and provenance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.engine import MemoryEngine
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, RawMemory, ReconcileAction, ReconcileOperation, content_hash
from memforge.storage.database import Database


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
    ) -> str:
        await self.db.insert_memory(memory)
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

    quality = classify_memory_candidate(_raw(
        "MemForge memories are loaded at SessionStart and used as warm context for l3-demo.",
        "",
    ))

    assert quality.keep is False
    assert quality.skip_reason == "self_referential"


def test_classifier_skips_candidate_citing_internal_memory_id():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(
        "Prefer sum() over manual accumulator loops (project convention, mem-a2229a2c).",
        "",
    ))

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
    engine = MemoryEngine(db=db, memory_store=DirectInsertStore(db))

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(METADATA_CONTENT, METADATA_CONTEXT)],
        source_type="confluence",
    )

    assert stats == {"inserted": 0, "corroborated": 0, "skipped": 1}
    assert await db.count_memories() == 0


@pytest.mark.asyncio
async def test_engine_skips_open_question_candidate(db: Database):
    engine = MemoryEngine(db=db, memory_store=DirectInsertStore(db))

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(OPEN_QUESTION_CONTENT, OPEN_QUESTION_CONTEXT)],
        source_type="confluence",
    )

    assert stats == {"inserted": 0, "corroborated": 0, "skipped": 1}
    assert await db.count_memories() == 0


@pytest.mark.asyncio
async def test_engine_keeps_conditional_ap_rule(db: Database):
    engine = MemoryEngine(db=db, memory_store=DirectInsertStore(db))

    stats = await engine.process_memories(
        doc_id="doc-acd",
        raw_memories=[_raw(CONDITIONAL_RULE_CONTENT, CONDITIONAL_RULE_CONTEXT)],
        source_type="confluence",
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
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence")
    engine = MemoryEngine(db=db, memory_store=DirectInsertStore(db), structured_llm_client=object())
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
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence")
    store = FailingUpdateAuditStore(db)
    engine = MemoryEngine(db=db, memory_store=store, structured_llm_client=object())
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
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence")
    collection = FakeCollection()
    store = MemoryStore(db=db, memory_collection=collection, embed_cfg={})
    engine = MemoryEngine(db=db, memory_store=store, structured_llm_client=object())

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[_raw(LINK_CONTENT, LINK_CONTEXT)],
        source_type="confluence",
        doc_type="design-doc",
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
    await db.add_memory_source(old_memory.id, doc.doc_id, "confluence")
    await db.add_memory_source(old_memory.id, other_doc.doc_id, "confluence")
    collection = FakeCollection()
    store = MemoryStore(db=db, memory_collection=collection, embed_cfg={})
    engine = MemoryEngine(db=db, memory_store=store, structured_llm_client=object())

    stats = await engine.reconcile_and_persist(
        doc_id=doc.doc_id,
        raw_memories=[_raw(LINK_CONTENT, LINK_CONTEXT)],
        source_type="confluence",
        doc_type="design-doc",
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
    )
    collection = FakeCollection()
    store = MemoryStore(db=db, memory_collection=collection, embed_cfg={})

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
    )
    collection = FakeCollection()
    store = MemoryStore(db=db, memory_collection=collection, embed_cfg={})

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
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt")

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
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt")

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
    assert artifacts["normalized_markdown"]["url"] == (
        "/api/documents/doc-artifact-url/artifacts/normalized_markdown"
    )
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
async def test_mcp_get_memory_omits_storage_uris_from_provenance(db: Database, tmp_path: Path):
    from mcp.types import CallToolRequest, CallToolRequestParams

    from memforge.server.mcp_server import create_mcp_server

    docs_dir = tmp_path / "memforge" / "documents"
    docs_dir.mkdir(parents=True)
    source_md = docs_dir / "source.md"
    source_pdf = docs_dir / "source.pdf"
    source_md.write_text("# Source\n\nDurable memory evidence.", encoding="utf-8")
    source_pdf.write_bytes(b"%PDF-1.4\n%memforge\n")

    doc = await _insert_document(
        db,
        doc_id="doc-mcp-artifact-url",
        normalized_content_uri=str(source_md),
        pdf_content_uri=str(source_pdf),
    )
    memory = await _insert_memory(
        db,
        mem_id="mem-mcp-artifact-url",
        content="MCP provenance uses service artifact URLs instead of storage paths.",
    )
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt")

    server = create_mcp_server(db, _config(tmp_path))
    handler = server.request_handlers[CallToolRequest]
    result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_memory",
            arguments={"memory_id": memory.id},
        ),
    ))

    payload = json.loads(result.root.content[0].text)
    source = payload["provenance"][0]
    assert source["content_url"] == "/api/documents/doc-mcp-artifact-url/content"
    assert source["pdf_url"] == "/api/documents/doc-mcp-artifact-url/pdf"
    assert "file_uri" not in source
    assert "pdf_uri" not in source


def test_mcp_json_ready_serializes_search_results_as_fields():
    from memforge.models import SearchResult
    from memforge.server.mcp_server import _json_ready

    payload = _json_ready({
        "results": [
            SearchResult(
                memory_id="mem-artifact",
                memory_type="fact",
                summary="Artifact URLs are structured.",
                confidence=0.9,
                relevance_score=1.0,
                source_doc_id="doc-artifact",
                content_url="/api/documents/doc-artifact/content",
                pdf_url="/api/documents/doc-artifact/pdf",
            )
        ]
    })

    result = payload["results"][0]
    assert result["memory_id"] == "mem-artifact"
    assert result["content_url"] == "/api/documents/doc-artifact/content"
    assert result["pdf_url"] == "/api/documents/doc-artifact/pdf"


@pytest.mark.asyncio
async def test_mcp_fetch_resource_file_streams_to_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from memforge.server import mcp_server

    real_async_client = httpx.AsyncClient

    def transport_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://memforge.test/api/documents/doc-stream/pdf"
        return httpx.Response(
            200,
            headers={
                "content-type": "application/pdf",
                "content-disposition": 'attachment; filename="stream.pdf"',
            },
            content=b"%PDF-1.4\n%streamed\n",
        )

    transport = httpx.MockTransport(transport_handler)

    def client_factory(*, timeout: float, follow_redirects: bool) -> httpx.AsyncClient:
        assert follow_redirects is False
        return real_async_client(
            transport=transport,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(mcp_server.httpx, "AsyncClient", client_factory)

    result = await mcp_server._fetch_resource_file_from_api(
        "http://memforge.test/api/documents/doc-stream/pdf",
        {},
        mcp_server._ResourceTarget(
            doc_id="doc-stream",
            kind="pdf",
            relative_url="/api/documents/doc-stream/pdf",
            request_url="http://memforge.test/api/documents/doc-stream/pdf",
        ),
    )

    local_path = Path(str(result["local_path"]))
    assert result["status_code"] == 200
    assert result["observed_size_bytes"] == len(b"%PDF-1.4\n%streamed\n")
    assert local_path.parent == tmp_path
    assert local_path.read_bytes() == b"%PDF-1.4\n%streamed\n"


@pytest.mark.asyncio
async def test_mcp_get_resource_reads_text_and_file_artifacts(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mcp.types import CallToolRequest, CallToolRequestParams

    from memforge.server.mcp_server import create_mcp_server

    artifact_cache = tmp_path / "artifact-cache"
    monkeypatch.setenv("MEMFORGE_ARTIFACT_CACHE_DIR", str(artifact_cache))
    fetch_calls: list[tuple[str, dict[str, str], int | None]] = []

    async def fake_fetch(
        url: str,
        headers: dict[str, str],
        *,
        max_bytes: int | None = None,
    ):
        fetch_calls.append((url, headers, max_bytes))
        parsed = urlparse(url)
        if parsed.path == "/api/documents/doc-resource-read/content":
            return {
                "status_code": 200,
                "headers": {
                    "content-type": "text/markdown; charset=utf-8",
                    "content-disposition": 'attachment; filename="source.md"',
                },
                "content": b"# Source\n\nDurable memory evidence.",
            }
        return {"status_code": 404, "headers": {}, "content": b""}

    async def fake_file_fetch(
        url: str,
        headers: dict[str, str],
        target,
    ):
        fetch_calls.append((url, headers, None))
        parsed = urlparse(url)
        if parsed.path == "/api/documents/doc-resource-read/pdf":
            local_path = artifact_cache / "source.pdf"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(b"%PDF-1.4\n%memforge\n")
            return {
                "status_code": 200,
                "headers": {
                    "content-type": "application/pdf",
                    "content-disposition": 'attachment; filename="source.pdf"',
                },
                "content": b"",
                "local_path": str(local_path),
                "observed_size_bytes": local_path.stat().st_size,
            }
        return {"status_code": 404, "headers": {}, "content": b""}

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setenv("MEMFORGE_API_TOKEN", "token-123")
    monkeypatch.setattr("memforge.server.mcp_server._fetch_resource_from_api", fake_fetch)
    monkeypatch.setattr("memforge.server.mcp_server._fetch_resource_file_from_api", fake_file_fetch)

    server = create_mcp_server(db, _config(tmp_path))
    handler = server.request_handlers[CallToolRequest]

    text_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/content",
                "mode": "text",
            },
        ),
    ))
    text_payload = json.loads(text_result.root.content[0].text)
    assert text_payload["text"] == "# Source\n\nDurable memory evidence."
    assert text_payload["kind"] == "content"
    assert text_payload["url"] == "/api/documents/doc-resource-read/content"
    assert fetch_calls[-1] == (
        "http://memforge.test/api/documents/doc-resource-read/content",
        {"Authorization": "Bearer token-123"},
        2_000_000,
    )

    file_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/pdf",
                "mode": "file",
            },
        ),
    ))
    file_payload = json.loads(file_result.root.content[0].text)
    local_path = Path(file_payload["local_path"])
    assert local_path.is_file()
    assert local_path.read_bytes() == b"%PDF-1.4\n%memforge\n"
    assert artifact_cache in local_path.parents
    assert fetch_calls[-1] == (
        "http://memforge.test/api/documents/doc-resource-read/pdf",
        {"Authorization": "Bearer token-123"},
        None,
    )


@pytest.mark.asyncio
async def test_mcp_get_resource_rejects_foreign_urls_and_bad_limits(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mcp.types import CallToolRequest, CallToolRequestParams

    from memforge.server.mcp_server import create_mcp_server

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    server = create_mcp_server(db, _config(tmp_path))
    handler = server.request_handlers[CallToolRequest]

    foreign_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "https://example.invalid/api/documents/doc-resource-read/content",
            },
        ),
    ))
    foreign_payload = json.loads(foreign_result.root.content[0].text)
    assert foreign_payload["error"] == "unsupported resource URL"

    dot_segment_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/../content",
            },
        ),
    ))
    dot_segment_payload = json.loads(dot_segment_result.root.content[0].text)
    assert dot_segment_payload["error"] == "unsupported resource URL"

    encoded_slash_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/artifacts/a%2Fb",
            },
        ),
    ))
    encoded_slash_payload = json.loads(encoded_slash_result.root.content[0].text)
    assert encoded_slash_payload["error"] == "unsupported resource URL"

    bad_limit_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/content",
                "max_bytes": 0,
            },
        ),
    ))
    bad_limit_payload = json.loads(bad_limit_result.root.content[0].text)
    assert bad_limit_payload["error"] == "invalid max_bytes"


@pytest.mark.asyncio
async def test_mcp_get_resource_handles_base64_truncation_and_binary_text_errors(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from mcp.types import CallToolRequest, CallToolRequestParams

    from memforge.server.mcp_server import create_mcp_server

    async def fake_fetch(
        url: str,
        headers: dict[str, str],
        *,
        max_bytes: int | None = None,
    ):
        parsed = urlparse(url)
        if parsed.path.endswith("/content"):
            if max_bytes is not None and max_bytes < len(b"abcdef"):
                return {
                    "status_code": 200,
                    "headers": {"content-type": "text/markdown; charset=utf-8"},
                    "content": b"abc",
                    "observed_size_bytes": len(b"abcdef"),
                    "exceeded_max_bytes": True,
                }
            return {
                "status_code": 200,
                "headers": {"content-type": "text/markdown; charset=utf-8"},
                "content": b"abcdef",
            }
        if parsed.path.endswith("/pdf"):
            return {
                "status_code": 200,
                "headers": {"content-type": "application/pdf"},
                "content": b"%PDF-1.4\n%memforge\n",
            }
        return {"status_code": 404, "headers": {}, "content": b""}

    monkeypatch.setenv("MEMFORGE_API_URL", "http://memforge.test")
    monkeypatch.setattr("memforge.server.mcp_server._fetch_resource_from_api", fake_fetch)
    server = create_mcp_server(db, _config(tmp_path))
    handler = server.request_handlers[CallToolRequest]

    base64_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/content",
                "mode": "base64",
            },
        ),
    ))
    base64_payload = json.loads(base64_result.root.content[0].text)
    assert base64_payload["data_base64"] == "YWJjZGVm"

    truncate_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/content",
                "mode": "text",
                "max_chars": 3,
            },
        ),
    ))
    truncate_payload = json.loads(truncate_result.root.content[0].text)
    assert truncate_payload["text"] == "abc"
    assert truncate_payload["truncated"] is True

    binary_text_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/pdf",
                "mode": "text",
            },
        ),
    ))
    binary_text_payload = json.loads(binary_text_result.root.content[0].text)
    assert binary_text_payload["error"] == "artifact is not text"

    max_bytes_result = await handler(CallToolRequest(
        params=CallToolRequestParams(
            name="get_resource",
            arguments={
                "url": "/api/documents/doc-resource-read/content",
                "max_bytes": 3,
            },
        ),
    ))
    max_bytes_payload = json.loads(max_bytes_result.root.content[0].text)
    assert max_bytes_payload["error"] == "artifact exceeds max_bytes"


@pytest.mark.asyncio
async def test_mcp_tools_guide_agentic_get_memory_or_get_resource_choice(db: Database, tmp_path: Path):
    from mcp.types import ListToolsRequest

    from memforge.server.mcp_server import create_mcp_server

    server = create_mcp_server(db, _config(tmp_path))
    handler = server.request_handlers[ListToolsRequest]
    result = await handler(ListToolsRequest())

    tools = {tool.name: tool for tool in result.root.tools}
    assert "get_memory" in tools
    assert "get_resource" in tools
    get_memory_description = tools["get_memory"].description or ""
    get_resource_description = tools["get_resource"].description or ""
    assert "complete provenance" in get_memory_description
    assert "use get_resource directly from search" in get_resource_description
    assert "call get_memory first" in get_resource_description
    assert tools["get_resource"].inputSchema["properties"]["mode"]["enum"] == [
        "text",
        "file",
        "base64",
    ]


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
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/memories", params={"search": "PAY-176425"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == memory.id


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
