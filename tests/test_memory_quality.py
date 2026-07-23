"""Tests for pre-persistence memory quality filtering and provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.store import MemoryStore
from memforge.models import DocumentRecord, Memory, RawMemory, content_hash
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

ATTACHMENT_EVENT_CONTENT = (
    "In SFPay, Donchev, Georgi attached 'screenshot-8.png' "
    "(attachment ID 11786339) to the payroll run timeout ticket on 2026-06-25."
)
ATTACHMENT_EVENT_CONTEXT = (
    '{"field":"Attachment","fieldtype":"jira","from":null,'
    '"to":"11786339","toString":"screenshot-8.png"}'
)

OPERATIONAL_HISTORY_CONTENT = (
    "In SFPay, the payroll timeout ticket due date changed from 2026-06-23 "
    "to 2026-06-17."
)
OPERATIONAL_HISTORY_CONTEXT = (
    '{"field":"duedate","fieldtype":"jira","fromString":"2026-06-23",'
    '"toString":"2026-06-17"}'
)

CONDITIONAL_RULE_CONTENT = (
    "If an employee's regular pay date is changed via a deviating payroll process and the employee is "
    "assigned to an on-demand AP group, the out-of-sequence validation should be repeated."
)
CONDITIONAL_RULE_CONTEXT = (
    "if the regular pay date ... has been changed via a deviating payroll process, the same validation "
    "should be repeated"
)


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
        extraction_context=context,
    )


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return config


async def _insert_document(
    db: Database,
    *,
    doc_id: str = "doc-acd",
    source: str = "src-confluence",
    raw_content_uri: str | None = "/tmp/source.raw",
    normalized_content_uri: str | None = "/tmp/source.md",
    pdf_content_uri: str | None = None,
) -> DocumentRecord:
    if await db.get_source(source) is None:
        await db.upsert_source(
            id=source,
            type="confluence",
            name="Confluence",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="dev",
        )
    now = datetime.now(timezone.utc)
    doc = DocumentRecord(
        doc_id=doc_id,
        source=source,
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


def test_classifier_skips_attachment_event_without_attachment_content_claim():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(_raw(ATTACHMENT_EVENT_CONTENT, ATTACHMENT_EVENT_CONTEXT))

    assert quality.keep is False
    assert quality.skip_reason == "attachment_event_only"


@pytest.mark.parametrize(
    ("semantic_class", "expected_reason"),
    [
        ("attachment_event", "attachment_event_only"),
        ("operational_transition", "operational_history_only"),
    ],
)
def test_classifier_consumes_provider_neutral_observation_semantics(
    semantic_class: str,
    expected_reason: str,
) -> None:
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw("A provider event occurred.", "opaque provider payload"),
        observation_semantic_class=semantic_class,
    )

    assert quality.keep is False
    assert quality.skip_reason == expected_reason


def test_classifier_keeps_claim_extracted_from_attachment_content():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "The attached screenshot shows that payroll triggers remained OPEN for ten minutes.",
            (
                '{"artifact_type":"image/png","attachment_id":"11786339",'
                '"fragment_id":"image-analysis-1"}'
            ),
        )
    )

    assert quality.keep is True
    assert quality.skip_reason is None


def test_classifier_rejects_attachment_claim_when_evidence_is_only_upload_event():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "The attached screenshot shows that payroll triggers remained OPEN for ten minutes.",
            ATTACHMENT_EVENT_CONTEXT,
        ),
        observation_semantic_class="attachment_event",
    )

    assert quality.keep is False
    assert quality.skip_reason == "attachment_event_only"


def test_classifier_skips_operational_field_history_without_durable_claim():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(OPERATIONAL_HISTORY_CONTENT, OPERATIONAL_HISTORY_CONTEXT),
        observation_semantic_class="operational_transition",
    )

    assert quality.keep is False
    assert quality.skip_reason == "operational_history_only"


@pytest.mark.parametrize("field", ["status", "priority", "assignee", "labels"])
def test_classifier_skips_pure_operational_history_for_shared_fields(field: str):
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            f"The ticket recorded an operational {field} field transition.",
            f'{{"field":"{field}","fromString":"old","toString":"new"}}',
        ),
        observation_semantic_class="operational_transition",
    )

    assert quality.keep is False
    assert quality.skip_reason == "operational_history_only"


def test_classifier_keeps_mixed_history_that_contains_a_durable_root_cause():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "The payroll trigger stayed OPEN because the scheduler lease expired.",
            (
                '[{"field":"status","fromString":"IN PROGRESS","toString":"OPEN"},'
                '{"field":"Root Cause","toString":"scheduler lease expired"}]'
            ),
        ),
        observation_semantic_class="domain_transition",
    )

    assert quality.keep is True
    assert quality.skip_reason is None


def test_classifier_skips_resolution_field_history_without_rationale():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "The incident resolution changed to Cannot Reproduce.",
            '{"field":"resolution","toString":"Cannot Reproduce"}',
        ),
        observation_semantic_class="operational_transition",
    )

    assert quality.keep is False
    assert quality.skip_reason == "operational_history_only"


def test_classifier_keeps_resolution_rationale_from_comment_evidence():
    from memforge.memory.quality import classify_memory_candidate

    quality = classify_memory_candidate(
        _raw(
            "The incident was closed as Cannot Reproduce because the failure was flaky.",
            "I am closing this as Cannot Reproduce because the environment failure is flaky.",
        )
    )

    assert quality.keep is True
    assert quality.skip_reason is None


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
        source_updated_at=None,
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
        source_updated_at=None,
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
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt", source_updated_at=None)

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
    await db.add_memory_source(memory.id, doc.doc_id, "confluence", excerpt="source excerpt", source_updated_at=None)

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
async def test_memory_detail_and_source_artifact_route_preserve_exact_image_evidence(
    db: Database,
    tmp_path: Path,
) -> None:
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.document_store import LocalDocumentStore

    config = _config(tmp_path)
    document_store = LocalDocumentStore(config.storage.docs_path)
    image = b"\x89PNG\r\n\x1a\nknown-evidence-image"
    digest = hashlib.sha256(image).hexdigest()
    uri = document_store.store_source_artifact(
        source_id="src-confluence",
        artifact_id="artifact-42",
        filename="diagram\u202foverview.png",
        content=image,
        content_type="image/png",
    )
    doc = await _insert_document(db, doc_id="doc-image")
    memory = await _insert_memory(
        db,
        mem_id="mem-image",
        content="The diagram records the accepted payroll flow.",
    )
    await db.add_memory_source(
        memory.id,
        doc.doc_id,
        "confluence",
        excerpt=None,
        source_updated_at=None,
    )
    now = datetime.now(timezone.utc).isoformat()
    artifact_metadata = {
        "source_artifact": {
            "artifact_id": "artifact-42",
            "parent_observation_id": "obs-page",
            "provider_revision": "3",
            "filename": "diagram\u202foverview.png",
            "media_type": "image/png",
            "size_bytes": len(image),
            "sha256": digest,
            "uri": uri,
        }
    }
    await db.db.execute(
        """INSERT INTO source_units
           (id, source_id, unit_type, provider_key, locator_json, current_revision_id, updated_at)
           VALUES (?, ?, ?, ?, '{}', NULL, ?)""",
        ("unit-image", "src-confluence", "confluence_page", "page-1", now),
    )
    await db.db.execute(
        """INSERT INTO source_observations
           (id, source_id, source_unit_id, observation_type, provider_key,
            locator_json, current_revision_id, updated_at)
           VALUES (?, ?, ?, ?, ?, '{}', ?, ?)""",
        (
            "obs-page",
            "src-confluence",
            "unit-image",
            "page_body",
            "page-1",
            "obsrev-page",
            now,
        ),
    )
    await db.db.execute(
        """INSERT INTO source_observation_revisions
           (id, observation_id, semantic_hash, content, metadata_json, observed_at, created_at)
           VALUES (?, ?, ?, ?, '{}', ?, ?)""",
        (
            "obsrev-page",
            "obs-page",
            "page-hash",
            "The page provides the primary claim.",
            now,
            now,
        ),
    )
    await db.db.execute(
        """INSERT INTO source_observations
           (id, source_id, source_unit_id, observation_type, provider_key,
            locator_json, current_revision_id, updated_at)
           VALUES (?, ?, ?, ?, ?, '{}', ?, ?)""",
        (
            "obs-image",
            "src-confluence",
            "unit-image",
            "binary_artifact",
            "artifact:42",
            "obsrev-image",
            now,
        ),
    )
    await db.db.execute(
        """INSERT INTO source_observation_revisions
           (id, observation_id, semantic_hash, content, metadata_json, observed_at, created_at)
           VALUES (?, ?, ?, '', ?, ?, ?)""",
        ("obsrev-image", "obs-image", digest, json.dumps(artifact_metadata), now, now),
    )
    await db.db.execute(
        """INSERT INTO evidence_units
           (id, source_id, doc_id, source_type, visibility, content, excerpt,
            evidence_provenance, access_context_hash, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'workspace', ?, NULL, 'extracted', ?, ?, ?)""",
        (
            "evidence-image",
            "src-confluence",
            doc.doc_id,
            "confluence",
            memory.content,
            "access-hash",
            now,
            now,
        ),
    )
    await db.db.execute(
        """INSERT INTO evidence_references
           (id, evidence_unit_id, role, anchor_kind, observation_id,
            observation_revision_id, created_at)
           VALUES (?, ?, 'primary', 'whole_observation', ?, ?, ?)""",
        ("eref-primary", "evidence-image", "obs-page", "obsrev-page", now),
    )
    await db.db.execute(
        """INSERT INTO evidence_references
           (id, evidence_unit_id, role, anchor_kind, observation_id,
            observation_revision_id, created_at)
           VALUES (?, ?, 'required', 'whole_observation', ?, ?, ?)""",
        ("eref-required", "evidence-image", "obs-page", "obsrev-page", now),
    )
    await db.db.execute(
        """INSERT INTO evidence_references
           (id, evidence_unit_id, role, anchor_kind, observation_id,
            observation_revision_id, created_at)
           VALUES (?, ?, 'context', 'whole_observation', ?, ?, ?)""",
        ("eref-image", "evidence-image", "obs-image", "obsrev-image", now),
    )
    await db.db.execute(
        """INSERT INTO memory_support_assertions
           (id, memory_id, evidence_reference_id, source_id, access_context_hash,
            active, created_at)
           VALUES (?, ?, ?, ?, ?, 1, ?)""",
        ("support-image", memory.id, "eref-primary", "src-confluence", "access-hash", now),
    )
    await db.db.execute(
        """INSERT INTO memory_support_assertions
           (id, memory_id, evidence_reference_id, source_id, access_context_hash,
            active, created_at)
           VALUES (?, ?, ?, ?, ?, 1, ?)""",
        (
            "support-image-required",
            memory.id,
            "eref-required",
            "src-confluence",
            "access-hash",
            now,
        ),
    )
    await db.db.commit()

    app = create_admin_app(db=db, config=config, document_store=document_store)
    with TestClient(app) as client:
        detail = client.get(f"/api/memories/{memory.id}")
        resource = client.get("/api/source-artifacts/obsrev-image")

    assert detail.status_code == 200
    [artifact] = detail.json()["evidence_artifacts"]
    assert artifact["observation_revision_id"] == "obsrev-image"
    assert artifact["evidence_reference_id"] == "eref-image"
    assert artifact["evidence_role"] == "context"
    assert artifact["content_type"] == "image/png"
    assert artifact["sha256"] == digest
    assert artifact["url"] == "/api/source-artifacts/obsrev-image"

    await db.db.execute(
        "UPDATE source_observations SET current_revision_id = ? WHERE id = ?",
        (None, "obs-page"),
    )
    await db.db.commit()
    with TestClient(app) as client:
        stale_support_detail = client.get(f"/api/memories/{memory.id}")
    assert stale_support_detail.status_code == 200
    assert stale_support_detail.json()["evidence_artifacts"] == []
    assert resource.status_code == 200
    assert resource.headers["content-type"] == "image/png"
    assert (
        resource.headers["content-disposition"]
        == "inline; filename=\"diagram_overview.png\"; "
        "filename*=UTF-8''diagram%E2%80%AFoverview.png"
    )
    assert resource.content == image

    await db.db.execute(
        "UPDATE source_observations SET current_revision_id = NULL WHERE id = ?",
        ("obs-image",),
    )
    await db.db.commit()
    with TestClient(app) as client:
        stale_revision = client.get("/api/source-artifacts/obsrev-image")
    assert stale_revision.status_code == 404

    await db.db.execute(
        "UPDATE source_observations SET current_revision_id = ? WHERE id = ?",
        ("obsrev-image", "obs-image"),
    )
    await db.db.execute(
        "UPDATE sources SET access_policy = 'private', owner_user_id = ? WHERE id = ?",
        ("different-user", "src-confluence"),
    )
    await db.db.commit()
    with TestClient(app) as client:
        unauthorized_replay = client.get("/api/source-artifacts/obsrev-image")
    assert unauthorized_replay.status_code == 404


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            r"..\secret.pdf",
            "inline; filename=\"secret.pdf\"; filename*=UTF-8''secret.pdf",
        ),
        (
            "../secret.pdf",
            "inline; filename=\"secret.pdf\"; filename*=UTF-8''secret.pdf",
        ),
        (
            'bad\r\nX-Evil: yes".png',
            "inline; filename=\"badX-Evil_yes.png\"; "
            "filename*=UTF-8''badX-Evil%3A%20yes%22.png",
        ),
    ],
)
def test_artifact_content_disposition_rejects_path_and_header_injection(
    filename: str,
    expected: str,
) -> None:
    from memforge.server.admin_api import _inline_content_disposition

    assert _inline_content_disposition(filename) == expected


@pytest.mark.asyncio
async def test_admin_document_content_uses_exact_document_identity_for_same_title(
    db: Database,
    tmp_path: Path,
) -> None:
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.document_store import LocalDocumentStore

    config = _config(tmp_path)
    document_store = LocalDocumentStore(config.storage.docs_path)
    first_doc_id = "doc-user-guide-a"
    second_doc_id = "doc-user-guide-b"
    first_uri = document_store.store_normalized(
        source_id="src-repository",
        doc_id=first_doc_id,
        title="User Guide",
        markdown="# First guide",
    )
    second_uri = document_store.store_normalized(
        source_id="src-repository",
        doc_id=second_doc_id,
        title="User Guide",
        markdown="# Second guide",
    )
    await _insert_document(
        db,
        doc_id=first_doc_id,
        source="src-repository",
        normalized_content_uri=first_uri,
        raw_content_uri=None,
    )
    await _insert_document(
        db,
        doc_id=second_doc_id,
        source="src-repository",
        normalized_content_uri=second_uri,
        raw_content_uri=None,
    )

    app = create_admin_app(db=db, config=config, document_store=document_store)
    with TestClient(app) as client:
        first = client.get("/api/documents/doc-user-guide-a/content")
        second = client.get("/api/documents/doc-user-guide-b/content")

    assert first.status_code == 200
    assert first.text == "# First guide"
    assert second.status_code == 200
    assert second.text == "# Second guide"


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
        memory.id, "doc-object-artifact-url", "jira", excerpt="source excerpt", source_updated_at=None
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
            self.deleted: list[str] = []

        def delete_artifact(self, uri: str) -> None:
            self.deleted.append(uri)

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

    class DatabaseMemoryStore:
        def __init__(self, database: Database) -> None:
            self.database = database

        async def delete_source_cascade(self, source_id: str):
            return await self.database.delete_source_cascade(source_id)

    async def fake_build_memory_store(*args, **kwargs):
        return DatabaseMemoryStore(db)

    monkeypatch.setattr(admin_api, "_build_memory_store", fake_build_memory_store)
    await db.upsert_source(
        "src-confluence", "confluence", "Delete Route Source", "{}", access_policy="workspace", owner_user_id="dev"
    )
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
    assert store.deleted == ["mem://doc.md"]


@pytest.mark.asyncio
async def test_delete_source_is_idempotent_when_source_is_already_absent(
    db: Database,
    tmp_path: Path,
):
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.delete("/api/sources/src-already-deleted")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "ok": True,
        "deleted_source": "src-already-deleted",
        "already_deleted": True,
    }


@pytest.mark.asyncio
async def test_delete_source_succeeds_and_retains_cleanup_task_when_artifact_delete_fails(
    db: Database,
    tmp_path: Path,
    monkeypatch,
):
    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class UnavailableDocumentStore:
        def delete_artifact(self, uri: str) -> None:
            raise RuntimeError("object store unavailable")

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

    class DatabaseMemoryStore:
        async def delete_source_cascade(self, source_id: str):
            return await db.delete_source_cascade(source_id)

    async def fake_build_memory_store(*args, **kwargs):
        return DatabaseMemoryStore()

    monkeypatch.setattr(admin_api, "_build_memory_store", fake_build_memory_store)
    await db.upsert_source(
        "src-cleanup-failure", "confluence", "Cleanup Failure", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await _insert_document(
        db,
        doc_id="doc-cleanup-failure",
        source="src-cleanup-failure",
        raw_content_uri=None,
        normalized_content_uri="object-store://workspace/documents/src-cleanup-failure/page.md",
    )

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        document_store=UnavailableDocumentStore(),
    )
    with TestClient(app) as client:
        response = client.delete("/api/sources/src-cleanup-failure")

    tasks = await db.list_source_artifact_cleanup_tasks(limit=10)
    assert response.status_code == 200, response.text
    assert await db.get_source("src-cleanup-failure") is None
    assert [(task.attempt_count, task.last_error) for task in tasks] == [(1, "object store unavailable")]


@pytest.mark.asyncio
async def test_delete_source_restores_previous_status_when_delete_transaction_fails(
    db: Database,
    tmp_path: Path,
    monkeypatch,
):
    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class FailingMemoryStore:
        async def delete_source_cascade(self, source_id: str):
            raise RuntimeError("delete transaction failed")

    async def fake_build_memory_store(*args, **kwargs):
        return FailingMemoryStore()

    monkeypatch.setattr(admin_api, "_build_memory_store", fake_build_memory_store)
    await db.upsert_source(
        "src-delete-rollback", "confluence", "Delete Rollback", "{}", access_policy="workspace", owner_user_id="dev"
    )
    await db.db.execute(
        "UPDATE sources SET status = 'paused' WHERE id = ?",
        ("src-delete-rollback",),
    )
    await db.db.commit()
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.delete("/api/sources/src-delete-rollback")

    source = await db.get_source("src-delete-rollback")
    assert response.status_code == 500
    assert source is not None
    assert source["status"] == "paused"


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
        source_updated_at=None,
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
            "source_filter": None,
            "time_range": None,
            "entities": None,
            "include_superseded": False,
            "top_k": 3,
            "request_scope": expected_scope,
            "offset": 0,
        }
    ]
    assert payload["results"][0]["memory_id"] == "mem-proxy-search"
    result = payload["results"][0]
    for field in ("source_doc_id", "source_doc_title", "source_url", "content_url", "pdf_url", "is_document_result"):
        assert field not in result


@pytest.mark.asyncio
async def test_get_memory_sources_orders_extracted_before_corroborated(db: Database):
    memory = await _insert_memory(
        db,
        mem_id="mem-source-order",
        content="The extracted source should be listed before corroborating sources.",
    )
    await _insert_document(db, doc_id="doc-corrob-new")
    await _insert_document(db, doc_id="doc-extracted")
    await _insert_document(db, doc_id="doc-corrob-tie-a")
    await _insert_document(db, doc_id="doc-corrob-tie-b")
    await _insert_document(db, doc_id="doc-corrob-old")
    await db.add_memory_source(
        memory.id,
        "doc-corrob-new",
        "confluence",
        support_kind="corroborated",
        source_updated_at=None,
    )
    await db.add_memory_source(
        memory.id,
        "doc-extracted",
        "confluence",
        support_kind="extracted",
        source_updated_at=None,
    )
    await db.add_memory_source(
        memory.id,
        "doc-corrob-old",
        "confluence",
        support_kind="corroborated",
        source_updated_at=None,
    )
    await db.add_memory_source(
        memory.id,
        "doc-corrob-tie-b",
        "confluence",
        support_kind="corroborated",
        source_updated_at=None,
    )
    await db.add_memory_source(
        memory.id,
        "doc-corrob-tie-a",
        "confluence",
        support_kind="corroborated",
        source_updated_at=None,
    )

    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        ("2026-06-23T12:00:00+00:00", memory.id, "doc-corrob-new"),
    )
    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        ("2026-06-22T12:00:00+00:00", memory.id, "doc-extracted"),
    )
    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        ("2026-06-21T12:00:00+00:00", memory.id, "doc-corrob-old"),
    )
    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        ("2026-06-22T00:00:00+00:00", memory.id, "doc-corrob-tie-b"),
    )
    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        ("2026-06-22T00:00:00+00:00", memory.id, "doc-corrob-tie-a"),
    )
    await db.db.commit()

    sources = await db.get_memory_sources(memory.id)

    assert [(source.doc_id, source.support_kind) for source in sources] == [
        ("doc-extracted", "extracted"),
        ("doc-corrob-new", "corroborated"),
        ("doc-corrob-tie-a", "corroborated"),
        ("doc-corrob-tie-b", "corroborated"),
        ("doc-corrob-old", "corroborated"),
    ]


@pytest.mark.asyncio
async def test_admin_memory_search_validates_source_ids_without_hydrating_admin_rows(
    db: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.adapters.context import LOCAL_DEV_USER_ID

    await db.upsert_source(
        "src-mounttai", "jira", "MountTai Defects", "{}", access_policy="workspace", owner_user_id="dev"
    )

    calls = []

    class FakeSearchEngine:
        async def search(self, **kwargs):
            calls.append(kwargs)
            return {"query": kwargs["query"], "results": []}

    class FakeRuntimeProvider:
        async def build_search_engine(self, _db, _config, *, audit_logger=None):
            return FakeSearchEngine()

    async def fail_admin_row_hydration(*args, **kwargs):
        raise AssertionError("search source-id validation must not hydrate all source admin rows")

    monkeypatch.setattr(
        "memforge.server.admin_api.list_source_admin_rows",
        fail_admin_row_hydration,
    )

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        runtime_provider=FakeRuntimeProvider(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/memories/search",
            json={
                "query": "payroll defect",
                "source_filter": {"source_ids": ["src-mounttai"]},
            },
        )

    assert response.status_code == 200
    assert calls[0]["source_filter"].source_ids == ("src-mounttai",)

    await db.set_source_subscription("src-mounttai", LOCAL_DEV_USER_ID, enabled=False)
    with TestClient(app) as client:
        response = client.post(
            "/api/memories/search",
            json={
                "query": "payroll defect",
                "source_filter": {"source_ids": ["src-mounttai", "src-missing"]},
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "unknown_or_unavailable_source_id"


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
