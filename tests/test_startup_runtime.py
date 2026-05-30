"""Startup/runtime wiring tests."""

from __future__ import annotations

import json
from pathlib import Path
import logging

import pytest
from rich.logging import RichHandler
from fastapi.testclient import TestClient

from meminception.config import AppConfig
from meminception.storage.database import Database

TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="


class FakeCollection:
    def query(self, **kwargs):
        return {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs):
        return None

    def delete(self, **kwargs):
        return None


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.enrichment_base_url = "http://localhost:6655/anthropic"
    cfg.llm.enrichment_api_key = "test-key"
    cfg.llm.request_timeout_s = 42.0
    cfg.llm.embedding_base_url = "http://localhost:6655/openai/v1"
    cfg.llm.embedding_api_key = "test-key"
    cfg.server.jwt_secret = "test-secret"
    return cfg


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "runtime.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_sync_runtime_wires_structured_llm_client_into_memory_engine(db, tmp_path, monkeypatch):
    """Normal sync startup must enable structured reconciliation and contradiction detection."""
    from meminception import runtime

    captured = {}

    class RecordingStructuredClient:
        def __init__(self, config):
            captured["config"] = config
            captured["client"] = self

    monkeypatch.setattr(runtime, "LiteLlmStructuredClient", RecordingStructuredClient)
    monkeypatch.setattr(runtime, "get_chroma_collection", lambda **kwargs: FakeCollection())

    sync_runtime = await runtime.build_sync_runtime(db, _config(tmp_path))

    assert sync_runtime.structured_llm_client is captured["client"]
    assert sync_runtime.memory_engine.structured_llm_client is captured["client"]
    assert sync_runtime.memory_engine.llm_model == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_sync_runtime_bounds_structured_request_timeout(db, tmp_path, monkeypatch):
    """Long-running sync model calls should fail as document errors, not hang forever."""
    from meminception import runtime

    configs = []

    class RecordingStructuredClient:
        def __init__(self, config):
            configs.append(config)

    monkeypatch.setattr(runtime, "LiteLlmStructuredClient", RecordingStructuredClient)
    monkeypatch.setattr(runtime, "get_chroma_collection", lambda **kwargs: FakeCollection())

    sync_runtime = await runtime.build_sync_runtime(db, _config(tmp_path))

    assert sync_runtime.structured_llm_client is not None
    assert configs
    assert all(config.timeout_s == 42.0 for config in configs)


@pytest.mark.asyncio
async def test_build_sync_runtime_wires_litellm_structured_source_support_client(db, tmp_path, monkeypatch):
    from meminception.config import AppConfig
    from meminception.runtime import build_sync_runtime

    captured = {}

    class RecordingStructuredClient:
        def __init__(self, config):
            captured["config"] = config
            captured["client"] = self

        async def verify_source_support(self, prompt: str):
            raise AssertionError("not called during runtime construction")

    monkeypatch.setattr("meminception.runtime.LiteLlmStructuredClient", RecordingStructuredClient)
    monkeypatch.setattr("meminception.runtime.get_chroma_collection", lambda **kwargs: FakeCollection())

    config = AppConfig()
    config.base_dir = tmp_path
    config.storage.db_path = str(tmp_path / "mem.db")
    config.storage.chroma_path = str(tmp_path / "chroma")
    config.storage.docs_path = str(tmp_path / "docs")
    config.llm.enrichment_model = "anthropic--claude-sonnet-latest"
    config.llm.enrichment_base_url = "http://localhost:6655/anthropic"
    config.llm.enrichment_api_key = "local-key"

    runtime = await build_sync_runtime(db, config)

    assert runtime.source_support_detector is not None
    assert runtime.source_support_detector.structured_llm_client is captured["client"]
    assert runtime.memory_extractor.structured_llm_client is captured["client"]
    assert runtime.enricher.structured_llm_client is captured["client"]
    assert runtime.memory_engine.structured_llm_client is captured["client"]
    assert captured["config"].model == "anthropic--claude-sonnet-latest"
    assert captured["config"].base_url == "http://localhost:6655/anthropic"
    assert captured["config"].api_key == "local-key"


def test_enrichment_and_extraction_clients_bound_request_timeout(monkeypatch):
    """Document LLM clients should use the configured request timeout."""
    from meminception.pipeline.enricher import Enricher
    from meminception.pipeline.memory_extractor import MemoryExtractor

    enricher = Enricher(api_key="test-key", request_timeout_s=42.0)
    extractor = MemoryExtractor(api_key="test-key", request_timeout_s=42.0)

    assert enricher.structured_llm_client.config.timeout_s == 42.0
    assert extractor.structured_llm_client.config.timeout_s == 42.0


def test_admin_app_lifespan_owns_database_and_sync_service(tmp_path):
    """The API startup path should open/close DB resources through FastAPI lifespan."""
    from meminception.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        assert app.state.db._db is not None
        assert app.state.sync_service is not None
        assert client.get("/api/health").status_code == 200

    assert app.state.db._db is None


@pytest.mark.asyncio
async def test_health_reports_recent_audit_failures_as_warning(db, tmp_path):
    from datetime import datetime, timezone

    from meminception.memory.audit import MemoryAuditEvent
    from meminception.server.admin_api import create_admin_app

    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_type="source_support_verification_failed",
            status="failed",
            doc_id="jira-PAY-176425",
            error="Extra data: line 9 column 1 (char 256)",
            occurred_at=datetime.now(timezone.utc),
        )
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["audit_failures"]["status"] == "warning"
    assert payload["audit_failures"]["payload"]["window_hours"] == 24
    assert payload["audit_failures"]["payload"]["counts_by_event_type"] == {
        "source_support_verification_failed": 1
    }
    assert payload["audit_failures"]["payload"]["total"] == 1
    assert payload["audit_failures"]["payload"]["last_seen_at"]


@pytest.mark.asyncio
async def test_health_ignores_old_audit_failures(db, tmp_path):
    from datetime import datetime, timedelta, timezone

    from meminception.memory.audit import MemoryAuditEvent
    from meminception.server.admin_api import create_admin_app

    await db.insert_memory_audit_event(
        MemoryAuditEvent(
            event_type="source_support_verification_failed",
            status="failed",
            doc_id="jira-PAY-old",
            error="Extra data: line 9 column 1 (char 256)",
            occurred_at=datetime.now(timezone.utc) - timedelta(days=3),
        )
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["audit_failures"]["status"] == "ok"
    assert payload["audit_failures"]["payload"]["counts_by_event_type"] == {}


def test_admin_app_scheduler_registers_expiry_maintenance(tmp_path):
    from meminception.scheduler import EXPIRY_JOB_ID
    from meminception.server.admin_api import create_admin_app

    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app):
        assert app.state.sync_scheduler.scheduler.get_job(EXPIRY_JOB_ID) is not None


def test_gene_config_schema_marks_advanced_source_fields(tmp_path):
    from meminception.server.admin_api import create_admin_app

    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/genes/confluence/config-schema")

    assert response.status_code == 200
    fields = {field["key"]: field for field in response.json()["fields"]}
    assert fields["pat"]["advanced"] is False
    assert fields["tls_ca_bundle"]["advanced"] is True


def test_admin_source_create_encrypts_and_redacts_pat(tmp_path, monkeypatch):
    import sqlite3

    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": "wiki-pat-secret",
                },
            },
        )
        assert response.status_code == 200
        source_id = response.json()["id"]
        sources_response = client.get("/api/sources")

    with sqlite3.connect(cfg.storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        stored = conn.execute(
            "SELECT config FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()

    stored_config = json.loads(stored["config"])
    assert "pat" not in stored_config
    assert stored_config["pat_encrypted"] != "wiki-pat-secret"

    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert "pat" not in source_payload["config"]
    assert "pat_encrypted" not in source_payload["config"]
    assert source_payload["config"]["pat_configured"] is True


def test_admin_source_update_preserves_encrypted_pat_when_blank(tmp_path, monkeypatch):
    import sqlite3

    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": "wiki-pat-secret",
                },
            },
        )
        assert create_response.status_code == 200
        source_id = create_response.json()["id"]

        with sqlite3.connect(cfg.storage.db_path) as conn:
            conn.row_factory = sqlite3.Row
            before = json.loads(conn.execute(
                "SELECT config FROM sources WHERE id = ?",
                (source_id,),
            ).fetchone()["config"])

        update_response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": "",
                },
            },
        )
        sources_response = client.get("/api/sources")

    assert update_response.status_code == 200
    with sqlite3.connect(cfg.storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        after = json.loads(conn.execute(
            "SELECT config FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()["config"])

    assert after["pat_encrypted"] == before["pat_encrypted"]
    assert "pat" not in after
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert "pat_encrypted" not in source_payload["config"]
    assert source_payload["config"]["pat_configured"] is True


def test_admin_source_save_rejects_missing_tls_ca_bundle(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": "wiki-pat-secret",
                    "tls_ca_bundle": str(tmp_path / "missing-ca.pem"),
                },
            },
        )

    assert response.status_code == 400
    assert "TLS CA bundle" in response.json()["detail"]


def test_admin_source_save_rejects_insecure_atlassian_base_url(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "http://jira.example.test",
                    "projects": ["PAY"],
                    "pat": "jira-pat-secret",
                },
            },
        )

    assert response.status_code == 400
    assert "HTTPS" in response.json()["detail"]


def test_admin_source_save_rejects_pat_without_base_url(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "config": {
                    "spaces": ["PAY"],
                    "pat": "wiki-pat-secret",
                },
            },
        )

    assert response.status_code == 400
    assert "base_url is required" in response.json()["detail"]


@pytest.mark.parametrize(
    ("source_type", "name", "config", "expected_detail"),
    [
        (
            "confluence",
            "Engineering Wiki",
            {
                "base_url": "https://wiki.example.test",
                "spaces": ["PAY"],
            },
            "Personal Access Token is required",
        ),
    ],
)
def test_admin_source_create_rejects_missing_atlassian_pat(
    source_type,
    name,
    config,
    expected_detail,
    tmp_path,
    monkeypatch,
):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": source_type,
                "name": name,
                "config": config,
            },
        )

    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


def test_admin_source_create_allows_jira_browser_session_without_source_cookie(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "projects": ["PAY"],
                    "auth_mode": "browser_cookie",
                },
            },
        )

    assert response.status_code == 200, response.text


def test_admin_source_create_rejects_missing_required_source_scope(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "pat": "jira-pat-secret",
                },
            },
        )

    assert response.status_code == 400
    assert "Projects to Sync is required" in response.json()["detail"]


def test_admin_source_create_ignores_forged_jira_cookie_configured_flag(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "projects": ["PAY"],
                    "auth_mode": "browser_cookie",
                    "jira_cookie_configured": True,
                },
            },
        )
        source_id = response.json().get("id")
        sources = client.get("/api/sources").json()["data"]

    assert response.status_code == 200, response.text
    source = next(source for source in sources if source["id"] == source_id)
    assert "jira_cookie_configured" not in source["config"]


def test_admin_source_create_rejects_forged_jira_pat_configured_flag(tmp_path, monkeypatch):
    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "projects": ["PAY"],
                    "auth_mode": "pat",
                    "pat_configured": True,
                },
            },
        )

    assert response.status_code == 400
    assert "Personal Access Token is required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_admin_sources_exposes_failed_and_partial_sync_status(db, tmp_path):
    from meminception.server.admin_api import create_admin_app

    await db.upsert_source(
        id="src-failed-visible",
        type="jira",
        name="Failed Jira",
        config_json=json.dumps({}),
    )
    await db.insert_sync_history(
        source="src-failed-visible",
        status="failed",
        docs_processed=0,
        docs_updated=0,
        docs_failed=0,
        memories_extracted=0,
        error_message="Jira PAT is required",
        failed_docs=None,
        started_at="2026-05-24T02:00:00+00:00",
        finished_at="2026-05-24T02:00:01+00:00",
    )
    await db.upsert_source(
        id="src-partial-visible",
        type="confluence",
        name="Partial Confluence",
        config_json=json.dumps({}),
    )
    await db.insert_sync_history(
        source="src-partial-visible",
        status="partial",
        docs_processed=4,
        docs_updated=4,
        docs_failed=2,
        memories_extracted=9,
        error_message="2 document(s) failed",
        failed_docs=[{"doc_id": "doc-1", "title": "Doc 1", "error": "boom"}],
        started_at="2026-05-24T02:01:00+00:00",
        finished_at="2026-05-24T02:01:10+00:00",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/sources")

    assert response.status_code == 200
    sources = {source["id"]: source for source in response.json()["data"]}
    assert sources["src-failed-visible"]["sync"]["status"] == "failed"
    assert sources["src-failed-visible"]["sync"]["error_message"] == "Jira PAT is required"
    assert sources["src-partial-visible"]["sync"]["status"] == "partial"
    assert sources["src-partial-visible"]["sync"]["docs_failed"] == 2
    assert sources["src-partial-visible"]["sync"]["error_message"] == "2 document(s) failed"
    assert sources["src-partial-visible"]["sync"]["failed_docs"] == [
        {"doc_id": "doc-1", "title": "Doc 1", "error": "boom"}
    ]


@pytest.mark.asyncio
async def test_admin_sources_exposes_running_stored_counts_separately(db, tmp_path):
    from meminception.server.admin_api import create_admin_app

    source_id = "src-running-counts"
    await db.upsert_source(
        id=source_id,
        type="github_pages",
        name="Runbook Source",
        config_json=json.dumps({}),
    )
    for index in range(2):
        doc_id = f"doc-{index}"
        await db.db.execute(
            """INSERT INTO documents (
                doc_id, source, source_url, title, space_or_project, author,
                last_modified, labels, version, content_hash, token_count,
                raw_content_uri, raw_content_type, normalized_content_uri,
                pdf_content_uri, last_synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                source_id,
                f"https://example.test/{doc_id}",
                f"Doc {index}",
                "org/repo",
                None,
                "2026-05-28T07:00:00+00:00",
                "[]",
                f"version-{index}",
                f"hash-{index}",
                10,
                None,
                "text/markdown",
                None,
                None,
                "2026-05-28T07:00:00+00:00",
            ),
        )
        await db.db.execute(
            """INSERT INTO memories (
                id, memory_type, content, content_hash, tags, scope,
                project_key, confidence, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"mem-{index}",
                "fact",
                f"Memory {index}",
                f"memory-hash-{index}",
                "[]",
                "team",
                None,
                0.8,
                "active",
                "2026-05-28T07:00:00+00:00",
                "2026-05-28T07:00:00+00:00",
            ),
        )
        await db.db.execute(
            "INSERT INTO memory_sources (memory_id, doc_id, source_type, excerpt) VALUES (?, ?, ?, ?)",
            (f"mem-{index}", doc_id, "github_pages", f"Excerpt {index}"),
        )
    await db.db.commit()

    class FakeSyncService:
        progress = {
            source_id: {
                "started_at": "2026-05-28T07:01:00+00:00",
                "phase": "processing",
                "docs_processed": 1,
                "docs_total": 3,
                "docs_updated": 1,
                "docs_failed": 0,
                "memories_extracted": 4,
                "title": "Current Doc",
            }
        }

        def is_running(self, checked_source_id: str):
            return checked_source_id == source_id

        async def shutdown(self):
            return None

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        app.state.sync_service = FakeSyncService()
        response = client.get("/api/sources")

    assert response.status_code == 200
    source = next(source for source in response.json()["data"] if source["id"] == source_id)
    assert source["doc_count"] == 2
    assert source["memory_count"] == 2
    assert source["sync"]["status"] == "running"
    assert source["sync"]["docs_processed"] == 1
    assert source["sync"]["docs_total"] == 3
    assert source["sync"]["docs_stored"] == 2
    assert source["sync"]["memories_stored"] == 2
    assert "docs_committed" not in source["sync"]
    assert "memories_committed" not in source["sync"]


def test_source_secret_field_policy_is_gene_driven_for_known_sources():
    from meminception.server.admin_api import _source_secret_fields, _validate_source_config

    assert _source_secret_fields("confluence") == ("pat",)
    assert _source_secret_fields("github_pages") == ("pat",)
    assert _source_secret_fields("teams") == ()
    assert _source_secret_fields("removed_confluence") == ("pat",)
    _validate_source_config("teams", {"base_url": "http://teams.internal"})
    with pytest.raises(ValueError, match="HTTPS"):
        _validate_source_config("removed_confluence", {"base_url": "http://wiki.internal", "pat": "legacy"})


@pytest.mark.asyncio
async def test_unknown_source_type_still_redacts_and_encrypts_secret_fields(db, tmp_path, monkeypatch):
    import sqlite3

    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    source_id = "src-removed-gene"
    await db.upsert_source(
        id=source_id,
        type="removed_confluence",
        name="Removed Gene",
        config_json=json.dumps({"base_url": "https://wiki.example.test", "pat": "legacy-plain"}),
    )

    app = create_admin_app(db=db, config=cfg)
    with TestClient(app) as client:
        sources_response = client.get("/api/sources")
        update_response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Removed Gene",
                "config": {"base_url": "https://wiki.example.test", "pat": "new-plain"},
            },
        )
        sources_after_update = client.get("/api/sources")

    assert sources_response.status_code == 200
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert "pat" not in source_payload["config"]
    assert "pat_encrypted" not in source_payload["config"]
    assert source_payload["config"]["pat_configured"] is True

    assert update_response.status_code == 200
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        stored = json.loads(conn.execute(
            "SELECT config FROM sources WHERE id = ?",
            (source_id,),
        ).fetchone()["config"])
    assert "pat" not in stored
    assert stored["pat_encrypted"] != "new-plain"

    source_after_update = next(s for s in sources_after_update.json()["data"] if s["id"] == source_id)
    assert "pat" not in source_after_update["config"]
    assert "pat_encrypted" not in source_after_update["config"]


@pytest.mark.parametrize("submitted_pat", ["", "wiki-pat-secret"])
@pytest.mark.asyncio
async def test_pat_source_noop_update_preserves_sync_cursor(db, tmp_path, monkeypatch, submitted_pat):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    source_id = "src-confluence"
    source_config = prepare_source_config_for_storage(
        {
            "base_url": "https://wiki.example.test",
            "spaces": ["PAY"],
            "pat": "wiki-pat-secret",
        },
    )
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Engineering Wiki",
        config_json=json.dumps(source_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=1,
        docs_updated=1,
        docs_failed=0,
        memories_extracted=1,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-22T08:00:00+00:00",
        finished_at="2026-05-22T08:01:00+00:00",
    )

    app = create_admin_app(db=db, config=cfg)
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": submitted_pat,
                },
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is not None
    assert len(await db.get_sync_history(source_id)) == 1
    updated = await db.get_source(source_id)
    assert updated["config"] == source_config


@pytest.mark.asyncio
async def test_pat_replacement_preserves_jira_sync_cursor_when_scope_is_unchanged(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    source_id = "src-jira-pat-refresh"
    scope_config = {
        "base_url": "https://jira.example.test",
        "projects": ["PAY"],
        "jql_filter": "updated >= -90d",
        "issue_types": ["Story", "Bug"],
        "include_comments": True,
        "request_interval_ms": 750,
    }
    stored_config = prepare_source_config_for_storage(
        {**scope_config, "pat": "old-jira-pat"},
        secret_fields=("pat",),
    )
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Delivery Board",
        config_json=json.dumps(stored_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=125,
            docs_updated=125,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=125,
        docs_updated=125,
        docs_failed=0,
        memories_extracted=20,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-21T08:00:00+00:00",
        finished_at="2026-05-21T08:01:00+00:00",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Delivery Board",
                "config": {**scope_config, "auth_mode": "pat", "request_interval_ms": 1000, "pat": "old-jira-pat"},
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is not None
    assert len(await db.get_sync_history(source_id)) == 1
    updated = await db.get_source(source_id)
    assert updated["config"]["pat_encrypted"] == stored_config["pat_encrypted"]
    assert updated["config"]["request_interval_ms"] == 1000


@pytest.mark.asyncio
async def test_jira_pat_replacement_resets_sync_cursor_when_secret_changes(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    source_id = "src-jira-pat-principal-change"
    scope_config = {
        "base_url": "https://jira.example.test",
        "projects": ["PAY"],
        "auth_mode": "pat",
        "jql_filter": "updated >= -90d",
        "issue_types": ["Story", "Bug"],
        "include_comments": True,
    }
    stored_config = prepare_source_config_for_storage(
        {**scope_config, "pat": "old-jira-pat"},
        secret_fields=("jira_cookie", "pat"),
    )
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Delivery Board",
        config_json=json.dumps(stored_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=125,
            docs_updated=125,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=125,
        docs_updated=125,
        docs_failed=0,
        memories_extracted=20,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-21T08:00:00+00:00",
        finished_at="2026-05-21T08:01:00+00:00",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Delivery Board",
                "config": {**scope_config, "pat": "new-jira-pat"},
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is None
    assert await db.get_sync_history(source_id) == []
    updated = await db.get_source(source_id)
    assert updated["config"]["pat_encrypted"] != stored_config["pat_encrypted"]


@pytest.mark.asyncio
async def test_jira_auth_mode_change_resets_sync_cursor(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    source_id = "src-jira-auth-mode-change"
    pat_scope_config = {
        "base_url": "https://jira.example.test",
        "projects": ["PAY"],
        "auth_mode": "pat",
        "jql_filter": "updated >= -90d",
        "issue_types": ["Story", "Bug"],
        "include_comments": True,
    }
    stored_config = prepare_source_config_for_storage(
        {**pat_scope_config, "pat": "old-jira-pat"},
        secret_fields=("jira_cookie", "pat"),
    )
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Delivery Board",
        config_json=json.dumps(stored_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=125,
            docs_updated=125,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=125,
        docs_updated=125,
        docs_failed=0,
        memories_extracted=20,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-21T08:00:00+00:00",
        finished_at="2026-05-21T08:01:00+00:00",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Delivery Board",
                "config": {
                    **pat_scope_config,
                    "auth_mode": "browser_cookie",
                },
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is None
    assert await db.get_sync_history(source_id) == []
    updated = await db.get_source(source_id)
    assert updated["config"]["auth_mode"] == "browser_cookie"
    assert "jira_cookie_configured" not in updated["config"]
    assert "jira_cookie_encrypted" not in updated["config"]
    assert "pat_configured" not in updated["config"]
    assert "pat_encrypted" not in updated["config"]


@pytest.mark.asyncio
async def test_source_base_url_update_releases_old_atlassian_limiter(db, tmp_path, monkeypatch):
    from meminception.server import admin_api
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    released: list[tuple[str, str]] = []
    monkeypatch.setattr(
        admin_api,
        "release_atlassian_request_limiter",
        lambda base_url, *, owner_id: released.append((base_url, owner_id)),
    )
    source_id = "src-jira-base-url-change"
    scope_config = {
        "base_url": "https://old-jira.example.test",
        "projects": ["PAY"],
        "auth_mode": "browser_cookie",
        "jql_filter": "updated >= -90d",
        "issue_types": ["Story", "Bug"],
        "include_comments": True,
    }
    stored_config = prepare_source_config_for_storage(
        {**scope_config, "jira_cookie": "JSESSIONID=old"},
        secret_fields=("jira_cookie", "pat"),
    )
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Delivery Board",
        config_json=json.dumps(stored_config),
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Delivery Board",
                "config": {
                    **scope_config,
                    "base_url": "https://new-jira.example.test",
                },
            },
        )

    assert response.status_code == 200
    assert released == [("https://old-jira.example.test", source_id)]


@pytest.mark.asyncio
async def test_non_secret_source_noop_update_preserves_sync_cursor(db, tmp_path):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app

    source_id = "src-agent-sessions"
    source_config = {"documents_dir": str(tmp_path / "sessions")}
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Agent Sessions",
        config_json=json.dumps(source_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=1,
            docs_updated=1,
        ),
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Agent Sessions",
                "config": source_config,
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is not None
    updated = await db.get_source(source_id)
    assert updated["config"] == source_config


@pytest.mark.asyncio
async def test_run_source_sync_leaves_authentication_to_orchestrator(monkeypatch, tmp_path):
    from meminception import runtime
    from meminception.models import SyncState

    class FakeGene:
        def __init__(self) -> None:
            self.auth_calls = 0

        async def authenticate(self) -> None:
            self.auth_calls += 1

    class FakeRuntime:
        def orchestrator(self):
            return self

        async def sync_gene(self, *, gene, source_name, source_id, progress_callback=None, force_full_sync=False):
            await gene.authenticate()
            return SyncState(source=source_id, last_sync_status="success")

    gene = FakeGene()
    monkeypatch.setattr(runtime, "create_gene", lambda **_kwargs: gene)

    await runtime.run_source_sync(
        db=None,
        config=_config(tmp_path),
        source={"id": "src-agent-sessions", "type": "agent_session", "name": "Agent Sessions", "config": {}},
        runtime=FakeRuntime(),
    )

    assert gene.auth_calls == 1


@pytest.mark.asyncio
async def test_run_source_sync_decrypts_gene_declared_secret_fields(monkeypatch, tmp_path):
    from meminception import runtime
    from meminception.genes import GENE_REGISTRY
    from meminception.genes.base import Gene
    from meminception.models import (
        ConfigField,
        ConfigFieldType,
        ConfigGroup,
        ContentItem,
        GeneConfigSchema,
        GeneMetadata,
        NormalizedContent,
        RawContent,
        SyncState,
    )
    from meminception.source_secrets import prepare_source_config_for_storage

    class ApiKeyGene(Gene):
        @classmethod
        def metadata(cls) -> GeneMetadata:
            return GeneMetadata(
                name="api_key_gene",
                display_name="API Key Gene",
                description="Test gene",
                default_sync_interval_minutes=60,
                auth_method="api_key",
                data_shape="document",
            )

        @classmethod
        def config_schema(cls) -> GeneConfigSchema:
            return GeneConfigSchema(
                groups=[ConfigGroup(key="connection", label="Connection")],
                fields=[
                    ConfigField(
                        key="api_key",
                        label="API Key",
                        field_type=ConfigFieldType.SECRET,
                        group="connection",
                    )
                ],
            )

        async def authenticate(self) -> None:
            return None

        async def discover(self, since=None):
            if False:
                yield None

        async def fetch(self, item: ContentItem) -> RawContent:
            raise NotImplementedError

        async def normalize(self, raw: RawContent) -> NormalizedContent:
            raise NotImplementedError

    class FakeRuntime:
        gene = None

        def orchestrator(self):
            return self

        async def sync_gene(self, *, gene, source_name, source_id, progress_callback=None, force_full_sync=False):
            self.gene = gene
            return SyncState(source=source_id, last_sync_status="success")

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    monkeypatch.setitem(GENE_REGISTRY, "api_key_gene", ApiKeyGene)
    source_config = prepare_source_config_for_storage(
        {"api_key": "runtime-secret"},
        secret_fields=("api_key",),
    )
    fake_runtime = FakeRuntime()

    await runtime.run_source_sync(
        db=None,
        config=_config(tmp_path),
        source={
            "id": "src-api-key",
            "type": "api_key_gene",
            "name": "API Key",
            "config": source_config,
        },
        runtime=fake_runtime,
    )

    assert fake_runtime.gene.config["api_key"] == "runtime-secret"
    assert "api_key_encrypted" not in fake_runtime.gene.config


@pytest.mark.asyncio
async def test_gene_discovery_preview_runs_configured_gene_without_saving_source(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from meminception.genes import GENE_REGISTRY
    from meminception.genes.base import Gene
    from meminception.models import (
        ConfigField,
        ConfigFieldType,
        ConfigGroup,
        ContentItem,
        GeneConfigSchema,
        GeneMetadata,
        NormalizedContent,
        RawContent,
    )
    from meminception.server.admin_api import create_admin_app

    class PreviewGene(Gene):
        @classmethod
        def metadata(cls) -> GeneMetadata:
            return GeneMetadata(
                name="preview_gene",
                display_name="Preview Gene",
                description="Test preview gene",
                default_sync_interval_minutes=60,
                auth_method="none",
                data_shape="document",
            )

        @classmethod
        def config_schema(cls) -> GeneConfigSchema:
            return GeneConfigSchema(
                groups=[ConfigGroup(key="connection", label="Connection")],
                fields=[
                    ConfigField(
                        key="base_url",
                        label="Base URL",
                        field_type=ConfigFieldType.URL,
                        group="connection",
                    )
                ],
            )

        async def authenticate(self) -> None:
            self.config["authenticated"] = True

        async def discover(self, since=None):
            for index in range(3):
                yield ContentItem(
                    item_id=f"doc-{index}",
                    title=f"Doc {index}",
                    source_url=f"https://docs.example.test/{index}",
                    last_modified=datetime(2026, 5, 20 + index, tzinfo=timezone.utc),
                )

        async def fetch(self, item: ContentItem) -> RawContent:
            raise NotImplementedError

        async def normalize(self, raw: RawContent) -> NormalizedContent:
            raise NotImplementedError

    monkeypatch.setitem(GENE_REGISTRY, "preview_gene", PreviewGene)

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/genes/preview_gene/preview-discovery",
            json={"config": {"base_url": "https://docs.example.test"}, "limit": 2},
        )
        sources_response = client.get("/api/sources")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_type"] == "preview_gene"
    assert payload["count"] == 3
    assert payload["truncated"] is True
    assert [item["title"] for item in payload["items"]] == ["Doc 0", "Doc 1"]
    assert payload["items"][0]["last_modified"] == "2026-05-20T00:00:00+00:00"
    assert sources_response.json()["data"] == []


@pytest.mark.asyncio
async def test_github_pages_source_config_requires_scope_url_for_selected_mode(db, tmp_path):
    from meminception.server.admin_api import create_admin_app

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        missing_page = client.post(
            "/api/sources",
            json={
                "name": "Payroll Docs",
                "type": "github_pages",
                "config": {
                    "base_url": "https://github-pages.example.test/pages/org/repo",
                    "auth_mode": "none",
                    "sync_mode": "single_page",
                },
            },
        )
        valid = client.post(
            "/api/sources",
            json={
                "name": "Payroll Docs",
                "type": "github_pages",
                "config": {
                    "base_url": "https://github-pages.example.test/pages/org/repo",
                    "auth_mode": "none",
                    "sync_mode": "single_page",
                    "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
                },
            },
        )
        wrong_site_path = client.post(
            "/api/sources",
            json={
                "name": "Other Docs",
                "type": "github_pages",
                "config": {
                    "base_url": "https://github-pages.example.test/pages/org/repo",
                    "auth_mode": "none",
                    "sync_mode": "single_page",
                    "page_url": "https://github-pages.example.test/pages/other/repo/process-tracking/",
                },
            },
        )
        missing_pat = client.post(
            "/api/sources",
            json={
                "name": "PAT Docs",
                "type": "github_pages",
                "config": {
                    "base_url": "https://github-pages.example.test/pages/org/repo",
                    "auth_mode": "github_pat",
                    "sync_mode": "single_page",
                    "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
                },
            },
        )

    assert missing_page.status_code == 400
    assert "Page URL is required" in missing_page.json()["detail"]
    assert valid.status_code == 200
    assert wrong_site_path.status_code == 400
    assert "configured site path" in wrong_site_path.json()["detail"]
    assert missing_pat.status_code == 400
    assert "Personal Access Token is required" in missing_pat.json()["detail"]


@pytest.mark.asyncio
async def test_source_config_update_resets_incremental_sync_cursor(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    source_id = "src-jira-rescope"
    old_config = {
        "base_url": "https://jira.example",
        "projects": ["PAY"],
        "jql_filter": "updated >= -180d",
    }
    stored_config = prepare_source_config_for_storage(
        {**old_config, "pat": "jira-pat-secret"},
        secret_fields=("pat",),
    )
    await db.upsert_source(
        id=source_id,
        type="jira",
        name="Delivery Board",
        config_json=json.dumps(stored_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=10,
            docs_updated=10,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=10,
        docs_updated=10,
        docs_failed=0,
        memories_extracted=4,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-21T08:00:00+00:00",
        finished_at="2026-05-21T08:01:00+00:00",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Delivery Board",
                "config": {
                    **old_config,
                    "jql_filter": "updated >= -90d",
                },
            },
        )
        sources_response = client.get("/api/sources")

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is None
    updated = await db.get_source(source_id)
    assert updated["last_sync"] is None
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert source_payload["sync"] is None


@pytest.mark.asyncio
async def test_force_resync_resets_cursor_and_starts_source_sync(db, tmp_path):
    from datetime import datetime, timezone

    from meminception.models import SyncState
    from meminception.server.admin_api import create_admin_app

    source_id = "src-force-resync"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Force Resync",
        config_json=json.dumps({"base_url": "https://wiki.example", "spaces": ["PAY"]}),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=7,
            docs_updated=2,
        ),
    )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=7,
        docs_updated=2,
        docs_failed=0,
        memories_extracted=3,
        error_message=None,
        failed_docs=None,
        started_at="2026-05-27T01:00:00+00:00",
        finished_at="2026-05-27T01:01:00+00:00",
    )

    class FakeSyncService:
        def __init__(self):
            self.started: list[str] = []

        def is_running(self, source_id: str):
            return False

        def start_source(self, started_source_id: str):
            self.started.append(started_source_id)

        async def shutdown(self):
            return None

    fake_sync_service = FakeSyncService()
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        app.state.sync_service = fake_sync_service
        response = client.post(f"/api/sources/{source_id}/force-resync")
        sources_response = client.get("/api/sources")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "message": "Force resync started", "source_id": source_id}
    assert fake_sync_service.started == [source_id]
    assert await db.get_sync_state(source_id) is None
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert source_payload["sync"] is None


def test_schedule_trigger_uses_configured_daily_time():
    from meminception.scheduler import build_schedule_trigger

    trigger = build_schedule_trigger({
        "enabled": True,
        "frequency": "daily",
        "time": "03:45",
        "day_of_week": 2,
        "timezone": "UTC",
    })

    fields = {field.name: str(field) for field in trigger.fields}
    assert fields["hour"] == "3"
    assert fields["minute"] == "45"


def test_config_env_overrides_startup_runtime_values(monkeypatch, tmp_path):
    from meminception.config import load_config

    monkeypatch.setenv("MEMINCEPTION_BASE_DIR", str(tmp_path / "env-base"))
    monkeypatch.setenv("MEMINCEPTION_ENRICHMENT_BASE_URL", "http://localhost:6655/anthropic")
    monkeypatch.setenv("MEMINCEPTION_EMBEDDING_BASE_URL", "http://localhost:6655/openai/v1")
    monkeypatch.setenv("MEMINCEPTION_ADMIN_API_PORT", "9876")

    cfg = load_config(base_dir=tmp_path / "ignored")

    assert cfg.base_dir == tmp_path / "env-base"
    assert cfg.llm.enrichment_base_url == "http://localhost:6655/anthropic"
    assert cfg.llm.embedding_base_url == "http://localhost:6655/openai/v1"
    assert cfg.server.admin_api_port == 9876


def test_cli_logging_uses_stderr_for_stdio_safety():
    from meminception.main import setup_logging

    root = logging.getLogger()
    previous_handlers = root.handlers[:]
    previous_level = root.level
    try:
        setup_logging(verbose=False)
        handler = next(h for h in root.handlers if isinstance(h, RichHandler))
        assert handler.console.stderr is True
    finally:
        root.handlers = previous_handlers
        root.setLevel(previous_level)
