"""Startup/runtime wiring tests."""

from __future__ import annotations

import json
from pathlib import Path
import logging

import pytest
from rich.logging import RichHandler
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Memory, Visibility, content_hash
from memforge.storage.database import Database

TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="


def _project_binding(project_key: str = "PAY") -> dict[str, str]:
    return {"mode": "fixed", "project_key": project_key}


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
    from memforge import runtime

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
    from memforge import runtime

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
    from memforge.config import AppConfig
    from memforge.runtime import build_sync_runtime

    captured = {}

    class RecordingStructuredClient:
        def __init__(self, config):
            captured["config"] = config
            captured["client"] = self

        async def verify_source_support(self, prompt: str):
            raise AssertionError("not called during runtime construction")

    monkeypatch.setattr("memforge.runtime.LiteLlmStructuredClient", RecordingStructuredClient)
    monkeypatch.setattr("memforge.runtime.get_chroma_collection", lambda **kwargs: FakeCollection())

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


@pytest.mark.asyncio
async def test_build_sync_runtime_wires_env_backed_provider_model_without_api_key(
    db,
    tmp_path,
    monkeypatch,
):
    from memforge.config import AppConfig
    from memforge.runtime import build_sync_runtime

    captured = {}

    class RecordingStructuredClient:
        def __init__(self, config):
            captured["config"] = config
            captured["client"] = self

    monkeypatch.setattr("memforge.runtime.LiteLlmStructuredClient", RecordingStructuredClient)
    monkeypatch.setattr("memforge.runtime.get_chroma_collection", lambda **kwargs: FakeCollection())

    config = AppConfig()
    config.base_dir = tmp_path
    config.storage.db_path = str(tmp_path / "mem.db")
    config.storage.chroma_path = str(tmp_path / "chroma")
    config.storage.docs_path = str(tmp_path / "docs")
    config.llm.enrichment_model = "provider/chat-model"
    config.llm.enrichment_base_url = ""
    config.llm.enrichment_api_key = ""

    runtime = await build_sync_runtime(db, config)

    assert runtime.structured_llm_client is captured["client"]
    assert runtime.memory_extractor.structured_llm_client is captured["client"]
    assert runtime.enricher.structured_llm_client is captured["client"]
    assert captured["config"].model == "provider/chat-model"
    assert captured["config"].base_url is None
    assert captured["config"].api_key is None


def test_enrichment_and_extraction_clients_bound_request_timeout(monkeypatch):
    """Document LLM clients should use the configured request timeout."""
    from memforge.pipeline.enricher import Enricher
    from memforge.pipeline.memory_extractor import MemoryExtractor

    enricher = Enricher(api_key="test-key", request_timeout_s=42.0)
    extractor = MemoryExtractor(api_key="test-key", request_timeout_s=42.0)

    assert enricher.structured_llm_client.config.timeout_s == 42.0
    assert extractor.structured_llm_client.config.timeout_s == 42.0


def test_sync_runtime_uses_injected_orchestrator_factory():
    from memforge.runtime import SyncRuntime

    class SentinelOrchestrator:
        pass

    sentinel = SentinelOrchestrator()
    runtime = SyncRuntime(
        db=object(),
        config=AppConfig(),
        doc_store=object(),
        enricher=object(),
        memory_extractor=object(),
        memory_store=object(),
        memory_engine=object(),
        vector_store=object(),
        embed_cfg={},
        structured_llm_client=None,
        llm_model="test-model",
        source_support_detector=None,
        orchestrator_factory=lambda _runtime: sentinel,
    )

    assert runtime.orchestrator() is sentinel


def test_admin_app_lifespan_owns_database_and_sync_service(tmp_path):
    """The API startup path should open/close DB resources through FastAPI lifespan."""
    from memforge.server.admin_api import create_admin_app

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

    from memforge.memory.audit import MemoryAuditEvent
    from memforge.server.admin_api import create_admin_app

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
    assert payload["audit_failures"]["payload"]["counts_by_event_type"] == {"source_support_verification_failed": 1}
    assert payload["audit_failures"]["payload"]["total"] == 1
    assert payload["audit_failures"]["payload"]["last_seen_at"]


@pytest.mark.asyncio
async def test_health_ignores_old_audit_failures(db, tmp_path):
    from datetime import datetime, timedelta, timezone

    from memforge.memory.audit import MemoryAuditEvent
    from memforge.server.admin_api import create_admin_app

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
    from memforge.scheduler import EXPIRY_JOB_ID
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app):
        assert app.state.sync_scheduler.scheduler.get_job(EXPIRY_JOB_ID) is not None


def test_admin_app_can_disable_scheduler_for_external_worker_mode(tmp_path):
    from memforge.server.admin_api import create_admin_app

    config = _config(tmp_path)
    config.sync.scheduler_enabled = False
    app = create_admin_app(config=config)

    with TestClient(app):
        assert app.state.sync_scheduler is None


def test_gene_config_schema_hides_runtime_transport_fields_from_ui(tmp_path):
    from memforge.server.admin_api import create_admin_app

    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/genes/confluence/config-schema")

    assert response.status_code == 200
    fields = {field["key"]: field for field in response.json()["fields"]}
    assert fields["pat"]["advanced"] is False
    assert "api_prefix" not in fields
    assert "tls_ca_bundle" not in fields


def test_admin_source_create_encrypts_and_redacts_pat(tmp_path, monkeypatch):
    import sqlite3

    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
                "project_binding": _project_binding(),
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


def test_admin_source_create_and_update_persist_sync_schedule(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Scheduled Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "spaces": ["PAY"],
                    "pat": "wiki-pat-secret",
                },
                "project_binding": _project_binding(),
                "sync_schedule": {"enabled": True, "interval_minutes": 60},
            },
        )
        assert create_response.status_code == 200, create_response.text
        source_id = create_response.json()["id"]
        created_sources = client.get("/api/sources").json()["data"]
        created = next(source for source in created_sources if source["id"] == source_id)
        next_run_at = created["sync_schedule"]["next_run_at"]

        update_response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Scheduled Wiki Renamed",
                "sync_schedule": {"enabled": True, "interval_minutes": 60},
            },
        )
        assert update_response.status_code == 200, update_response.text
        updated_sources = client.get("/api/sources").json()["data"]
        updated = next(source for source in updated_sources if source["id"] == source_id)

        disable_response = client.put(
            f"/api/sources/{source_id}",
            json={"sync_schedule": None},
        )
        assert disable_response.status_code == 200, disable_response.text
        disabled_sources = client.get("/api/sources").json()["data"]
        disabled = next(source for source in disabled_sources if source["id"] == source_id)

    assert created["sync_schedule"]["enabled"] is True
    assert created["sync_schedule"]["interval_minutes"] == 60
    assert next_run_at is not None
    assert updated["name"] == "Scheduled Wiki Renamed"
    assert updated["sync_schedule"]["next_run_at"] == next_run_at
    assert disabled["sync_schedule"]["enabled"] is False
    assert disabled["sync_schedule"]["next_run_at"] is None


def test_admin_source_update_preserves_encrypted_pat_when_blank(tmp_path, monkeypatch):
    import sqlite3

    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
                "project_binding": _project_binding(),
            },
        )
        assert create_response.status_code == 200
        source_id = create_response.json()["id"]

        with sqlite3.connect(cfg.storage.db_path) as conn:
            conn.row_factory = sqlite3.Row
            before = json.loads(
                conn.execute(
                    "SELECT config FROM sources WHERE id = ?",
                    (source_id,),
                ).fetchone()["config"]
            )

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
        after = json.loads(
            conn.execute(
                "SELECT config FROM sources WHERE id = ?",
                (source_id,),
            ).fetchone()["config"]
        )

    assert after["pat_encrypted"] == before["pat_encrypted"]
    assert "pat" not in after
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert "pat_encrypted" not in source_payload["config"]
    assert source_payload["config"]["pat_configured"] is True


def test_admin_source_update_persists_status(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

        update_response = client.put(
            f"/api/sources/{source_id}",
            json={"status": "paused"},
        )
        sources_response = client.get("/api/sources")

    assert update_response.status_code == 200
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert source_payload["status"] == "paused"


def test_admin_source_update_rejects_unknown_status(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

        update_response = client.put(
            f"/api/sources/{source_id}",
            json={"status": "disabled"},
        )

    assert update_response.status_code == 400
    assert "Invalid source status" in update_response.json()["detail"]


def test_admin_source_sync_rejects_paused_source(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
        assert client.put(f"/api/sources/{source_id}", json={"status": "paused"}).status_code == 200

        sync_response = client.post(f"/api/sources/{source_id}/sync")

    assert sync_response.status_code == 400
    assert sync_response.json()["detail"] == "Source is paused"


@pytest.mark.asyncio
async def test_agent_session_document_intake_is_retired_before_source_status(db, tmp_path):
    from memforge.agent_sessions import agent_session_source_id
    from memforge.server.admin_api import create_admin_app

    source_id = agent_session_source_id("codex")
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Codex Session",
        config_json=json.dumps({"documents_dir": str(tmp_path / "sessions"), "client": "codex"}),
        status="paused",
    )

    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/agent-sessions/documents",
            json={
                "client": "codex",
                "session_id": "session-1",
                "trigger": "compact",
                "workspace": "/repo",
                "document_markdown": "# Summary\n\nUseful durable notes.",
                "process_now": True,
            },
        )

    assert response.status_code == 410
    assert "agent-session document intake has been retired" in response.json()["detail"]


@pytest.mark.asyncio
async def test_agent_session_window_intake_rejects_paused_source_before_llm(
    db,
    tmp_path,
):
    from memforge.agent_sessions import agent_session_source_id
    from memforge.server.admin_api import create_admin_app

    source_id = agent_session_source_id("claude-code")
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Claude Code Session",
        config_json=json.dumps({"documents_dir": str(tmp_path / "sessions"), "client": "claude-code"}),
        status="paused",
    )

    class FailingWindowClient:
        async def generate_agent_knowledge_patch(self, *args, **kwargs):
            raise AssertionError("paused source should be rejected before LLM work")

    app = create_admin_app(db=db, config=_config(tmp_path))
    app.state.agent_session_window_client = FailingWindowClient()
    with TestClient(app) as client:
        response = client.post(
            "/api/agent-sessions/windows",
            json={
                "client": "claude-code",
                "session_id": "session-1",
                "trigger": "compact",
                "workspace": "/repo",
                "events": [{"kind": "decision", "text": "Keep the source paused."}],
                "process_now": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Source is paused"


@pytest.mark.asyncio
async def test_sync_service_rejects_paused_source_at_start_boundary(db, tmp_path):
    from memforge.runtime import SourcePausedError, SyncService

    source_id = "src-paused-boundary"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Paused Boundary",
        config_json=json.dumps({"base_url": "https://wiki.example.test", "spaces": ["PAY"]}),
        status="paused",
    )

    sync_service = SyncService(db, _config(tmp_path))

    with pytest.raises(SourcePausedError):
        await sync_service.start_source(source_id)

    assert not sync_service.is_running(source_id)


@pytest.mark.asyncio
async def test_sync_service_does_not_queue_paused_source(db, tmp_path):
    from memforge.runtime import SyncService

    source_id = "src-paused-queued"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Paused Queued",
        config_json=json.dumps({"base_url": "https://wiki.example.test", "spaces": ["PAY"]}),
        status="paused",
    )

    sync_service = SyncService(db, _config(tmp_path))

    assert await sync_service.request_source_sync(source_id) is False


@pytest.mark.asyncio
async def test_queued_sync_stops_quietly_if_source_is_paused_before_execution(db, tmp_path, monkeypatch):
    import memforge.runtime as runtime
    from memforge.runtime import SourceSyncWorker, SyncService

    source_id = "src-paused-after-queue"
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Paused After Queue",
        config_json=json.dumps({"base_url": "https://wiki.example.test", "spaces": ["PAY"]}),
        status="active",
    )

    logged_errors: list[tuple] = []
    monkeypatch.setattr(
        runtime.logger,
        "exception",
        lambda *args, **kwargs: logged_errors.append((args, kwargs)),
    )
    sync_service = SyncService(db, _config(tmp_path))

    assert await sync_service.request_source_sync(source_id, delay_seconds=0) is True
    queued_run = await db.enqueue_source_sync_run(source_id=source_id, trigger="request")
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="Paused After Queue",
        config_json=json.dumps({"base_url": "https://wiki.example.test", "spaces": ["PAY"]}),
        status="paused",
    )
    worker = SourceSyncWorker(db, _config(tmp_path), worker_id="test-worker")
    await worker.run_once()
    completed = await db.get_source_sync_run(queued_run.run_id)

    assert logged_errors == []
    assert completed is not None
    assert completed.status == "failed"
    assert completed.error_message == f"Source is paused: {source_id}"
    assert not sync_service.is_running(source_id)


def test_admin_source_save_rejects_missing_tls_ca_bundle(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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


def test_admin_source_create_accepts_confluence_page_url_without_spaces(tmp_path, monkeypatch):
    import sqlite3

    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    cfg = _config(tmp_path)
    app = create_admin_app(config=cfg)

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Payroll Architecture",
                "config": {
                    "base_url": ("https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll"),
                    "pat": "wiki-pat-secret",
                },
                "project_binding": _project_binding(),
            },
        )

    assert response.status_code == 200
    with sqlite3.connect(cfg.storage.db_path) as conn:
        conn.row_factory = sqlite3.Row
        stored = conn.execute(
            "SELECT config FROM sources WHERE id = ?",
            (response.json()["id"],),
        ).fetchone()

    stored_config = json.loads(stored["config"])
    assert stored_config["base_url"] == "https://wiki.company.example"
    assert stored_config["api_prefix"] == "/wiki"
    assert stored_config["spaces"] == ["PAY"]
    assert stored_config["page_tree_root"] == "5695886009"
    assert stored_config["sync_mode"] == "page_tree"


def test_admin_source_create_requires_spaces_for_confluence_space_scope(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "sync_mode": "space",
                    "pat": "wiki-pat-secret",
                },
            },
        )

    assert response.status_code == 400
    assert "Spaces to Sync is required" in response.json()["detail"]


def test_admin_source_create_allows_jira_browser_session_without_source_cookie(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
                "project_binding": _project_binding(),
            },
        )

    assert response.status_code == 200, response.text


def test_admin_source_create_rejects_missing_required_source_scope(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
                "project_binding": _project_binding(),
            },
        )
        source_id = response.json().get("id")
        sources = client.get("/api/sources").json()["data"]

    assert response.status_code == 200, response.text
    source = next(source for source in sources if source["id"] == source_id)
    assert "jira_cookie_configured" not in source["config"]


def test_admin_source_create_rejects_forged_jira_pat_configured_flag(tmp_path, monkeypatch):
    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server.admin_api import create_admin_app

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
    from memforge.server.admin_api import create_admin_app

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
                id, memory_type, content, content_hash, tags,
                project_key, confidence, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"mem-{index}",
                "fact",
                f"Memory {index}",
                f"memory-hash-{index}",
                "[]",
                None,
                0.8,
                "active",
                "2026-05-28T07:00:00+00:00",
                "2026-05-28T07:00:00+00:00",
            ),
        )
        await db.db.execute(
            """
            INSERT INTO memory_sources (
                memory_id, doc_id, source_type, source_id, excerpt
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (f"mem-{index}", doc_id, "github_pages", source_id, f"Excerpt {index}"),
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


@pytest.mark.asyncio
async def test_admin_source_memory_count_matches_viewer_scoped_memory_list(db, tmp_path):
    """Source counts and memory-list totals must share one visibility contract.

    A sync history row reports how many memories the latest run extracted. The
    source ``memory_count`` is different: it is the durable count visible to the
    current viewer for that source. This pins the route-level contract so every
    store adapter implements the same source-provenance + visibility semantics.
    """

    from memforge.server.admin_api import create_admin_app

    source_id = "src-private-sessions"
    now = "2026-06-20T10:00:00+00:00"
    await db.upsert_source(
        id=source_id,
        type="agent_session",
        name="Codex Session",
        config_json=json.dumps({}),
    )
    for index, owner in enumerate(["viewer-a", "viewer-a", "viewer-b"], start=1):
        doc_id = f"doc-private-{index}"
        memory_id = f"mem-private-{index}"
        await db.db.execute(
            """INSERT INTO documents
               (doc_id, source, source_url, title, space_or_project,
                last_modified, version, content_hash, last_synced, client)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id,
                source_id,
                f"agent-session://codex/session-{index}",
                f"Session {index}",
                "github.com/example/repo",
                now,
                "1",
                f"doc-hash-{index}",
                now,
                "codex",
            ),
        )
        await db.insert_memory(
            Memory(
                id=memory_id,
                memory_type="fact",
                content=f"Private memory {index}",
                content_hash=content_hash(f"Private memory {index}"),
                visibility=Visibility.PRIVATE.value,
                owner_user_id=owner,
                confidence=0.9,
                status="active",
            )
        )
        await db.add_memory_source(
            memory_id,
            doc_id,
            "agent_session",
            f"Excerpt {index}",
            source_updated_at=None,
        )
    await db.insert_sync_history(
        source=source_id,
        status="success",
        docs_processed=1,
        docs_updated=1,
        docs_failed=0,
        memories_extracted=3,
        error_message=None,
        failed_docs=[],
        started_at="2026-06-20T10:01:00+00:00",
        finished_at="2026-06-20T10:02:00+00:00",
    )
    await db.db.commit()

    app = create_admin_app(
        db=db,
        config=_config(tmp_path),
        principal_resolver=lambda request: "viewer-a",
    )

    with TestClient(app) as client:
        sources_response = client.get("/api/sources")
        memories_response = client.get(
            "/api/memories",
            params={
                "source": source_id,
                "include_private": "true",
                "limit": 20,
            },
        )

    assert sources_response.status_code == 200
    assert memories_response.status_code == 200
    source = next(item for item in sources_response.json()["data"] if item["id"] == source_id)
    memories = memories_response.json()
    assert source["sync"]["memories_extracted"] == 3
    assert memories["total"] == 2
    assert source["memory_count"] == 2
    assert {item["id"] for item in memories["data"]} == {
        "mem-private-1",
        "mem-private-2",
    }


def test_source_secret_field_policy_is_gene_driven_for_known_sources():
    from memforge.server.admin_api import _source_secret_fields, _validate_source_config

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

    from memforge.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
        stored = json.loads(
            conn.execute(
                "SELECT config FROM sources WHERE id = ?",
                (source_id,),
            ).fetchone()["config"]
        )
    assert "pat" not in stored
    assert stored["pat_encrypted"] != "new-plain"

    source_after_update = next(s for s in sources_after_update.json()["data"] if s["id"] == source_id)
    assert "pat" not in source_after_update["config"]
    assert "pat_encrypted" not in source_after_update["config"]


@pytest.mark.parametrize("submitted_pat", ["", "wiki-pat-secret"])
@pytest.mark.asyncio
async def test_pat_source_noop_update_preserves_sync_cursor(db, tmp_path, monkeypatch, submitted_pat):
    from datetime import datetime, timezone

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

    from memforge.local_adapter import default_local_adapter_inbox
    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.storage.adapters.context import LOCAL_DEV_USER_ID

    source_id = "src-local-markdown"
    cfg = _config(tmp_path)
    source_config = {
        "root": str(tmp_path / "engineering-notes"),
        "vault_id": "engineering-notes",
        "documents_dir": str(default_local_adapter_inbox(cfg, source_id)),
    }
    await db.upsert_source(
        id=source_id,
        type="local_markdown",
        name="Engineering Notes",
        config_json=json.dumps(source_config),
        created_by_user_id=LOCAL_DEV_USER_ID,
        execution_owner_user_id=LOCAL_DEV_USER_ID,
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

    app = create_admin_app(db=db, config=cfg)
    with TestClient(app) as client:
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "Engineering Notes",
                "config": source_config,
            },
        )

    assert response.status_code == 200
    assert await db.get_sync_state(source_id) is not None
    updated = await db.get_source(source_id)
    assert updated["config"] == source_config


@pytest.mark.asyncio
async def test_run_source_sync_leaves_authentication_to_orchestrator(monkeypatch, tmp_path):
    from memforge import runtime
    from memforge.models import SyncState

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
    from memforge import runtime
    from memforge.genes import GENE_REGISTRY
    from memforge.genes.base import Gene
    from memforge.models import (
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
    from memforge.source_secrets import prepare_source_config_for_storage

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

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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

    from memforge.genes import GENE_REGISTRY
    from memforge.genes.base import Gene
    from memforge.models import (
        ConfigField,
        ConfigFieldType,
        ConfigGroup,
        ContentItem,
        GeneConfigSchema,
        GeneMetadata,
        NormalizedContent,
        RawContent,
    )
    from memforge.server.admin_api import create_admin_app

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
    from memforge.server.admin_api import create_admin_app

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
                "project_binding": _project_binding(),
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

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
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
async def test_source_scope_update_cancels_active_sync_before_reset(db, tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app
    from memforge.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    source_id = "src-confluence-rescope"
    old_config = {
        "base_url": "https://wiki.example",
        "sync_mode": "page_tree",
        "page_tree_root": "5695886009",
        "include_children": True,
        "spaces": ["SFPAY"],
    }
    stored_config = prepare_source_config_for_storage(
        {**old_config, "pat": "confluence-pat-secret"},
        secret_fields=("pat",),
    )
    await db.upsert_source(
        id=source_id,
        type="confluence",
        name="SFPAY Arch",
        config_json=json.dumps(stored_config),
    )
    await db.upsert_sync_state(
        SyncState(
            source=source_id,
            last_sync_at=datetime.now(timezone.utc),
            last_sync_status="success",
            docs_processed=17,
            docs_updated=10,
        ),
    )

    class FakeSyncService:
        def __init__(self):
            self.cancelled: list[str] = []

        async def cancel_source(self, cancelled_source_id: str):
            self.cancelled.append(cancelled_source_id)

        async def shutdown(self):
            return None

    fake_sync_service = FakeSyncService()
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        app.state.sync_service = fake_sync_service
        response = client.put(
            f"/api/sources/{source_id}",
            json={
                "name": "SFPAY Arch",
                "config": {
                    **old_config,
                    "page_tree_root": "5625394036",
                },
            },
        )

    assert response.status_code == 200
    assert fake_sync_service.cancelled == [source_id]
    assert await db.get_sync_state(source_id) is None


@pytest.mark.asyncio
async def test_force_resync_preserves_existing_sync_state_until_new_run_succeeds(db, tmp_path):
    from datetime import datetime, timezone

    from memforge.models import SyncState
    from memforge.server.admin_api import create_admin_app

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
            self.enqueued: list[tuple[str, str, bool]] = []

        def is_running(self, source_id: str):
            return False

        async def enqueue_source(
            self,
            enqueued_source_id: str,
            *,
            trigger: str = "manual",
            force_full_sync: bool = False,
            workspace_id: str = "default",
        ):
            del workspace_id
            self.enqueued.append((enqueued_source_id, trigger, force_full_sync))

            class Run:
                run_id = "run-force-resync"
                status = "pending"
                coalesced = False

            return Run()

        async def shutdown(self):
            return None

    fake_sync_service = FakeSyncService()
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        app.state.sync_service = fake_sync_service
        response = client.post(f"/api/sources/{source_id}/force-resync")
        sources_response = client.get("/api/sources")

    assert response.status_code == 202
    assert response.json() == {
        "ok": True,
        "message": "Sync enqueued",
        "source_id": source_id,
        "run_id": "run-force-resync",
        "status": "pending",
        "coalesced": False,
    }
    assert fake_sync_service.enqueued == [(source_id, "force", True)]
    assert await db.get_sync_state(source_id) is not None
    source_payload = next(s for s in sources_response.json()["data"] if s["id"] == source_id)
    assert source_payload["sync"]["status"] == "success"


def test_schedule_trigger_uses_configured_daily_time():
    from memforge.scheduler import build_schedule_trigger

    trigger = build_schedule_trigger(
        {
            "enabled": True,
            "frequency": "daily",
            "time": "03:45",
            "day_of_week": 2,
            "timezone": "UTC",
        }
    )

    fields = {field.name: str(field) for field in trigger.fields}
    assert fields["hour"] == "3"
    assert fields["minute"] == "45"


def test_config_env_overrides_startup_runtime_values(monkeypatch, tmp_path):
    from memforge.config import load_config

    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "env-base"))
    monkeypatch.setenv("MEMFORGE_ENRICHMENT_BASE_URL", "http://localhost:6655/anthropic")
    monkeypatch.setenv("MEMFORGE_EMBEDDING_BASE_URL", "http://localhost:6655/openai/v1")
    monkeypatch.setenv("MEMFORGE_ADMIN_API_PORT", "9876")

    cfg = load_config(base_dir=tmp_path / "ignored")

    assert cfg.base_dir == tmp_path / "env-base"
    assert cfg.llm.enrichment_base_url == "http://localhost:6655/anthropic"
    assert cfg.llm.embedding_base_url == "http://localhost:6655/openai/v1"
    assert cfg.server.admin_api_port == 9876


@pytest.mark.asyncio
async def test_llm_config_probe_fetches_model_ids(db, tmp_path, monkeypatch):
    import httpx

    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class ModelListClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-5-mini"},
                        {"id": "text-embedding-3-small"},
                    ]
                },
                request=request,
            )

    monkeypatch.setattr(admin_api.httpx, "AsyncClient", lambda **kwargs: ModelListClient())
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/llm-config/probe",
            json={
                "kind": "embedding",
                "base_url": "https://proxy.example.test/v1",
                "api_key": "proxy-key",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["models_supported"] is True
    assert [model["id"] for model in payload["models"]] == [
        "gpt-5-mini",
        "text-embedding-3-small",
    ]


@pytest.mark.asyncio
async def test_llm_config_probe_treats_missing_models_as_manual_fallback(db, tmp_path, monkeypatch):
    import httpx

    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class MissingModelsClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            return httpx.Response(404, json={"error": "not found"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(admin_api.httpx, "AsyncClient", lambda **kwargs: MissingModelsClient())
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/llm-config/probe",
            json={"kind": "enrichment", "base_url": "https://proxy.example.test", "api_key": "proxy-key"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["models_supported"] is False
    assert payload["models"] == []
    assert "does not expose a model list" in payload["message"]


@pytest.mark.asyncio
async def test_llm_config_probe_falls_through_from_html_models_to_v1_models(db, tmp_path, monkeypatch):
    import httpx

    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class HtmlThenModelsClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            request = httpx.Request("GET", url)
            if url.endswith("/v1/models"):
                return httpx.Response(200, json={"data": [{"id": "model-from-v1"}]}, request=request)
            return httpx.Response(200, content=b"<html>not json</html>", request=request)

    monkeypatch.setattr(admin_api.httpx, "AsyncClient", lambda **kwargs: HtmlThenModelsClient())
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/llm-config/probe",
            json={"kind": "enrichment", "base_url": "https://proxy.example.test", "api_key": "proxy-key"},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["models_supported"] is True
    assert [model["id"] for model in payload["models"]] == ["model-from-v1"]


@pytest.mark.asyncio
async def test_llm_config_probe_suggests_host_docker_internal(db, tmp_path, monkeypatch):
    import httpx

    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class ConnectionFailureClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(admin_api.httpx, "AsyncClient", lambda **kwargs: ConnectionFailureClient())
    monkeypatch.setattr(admin_api, "_is_running_in_container", lambda: True)
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/llm-config/probe",
            json={"kind": "embedding", "base_url": "http://localhost:6655/openai/v1", "api_key": None},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["stage"] == "connect"
    assert payload["suggested_base_url"] == "http://host.docker.internal:6655/openai/v1"


@pytest.mark.asyncio
async def test_llm_config_probe_auth_error_without_key_is_actionable(db, tmp_path, monkeypatch):
    import httpx

    from memforge.server import admin_api
    from memforge.server.admin_api import create_admin_app

    class AuthFailureClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers):
            return httpx.Response(401, json={"error": "missing key"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(admin_api.httpx, "AsyncClient", lambda **kwargs: AuthFailureClient())
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/llm-config/probe",
            json={"kind": "enrichment", "base_url": "https://api.example.test/v1", "api_key": None},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert payload["stage"] == "auth"
    assert payload["message"] == "Add an API key, then test again."


@pytest.mark.asyncio
async def test_llm_config_put_can_preserve_and_clear_keys(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    await db.set_llm_config(
        {
            "enrichment_model": "chat-model",
            "enrichment_base_url": "https://chat.example.test/v1",
            "enrichment_api_key": "chat-secret",
            "embedding_model": "embed-model",
            "embedding_base_url": "https://embed.example.test/v1",
            "embedding_api_key": "embed-secret",
        }
    )
    app = create_admin_app(db=db, config=_config(tmp_path))

    with TestClient(app) as client:
        preserve_response = client.put("/api/llm-config", json={"embedding_api_key": None})
        clear_response = client.put("/api/llm-config", json={"enrichment_api_key": ""})

    stored = await db.get_llm_config()
    assert preserve_response.status_code == 200
    assert clear_response.status_code == 200
    assert stored["embedding_api_key"] == "embed-secret"
    assert stored["enrichment_api_key"] is None


@pytest.mark.asyncio
async def test_llm_config_put_can_be_disabled_for_deployment_managed_config(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    await db.set_llm_config(
        {
            "enrichment_model": "chat-model",
            "enrichment_base_url": "https://chat.example.test/v1",
            "enrichment_api_key": "chat-secret",
            "embedding_model": "embed-model",
            "embedding_base_url": "https://embed.example.test/v1",
            "embedding_api_key": "embed-secret",
        }
    )
    cfg = _config(tmp_path)
    cfg.server.llm_config_writable = False
    app = create_admin_app(db=db, config=cfg)

    with TestClient(app) as client:
        response = client.put(
            "/api/llm-config",
            json={"enrichment_model": "replacement-model"},
        )

    stored = await db.get_llm_config()
    assert response.status_code == 405
    assert response.json()["detail"] == "LLM settings are managed by the deployment environment"
    assert stored["enrichment_model"] == "chat-model"


def test_cli_logging_uses_stderr_for_stdio_safety():
    from memforge.main import setup_logging

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
