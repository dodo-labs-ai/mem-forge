"""Local Markdown Source configuration and local-agent Artifact uploads through the admin API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from tests.leased_job_store import LeasedJobStore
from memforge.config import AppConfig
from memforge.local_agent.source_contract import LocalAgentLease
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    # These tests exercise the adapter HTTP/package boundary. Background
    # scheduling and durable-worker behavior have dedicated contract suites;
    # starting them here adds unrelated concurrent database activity and makes
    # TestClient shutdown nondeterministic on Linux runners.
    cfg.sync.scheduler_enabled = False
    cfg.sync.worker_enabled = False
    return cfg


def _connect_database(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    return database


def _create_local_markdown_source(
    client: TestClient, *, name: str = "Engineering notes", vault_id: str = "engineering"
) -> dict:
    response = client.post(
        "/api/v1/sources",
        json={
            "type": "local_markdown",
            "name": name,
            "access_policy": "private",
            "config": {
                "root": "/Users/test/engineering-notes",
                "vault_id": vault_id,
                "display_label": "Engineering notes",
            },
            "project_binding": {"mode": "fixed", "project_key": vault_id.upper()},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_teams_source(client: TestClient, *, name: str = "Teams Channel") -> dict:
    response = client.post(
        "/api/v1/sources",
        json={
            "type": "teams",
            "name": name,
            "access_policy": "private",
            "config": {
                "region": "emea",
                "channels": ["19:channel@example.test"],
                "conversation_gap_minutes": 60,
            },
            "project_binding": {"mode": "fixed", "project_key": "TEAMS"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_create_local_markdown_source_generates_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_jobs=LeasedJobStore(database, _allow_local_agent_lease))
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/v1/sources",
                json={
                    "type": "local_markdown",
                    "name": "Engineering notes",
                    "access_policy": "private",
                    "config": {
                        "root": "/Users/test/engineering-notes",
                        "display_label": "Engineering notes",
                    },
                },
            )
            assert response.status_code == 200, response.text
            source_id = response.json()["id"]
            row = asyncio.run(database.get_source(source_id))
        assert row is not None
        config = row["config"]
        assert config["vault_id"].startswith("local-")
        assert config["display_label"] == "Engineering notes"
        # The folder is read on the Source's execution device; the server keeps no inbox for it.
        assert "documents_dir" not in config
    finally:
        asyncio.run(database.close())


def test_update_local_markdown_source_preserves_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_jobs=LeasedJobStore(database, _allow_local_agent_lease))
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client, vault_id="engineering")
            response = client.put(
                f"/api/v1/sources/{created['id']}",
                json={
                    "name": "Engineering docs",
                    "config": {
                        "root": "/Users/test/engineering-docs",
                        "display_label": "Engineering docs",
                    },
                },
            )
            assert response.status_code == 200, response.text
            row = asyncio.run(database.get_source(created["id"]))
        assert row is not None
        assert row["config"]["vault_id"] == "engineering"
        assert row["config"]["display_label"] == "Engineering docs"
    finally:
        asyncio.run(database.close())


def test_to_markdown_conversions():
    """The gene's content-type converter handles every supported file type."""
    from memforge.genes.local_markdown_gene import _to_markdown

    assert _to_markdown("text/markdown", "# Keep\n\nx").startswith("# Keep")
    assert _to_markdown("text/plain", "plain") == "plain"
    assert _to_markdown("application/json", '{"a": 1}') == '```json\n{"a": 1}\n```\n'
    assert _to_markdown("text/html", "<h1>T</h1><p>hi</p>") == "# T\n\nhi"
    # content-type parameters and unknown types degrade to plain text
    assert _to_markdown("text/markdown; charset=utf-8", "# H") == "# H"
    assert _to_markdown("application/octet-stream", "raw") == "raw"


async def _allow_local_agent_lease(*args, **kwargs) -> LocalAgentLease:
    return LocalAgentLease(payload={}, leased_at=datetime.now(timezone.utc))


class LeaseAwareTestClient(TestClient):
    def post(self, url, *args, **kwargs):
        if "/adapter/packages" in url or "/adapter/manifest" in url or url.endswith("/process"):
            body = dict(kwargs.get("json") or {})
            body.setdefault("local_agent_job_id", "test-local-agent-job")
            body.setdefault("local_agent_attempt_count", 1)
            kwargs["json"] = body
        return super().post(url, *args, **kwargs)


class _FailOnceThenCleanUp:
    """Fails the first sync-input record; runs artifact cleanup right before the next one.

    The second call lands in the window between writing an object and recording
    the reference to it, where cleanup must not be able to delete that object.
    """

    def __init__(self, database: Database, document_store: LocalDocumentStore) -> None:
        self.database = database
        self.document_store = document_store
        self.record = database.create_source_sync_input
        self.calls = 0
        self.cleaned = 0

    async def __call__(self, **kwargs):
        from memforge.storage.source_cleanup import SourceArtifactCleanupService

        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("sync input store unavailable")
        self.cleaned += await SourceArtifactCleanupService(self.database, self.document_store).run_pending()
        return await self.record(**kwargs)


def _cleanup_uris(database: Database) -> list[str]:
    return [task.artifact_uri for task in asyncio.run(database.list_source_artifact_cleanup_tasks(limit=10))]


def test_a_retried_artifact_upload_survives_cleanup_of_the_failed_attempt(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)
    image_bytes = b"\x89PNG\r\n\x1a\nretried-upload"
    params = {
        "source_unit_key": "Payroll Processing/architecture.png",
        "provider_key": "Payroll Processing/architecture.png",
        "provider_revision": "blob-image-sha",
        "parent_observation_type": "file_content",
        "parent_provider_key": "content",
        "filename": "architecture.png",
        "media_type": "image/png",
        "local_agent_job_id": "test-local-agent-job",
        "local_agent_attempt_count": 1,
    }
    try:
        app = create_admin_app(
            db=database, config=cfg, document_store=document_store,
            local_agent_jobs=LeasedJobStore(database, _allow_local_agent_lease),
        )
        record = _FailOnceThenCleanUp(database, document_store)
        database.create_source_sync_input = record
        with LeaseAwareTestClient(app, raise_server_exceptions=False) as client:
            source_id = _create_teams_source(client)["id"]
            route = f"/api/v1/sources/{source_id}/adapter/artifacts"
            failed = client.post(route, params=params, content=image_bytes, headers={"content-type": "image/png"})
            [abandoned] = _cleanup_uris(database)
            retried = client.post(route, params=params, content=image_bytes, headers={"content-type": "image/png"})

        assert failed.status_code == 500
        assert retried.status_code == 200, retried.text
        assert record.cleaned == 1
        [retained] = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert retained.raw_uri != abandoned
        assert document_store.read_artifact(retained.raw_uri) == image_bytes
        assert document_store.get_artifact(abandoned, "image/png") is None
    finally:
        asyncio.run(database.close())


