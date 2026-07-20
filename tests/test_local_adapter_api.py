"""Tests for the local CLI adapter document push contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.local_agent.source_contract import source_with_sync_inputs
from memforge.memory.lifecycle_plan import (
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
)
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def _connect_database(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    return database


def _project_source_inputs(database: Database, source: dict) -> dict:
    inputs = asyncio.run(database.list_source_sync_inputs(source_id=source["id"]))
    return source_with_sync_inputs(source, inputs)


def _create_local_markdown_source(
    client: TestClient, *, name: str = "Engineering notes", vault_id: str = "engineering"
) -> dict:
    response = client.post(
        "/api/sources",
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


def _create_unmapped_local_markdown_source(
    client: TestClient, *, name: str = "Engineering notes", vault_id: str = "engineering"
) -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "local_markdown",
            "name": name,
            "access_policy": "private",
            "config": {
                "root": "/Users/test/engineering-notes",
                "vault_id": vault_id,
                "display_label": "Engineering notes",
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_github_repo_source(
    client: TestClient,
    *,
    name: str = "Matterhorn Architecture",
    connection_mode: str = "local_push",
    max_files: int = 500,
    exclude_paths: list[str] | None = None,
) -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "github_repo",
            "name": name,
            "access_policy": "private",
            "config": {
                "connection_mode": connection_mode,
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing/"],
                "exclude_paths": exclude_paths or [],
                "include_extensions": ["md"],
                "max_files": max_files,
            },
            "project_binding": {"mode": "fixed", "project_key": "MATTERHORN"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_jira_source(
    client: TestClient,
    *,
    name: str = "Payroll Jira",
    sync_mode: str = "local_agent",
) -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "jira",
            "name": name,
            "access_policy": "private",
            "config": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": sync_mode,
                "projects": ["PAY"],
                "issue_types": ["Task"],
                "include_comments": True,
            },
            "project_binding": {"mode": "fixed", "project_key": "PAY"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_teams_source(client: TestClient, *, name: str = "Teams Channel") -> dict:
    response = client.post(
        "/api/sources",
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


def test_create_local_markdown_source_populates_inbox_path(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert row["type"] == "local_markdown"
        config = row["config"]
        assert config["vault_id"] == "engineering"
        documents_dir = Path(config["documents_dir"])
        assert documents_dir.exists()
        expected_root = Path(cfg.storage.docs_path).parent / "local-adapter-submissions"
        assert documents_dir.is_relative_to(expected_root)
    finally:
        asyncio.run(database.close())


def test_create_local_markdown_source_generates_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/sources",
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
        assert Path(config["documents_dir"]).exists()
    finally:
        asyncio.run(database.close())


def test_update_local_markdown_source_preserves_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client, vault_id="engineering")
            response = client.put(
                f"/api/sources/{created['id']}",
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


def test_create_github_repo_source_populates_inbox_path_for_local_push(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_github_repo_source(client)
            row = asyncio.run(database.get_source(created["id"]))
        assert row is not None
        assert row["type"] == "github_repo"
        assert row["config"]["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
        documents_dir = Path(row["config"]["documents_dir"])
        assert documents_dir.exists()
        expected_root = Path(cfg.storage.docs_path).parent / "local-adapter-submissions"
        assert documents_dir.is_relative_to(expected_root)
    finally:
        asyncio.run(database.close())


def test_jira_adapter_document_push_uses_one_canonical_artifact(tmp_path):
    from memforge.genes.jira_gene import JiraGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_jira_source(client)
            source_id = created["id"]
            raw_payload = {
                "id": "10001",
                "key": "PAY-1",
                "fields": {
                    "summary": "Create daemon source support",
                    "description": "Create daemon source support.",
                    "status": {"name": "Open"},
                    "issuetype": {"name": "Task"},
                    "priority": {"name": "Medium"},
                    "assignee": {"displayName": "Ada"},
                    "labels": [],
                    "resolution": None,
                    "updated": "2026-07-10T08:00:00+00:00",
                    "issuelinks": [],
                    "subtasks": [],
                },
                "_comments": [],
                "_comments_included": True,
                "_comments_total": 0,
                "changelog": {"startAt": 0, "histories": [], "total": 0},
            }
            initial_row = asyncio.run(database.get_source(source_id))
            assert initial_row is not None
            initial_documents_dir = Path(initial_row["config"]["local_agent_documents_dir"])
            assert initial_documents_dir.exists()
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "base_url": "https://jira.example.test",
                    "issue_key": "PAY-1",
                    "source_url": "https://jira.example.test/browse/PAY-1",
                    "title": "Create daemon source support",
                    "raw_payload": raw_payload,
                    "sync_snapshot_id": "test-local-agent-job:attempt:1",
                    "submitted_at": "2026-07-10T08:00:00+00:00",
                },
            )
            repeated_response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "base_url": "https://jira.example.test",
                    "issue_key": "PAY-1",
                    "source_url": "https://jira.example.test/browse/PAY-1",
                    "title": "Create daemon source support",
                    "raw_payload": raw_payload,
                    "sync_snapshot_id": "test-local-agent-job:attempt:1",
                    "submitted_at": "2026-07-10T08:05:00+00:00",
                },
            )
            process_response = client.post(
                f"/api/sources/{source_id}/process",
                json={
                    "force_full_sync": False,
                    "sync_snapshot_id": "test-local-agent-job:attempt:1",
                },
            )

        assert response.status_code == 200, response.text
        assert repeated_response.status_code == 200, repeated_response.text
        payload = response.json()
        repeated_payload = repeated_response.json()
        assert payload["issue_key"] == "PAY-1"
        assert payload["package_uri"]
        assert repeated_payload["package_uri"] == payload["package_uri"]
        first_package = json.loads(Path(payload["package_uri"]).read_text(encoding="utf-8"))
        assert first_package["submitted_at"] == "2026-07-10T08:00:00+00:00"
        package_artifacts = list(Path(cfg.storage.docs_path).rglob("*package*.json"))
        assert package_artifacts == [Path(payload["package_uri"])]
        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        documents_dir = Path(row["config"]["local_agent_documents_dir"])
        package_path = documents_dir / f"{payload['doc_id']}.json"
        assert payload["package_path"] is None
        assert not package_path.exists()
        inputs = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert len(inputs) == 1
        assert inputs[0].raw_uri == payload["package_uri"]
        assert inputs[0].metadata["doc_id"] == payload["doc_id"]
        assert inputs[0].metadata["manifest_entry"]["doc_id"] == payload["doc_id"]
        snapshot_inputs = asyncio.run(
            database.list_source_sync_inputs(
                source_id=source_id,
                input_snapshot_id="test-local-agent-job:attempt:1",
            )
        )
        assert [item.input_id for item in snapshot_inputs] == [inputs[0].input_id]
        repeated_snapshot_inputs = asyncio.run(
            database.list_source_sync_inputs(
                source_id=source_id,
                input_snapshot_id="test-local-agent-job:attempt:1",
            )
        )
        assert [item.input_id for item in repeated_snapshot_inputs] == [inputs[0].input_id]
        assert process_response.status_code == 202, process_response.text
        run = asyncio.run(database.get_source_sync_run(process_response.json()["run_id"]))
        assert run is not None
        assert run.input_snapshot_id == "test-local-agent-job:attempt:1"

        projected = _project_source_inputs(database, row)
        gene = JiraGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def _read_package():
            await gene.authenticate()
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_read_package())
        assert [item.extra["issue_key"] for item in items] == ["PAY-1"]
        assert "Create daemon source support." in normalized.markdown_body
        assert normalized.source_semantics["status"] == "Open"
        assert normalized.source_semantics["issue_key"] == "PAY-1"
    finally:
        asyncio.run(database.close())


def test_jira_adapter_rejects_comment_without_stable_provider_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_jira_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "base_url": "https://jira.example.test",
                    "issue_key": "PAY-1",
                    "raw_payload": {
                        "key": "PAY-1",
                        "fields": {"summary": "Issue"},
                        "_comments": [{"body": "Decision"}],
                    },
                },
            )

        assert response.status_code == 400
        assert "stable provider id" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_jira_local_agent_mode_rejects_pat_auth(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/sources",
                json={
                    "type": "jira",
                    "name": "Payroll Jira",
                    "access_policy": "private",
                    "config": {
                        "base_url": "https://jira.example.test",
                        "auth_mode": "pat",
                        "sync_mode": "local_agent",
                        "pat": "local-pat",
                        "projects": ["PAY"],
                    },
                    "project_binding": {"mode": "fixed", "project_key": "PAY"},
                },
            )

        assert response.status_code == 400, response.text
        assert "Browser session" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_update_jira_local_agent_source_preserves_sync_mode_when_omitted(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_jira_source(client)
            source_id = created["id"]
            before = asyncio.run(database.get_source(source_id))
            assert before is not None
            initial_documents_dir = before["config"]["local_agent_documents_dir"]
            response = client.put(
                f"/api/sources/{source_id}",
                json={
                    "config": {
                        "base_url": "https://jira.example.test",
                        "auth_mode": "browser_cookie",
                        "projects": ["PAY", "ENG"],
                        "issue_types": ["Task"],
                        "include_comments": True,
                    }
                },
            )

        assert response.status_code == 200, response.text
        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert row["config"]["sync_mode"] == "local_agent"
        assert row["config"]["local_agent_documents_dir"] == initial_documents_dir
    finally:
        asyncio.run(database.close())


def test_jira_local_agent_package_discovery_normalizes_naive_timestamps(tmp_path):
    from datetime import datetime, timezone

    from memforge.genes.jira_gene import JiraGene

    documents_dir = tmp_path / "jira-inbox"
    documents_dir.mkdir()
    (documents_dir / "pay-1.json").write_text(
        json.dumps(
            {
                "package_kind": "jira_document",
                "doc_id": "jira-pay-1",
                "issue_key": "PAY-1",
                "title": "Naive timestamp package",
                "source_url": "https://jira.example.test/browse/PAY-1",
                "last_modified": "2026-07-07T12:00:00",
                "raw_payload": {
                    "key": "PAY-1",
                    "fields": {
                        "summary": "Naive timestamp package",
                        "description": "PAY-1",
                        "status": {"name": "Open"},
                        "issuetype": {"name": "Task"},
                        "issuelinks": [],
                        "subtasks": [],
                    },
                    "_comments": [],
                },
            }
        ),
        encoding="utf-8",
    )

    gene = JiraGene(
        {
            "base_url": "https://jira.example.test",
            "auth_mode": "browser_cookie",
            "sync_mode": "local_agent",
            "local_agent_documents_dir": str(documents_dir),
        },
        "src-jira",
    )

    async def _discover():
        await gene.authenticate()
        return [item async for item in gene.discover(datetime(2026, 7, 7, 11, 0, tzinfo=timezone.utc))]

    items = asyncio.run(_discover())
    assert [item.extra["issue_key"] for item in items] == ["PAY-1"]
    assert items[0].last_modified.tzinfo is timezone.utc


def test_jira_adapter_document_push_requires_local_agent_mode(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_jira_source(client, sync_mode="cloud")
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "base_url": "https://jira.example.test",
                    "issue_key": "PAY-1",
                    "markdown_body": "# PAY-1",
                },
            )

        assert response.status_code == 400, response.text
        assert "sync_mode=local_agent" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_writes_package(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/cutoff.md",
                    "title": "Cutoff Decision",
                    "markdown_body": "# Cutoff\n\nDeploy on Tuesday.",
                    "raw_hash": "deadbeef",
                    "submitted_by": "cli-adapter",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source_id"] == source_id
        assert body["doc_id"].startswith("local-md-")
        assert body["package_uri"]
        assert body["sync_started"] is False
        assert body["relative_path"] == "decisions/cutoff.md"

        assert body["package_path"] is None
        package_path = Path(body["package_uri"])
        package = json.loads(package_path.read_text())
        assert package["package_kind"] == "local_markdown_document"
        assert package["vault_id"] == "engineering"
        assert package["relative_path"] == "decisions/cutoff.md"
        assert package["title"] == "Cutoff Decision"
        assert package["markdown"].startswith("# Cutoff")

        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        inputs = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert len(inputs) == 1
        assert inputs[0].raw_uri == body["package_uri"]
        assert inputs[0].metadata["doc_id"] == body["doc_id"]
        assert inputs[0].metadata["manifest_entry"]["doc_id"] == body["doc_id"]
        assert inputs[0].metadata["package_sha256"] == body["package_sha256"]
        assert inputs[0].metadata["manifest_entry"]["package_sha256"] == body["package_sha256"]

        projected = _project_source_inputs(database, row)
        gene = LocalMarkdownGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def _read_package():
            await gene.authenticate()
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_read_package())
        assert items[0].extra["package_uri"] == body["package_uri"]
        assert normalized.markdown_body.startswith("# Cutoff")
    finally:
        asyncio.run(database.close())


def test_duplicate_local_package_attests_the_retained_artifact_not_the_new_upload(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            first = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/retained.md",
                    "markdown_body": "# Decision\n\nKeep the retained artifact.",
                    "submitted_at": "2026-07-15T10:00:00+00:00",
                    "process_now": False,
                },
            )
            assert first.status_code == 200, first.text
            retained_uri = first.json()["package_uri"]
            retained_sha = hashlib.sha256(Path(retained_uri).read_bytes()).hexdigest()
            [legacy_input] = asyncio.run(
                database.list_source_sync_inputs(source_id=source_id)
            )
            legacy_metadata = dict(legacy_input.metadata)
            legacy_metadata.pop("package_sha256")
            manifest_entry = dict(legacy_metadata["manifest_entry"])
            manifest_entry.pop("package_sha256", None)
            legacy_metadata["manifest_entry"] = manifest_entry
            asyncio.run(
                database.db.execute(
                    "UPDATE source_sync_inputs SET metadata_json = ? WHERE input_id = ?",
                    (json.dumps(legacy_metadata, sort_keys=True), legacy_input.input_id),
                )
            )
            asyncio.run(database.db.commit())

            duplicate = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/retained.md",
                    "markdown_body": "# Decision\n\nKeep the retained artifact.",
                    "submitted_at": "2026-07-16T10:00:00+00:00",
                    "process_now": False,
                },
            )

        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["package_uri"] == retained_uri
        assert duplicate.json()["package_sha256"] == retained_sha
        [attested] = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert attested.input_id == legacy_input.input_id
        assert attested.raw_uri == retained_uri
        assert attested.metadata["package_sha256"] == retained_sha
        assert attested.metadata["manifest_entry"]["package_sha256"] == retained_sha
        assert Path(retained_uri).exists()
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == [
            Path(retained_uri)
        ]
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("retained_failure", ["corrupt", "missing"])
def test_duplicate_local_package_does_not_attest_an_invalid_retained_artifact(
    tmp_path,
    retained_failure,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            first = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/corrupt.md",
                    "markdown_body": "# Original",
                    "submitted_at": "2026-07-15T10:00:00+00:00",
                    "process_now": False,
                },
            )
            assert first.status_code == 200, first.text
            retained_uri = first.json()["package_uri"]
            [legacy_input] = asyncio.run(
                database.list_source_sync_inputs(source_id=source_id)
            )
            legacy_metadata = dict(legacy_input.metadata)
            legacy_metadata.pop("package_sha256")
            manifest_entry = dict(legacy_metadata["manifest_entry"])
            manifest_entry.pop("package_sha256", None)
            legacy_metadata["manifest_entry"] = manifest_entry
            asyncio.run(
                database.db.execute(
                    "UPDATE source_sync_inputs SET metadata_json = ? WHERE input_id = ?",
                    (json.dumps(legacy_metadata, sort_keys=True), legacy_input.input_id),
                )
            )
            asyncio.run(database.db.commit())
            if retained_failure == "corrupt":
                Path(retained_uri).write_bytes(b"{}")
            else:
                Path(retained_uri).unlink()

            duplicate = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/corrupt.md",
                    "markdown_body": "# Original",
                    "submitted_at": "2026-07-16T10:00:00+00:00",
                    "process_now": False,
                },
            )

        assert duplicate.status_code == 409, duplicate.text
        assert duplicate.json()["detail"] == (
            "source_lifecycle_local_replay_artifact_invalid"
        )
        [unchanged] = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert "package_sha256" not in unchanged.metadata
        assert "package_sha256" not in unchanged.metadata["manifest_entry"]
        if retained_failure == "corrupt":
            assert Path(retained_uri).read_bytes() == b"{}"
            assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == [
                Path(retained_uri)
            ]
        else:
            assert not Path(retained_uri).exists()
            assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
    finally:
        asyncio.run(database.close())


def test_normal_local_source_fetch_rejects_tampered_canonical_package(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "tampered.md",
                    "markdown_body": "# Original",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        artifact = Path(body["package_uri"])
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        source = asyncio.run(database.get_source(source_id))
        projected = _project_source_inputs(database, source)
        gene = LocalMarkdownGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def fetch_package():
            items = [item async for item in gene.discover(None)]
            return await gene.fetch(items[0])

        with pytest.raises(
            ValueError,
            match="source_lifecycle_local_replay_artifact_invalid",
        ):
            asyncio.run(fetch_package())
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("source_type", ["local_markdown", "github_repo"])
def test_local_file_package_attests_explicit_empty_content(tmp_path, source_type):
    from memforge.genes.github_repo_gene import GitHubRepoGene
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            if source_type == "local_markdown":
                source_id = _create_local_markdown_source(client)["id"]
                payload = {
                    "vault_id": "engineering",
                    "relative_path": "emptied.md",
                    "markdown_body": "",
                    "process_now": False,
                }
            else:
                source_id = _create_github_repo_source(client)["id"]
                payload = {
                    "repo_url": ("https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"),
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/emptied.md",
                    "markdown_body": "",
                    "process_now": False,
                }
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json=payload,
            )
        assert response.status_code == 200, response.text
        source = asyncio.run(database.get_source(source_id))
        projected = _project_source_inputs(database, source)
        gene = (
            LocalMarkdownGene(projected["config"], source_id)
            if source_type == "local_markdown"
            else GitHubRepoGene(projected["config"], source_id)
        )
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def fetch_package():
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return raw, normalized

        raw, normalized = asyncio.run(fetch_package())
        assert raw.body.strip()
        assert raw.authoritative_empty is True
        assert raw.empty_evidence
        assert normalized.markdown_body == ""
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_writes_package(tmp_path):
    from memforge.genes.github_repo_gene import GitHubRepoGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_github_repo_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/README.md",
                    "title": "Payroll Processing",
                    "markdown_body": "# Payroll Processing\n\nArchitecture notes.",
                    "content_type": "text/markdown",
                    "blob_sha": "blob-sha-1",
                    "raw_hash": "raw-hash-1",
                    "submitted_by": "github-adapter",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source_id"] == source_id
        assert body["doc_id"] == build_github_repo_doc_id(
            source_id=source_id,
            repo_url="https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            repo_ref="main",
            relative_path="Payroll Processing/README.md",
        )
        assert body["package_uri"]
        assert body["relative_path"] == "Payroll Processing/README.md"

        assert body["package_path"] is None
        package_path = Path(body["package_uri"])
        package = json.loads(package_path.read_text())
        assert package["package_kind"] == "github_repo_document"
        assert package["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
        assert package["repo_ref"] == "main"
        assert package["relative_path"] == "Payroll Processing/README.md"
        assert package["blob_sha"] == "blob-sha-1"
        assert package["markdown"].startswith("# Payroll Processing")

        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        inputs = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert len(inputs) == 1
        assert inputs[0].raw_uri == body["package_uri"]
        assert inputs[0].metadata["doc_id"] == body["doc_id"]
        assert inputs[0].metadata["manifest_entry"]["doc_id"] == body["doc_id"]

        projected = _project_source_inputs(database, row)
        gene = GitHubRepoGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def _read_package():
            await gene.authenticate()
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_read_package())
        assert items[0].extra["package_uri"] == body["package_uri"]
        assert normalized.markdown_body.startswith("# Payroll Processing")
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_requires_local_push_mode(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_github_repo_source(client, connection_mode="cloud_pull")["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert "Internal network / VPN" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_rejects_out_of_scope_request(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_github_repo_source(
                client,
                exclude_paths=["Payroll Processing/archived"],
            )["id"]
            wrong_ref = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "feature",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
            wrong_path = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Flexible Payroll/README.md",
                    "markdown_body": "# Flexible Payroll",
                    "process_now": False,
                },
            )
            wrong_extension = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/diagram.png",
                    "markdown_body": "png bytes",
                    "process_now": False,
                },
            )
            excluded_path = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/archived/README.md",
                    "markdown_body": "# Archived",
                    "process_now": False,
                },
            )
        assert wrong_ref.status_code == 400
        assert "configured ref" in wrong_ref.json()["detail"]
        assert wrong_path.status_code == 400
        assert "repository scope" in wrong_path.json()["detail"]
        assert wrong_extension.status_code == 400
        assert "include_extensions" in wrong_extension.json()["detail"]
        assert excluded_path.status_code == 400
        assert "repository scope" in excluded_path.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_enforces_max_files(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_github_repo_source(client, max_files=1)["id"]
            first = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
            second = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/Second.md",
                    "markdown_body": "# Second",
                    "process_now": False,
                },
            )
        assert first.status_code == 200, first.text
        assert second.status_code == 400
        assert "max_files" in second.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_max_files_counts_current_scope_only(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_github_repo_source(client, max_files=1)["id"]
            row = asyncio.run(database.get_source(source_id))
            inbox = Path(row["config"]["documents_dir"])
            inbox.mkdir(parents=True, exist_ok=True)
            (inbox / "stale.json").write_text(
                json.dumps(
                    {
                        "package_kind": "github_repo_document",
                        "doc_id": "stale",
                        "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                        "repo_ref": "main",
                        "relative_path": "Flexible Payroll/README.md",
                        "content_type": "text/markdown",
                    }
                ),
                encoding="utf-8",
            )
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_requires_execution_owner(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)

    async def _seed_source() -> None:
        await database.upsert_source(
            id="src-owned-local",
            type="local_markdown",
            name="Owned notes",
            config_json=json.dumps(
                {
                    "vault_id": "engineering",
                    "documents_dir": str(tmp_path / "local-inbox"),
                }
            ),
            created_by_user_id="alice",
            execution_owner_user_id="alice",
            access_policy="workspace",
            owner_user_id="alice",
        )

    asyncio.run(_seed_source())
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            principal_resolver=lambda request: "bob",
            workspace_role_resolver=lambda request: "member",
        )
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/sources/src-owned-local/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/cutoff.md",
                    "markdown_body": "# Cutoff\n\nDeploy on Tuesday.",
                    "process_now": False,
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "local_agent_sync_execution_owner_forbidden"
    finally:
        asyncio.run(database.close())


def test_local_adapter_revalidates_lease_before_committing_snapshot_membership(tmp_path):
    from memforge.server.admin_api import create_admin_app

    database = _connect_database(tmp_path)
    calls: list[tuple] = []

    async def validate(*args):
        calls.append(args)
        return True

    try:
        app = create_admin_app(
            db=database,
            config=_config(tmp_path),
            local_agent_lease_validator=validate,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "lease-check.md",
                    "markdown_body": "# Lease check",
                    "sync_snapshot_id": "test-local-agent-job:attempt:1",
                },
            )

        assert response.status_code == 200, response.text
        assert len(calls) == 2
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_allows_unmapped_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_unmapped_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes/unmapped.md",
                    "markdown_body": "# Unmapped\n\nStill ingest this.",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert row["project_binding"] is None
        assert body["package_path"] is None
        assert Path(body["package_uri"]).exists()
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_is_idempotent_on_doc_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            payload = {
                "vault_id": "engineering",
                "relative_path": "notes/index.md",
                "markdown_body": "# Index\n\nTop level.",
                "process_now": False,
            }
            first = client.post(f"/api/sources/{source_id}/adapter/packages", json=payload).json()
            second = client.post(f"/api/sources/{source_id}/adapter/packages", json=payload).json()
        assert first["doc_id"] == second["doc_id"]
        assert first["package_path"] == second["package_path"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_attributes_input_to_app_workspace(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
            workspace_id="workspace-a",
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes/index.md",
                    "markdown_body": "# Index\n\nTop level.",
                    "process_now": False,
                },
            )

        assert response.status_code == 200, response.text
        inputs = asyncio.run(
            database.list_source_sync_inputs(
                source_id=source_id,
                workspace_id="workspace-a",
            )
        )
        assert len(inputs) == 1
        assert inputs[0].workspace_id == "workspace-a"
    finally:
        asyncio.run(database.close())


def test_teams_adapter_push_writes_window_package(tmp_path):
    from memforge.local_agent.teams_ledger import build_teams_window_id
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_teams_source(client)["id"]
            window_id = build_teams_window_id(
                source_id=source_id,
                conversation_id="19:channel@example.test",
                root_or_anchor_message_id="root-1",
                window_type="thread",
            )
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "conversation_id": "19:channel@example.test",
                    "window_id": window_id,
                    "revision_hash": "rev-1",
                    "root_message_id": "root-1",
                    "window_type": "thread",
                    "title": "Teams decision",
                    "source_url": "teams-window://src/conversation/window/rev-1",
                    "raw_payload": {
                        "conversation_id": "19:channel@example.test",
                        "window_id": window_id,
                        "conversation_type": "channel",
                        "title": "Teams decision",
                        "channel_name": "architecture",
                        "team_name": "Engineering",
                        "messages": [
                            {
                                "id": "root-1",
                                "from": "Alice",
                                "content": "Use rootMessageId for channel threads.",
                                "time": "2026-07-08T10:00:00+00:00",
                                "is_root": True,
                            },
                            {
                                "id": "reply-1",
                                "from": "Bob",
                                "content": "Agreed.",
                                "time": "2026-07-08T10:05:00+00:00",
                            },
                        ],
                        "participants": ["Alice", "Bob"],
                        "first_message_time": "2026-07-08T10:00:00+00:00",
                        "last_message_time": "2026-07-08T10:05:00+00:00",
                    },
                    "process_now": False,
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source_id"] == source_id
        assert body["doc_id"].startswith("teams-")
        assert body["package_path"] is None
        package_path = Path(body["package_uri"])
        package = json.loads(package_path.read_text(encoding="utf-8"))
        assert package["package_kind"] == "teams_window_document"
        assert package["content_role"] == "teams_conversation_window"
        assert package["window_id"] == window_id
        assert package["revision_hash"] == "rev-1"
        assert package["conversation_id"] == "19:channel@example.test"
        assert "markdown" not in package
        assert package["raw_payload"]["messages"][0]["content"] == "Use rootMessageId for channel threads."

        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert "local_agent_documents_dir" not in row["config"]
        inputs = asyncio.run(database.list_source_sync_inputs(source_id=source_id))
        assert len(inputs) == 1
        assert inputs[0].raw_uri == body["package_uri"]
        assert inputs[0].metadata["doc_id"] == body["doc_id"]
        manifest_entry = inputs[0].metadata["manifest_entry"]
        assert manifest_entry["doc_id"] == body["doc_id"]
        assert manifest_entry["window_id"] == window_id
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize(
    "raw_payload",
    [
        {
            "conversation_id": "19:channel@example.test",
            "window_id": "window-a",
            "messages": [{"id": "message-a", "content": "Decision"}],
        },
        {
            "conversation_id": "19:other@example.test",
            "window_id": "window-a",
            "messages": [
                {
                    "id": "message-a",
                    "content": "Decision",
                    "time": "2026-07-16T09:00:00+00:00",
                }
            ],
        },
    ],
)
def test_teams_adapter_rejects_ambiguous_message_evidence(
    tmp_path,
    raw_payload,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_teams_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "conversation_id": "19:channel@example.test",
                    "window_id": "window-a",
                    "revision_hash": "revision-a",
                    "raw_payload": raw_payload,
                    "process_now": False,
                },
            )

        assert response.status_code == 400
    finally:
        asyncio.run(database.close())


def test_teams_adapter_rejects_window_locator_bound_to_another_conversation(tmp_path):
    from memforge.local_agent.teams_ledger import build_teams_window_id
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_teams_source(client)["id"]
            window_id = build_teams_window_id(
                source_id=source_id,
                conversation_id="19:conversation-a@example.test",
                root_or_anchor_message_id="message-a",
                window_type="time_block",
            )
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "conversation_id": "19:conversation-b@example.test",
                    "window_id": window_id,
                    "revision_hash": "revision-a",
                    "root_message_id": "message-a",
                    "window_type": "time_block",
                    "raw_payload": {
                        "conversation_id": "19:conversation-b@example.test",
                        "window_id": window_id,
                        "messages": [
                            {
                                "id": "message-a",
                                "content": "Decision",
                                "time": "2026-07-16T09:00:00+00:00",
                            }
                        ],
                    },
                },
            )

        assert response.status_code == 400
        assert "locator" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_vault_mismatch(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "wrong-vault",
                    "relative_path": "notes/x.md",
                    "markdown_body": "# X",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert "vault_id" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_path_traversal(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "../escape.md",
                    "markdown_body": "# Nope",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert ".." in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_teams_gene_discovers_local_agent_window_packages(tmp_path):
    from memforge.genes import create_gene
    from memforge.local_agent.teams_ledger import build_teams_window_id
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_teams_source(client)["id"]
            window_id = build_teams_window_id(
                source_id=source_id,
                conversation_id="19:channel@example.test",
                root_or_anchor_message_id="anchor-1",
                window_type="time_block",
            )
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "conversation_id": "19:channel@example.test",
                    "window_id": window_id,
                    "revision_hash": "rev-block-1",
                    "root_message_id": "anchor-1",
                    "window_type": "time_block",
                    "title": "Group: July 8, 10:00-10:45",
                    "raw_payload": {
                        "conversation_id": "19:channel@example.test",
                        "window_id": window_id,
                        "conversation_type": "group_chat",
                        "title": "Group: July 8, 10:00-10:45",
                        "team_name": "Planning",
                        "messages": [
                            {
                                "id": "anchor-1",
                                "from": "Alice",
                                "content": "Decision captured.",
                                "time": "2026-07-08T10:00:00+00:00",
                                "is_root": True,
                            },
                        ],
                        "participants": ["Alice"],
                        "first_message_time": "2026-07-08T10:00:00+00:00",
                        "last_message_time": "2026-07-08T10:00:00+00:00",
                    },
                    "process_now": False,
                },
            )
            assert response.status_code == 200, response.text

        row = asyncio.run(database.get_source(source_id))
        projected = _project_source_inputs(database, row)
        gene = create_gene("teams", projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))
        asyncio.run(gene.authenticate())

        async def _collect_normalized():
            items = []
            async for item in gene.discover():
                items.append(item)
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_collect_normalized())
        assert len(items) == 1
        assert items[0].item_id.startswith("teams-")
        assert items[0].version == "rev-block-1"
        assert items[0].extra["window_id"] == window_id
        assert "Decision captured." in normalized.markdown_body
        assert normalized.source_semantics["source_kind"] == "teams"
        assert normalized.source_semantics["window_id"] == window_id
        assert normalized.source_semantics["message_count"] == 1
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_paused_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            assert client.put(f"/api/sources/{source_id}", json={"status": "paused"}).status_code == 200
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert response.json()["detail"] == "Source is paused"
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_lifecycle_maintenance_before_artifact_write(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            asyncio.run(
                database.create_lifecycle_backfill_job(
                    LifecycleBackfillJob(
                        id="lifecycle-package-fence",
                        source_id=source_id,
                        status=LifecycleBackfillJobStatus.QUEUED,
                    )
                )
            )
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )

        assert response.status_code == 409
        assert response.json()["detail"] == ("source lifecycle maintenance active: lifecycle-package-fence")
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_cleans_artifact_when_lease_expires_after_write(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    validation_count = 0

    async def validate_lease(*args, **kwargs) -> bool:
        nonlocal validation_count
        validation_count += 1
        return validation_count == 1

    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=validate_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )

        assert response.status_code == 409
        assert response.json()["detail"] == "local_agent_lease_not_current"
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
        assert asyncio.run(database.list_source_sync_inputs(source_id=source_id)) == []
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_epoch_cas_rejects_maintenance_after_lease_check(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    original_create_input = database.create_source_sync_input
    maintenance_started = False

    async def create_input_after_maintenance_started(**kwargs):
        nonlocal maintenance_started
        if not maintenance_started:
            maintenance_started = True
            await database.create_lifecycle_backfill_job(
                LifecycleBackfillJob(
                    id="lifecycle-after-package-lease-check",
                    source_id=str(kwargs["source_id"]),
                    status=LifecycleBackfillJobStatus.QUEUED,
                )
            )
            await database.fail_lifecycle_backfill_job(
                "lifecycle-after-package-lease-check",
                error="maintenance completed before stale upload",
            )
        return await original_create_input(**kwargs)

    database.create_source_sync_input = create_input_after_maintenance_started
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )

        assert response.status_code == 409, response.text
        assert "source activity epoch changed" in response.json()["detail"]
        assert asyncio.run(database.list_source_sync_inputs(source_id=source_id)) == []
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_requires_local_adapter_source(tmp_path):
    """Pushing to a non-local-adapter source must be rejected."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)

    async def _seed_other_source() -> str:
        await database.upsert_source(
            id="src-other",
            type="agent_session",
            name="Agent Sessions",
            config_json=json.dumps({"documents_dir": str(tmp_path / "as-inbox")}),
            access_policy="private",
            owner_user_id="dev",
        )
        return "src-other"

    try:
        source_id = asyncio.run(_seed_other_source())
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "x",
                    "relative_path": "x.md",
                    "markdown_body": "# X",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert "local adapter source" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_markdown_gene_discovers_pushed_packages(tmp_path):
    """The gene discovers the JSON packages the push endpoint writes."""
    from memforge.genes import create_gene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            client.post(
                f"/api/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "release.md",
                    "markdown_body": "# Release\n\nCut on Tuesday.",
                    "process_now": False,
                },
            )

            row = asyncio.run(database.get_source(source_id))
            projected = _project_source_inputs(database, row)
            gene = create_gene("local_markdown", projected["config"], source_id)
            gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))
            asyncio.run(gene.authenticate())

        async def _collect():
            items = []
            async for item in gene.discover():
                items.append(item)
            return items

        items = asyncio.run(_collect())
        assert len(items) == 1
        assert items[0].title == "Release"
        assert items[0].space_or_project == "engineering"

        raw = asyncio.run(gene.fetch(items[0]))
        normalized = asyncio.run(gene.normalize(raw))
        assert normalized.markdown_body.startswith("# Release")
        assert normalized.source_semantics["vault_id"] == "engineering"
        assert normalized.source_semantics["relative_path"] == "release.md"
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


def test_local_markdown_gene_converts_by_content_type(tmp_path):
    """HTML and JSON pushes convert to markdown server-side; text passes through."""
    from memforge.genes import create_gene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            docs = [
                ("page.html", "text/html", "<h1>Decision</h1><p>Ship on <b>Tuesday</b>.</p>"),
                ("data.json", "application/json", '{"deploy": "tuesday"}'),
                ("note.txt", "text/plain", "plain reminder"),
            ]
            for rel, ctype, body in docs:
                resp = client.post(
                    f"/api/sources/{source_id}/adapter/packages",
                    json={
                        "vault_id": "engineering",
                        "relative_path": rel,
                        "markdown_body": body,
                        "content_type": ctype,
                        "process_now": False,
                    },
                )
                assert resp.status_code == 200, resp.text

            row = asyncio.run(database.get_source(source_id))
            projected = _project_source_inputs(database, row)
            gene = create_gene("local_markdown", projected["config"], source_id)
            gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))
            asyncio.run(gene.authenticate())

        async def _normalized_by_path():
            out = {}
            async for item in gene.discover():
                normalized = await gene.normalize(await gene.fetch(item))
                out[normalized.source_semantics["relative_path"]] = normalized
            return out

        by_path = asyncio.run(_normalized_by_path())
        assert by_path["page.html"].markdown_body == "# Decision\n\nShip on **Tuesday**."
        assert by_path["page.html"].source_semantics["content_type"] == "text/html"
        assert by_path["data.json"].markdown_body == '```json\n{"deploy": "tuesday"}\n```\n'
        assert by_path["note.txt"].markdown_body == "plain reminder"
    finally:
        asyncio.run(database.close())


async def _allow_local_agent_lease(*args, **kwargs) -> bool:
    return True


class LeaseAwareTestClient(TestClient):
    def post(self, url, *args, **kwargs):
        if "/adapter/packages" in url or url.endswith("/process"):
            body = dict(kwargs.get("json") or {})
            body.setdefault("local_agent_job_id", "test-local-agent-job")
            body.setdefault("local_agent_attempt_count", 1)
            kwargs["json"] = body
        return super().post(url, *args, **kwargs)
