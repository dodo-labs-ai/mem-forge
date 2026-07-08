"""Tests for the local CLI adapter document push contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.storage.database import Database


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


def _create_local_markdown_source(client: TestClient, *, name: str = "Engineering notes",
                                  vault_id: str = "engineering") -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "local_markdown",
            "name": name,
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


def _create_unmapped_local_markdown_source(client: TestClient, *, name: str = "Engineering notes",
                                           vault_id: str = "engineering") -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "local_markdown",
            "name": name,
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
) -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "github_repo",
            "name": name,
            "config": {
                "connection_mode": connection_mode,
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing/"],
                "include_extensions": ["md"],
                "max_files": max_files,
            },
            "project_binding": {"mode": "fixed", "project_key": "MATTERHORN"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_jira_source(client: TestClient, *, name: str = "Payroll Jira") -> dict:
    response = client.post(
        "/api/sources",
        json={
            "type": "jira",
            "name": name,
            "config": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": "local_agent",
                "projects": ["PAY"],
                "issue_types": ["Task"],
                "include_comments": True,
            },
            "project_binding": {"mode": "fixed", "project_key": "PAY"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_create_local_markdown_source_populates_inbox_path(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/sources",
                json={
                    "type": "local_markdown",
                    "name": "Engineering notes",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
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


def test_jira_adapter_document_push_populates_local_agent_inbox(tmp_path):
    from memforge.genes.jira_gene import JiraGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_jira_source(client)
            source_id = created["id"]
            initial_row = asyncio.run(database.get_source(source_id))
            assert initial_row is not None
            initial_documents_dir = Path(initial_row["config"]["local_agent_documents_dir"])
            assert initial_documents_dir.exists()
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "base_url": "https://jira.example.test",
                    "issue_key": "PAY-1",
                    "source_url": "https://jira.example.test/browse/PAY-1",
                    "title": "Create daemon source support",
                    "markdown_body": "# PAY-1\n\nCreate daemon source support.",
                    "source_semantics": {"status": "Open", "issue_type": "Task"},
                    "process_now": False,
                },
            )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["issue_key"] == "PAY-1"
        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        documents_dir = Path(row["config"]["local_agent_documents_dir"])
        package_path = documents_dir / f"{payload['doc_id']}.json"
        assert package_path.exists()

        gene = JiraGene(row["config"], source_id)

        async def _read_package():
            await gene.authenticate()
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_read_package())
        assert [item.extra["issue_key"] for item in items] == ["PAY-1"]
        assert normalized.markdown_body == "# PAY-1\n\nCreate daemon source support."
        assert normalized.source_semantics["status"] == "Open"
        assert normalized.source_semantics["issue_key"] == "PAY-1"
    finally:
        asyncio.run(database.close())


def test_jira_local_agent_mode_rejects_pat_auth(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                "/api/sources",
                json={
                    "type": "jira",
                    "name": "Payroll Jira",
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
                "markdown": "# PAY-1",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_jira_source(client)
            source_id = created["id"]
            update = client.put(
                f"/api/sources/{source_id}",
                json={
                    "config": {
                        "base_url": "https://jira.example.test",
                        "auth_mode": "browser_cookie",
                        "sync_mode": "cloud",
                        "projects": ["PAY"],
                    }
                },
            )
            assert update.status_code == 200, update.text
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        assert body["sync_started"] is False
        assert body["relative_path"] == "decisions/cutoff.md"

        package_path = Path(body["package_path"])
        assert package_path.exists()
        package = json.loads(package_path.read_text())
        assert package["package_kind"] == "local_markdown_document"
        assert package["vault_id"] == "engineering"
        assert package["relative_path"] == "decisions/cutoff.md"
        assert package["title"] == "Cutoff Decision"
        assert package["markdown"].startswith("# Cutoff")
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_writes_package(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_github_repo_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        assert body["relative_path"] == "Payroll Processing/README.md"

        package = json.loads(Path(body["package_path"]).read_text())
        assert package["package_kind"] == "github_repo_document"
        assert package["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
        assert package["repo_ref"] == "main"
        assert package["relative_path"] == "Payroll Processing/README.md"
        assert package["blob_sha"] == "blob-sha-1"
        assert package["markdown"].startswith("# Payroll Processing")
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_requires_local_push_mode(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            source_id = _create_github_repo_source(client, connection_mode="cloud_pull")["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            source_id = _create_github_repo_source(client)["id"]
            wrong_ref = client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "feature",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
            wrong_path = client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Flexible Payroll/README.md",
                    "markdown_body": "# Flexible Payroll",
                    "process_now": False,
                },
            )
            wrong_extension = client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/diagram.png",
                    "markdown_body": "png bytes",
                    "process_now": False,
                },
            )
        assert wrong_ref.status_code == 400
        assert "configured ref" in wrong_ref.json()["detail"]
        assert wrong_path.status_code == 400
        assert "include_paths" in wrong_path.json()["detail"]
        assert wrong_extension.status_code == 400
        assert "include_extensions" in wrong_extension.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_github_repo_adapter_document_push_enforces_max_files(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            source_id = _create_github_repo_source(client, max_files=1)["id"]
            first = client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "repo_ref": "main",
                    "relative_path": "Payroll Processing/README.md",
                    "markdown_body": "# Payroll Processing",
                    "process_now": False,
                },
            )
            second = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
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
                f"/api/sources/{source_id}/adapter/documents",
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


def test_local_adapter_document_push_requires_source_management(tmp_path):
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
        )

    asyncio.run(_seed_source())
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            principal_resolver=lambda request: "bob",
            workspace_role_resolver=lambda request: "member",
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-owned-local/adapter/documents",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/cutoff.md",
                    "markdown_body": "# Cutoff\n\nDeploy on Tuesday.",
                    "process_now": False,
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"]["error"] == "source_management_forbidden"
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_allows_unmapped_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_unmapped_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        assert Path(body["package_path"]).exists()
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_is_idempotent_on_doc_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            payload = {
                "vault_id": "engineering",
                "relative_path": "notes/index.md",
                "markdown_body": "# Index\n\nTop level.",
                "process_now": False,
            }
            first = client.post(f"/api/sources/{source_id}/adapter/documents", json=payload).json()
            second = client.post(f"/api/sources/{source_id}/adapter/documents", json=payload).json()
        assert first["doc_id"] == second["doc_id"]
        assert first["package_path"] == second["package_path"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_vault_mismatch(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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


def test_local_adapter_push_rejects_paused_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            assert client.put(f"/api/sources/{source_id}", json={"status": "paused"}).status_code == 200
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        )
        return "src-other"

    try:
        source_id = asyncio.run(_seed_other_source())
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            response = client.post(
                f"/api/sources/{source_id}/adapter/documents",
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            client.post(
                f"/api/sources/{source_id}/adapter/documents",
                json={
                    "vault_id": "engineering",
                    "relative_path": "release.md",
                    "markdown_body": "# Release\n\nCut on Tuesday.",
                    "process_now": False,
                },
            )

        row = asyncio.run(database.get_source(source_id))
        gene = create_gene("local_markdown", row["config"], source_id)
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
        app = create_admin_app(db=database, config=cfg)
        with TestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            docs = [
                ("page.html", "text/html", "<h1>Decision</h1><p>Ship on <b>Tuesday</b>.</p>"),
                ("data.json", "application/json", '{"deploy": "tuesday"}'),
                ("note.txt", "text/plain", "plain reminder"),
            ]
            for rel, ctype, body in docs:
                resp = client.post(
                    f"/api/sources/{source_id}/adapter/documents",
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
        gene = create_gene("local_markdown", row["config"], source_id)
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
