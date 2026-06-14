"""Tests for the local CLI adapter document push contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from memforge.config import AppConfig
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
            "config": {"vault_id": vault_id, "display_label": "Engineering notes"},
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
            "config": {"vault_id": vault_id, "display_label": "Engineering notes"},
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


def test_local_adapter_push_requires_local_markdown_source(tmp_path):
    """Pushing to a non-local-markdown source must be rejected."""
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
        assert "local_markdown" in response.json()["detail"]
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
