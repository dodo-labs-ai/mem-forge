from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app
from memforge.server.principal import resolve_workspace_role
from memforge.storage.database import Database


def _app(tmp_path: Path):
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    database = Database(str(tmp_path / "memforge.sqlite"))
    asyncio.run(database.connect())
    return create_admin_app(db=database, config=config), database


def test_self_hosted_workspace_directory_exposes_readable_local_workspace(
    tmp_path: Path,
) -> None:
    app, database = _app(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/workspaces")

        assert response.status_code == 200
        assert response.json() == {
            "workspaces": [
                {
                    "workspace_id": "local",
                    "name": "Local workspace",
                    "role": "owner",
                    "status": "active",
                    "selectable": True,
                }
            ]
        }
    finally:
        asyncio.run(database.close())


def test_self_hosted_workspace_role_matches_directory_owner() -> None:
    assert resolve_workspace_role(object()) == "owner"  # type: ignore[arg-type]


def test_self_hosted_default_workspace_route_is_absent(
    tmp_path: Path,
) -> None:
    app, database = _app(tmp_path)
    try:
        with TestClient(app) as client:
            response = client.put(
                "/api/v1/me/default-workspace",
                json={"workspace_id": "local"},
            )

        assert response.status_code == 404
    finally:
        asyncio.run(database.close())


def test_self_hosted_data_plane_resolves_omitted_or_explicit_local_workspace(
    tmp_path: Path,
) -> None:
    app, database = _app(tmp_path)
    try:
        with TestClient(app) as client:
            implicit = client.get("/api/v1/projects")
            explicit = client.get(
                "/api/v1/projects",
                params={"workspace_id": "local"},
            )
            inaccessible = client.get(
                "/api/v1/projects",
                params={"workspace_id": "another-workspace"},
            )
            empty = client.get("/api/v1/projects?workspace_id=")
            duplicate = client.get("/api/v1/projects?workspace_id=local&workspace_id=local")
            obsolete = client.get("/api/projects")

        assert implicit.status_code == 200
        assert implicit.headers["MemForge-Workspace"] == "local"
        assert explicit.status_code == 200
        assert explicit.headers["MemForge-Workspace"] == "local"
        assert inaccessible.status_code == 404
        assert inaccessible.json() == {
            "code": "workspace_not_found_or_inaccessible",
            "detail": "Workspace not found or inaccessible.",
        }
        assert empty.status_code == 400
        assert empty.json()["code"] == "invalid_workspace_selector"
        assert duplicate.status_code == 400
        assert duplicate.json()["code"] == "invalid_workspace_selector"
        assert obsolete.status_code == 404
    finally:
        asyncio.run(database.close())
