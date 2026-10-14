from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database


@pytest.fixture
def client_for(tmp_path: Path) -> Iterator:
    databases: list[Database] = []

    def build(cors_origins: str | None = None) -> TestClient:
        config = AppConfig(base_dir=tmp_path / "memforge")
        config.sync.worker_enabled = False
        if cors_origins is not None:
            config.server.cors_origins = cors_origins
        database = Database(str(tmp_path / "memforge.sqlite"))
        asyncio.run(database.connect())
        databases.append(database)
        return TestClient(create_admin_app(db=database, config=config))

    yield build
    for database in databases:
        asyncio.run(database.close())


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/workspaces",
        headers={"Origin": origin, "Access-Control-Request-Method": "DELETE"},
    )


def test_default_answers_only_the_admin_ui_loopback_origins(client_for) -> None:
    with client_for() as client:
        for origin in ("http://localhost:5174", "http://127.0.0.1:5174"):
            response = _preflight(client, origin)
            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin
            assert response.headers["access-control-allow-credentials"] == "true"

        foreign = _preflight(client, "https://example.com")
        assert foreign.status_code == 400
        assert "access-control-allow-origin" not in foreign.headers

        read = client.get("/api/v1/workspaces", headers={"Origin": "https://example.com"})
        assert "access-control-allow-origin" not in read.headers


def test_configured_origins_replace_the_default(client_for) -> None:
    with client_for(" https://memforge.example , ") as client:
        assert _preflight(client, "https://memforge.example").status_code == 200
        assert _preflight(client, "http://localhost:5174").status_code == 400


def test_wildcard_answers_every_origin_without_credentials(client_for) -> None:
    with client_for("*") as client:
        response = client.get(
            "/api/v1/workspaces",
            headers={"Origin": "https://example.com", "Cookie": "session=1"},
        )
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in response.headers
