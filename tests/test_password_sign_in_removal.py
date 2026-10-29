"""Removing password sign-in (migration 132): no route, account or password hash remains."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.requests import Request

from memforge.config import AppConfig
from memforge.storage.database import Database, latest_schema_version

REMOVAL = 132

PRIOR_USERS_TABLE = """
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    last_login      TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
INSERT INTO users (username, display_name, password_hash, role) VALUES ('admin', 'Admin', 'hash', 'admin');
"""


def _tables(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _schema(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0])


async def _open_and_close(path: Path) -> None:
    db = Database(str(path))
    await db.connect()
    await db.close()


async def test_a_fresh_database_keeps_no_users(tmp_path) -> None:
    path = tmp_path / "fresh.db"
    await _open_and_close(path)

    assert _schema(path) == latest_schema_version()
    assert "users" not in _tables(path)


async def test_upgrade_drops_the_users_table_and_its_password_hashes(tmp_path) -> None:
    path = tmp_path / "prior.db"
    await _open_and_close(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(PRIOR_USERS_TABLE)
        connection.execute("DELETE FROM schema_migrations WHERE version >= ?", (REMOVAL,))
        connection.commit()

    await _open_and_close(path)

    assert _schema(path) == latest_schema_version()
    assert "users" not in _tables(path)


async def test_the_password_sign_in_route_is_gone(tmp_path) -> None:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    db = Database(str(tmp_path / "api.db"))
    await db.connect()
    try:
        from memforge.server.admin_api import create_admin_app

        def principal(_request: Request) -> str:
            return "user-1"

        app = create_admin_app(db=db, config=config, principal_resolver=principal)
        with TestClient(app) as client:
            login = client.post("/api/v1/auth/login", json={"username": "admin", "password": "secret"})
            openapi = client.get("/openapi.json").json()
    finally:
        await db.close()

    assert login.status_code == 404
    assert "/api/v1/auth/login" not in openapi["paths"]
    assert "LoginRequest" not in openapi["components"]["schemas"]
