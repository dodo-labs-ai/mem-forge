from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Memory, content_hash
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def _connect_database(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    return database


def _principal(request: Request) -> str:
    return request.headers.get("x-test-user", "owner-user")


def _workspace_role(request: Request) -> str:
    return request.headers.get("x-test-workspace-role", "member")


def _app(tmp_path: Path, database: Database):
    from memforge.server.admin_api import create_admin_app

    return create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=_principal,
        workspace_role_resolver=_workspace_role,
    )


def _confluence_payload(name: str = "Architecture Wiki") -> dict:
    return {
        "type": "confluence",
        "name": name,
        "config": {
            "base_url": "https://wiki.example.test/wiki/spaces/ARCH/pages/12345/Home",
            "pat": "super-secret-token",
            "sync_mode": "page_tree",
            "page_tree_root": "12345",
            "include_children": True,
        },
    }


def _teams_payload(name: str = "PCC Agent Dev") -> dict:
    return {
        "type": "teams",
        "name": name,
        "config": {
            "region": "emea",
            "conversation_ids": ["19:conversation-a@example.test"],
            "conversation_gap_minutes": 60,
        },
    }


async def _insert_source_backed_memory(
    database: Database,
    *,
    source_id: str,
    memory_id: str = "mem-source-backed",
) -> None:
    now = datetime.now(timezone.utc)
    await database.db.execute(
        """INSERT INTO documents (
           doc_id, source, source_url, title, space_or_project, last_modified,
           version, content_hash, last_synced
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "doc-source-backed",
            source_id,
            "https://wiki.example.test/page",
            "Architecture",
            "ARCH",
            now.isoformat(),
            "1",
            "doc-hash",
            now.isoformat(),
        ),
    )
    await database.db.commit()
    content = "Shared source memory should honor per-user subscription."
    await database.insert_memory(
        Memory(
            id=memory_id,
            memory_type="fact",
            content=content,
            content_hash=content_hash(content),
            confidence=0.9,
            created_at=now,
            updated_at=now,
            status="active",
        )
    )
    await database.add_memory_source(memory_id, "doc-source-backed", "confluence", source_updated_at=None)


def test_source_list_exposes_capabilities_and_redacts_config_for_non_owner_member(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text

            response = client.get(
                "/api/sources",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )

        assert response.status_code == 200, response.text
        source = response.json()["data"][0]
        assert source["ownership"] == {
            "created_by_user_id": "owner-user",
            "execution_owner_user_id": None,
            "viewer_role": "member",
            "viewer_relationship": "member",
        }
        assert source["capabilities"] == {
            "can_subscribe": True,
            "can_configure": False,
            "can_configure_connection": False,
            "can_sync": False,
            "can_force_resync": False,
            "can_delete": False,
        }
        assert source["config"] == {}
    finally:
        asyncio.run(database.close())


def test_source_creator_can_manage_and_receives_redacted_config(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            updated = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json={"name": "Architecture Wiki Updated"},
            )
            listed = client.get(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )

        assert updated.status_code == 200, updated.text
        source = listed.json()["data"][0]
        assert source["name"] == "Architecture Wiki Updated"
        assert source["ownership"]["viewer_relationship"] == "creator"
        assert source["capabilities"]["can_configure"] is True
        assert source["capabilities"]["can_sync"] is True
        assert source["capabilities"]["can_delete"] is True
        assert source["config"]["pat_configured"] is True
        assert "pat" not in source["config"]
        assert "pat_encrypted" not in source["config"]
    finally:
        asyncio.run(database.close())


def test_non_owner_member_cannot_manage_source(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            updated = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
                json={"name": "Should Not Change"},
            )
            deleted = client.delete(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )

        assert updated.status_code == 403
        assert updated.json()["detail"] == {
            "error": "source_management_forbidden",
            "message": "Only the source creator or a workspace admin can manage this source.",
        }
        assert deleted.status_code == 403
    finally:
        asyncio.run(database.close())


def test_workspace_admin_can_manage_source_they_do_not_own(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            updated = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "admin-user", "x-test-workspace-role": "workspace_admin"},
                json={"name": "Admin Updated"},
            )
            listed = client.get(
                "/api/sources",
                headers={"x-test-user": "admin-user", "x-test-workspace-role": "workspace_admin"},
            )

        assert updated.status_code == 200, updated.text
        source = listed.json()["data"][0]
        assert source["name"] == "Admin Updated"
        assert source["ownership"]["viewer_relationship"] == "workspace_admin"
        assert source["capabilities"]["can_delete"] is True
    finally:
        asyncio.run(database.close())


def test_local_source_creator_becomes_execution_owner_and_client_cannot_override(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        payload = _teams_payload()
        payload["execution_owner_user_id"] = "admin-b"
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=payload,
            )

        assert created.status_code == 200, created.text
        source = asyncio.run(database.get_source(created.json()["id"]))
        assert source is not None
        assert source["created_by_user_id"] == "owner-a"
        assert source["execution_owner_user_id"] == "owner-a"
    finally:
        asyncio.run(database.close())


def test_non_owner_admin_can_manage_local_source_but_cannot_change_connection(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_teams_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            managed = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
                json={
                    "name": "Admin Renamed",
                    "project_binding": {"mode": "fixed", "project_key": "PAY"},
                    "sync_schedule": {"enabled": True, "interval_minutes": 60},
                },
            )
            connection_update = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
                json={
                    "config": {
                        "region": "emea",
                        "conversation_ids": ["19:conversation-b@example.test"],
                        "conversation_gap_minutes": 60,
                    }
                },
            )

        assert managed.status_code == 200, managed.text
        assert connection_update.status_code == 403, connection_update.text
        assert connection_update.json()["detail"]["error"] == (
            "local_agent_source_connection_owner_forbidden"
        )
        source = asyncio.run(database.get_source(source_id))
        assert source is not None
        assert source["name"] == "Admin Renamed"
        assert source["project_binding"] == {"mode": "fixed", "project_key": "PAY"}
        assert source["execution_owner_user_id"] == "owner-a"
    finally:
        asyncio.run(database.close())


def test_non_owner_admin_cannot_trigger_local_source_sync(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_teams_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            synced = client.post(
                f"/api/sources/{source_id}/sync",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
            )
            force_synced = client.post(
                f"/api/sources/{source_id}/force-resync",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
            )

        assert synced.status_code == 403, synced.text
        assert synced.json()["detail"] == "local_agent_sync_execution_owner_forbidden"
        assert force_synced.status_code == 403, force_synced.text
        assert force_synced.json()["detail"] == "local_agent_sync_execution_owner_forbidden"
    finally:
        asyncio.run(database.close())


def test_local_source_owner_can_change_connection_without_transferring_ownership(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_teams_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]
            updated = client.put(
                f"/api/sources/{source_id}",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={
                    "config": {
                        "region": "emea",
                        "conversation_ids": ["19:conversation-b@example.test"],
                        "conversation_gap_minutes": 60,
                    },
                    "execution_owner_user_id": "admin-b",
                },
            )

        assert updated.status_code == 200, updated.text
        source = asyncio.run(database.get_source(source_id))
        assert source is not None
        assert source["config"]["conversation_ids"] == ["19:conversation-b@example.test"]
        assert source["execution_owner_user_id"] == "owner-a"
    finally:
        asyncio.run(database.close())


def test_source_execution_owner_migration_backfills_creator(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE sources (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            config TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            last_sync TEXT,
            doc_count INTEGER DEFAULT 0,
            project_binding TEXT,
            created_by_user_id TEXT,
            sync_schedule_enabled INTEGER NOT NULL DEFAULT 0,
            sync_schedule_interval_minutes INTEGER NOT NULL DEFAULT 1440,
            sync_schedule_next_at TEXT,
            sync_schedule_updated_at TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY,
            description TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO sources (id, type, name, config, created_by_user_id)
        VALUES ('src-legacy-teams', 'teams', 'Legacy Teams', '{}', 'owner-a');
        """
    )
    connection.executemany(
        "INSERT INTO schema_migrations (version, description) VALUES (?, ?)",
        [(version, "already applied") for version in range(1, 35)],
    )
    connection.commit()
    connection.close()

    database = Database(str(db_path))
    asyncio.run(database.connect())
    try:
        source = asyncio.run(database.get_source("src-legacy-teams"))
        assert source is not None
        assert source["execution_owner_user_id"] == "owner-a"
    finally:
        asyncio.run(database.close())


def test_member_can_toggle_their_own_source_subscription(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            default_list = client.get(
                "/api/sources",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            disabled = client.put(
                f"/api/sources/{source_id}/subscription",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
                json={"enabled": False},
            )
            disabled_list = client.get(
                "/api/sources",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            owner_list = client.get(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )

        assert default_list.status_code == 200
        assert default_list.json()["data"][0]["subscription"] == {"enabled": True}
        assert default_list.json()["data"][0]["enabled_for_me"] is True
        assert disabled.status_code == 200, disabled.text
        assert disabled.json() == {
            "ok": True,
            "source_id": source_id,
            "subscription": {"enabled": False},
        }
        assert disabled_list.json()["data"][0]["subscription"] == {"enabled": False}
        assert disabled_list.json()["data"][0]["enabled_for_me"] is False
        assert owner_list.json()["data"][0]["subscription"] == {"enabled": True}
        assert owner_list.json()["data"][0]["enabled_for_me"] is True
    finally:
        asyncio.run(database.close())


def test_disabled_source_is_removed_from_member_memory_list(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]
            asyncio.run(_insert_source_backed_memory(database, source_id=source_id))

            owner_before = client.get(
                "/api/memories",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )
            member_before = client.get(
                "/api/memories",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            disabled = client.put(
                f"/api/sources/{source_id}/subscription",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
                json={"enabled": False},
            )
            member_after = client.get(
                "/api/memories",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            owner_after = client.get(
                "/api/memories",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )

        assert owner_before.json()["total"] == 1
        assert member_before.json()["total"] == 1
        assert disabled.status_code == 200, disabled.text
        assert member_after.json()["total"] == 0
        assert owner_after.json()["total"] == 1
    finally:
        asyncio.run(database.close())
