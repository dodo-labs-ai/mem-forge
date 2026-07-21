from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.lifecycle_plan import (
    LifecycleGate,
    LifecycleGateState,
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
)
from memforge.models import (
    ContentItem,
    Memory,
    NormalizedContent,
    RawContent,
    content_hash,
)
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.server.source_admin_service import (
    _current_lifecycle_maintenance_payload,
)
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
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


def _confluence_payload(
    name: str = "Architecture Wiki",
    *,
    access_policy: str = "workspace",
) -> dict:
    return {
        "type": "confluence",
        "name": name,
        "access_policy": access_policy,
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
        "access_policy": "workspace",
        "config": {
            "region": "emea",
            "conversation_ids": ["19:conversation-a@example.test"],
            "conversation_gap_minutes": 60,
        },
    }


def _github_repo_payload(*, connection_mode: str) -> dict:
    return {
        "type": "github_repo",
        "name": "Internal Cookbook",
        "access_policy": "workspace",
        "config": {
            "repo_url": "https://github.example.test/platform/cookbook",
            "ref": "main",
            "connection_mode": connection_mode,
        },
    }


def _jira_payload(*, sync_mode: str) -> dict:
    return {
        "type": "jira",
        "name": "Payroll Jira",
        "access_policy": "workspace",
        "config": {
            "base_url": "https://jira.example.test",
            "auth_mode": "browser_cookie",
            "sync_mode": sync_mode,
            "projects": ["PAY"],
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


def test_create_source_requires_explicit_access_policy(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        payload = _confluence_payload()
        payload.pop("access_policy")

        with TestClient(app) as client:
            response = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=payload,
            )

        assert response.status_code == 422, response.text
    finally:
        asyncio.run(database.close())


def test_private_source_is_undiscoverable_to_members_and_admins(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
                json=_confluence_payload(access_policy="private"),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]

            member_list = client.get(
                "/api/sources",
                headers={"x-test-user": "member-b", "x-test-workspace-role": "member"},
            )
            admin_list = client.get(
                "/api/sources",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
            )
            direct_subscription = client.put(
                f"/api/sources/{source_id}/subscription",
                headers={"x-test-user": "member-b", "x-test-workspace-role": "member"},
                json={"enabled": True},
            )

        assert member_list.status_code == 200
        assert member_list.json()["data"] == []
        assert admin_list.status_code == 200
        assert admin_list.json()["data"] == []
        assert direct_subscription.status_code == 404
    finally:
        asyncio.run(database.close())


def test_owner_can_share_private_source_with_idempotent_transition(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        owner_headers = {
            "x-test-user": "owner-user",
            "x-test-workspace-role": "member",
        }
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers=owner_headers,
                json=_confluence_payload(access_policy="private"),
            )
            source_id = created.json()["id"]
            transition_headers = {
                **owner_headers,
                "Idempotency-Key": "share-source-once",
            }
            started = client.post(
                f"/api/sources/{source_id}/access-transitions",
                headers=transition_headers,
                json={"target_policy": "workspace"},
            )
            repeated = client.post(
                f"/api/sources/{source_id}/access-transitions",
                headers=transition_headers,
                json={"target_policy": "workspace"},
            )
            member_list = client.get(
                "/api/sources",
                headers={
                    "x-test-user": "other-user",
                    "x-test-workspace-role": "member",
                },
            )

        assert started.status_code == 202, started.text
        assert repeated.status_code == 202, repeated.text
        assert repeated.json()["operation_id"] == started.json()["operation_id"]
        source = asyncio.run(database.get_source(source_id))
        assert source["access_policy"] == "workspace"
        assert source["access_state"] == "active"
        assert source_id in {row["id"] for row in member_list.json()["data"]}
    finally:
        asyncio.run(database.close())


def test_private_source_access_transition_does_not_leak_to_workspace_admin(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={
                    "x-test-user": "owner-user",
                    "x-test-workspace-role": "member",
                },
                json=_confluence_payload(access_policy="private"),
            )
            source_id = created.json()["id"]
            response = client.post(
                f"/api/sources/{source_id}/access-transitions",
                headers={
                    "x-test-user": "admin-b",
                    "x-test-workspace-role": "workspace_admin",
                    "Idempotency-Key": "unauthorized-share",
                },
                json={"target_policy": "workspace"},
            )

        assert response.status_code == 404
    finally:
        asyncio.run(database.close())


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
            "owner_user_id": "owner-user",
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
            "can_change_access": False,
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
        assert source["ownership"]["viewer_relationship"] == "owner"
        assert source["capabilities"]["can_configure"] is True
        assert source["capabilities"]["can_sync"] is True
        assert source["capabilities"]["can_delete"] is True
        assert source["config"]["pat_configured"] is True
        assert "pat" not in source["config"]
        assert "pat_encrypted" not in source["config"]
    finally:
        asyncio.run(database.close())


def test_agent_session_rejects_ordinary_sync_and_hides_historical_sync_failure(tmp_path):
    database = _connect_database(tmp_path)
    source_id = "src-agent-session-sync-policy"
    owner_headers = {
        "x-test-user": "owner-user",
        "x-test-workspace-role": "member",
    }
    try:
        asyncio.run(
            database.upsert_source(
                id=source_id,
                type="agent_session",
                name="Codex Session",
                config_json='{"documents_dir":"/missing/local-only-path"}',
                access_policy="private",
                owner_user_id="owner-user",
            )
        )
        asyncio.run(
            database.insert_sync_history(
                source=source_id,
                status="failed",
                docs_processed=0,
                docs_updated=0,
                docs_failed=0,
                memories_extracted=0,
                error_message="Agent session documents directory does not exist",
                failed_docs=None,
                started_at="2026-07-15T06:57:00+00:00",
                finished_at="2026-07-15T06:57:01+00:00",
            )
        )

        with TestClient(_app(tmp_path, database)) as client:
            listed = client.get("/api/sources", headers=owner_headers)
            synced = client.post(
                f"/api/sources/{source_id}/sync",
                headers=owner_headers,
            )
            force_synced = client.post(
                f"/api/sources/{source_id}/force-resync",
                headers=owner_headers,
            )
            scheduled = client.put(
                f"/api/sources/{source_id}/schedule",
                headers=owner_headers,
                json={"enabled": True, "interval_minutes": 60},
            )
            inventory = client.get(
                f"/api/sources/{source_id}/projection-inventory",
                headers=owner_headers,
            )

        assert listed.status_code == 200, listed.text
        source = next(row for row in listed.json()["data"] if row["id"] == source_id)
        assert source["capabilities"]["can_sync"] is False
        assert source["capabilities"]["can_force_resync"] is False
        assert source["sync"] is None
        for response in (synced, force_synced, scheduled):
            assert response.status_code == 409, response.text
            assert response.json()["detail"] == "source_sync_not_supported"
        assert inventory.status_code == 200, inventory.text
        assert asyncio.run(database.get_latest_source_sync_run(source_id=source_id)) is None
        history = asyncio.run(database.get_sync_history(source=source_id, limit=1))
        assert history[0]["status"] == "failed"
    finally:
        asyncio.run(database.close())


def test_source_list_projects_active_lifecycle_maintenance(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        owner_headers = {
            "x-test-user": "owner-user",
            "x-test-workspace-role": "member",
        }
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers=owner_headers,
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]
            now = datetime.now(timezone.utc).isoformat()
            asyncio.run(
                database.create_lifecycle_backfill_job(
                    LifecycleBackfillJob(
                        id="maintenance-job",
                        source_id=source_id,
                        status=LifecycleBackfillJobStatus.QUEUED,
                        created_at=now,
                    )
                )
            )
            asyncio.run(database.start_lifecycle_backfill_job("maintenance-job"))

            listed = client.get("/api/sources", headers=owner_headers)

        assert listed.status_code == 200, listed.text
        source = listed.json()["data"][0]
        assert source["lifecycle_maintenance"] == {
            "status": "running",
            "created_at": now,
            "started_at": source["lifecycle_maintenance"]["started_at"],
            "finished_at": None,
        }
        assert source["lifecycle_maintenance"]["started_at"] is not None
    finally:
        asyncio.run(database.close())


def test_source_list_suppresses_resolved_lifecycle_failure_but_keeps_history(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        owner_headers = {
            "x-test-user": "owner-user",
            "x-test-workspace-role": "member",
        }
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers=owner_headers,
                json=_confluence_payload(),
            )
            assert created.status_code == 200, created.text
            source_id = created.json()["id"]
            asyncio.run(
                database.create_lifecycle_backfill_job(
                    LifecycleBackfillJob(
                        id="historical-failed-maintenance",
                        source_id=source_id,
                        status=LifecycleBackfillJobStatus.QUEUED,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
            )
            asyncio.run(
                database.start_lifecycle_backfill_job(
                    "historical-failed-maintenance"
                )
            )
            asyncio.run(
                database.fail_lifecycle_backfill_job(
                    "historical-failed-maintenance",
                    error="operator recovered stale lifecycle job",
                )
            )
            asyncio.run(
                database.gate_destructive_lifecycle(
                    source_id,
                    reason="maintenance failure still requires operator action",
                )
            )
            actionable = client.get("/api/sources", headers=owner_headers)
            asyncio.run(database.enable_lifecycle_gate(source_id))

            listed = client.get("/api/sources", headers=owner_headers)
            lifecycle = client.get(
                f"/api/sources/{source_id}/memory-lifecycle",
                headers=owner_headers,
            )

        assert actionable.status_code == 200, actionable.text
        assert (
            actionable.json()["data"][0]["lifecycle_maintenance"]["status"]
            == "failed"
        )
        assert listed.status_code == 200, listed.text
        assert listed.json()["data"][0]["lifecycle_maintenance"] is None
        assert lifecycle.status_code == 200, lifecycle.text
        assert lifecycle.json()["jobs"][0]["status"] == "failed"
        assert (
            lifecycle.json()["jobs"][0]["error"]
            == "operator recovered stale lifecycle job"
        )
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize(
    ("has_open_finding", "has_vector_task"),
    [(True, False), (False, True)],
)
def test_failed_lifecycle_maintenance_keeps_each_current_blocker_actionable(
    has_open_finding,
    has_vector_task,
):
    class LifecycleAttentionReader:
        async def get_lifecycle_gate(self, source_id):
            return LifecycleGate(
                source_id=source_id,
                state=LifecycleGateState.ENABLED,
            )

        async def list_lifecycle_cutover_findings(self, source_id, *, status=None):
            return [object()] if has_open_finding else []

        async def list_lifecycle_vector_tasks(
            self,
            *,
            source_id=None,
            lifecycle_plan_id=None,
            limit=100,
        ):
            return [object()] if has_vector_task else []

    failed_job = LifecycleBackfillJob(
        id="failed-maintenance",
        source_id="src-actionable",
        status=LifecycleBackfillJobStatus.FAILED,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    projected = asyncio.run(
        _current_lifecycle_maintenance_payload(
            LifecycleAttentionReader(),
            source_id=failed_job.source_id,
            latest_job=failed_job,
        )
    )

    assert projected is not None
    assert projected["status"] == "failed"


def test_member_can_pin_source_only_for_their_own_source_list(tmp_path):
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

            pinned = client.put(
                f"/api/sources/{source_id}/pin",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            pinned_again = client.put(
                f"/api/sources/{source_id}/pin",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            other_list = client.get(
                "/api/sources",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            owner_list = client.get(
                "/api/sources",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )
            unpinned = client.delete(
                f"/api/sources/{source_id}/pin",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            unpinned_again = client.delete(
                f"/api/sources/{source_id}/pin",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )

        assert pinned.status_code == 200, pinned.text
        assert pinned.json() == {"source_id": source_id, "pinned": True}
        assert pinned_again.status_code == 200, pinned_again.text
        assert other_list.json()["data"][0]["pinned_for_me"] is True
        assert owner_list.json()["data"][0]["pinned_for_me"] is False
        assert unpinned.json() == {"source_id": source_id, "pinned": False}
        assert unpinned_again.json() == {"source_id": source_id, "pinned": False}
        assert asyncio.run(database.is_source_pinned_for_user(source_id, "other-user")) is False

        asyncio.run(database.set_source_pinned_for_user(source_id, "other-user", True))
        asyncio.run(database.delete_source_cascade(source_id))
        assert asyncio.run(database.is_source_pinned_for_user(source_id, "other-user")) is False
    finally:
        asyncio.run(database.close())


def test_source_list_sort_preference_is_personal_and_validated(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            default_response = client.get(
                "/api/source-list/preferences",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            updated = client.put(
                "/api/source-list/preferences",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
                json={"sort_mode": "name"},
            )
            other_response = client.get(
                "/api/source-list/preferences",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
            )
            owner_response = client.get(
                "/api/source-list/preferences",
                headers={"x-test-user": "owner-user", "x-test-workspace-role": "member"},
            )
            invalid = client.put(
                "/api/source-list/preferences",
                headers={"x-test-user": "other-user", "x-test-workspace-role": "member"},
                json={"sort_mode": "manual"},
            )

        assert default_response.status_code == 200
        assert default_response.json() == {"sort_mode": "newest"}
        assert updated.status_code == 200
        assert updated.json() == {"sort_mode": "name"}
        assert other_response.json() == {"sort_mode": "name"}
        assert owner_response.json() == {"sort_mode": "newest"}
        assert invalid.status_code == 422
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
            "message": "Only the source owner or a workspace admin can manage this source.",
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
        with TestClient(app) as client:
            listed = client.get(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
            )
        assert listed.status_code == 200, listed.text
        assert listed.json()["data"][0]["execution"] == {
            "kind": "local_agent",
            "operation": "teams_sync",
            "immutable_config_fields": [],
        }
    finally:
        asyncio.run(database.close())


def test_local_markdown_source_requires_root_before_execution_classification(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            response = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={
                    "type": "local_markdown",
                    "name": "Incomplete local source",
                    "access_policy": "workspace",
                    "config": {},
                },
            )

        assert response.status_code == 400, response.text
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
        assert connection_update.json()["detail"]["error"] == ("local_agent_source_connection_owner_forbidden")
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


def test_local_source_sync_enqueues_canonical_owner_job(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        payload = _teams_payload()
        payload["config"]["access_token"] = "must-not-reach-daemon"
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=payload,
            )
            assert created.status_code == 200, created.text

            triggered = client.post(
                f"/api/sources/{created.json()['id']}/sync",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"force_full_sync": True},
            )
            assert triggered.status_code == 202, triggered.text
            repeated = client.post(
                f"/api/sources/{created.json()['id']}/sync",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"force_full_sync": True},
            )
            assert repeated.status_code == 202, repeated.text
            assert repeated.json()["job_id"] == triggered.json()["job_id"]
            assert repeated.json()["coalesced"] is True

            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"limit": 5, "lease_seconds": 60},
            )

        assert leased.status_code == 200, leased.text
        jobs = leased.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["execution_owner_user_id"] == "owner-a"
        job_payload = dict(jobs[0]["payload"])
        assert job_payload.pop("source_config_revision")
        assert job_payload.pop("source_activity_epoch") == 0
        assert job_payload == {
            "region": "emea",
            "conversation_ids": ["19:conversation-a@example.test"],
            "conversation_gap_minutes": 60,
            "source_id": created.json()["id"],
            "source_type": "teams",
            "force_full_sync": True,
        }
    finally:
        asyncio.run(database.close())


def test_teams_scope_transition_is_attached_to_collection_job(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        headers = {"x-test-user": "owner-a", "x-test-workspace-role": "member"}
        with TestClient(app) as client:
            created = client.post("/api/sources", headers=headers, json=_teams_payload())
            source_id = created.json()["id"]
            updated = client.put(
                f"/api/sources/{source_id}",
                headers=headers,
                json={
                    "config": {
                        "region": "emea",
                        "conversation_ids": ["19:conversation-a@example.test"],
                        "conversation_gap_minutes": 60,
                        "max_age_days": 30,
                    }
                },
            )
            triggered = client.post(f"/api/sources/{source_id}/sync", headers=headers)
            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers=headers,
                json={"limit": 1, "lease_seconds": 60},
            )

        assert updated.status_code == 200, updated.text
        assert triggered.status_code == 202, triggered.text
        transition = leased.json()["jobs"][0]["payload"]["projection_scope_transition"]
        assert transition["previous_scope"] == {
            "conversation_ids": ["19:conversation-a@example.test"],
            "conversation_gap_minutes": 60,
        }
        assert transition["target_scope"] == {
            "conversation_ids": ["19:conversation-a@example.test"],
            "conversation_gap_minutes": 60,
            "max_age_days": 30,
        }
    finally:
        asyncio.run(database.close())


def test_local_agent_data_plane_is_lease_fenced_and_completion_is_idempotent(tmp_path):
    from memforge.local_agent.teams_ledger import build_teams_window_id

    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        headers = {"x-test-user": "owner-a", "x-test-workspace-role": "member"}
        with TestClient(app) as client:
            created = client.post("/api/sources", headers=headers, json=_teams_payload())
            source_id = created.json()["id"]
            client.post(f"/api/sources/{source_id}/sync", headers=headers)
            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers=headers,
                json={"limit": 1, "lease_seconds": 60},
            ).json()["jobs"][0]
            context = {
                "local_agent_job_id": leased["job_id"],
                "local_agent_attempt_count": leased["attempt_count"],
            }
            window_id = build_teams_window_id(
                source_id=source_id,
                conversation_id="19:conversation-a@example.test",
                root_or_anchor_message_id="m1",
                window_type="time_block",
            )
            accepted = client.post(
                f"/api/sources/{source_id}/adapter/packages",
                headers=headers,
                json={
                    **context,
                    "conversation_id": "19:conversation-a@example.test",
                    "window_id": window_id,
                    "revision_hash": "revision-a",
                    "root_message_id": "m1",
                    "window_type": "time_block",
                    "raw_payload": {
                        "conversation_id": "19:conversation-a@example.test",
                        "window_id": window_id,
                        "messages": [
                            {
                                "id": "m1",
                                "content": "hello",
                                "time": "2026-07-16T09:00:00+00:00",
                            }
                        ],
                    },
                },
            )
            accepted_process = client.post(
                f"/api/sources/{source_id}/process",
                headers=headers,
                json=context,
            )
            stale = client.post(
                f"/api/sources/{source_id}/process",
                headers=headers,
                json={**context, "local_agent_attempt_count": leased["attempt_count"] + 1},
            )
            completion_body = {
                "attempt_count": leased["attempt_count"],
                "status": "succeeded",
                "result": {"ok": True},
            }
            first_complete = client.post(
                f"/api/cloud/local-agent/jobs/{leased['job_id']}/complete",
                headers=headers,
                json=completion_body,
            )
            repeated_complete = client.post(
                f"/api/cloud/local-agent/jobs/{leased['job_id']}/complete",
                headers=headers,
                json=completion_body,
            )
            forbidden_repeat = client.post(
                f"/api/cloud/local-agent/jobs/{leased['job_id']}/complete",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
                json=completion_body,
            )

        assert accepted.status_code == 200, accepted.text
        assert accepted_process.status_code == 202, accepted_process.text
        expected_attempt_id = f"{leased['job_id']}:attempt:{leased['attempt_count']}"
        retained_inputs = asyncio.run(
            database.list_source_sync_inputs(
                source_id=source_id,
                input_snapshot_id=expected_attempt_id,
            )
        )
        assert len(retained_inputs) == 1
        process_run = asyncio.run(
            database.get_source_sync_run(accepted_process.json()["run_id"])
        )
        assert process_run is not None
        assert process_run.input_snapshot_id == expected_attempt_id
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"] == "local_agent_lease_not_current"
        assert first_complete.status_code == 200, first_complete.text
        assert repeated_complete.status_code == 200, repeated_complete.text
        assert repeated_complete.json()["status"] == "succeeded"
        assert forbidden_repeat.status_code == 404, forbidden_repeat.text
    finally:
        asyncio.run(database.close())


def test_source_config_change_fences_leased_job_and_enqueues_successor(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        headers = {"x-test-user": "owner-a", "x-test-workspace-role": "member"}
        with TestClient(app) as client:
            created = client.post("/api/sources", headers=headers, json=_jira_payload(sync_mode="local_agent"))
            source_id = created.json()["id"]
            first = client.post(f"/api/sources/{source_id}/sync", headers=headers)
            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers=headers,
                json={"limit": 1, "lease_seconds": 60},
            ).json()["jobs"][0]
            updated_payload = _jira_payload(sync_mode="local_agent")
            updated_payload["config"]["projects"] = ["PAY", "TIME"]
            updated = client.put(f"/api/sources/{source_id}", headers=headers, json=updated_payload)
            stale = client.post(
                f"/api/sources/{source_id}/process",
                headers=headers,
                json={
                    "local_agent_job_id": leased["job_id"],
                    "local_agent_attempt_count": leased["attempt_count"],
                },
            )
            successor = client.post(f"/api/sources/{source_id}/sync", headers=headers)

        assert first.status_code == 202, first.text
        assert updated.status_code == 200, updated.text
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"] == "local_agent_lease_not_current"
        assert successor.status_code == 202, successor.text
        assert successor.json()["job_id"] != leased["job_id"]
    finally:
        asyncio.run(database.close())


def test_local_agent_sync_job_can_only_be_created_and_leased_by_source_owner(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_teams_payload(),
            )
            source_id = created.json()["id"]
            request_body = {
                "source_id": source_id,
                "source_type": "teams",
                "operation": "teams_sync",
                "payload": {
                    "execution_owner_user_id": "admin-b",
                    "config": {"conversation_ids": ["attacker-value"]},
                },
            }

            forbidden = client.post(
                "/api/cloud/local-agent/jobs",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
                json=request_body,
            )
            accepted = client.post(
                "/api/cloud/local-agent/jobs",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=request_body,
            )
            other_jobs = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers={"x-test-user": "admin-b", "x-test-workspace-role": "workspace_admin"},
                json={"limit": 5, "lease_seconds": 60},
            )
            owner_jobs = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"limit": 5, "lease_seconds": 60},
            )

        assert forbidden.status_code == 403, forbidden.text
        assert accepted.status_code == 201, accepted.text
        assert other_jobs.json() == {"jobs": []}
        assert [job["job_id"] for job in owner_jobs.json()["jobs"]] == [accepted.json()["job_id"]]
        assert owner_jobs.json()["jobs"][0]["payload"]["conversation_ids"] == ["19:conversation-a@example.test"]
        assert "config" not in owner_jobs.json()["jobs"][0]["payload"]
    finally:
        asyncio.run(database.close())


def test_local_agent_setup_job_allows_member_but_forbids_viewer(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        body = {
            "source_type": "local_markdown",
            "operation": "local_markdown_pick_root",
            "payload": {"title": "Choose folder"},
        }
        with TestClient(app) as client:
            accepted = client.post(
                "/api/cloud/local-agent/jobs",
                headers={"x-test-user": "member-a", "x-test-workspace-role": "member"},
                json=body,
            )
            forbidden = client.post(
                "/api/cloud/local-agent/jobs",
                headers={"x-test-user": "viewer-a", "x-test-workspace-role": "viewer"},
                json=body,
            )
            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers={"x-test-user": "member-a", "x-test-workspace-role": "member"},
                json={"limit": 5, "lease_seconds": 60},
            )

        assert accepted.status_code == 201, accepted.text
        assert forbidden.status_code == 403, forbidden.text
        assert leased.json()["jobs"][0]["execution_owner_user_id"] == "member-a"
    finally:
        asyncio.run(database.close())


def test_local_source_sync_without_execution_owner_returns_contract_error(tmp_path):
    database = _connect_database(tmp_path)
    try:
        asyncio.run(
            database.upsert_source(
                id="src-ownerless-local",
                type="local_markdown",
                name="Ownerless Local",
                config_json='{"vault_id":"notes"}',
                access_policy="workspace",
                owner_user_id="owner-a",
            )
        )
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-ownerless-local/sync",
                headers={"x-test-user": "member-a", "x-test-workspace-role": "member"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "local_agent_sync_execution_owner_required"
    finally:
        asyncio.run(database.close())


def test_local_source_force_resync_collects_fresh_raw_data_before_processing(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_teams_payload(),
            )
            force = client.post(
                f"/api/sources/{created.json()['id']}/force-resync",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
            )
            leased = client.post(
                "/api/cloud/local-agent/jobs/lease",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"limit": 5, "lease_seconds": 60},
            )

        assert force.status_code == 202, force.text
        assert force.json()["status"] == "queued"
        assert "run_id" not in force.json()
        assert leased.json()["jobs"][0]["operation"] == "teams_sync"
        assert leased.json()["jobs"][0]["payload"]["force_full_sync"] is True
    finally:
        asyncio.run(database.close())


def test_projection_inventory_returns_server_owned_active_units(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        owner_headers = {
            "x-test-user": "owner-a",
            "x-test-workspace-role": "member",
        }
        with TestClient(app) as client:
            created = client.post(
                "/api/sources",
                headers=owner_headers,
                json=_teams_payload(),
            )
            source_id = created.json()["id"]

            item = ContentItem(
                item_id="window-1",
                title="Teams window",
                source_url="https://teams.example.test/window-1",
                last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
                version="revision-1",
                extra={
                    "conversation_id": "19:conversation-a@example.test",
                    "window_id": "window-1",
                    "root_message_id": "message-1",
                },
            )
            payload = {
                "messages": [
                    {
                        "id": "message-1",
                        "content": "Current answer",
                        "time": "2026-07-16T09:00:00Z",
                    }
                ]
            }
            projection = project_source_item(
                source_id=source_id,
                source_type="teams",
                run_id="run-inventory-route",
                item=item,
                raw=RawContent(
                    item=item,
                    body=json.dumps(payload).encode(),
                    content_type="application/json",
                ),
                normalized=NormalizedContent(
                    item=item,
                    markdown_body="Current answer",
                ),
            )
            asyncio.run(database.record_source_projection(projection))

            response = client.get(
                f"/api/sources/{source_id}/projection-inventory",
                headers=owner_headers,
            )
            forbidden = client.get(
                f"/api/sources/{source_id}/projection-inventory",
                headers={
                    "x-test-user": "viewer-a",
                    "x-test-workspace-role": "viewer",
                },
            )

        assert response.status_code == 200, response.text
        assert response.json() == {
            "source_id": source_id,
            "source_type": "teams",
            "next_cursor": None,
            "projection_scope_transition": None,
            "units": [
                {
                    "source_unit_id": projection.source_units[0].id,
                    "unit_type": "teams_window",
                    "provider_key": "window-1",
                    "locator": dict(projection.source_units[0].locator),
                }
            ],
        }
        assert forbidden.status_code == 403
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


def test_existing_source_cannot_change_between_server_and_local_execution(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            cloud_source = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_github_repo_payload(connection_mode="cloud_pull"),
            )
            local_source = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_github_repo_payload(connection_mode="local_push"),
            )
            assert cloud_source.status_code == 200, cloud_source.text
            assert local_source.status_code == 200, local_source.text

            to_local = client.put(
                f"/api/sources/{cloud_source.json()['id']}",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"config": _github_repo_payload(connection_mode="local_push")["config"]},
            )
            to_cloud = client.put(
                f"/api/sources/{local_source.json()['id']}",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"config": _github_repo_payload(connection_mode="cloud_pull")["config"]},
            )

        assert to_local.status_code == 409, to_local.text
        assert to_cloud.status_code == 409, to_cloud.text
        assert to_local.json()["detail"] == "source_execution_mode_immutable"
        assert to_cloud.json()["detail"] == "source_execution_mode_immutable"
    finally:
        asyncio.run(database.close())


def test_existing_jira_source_cannot_change_between_cloud_and_local_agent(tmp_path):
    database = _connect_database(tmp_path)
    try:
        app = _app(tmp_path, database)
        with TestClient(app) as client:
            cloud_source = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_jira_payload(sync_mode="cloud"),
            )
            local_source = client.post(
                "/api/sources",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json=_jira_payload(sync_mode="local_agent"),
            )
            assert cloud_source.status_code == 200, cloud_source.text
            assert local_source.status_code == 200, local_source.text

            to_local = client.put(
                f"/api/sources/{cloud_source.json()['id']}",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"config": _jira_payload(sync_mode="local_agent")["config"]},
            )
            to_cloud = client.put(
                f"/api/sources/{local_source.json()['id']}",
                headers={"x-test-user": "owner-a", "x-test-workspace-role": "member"},
                json={"config": _jira_payload(sync_mode="cloud")["config"]},
            )

        assert to_local.status_code == 409, to_local.text
        assert to_cloud.status_code == 409, to_cloud.text
        assert to_local.json()["detail"] == "source_execution_mode_immutable"
        assert to_cloud.json()["detail"] == "source_execution_mode_immutable"
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
