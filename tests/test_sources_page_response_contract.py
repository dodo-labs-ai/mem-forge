"""The typed response contract of the routes behind the Sources page.

Each route declares a response model, so the model decides which keys reach
the wire. These tests pin the exact key sets that storage-backed rows produce,
so a field a client reads cannot disappear from the response unnoticed.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database

SOURCE_KEYS = {
    "id",
    "type",
    "name",
    "config",
    "status",
    "owner_user_id",
    "access_policy",
    "access_state",
    "access_transition",
    "last_sync",
    "doc_count",
    "memory_count",
    "created_at",
    "project_binding",
    "sync_schedule",
    "sync",
    "client",
    "ownership",
    "capabilities",
    "execution",
    "subscription",
    "enabled_for_me",
    "pinned_for_me",
}
DURABLE_SYNC_KEYS = {
    "run_id",
    "status",
    "trigger",
    "force_full_sync",
    "created_at",
    "started_at",
    "finished_at",
    "next_attempt_at",
    "recovery_count",
    "error_message",
    "progress",
    "progress_revision",
    "progress_updated_at",
}
IN_PROCESS_SYNC_KEYS = {
    "status",
    "phase",
    "started_at",
    "finished_at",
    "docs_processed",
    "docs_total",
    "docs_updated",
    "docs_failed",
    "memories_extracted",
    "docs_stored",
    "memories_stored",
    "current_title",
    "error_message",
    "progress",
}
HISTORY_SYNC_KEYS = {
    "run_id",
    "status",
    "started_at",
    "finished_at",
    "docs_processed",
    "docs_updated",
    "docs_failed",
    "memories_extracted",
    "error_message",
    "failed_docs",
    "progress",
}
ACCESS_TRANSITION_KEYS = {
    "operation_id",
    "source_id",
    "previous_policy",
    "target_policy",
    "status",
    "total_memories",
    "processed_memories",
    "error_code",
    "error_message",
    "created_at",
    "updated_at",
    "completed_at",
}
LOCAL_AGENT_JOB_KEYS = {
    "job_id",
    "workspace_id",
    "source_id",
    "source_type",
    "operation",
    "status",
    "payload",
    "execution_owner_user_id",
    "result",
    "last_error",
    "next_attempt_at",
    "leased_until",
    "attempt_count",
    "created_at",
    "updated_at",
    "finished_at",
}
# Storage columns the list rows are read from. The response carries the
# schedule and ownership columns in ``sync_schedule`` and ``ownership``; the
# HANA row's own ``updated_at`` is not part of the contract.
STORAGE_ONLY_SOURCE_KEYS = {
    "created_by_user_id",
    "execution_owner_user_id",
    "sync_schedule_enabled",
    "sync_schedule_interval_minutes",
    "sync_schedule_next_at",
    "sync_schedule_updated_at",
    "updated_at",
}


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _principal(request: Request) -> str:
    return request.headers.get("x-test-user", "owner-user")


def _workspace_role(request: Request) -> str:
    return request.headers.get("x-test-workspace-role", "member")


def _confluence(name: str, root: str, *, access_policy: str = "workspace") -> dict:
    return {
        "type": "confluence",
        "name": name,
        "access_policy": access_policy,
        "config": {
            "base_url": f"https://wiki-{root}.example.test",
            "pat": "secret-token",
            "sync_mode": "page_tree",
            "page_tree_root": root,
            "include_children": True,
        },
    }


def _local_jira() -> dict:
    return {
        "type": "jira",
        "name": "Payroll Jira",
        "access_policy": "workspace",
        "config": {
            "base_url": "https://jira.example.test",
            "auth_mode": "browser_cookie",
            "sync_mode": "local_agent",
            "projects": ["PAY"],
        },
    }


def test_source_list_rows_carry_exactly_the_contract_fields(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=_principal,
        workspace_role_resolver=_workspace_role,
    )
    try:
        with TestClient(app) as client:
            ids = {}
            for key, payload in {
                "history": {
                    **_confluence("History Wiki", "1"),
                    "sync_schedule": {"enabled": True, "interval_minutes": 60},
                },
                "durable": _confluence("Durable Wiki", "2", access_policy="private"),
                "in_process": _confluence("Running Wiki", "3"),
                "changing": _confluence("Changing Wiki", "4"),
                "jira": _local_jira(),
            }.items():
                created = client.post("/api/v1/sources", json=payload)
                assert created.status_code == 200, created.text
                ids[key] = created.json()["id"]
            asyncio.run(
                database.upsert_source(
                    id="src-agent",
                    type="agent_session",
                    name="Codex Session",
                    config_json=json.dumps({"client": "codex"}),
                    access_policy="private",
                    owner_user_id="owner-user",
                )
            )
            asyncio.run(
                database.insert_sync_history(
                    source=ids["history"],
                    status="partial",
                    docs_processed=3,
                    docs_updated=2,
                    docs_failed=1,
                    memories_extracted=4,
                    error_message="one failed",
                    failed_docs=[{"doc_id": "doc-1", "title": "Design", "error": "boom"}],
                    started_at="2026-06-13T00:00:01+00:00",
                    finished_at="2026-06-13T00:00:10+00:00",
                    run_id="run-history",
                )
            )
            asyncio.run(
                database.enqueue_source_sync_run(
                    source_id=ids["durable"],
                    workspace_id=app.state.sync_service.workspace_id,
                )
            )
            asyncio.run(
                database.create_source_access_transition(
                    operation_id="op-1",
                    source_id=ids["changing"],
                    idempotency_key="key-1",
                    actor_user_id="owner-user",
                    target_policy="private",
                )
            )
            sync_service = app.state.sync_service
            sync_service.progress[ids["in_process"]] = {
                "started_at": "2026-06-13T00:00:00+00:00",
                "phase": "processing",
                "docs_processed": 2,
                "docs_total": 5,
                "docs_updated": 1,
                "docs_failed": 0,
                "memories_extracted": 3,
                "title": "Page 2",
            }
            sync_service.is_running = lambda source_id: source_id == ids["in_process"]

            response = client.get("/api/v1/sources")
    finally:
        asyncio.run(database.close())

    assert response.status_code == 200, response.text
    rows = {row["name"]: row for row in response.json()["data"]}
    assert set(rows) == {
        "History Wiki",
        "Durable Wiki",
        "Running Wiki",
        "Changing Wiki",
        "Payroll Jira",
        "Codex Session",
    }
    for name, row in rows.items():
        expected = SOURCE_KEYS | ({"connection_status"} if name == "Payroll Jira" else set())
        assert set(row) == expected, name
        assert set(row["ownership"]) == {
            "created_by_user_id",
            "owner_user_id",
            "execution_owner_user_id",
            "viewer_role",
            "viewer_relationship",
        }
        assert set(row["execution"]) == {"kind", "operation", "immutable_config_fields"}

    history = rows["History Wiki"]
    assert set(history["sync"]) == HISTORY_SYNC_KEYS
    assert history["sync"]["failed_docs"] == [{"doc_id": "doc-1", "title": "Design", "error": "boom"}]
    assert history["sync"]["progress"] == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 3, "unit": "page"},
        "counts": {"changed": 2, "failed": 1, "memories_created": 4},
    }
    assert history["sync_schedule"]["enabled"] is True
    assert history["sync_schedule"]["interval_minutes"] == 60
    assert history["config"] == {
        "base_url": "https://wiki-1.example.test",
        "include_children": True,
        "page_tree_root": "1",
        "pat_configured": True,
        "sync_mode": "page_tree",
    }
    assert set(rows["Durable Wiki"]["sync"]) == DURABLE_SYNC_KEYS
    assert rows["Durable Wiki"]["sync"]["status"] == "pending"
    assert set(rows["Running Wiki"]["sync"]) == IN_PROCESS_SYNC_KEYS
    assert rows["Running Wiki"]["sync"]["progress"] == {
        "schema_version": 1,
        "phase": "processing",
        "progress": {"completed": 2, "total": 5, "unit": "page"},
        "counts": {"changed": 1, "failed": 0, "memories_created": 3},
    }
    assert set(rows["Changing Wiki"]["access_transition"]) == ACCESS_TRANSITION_KEYS
    assert rows["Changing Wiki"]["access_transition"]["status"] == "queued"
    assert rows["Payroll Jira"]["connection_status"] == {
        "state": "action_required",
        "reason": "authentication",
    }
    assert rows["Payroll Jira"]["execution"]["kind"] == "local_agent"
    assert rows["Codex Session"]["client"] == "codex"
    assert rows["Codex Session"]["sync"] is None


def test_source_list_validates_rows_shaped_like_the_hana_workspace_store(tmp_path):
    """Rows carry every SOURCES column, lower-cased, as the HANA store returns them."""

    class HanaShapedReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_: object) -> list[dict]:
            return []

        async def list_sources(self) -> list[dict]:
            return [
                {
                    "id": "src-hana",
                    "type": "confluence",
                    "name": "HANA Wiki",
                    "config": {"base_url": "https://wiki.example.test"},
                    "project_binding": None,
                    "status": "paused",
                    "last_sync": None,
                    "doc_count": 0,
                    "created_by_user_id": None,
                    "owner_user_id": "cloud-user",
                    "access_policy": "workspace",
                    "access_state": "changing",
                    "execution_owner_user_id": None,
                    "sync_schedule_enabled": 0,
                    "sync_schedule_interval_minutes": 1440,
                    "sync_schedule_next_at": None,
                    "sync_schedule_updated_at": None,
                    "created_at": None,
                    "updated_at": "2026-06-13T00:00:00+00:00",
                    "sync_schedule": {
                        "enabled": False,
                        "interval_minutes": 1440,
                        "next_run_at": None,
                        "updated_at": None,
                    },
                }
            ]

        async def count_source_memories(self, *_: object, **__: object) -> int:
            return 0

        async def count_documents(self, *_: object, **__: object) -> int:
            return 0

        async def get_latest_source_sync_run(self, **_: object):
            return None

        async def get_sync_history(self, **_: object) -> list[dict]:
            return [
                {
                    "id": 7,
                    "source": "src-hana",
                    "status": "failed",
                    "docs_processed": 0,
                    "docs_updated": 0,
                    "docs_failed": 0,
                    "memories_extracted": 0,
                    "error_message": "auth expired",
                    "failed_docs": [],
                    "started_at": "2026-06-13T00:00:01+00:00",
                    "finished_at": "2026-06-13T00:00:02+00:00",
                    "run_id": None,
                }
            ]

        async def get_active_source_access_transition(self, source_id: str):
            return {
                "operation_id": "op-hana",
                "source_id": source_id,
                "idempotency_key": "key-hana",
                "actor_user_id": "cloud-user",
                "previous_policy": "workspace",
                "target_policy": "private",
                "previous_source_status": "active",
                "status": "failed",
                "total_memories": 4,
                "processed_memories": 1,
                "error_code": "vector_write_failed",
                "error_message": "retry",
                "created_at": "2026-06-13T00:00:00+00:00",
                "updated_at": "2026-06-13T00:00:03+00:00",
                "completed_at": None,
            }

        async def is_source_enabled_for_user(self, *_: object) -> bool:
            return True

        async def is_source_pinned_for_user(self, *_: object) -> bool:
            return False

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            raise AssertionError("not used by source list")

    app = create_admin_app(
        db=HanaShapedReader(),
        config=_config(tmp_path),
        principal_resolver=lambda request: "cloud-user",
        workspace_role_resolver=lambda request: "workspace_admin",
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/sources")

    assert response.status_code == 200, response.text
    row = response.json()["data"][0]
    assert set(row) == SOURCE_KEYS
    assert not STORAGE_ONLY_SOURCE_KEYS & set(row)
    assert row["created_at"] is None
    assert set(row["sync"]) == HISTORY_SYNC_KEYS
    assert row["sync"]["run_id"] is None
    assert set(row["access_transition"]) == ACCESS_TRANSITION_KEYS


def test_sync_receipts_and_local_agent_routes_keep_their_shapes(tmp_path):
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=_principal,
        workspace_role_resolver=_workspace_role,
    )
    try:
        with TestClient(app) as client:
            server_id = client.post("/api/v1/sources", json=_confluence("Server Wiki", "1")).json()["id"]
            jira_id = client.post("/api/v1/sources", json=_local_jira()).json()["id"]

            run_receipt = client.post(f"/api/v1/sources/{server_id}/sync")
            coalesced_receipt = client.post(f"/api/v1/sources/{server_id}/sync")
            local_receipt = client.post(f"/api/v1/sources/{jira_id}/sync")

            offline = client.get("/api/cloud/local-agent/status")
            sync_job = client.post(
                "/api/cloud/local-agent/jobs",
                json={"source_id": jira_id, "source_type": "jira", "operation": "jira_sync", "payload": {}},
            )
            setup_job = client.post(
                "/api/cloud/local-agent/jobs",
                json={"source_type": "teams", "operation": "teams_auth", "payload": {}},
            )
            current = client.get("/api/cloud/local-agent/jobs/current")
            leased = client.post("/api/cloud/local-agent/jobs/lease", json={"limit": 5}).json()["jobs"]
            online = client.get("/api/cloud/local-agent/status")

            default_preferences = client.get("/api/v1/source-list/preferences")
            saved_preferences = client.put("/api/v1/source-list/preferences", json={"sort_mode": "recently_synced"})
            read_preferences = client.get("/api/v1/source-list/preferences")
    finally:
        asyncio.run(database.close())

    assert run_receipt.status_code == 202, run_receipt.text
    assert set(run_receipt.json()) == {"ok", "message", "source_id", "run_id", "status", "created_at", "coalesced"}
    assert run_receipt.json()["status"] == "pending"
    assert coalesced_receipt.json()["coalesced"] is True
    assert local_receipt.status_code == 202, local_receipt.text
    assert local_receipt.json() == {
        "ok": True,
        "message": "Local collection enqueued",
        "source_id": jira_id,
        "job_id": local_receipt.json()["job_id"],
        "status": "queued",
        "coalesced": False,
    }

    assert sync_job.status_code == 201, sync_job.text
    assert set(sync_job.json()) == {"job_id", "status", "created_at", "coalesced"}
    assert setup_job.status_code == 201, setup_job.text
    assert set(setup_job.json()) == {"job_id", "status", "coalesced"}
    assert current.status_code == 200, current.text
    assert [set(job) for job in current.json()["data"]] == [LOCAL_AGENT_JOB_KEYS]
    assert current.json()["data"][0]["result"] == {}
    assert leased

    assert set(offline.json()) == {"status", "last_seen_at", "checked_at", "stale_after_seconds"}
    assert offline.json()["status"] == "offline"
    assert offline.json()["last_seen_at"] is None
    assert online.json()["status"] == "online"

    assert default_preferences.json() == {"sort_mode": "newest"}
    assert saved_preferences.json() == {"sort_mode": "recently_synced"}
    assert read_preferences.json() == {"sort_mode": "recently_synced"}


def test_openapi_types_the_sources_page_responses(tmp_path):
    document = create_admin_app(config=_config(tmp_path)).openapi()

    def response_schema(path: str, method: str, status: str) -> dict:
        return document["paths"][path][method]["responses"][status]["content"]["application/json"]["schema"]

    assert response_schema("/api/v1/sources", "get", "200") == {"$ref": "#/components/schemas/SourceListResponse"}
    for method in ("get", "put"):
        assert response_schema("/api/v1/source-list/preferences", method, "200") == {
            "$ref": "#/components/schemas/SourceListPreferencesResponse"
        }
    sync_receipt = response_schema("/api/v1/sources/{source_id}/sync", "post", "202")
    assert {option["$ref"] for option in sync_receipt["anyOf"]} == {
        "#/components/schemas/SourceSyncRunReceiptResponse",
        "#/components/schemas/LocalCollectionReceiptResponse",
    }
    assert response_schema("/api/cloud/local-agent/status", "get", "200") == {
        "$ref": "#/components/schemas/LocalAgentDaemonStatusResponse"
    }
    assert response_schema("/api/cloud/local-agent/jobs/current", "get", "200") == {
        "$ref": "#/components/schemas/LocalAgentJobListResponse"
    }
    assert response_schema("/api/cloud/local-agent/jobs", "post", "201") == {
        "$ref": "#/components/schemas/LocalAgentJobCreateResponse"
    }
    source = document["components"]["schemas"]["SourceResponse"]
    assert set(source["properties"]) == SOURCE_KEYS | {"connection_status"}
    assert set(source["required"]) == SOURCE_KEYS - {
        "last_sync",
        "created_at",
        "project_binding",
        "sync_schedule",
    }
    job_result = document["components"]["schemas"]["LocalAgentJobResponse"]["properties"]["result"]
    assert job_result["type"] == "object"
    assert "anyOf" not in job_result
