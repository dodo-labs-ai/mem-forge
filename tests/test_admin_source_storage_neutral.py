from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def test_source_list_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def list_sources(self) -> list[dict]:
            return [
                {
                    "id": "src-neutral",
                    "type": "confluence",
                    "name": "Neutral Source",
                    "config": {"base_url": "https://wiki.example.test", "pat": "secret"},
                    "status": "active",
                    "access_policy": "private",
                    "access_state": "active",
                    "owner_user_id": "dev",
                    "last_sync": None,
                    "doc_count": 99,
                    "created_at": "2026-06-13T00:00:00+00:00",
                    "updated_at": "2026-06-13T00:00:00+00:00",
                    "sync_schedule": {
                        "enabled": True,
                        "interval_minutes": 60,
                        "next_run_at": "2026-06-13T01:00:00+00:00",
                        "updated_at": "2026-06-13T00:00:00+00:00",
                    },
                }
            ]

        async def count_source_memories(
            self,
            source_id: str,
            *,
            include_private: bool = False,
            owner_user_id: str | None = None,
        ) -> int:
            assert source_id == "src-neutral"
            assert include_private is True
            assert owner_user_id == "dev"
            return 7

        async def count_documents(self, source: str | None = None) -> int:
            assert source == "src-neutral"
            return 3

        async def get_sync_history(self, source: str | None = None, limit: int = 20) -> list[dict]:
            assert source == "src-neutral"
            assert limit == 1
            return [
                {
                    "status": "partial",
                    "started_at": "2026-06-13T00:00:01+00:00",
                    "finished_at": "2026-06-13T00:00:10+00:00",
                    "docs_processed": 3,
                    "docs_updated": 2,
                    "docs_failed": 1,
                    "memories_extracted": 4,
                    "error_message": "one failed",
                    "failed_docs": [{"doc_id": "doc-1", "error": "boom"}],
                }
            ]

        async def get_latest_source_sync_run(
            self,
            *,
            source_id: str,
            workspace_id: str = "default",
        ):
            assert source_id == "src-neutral"
            assert workspace_id == "default"
            return None

        async def get_active_source_access_transition(self, source_id: str):
            assert source_id == "src-neutral"
            return None

        async def is_source_enabled_for_user(self, source_id: str, user_id: str) -> bool:
            assert source_id == "src-neutral"
            return True

        async def is_source_pinned_for_user(self, source_id: str, user_id: str) -> bool:
            assert source_id == "src-neutral"
            return False

        async def set_source_subscription(self, source_id: str, user_id: str, enabled: bool) -> None:
            raise AssertionError("not used by source list")

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            raise AssertionError("not used by source list")

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/sources")

    assert response.status_code == 200
    source = response.json()["data"][0]
    assert source["id"] == "src-neutral"
    assert source["config"] == {
        "base_url": "https://wiki.example.test",
        "pat_configured": True,
    }
    assert source["doc_count"] == 3
    assert source["memory_count"] == 7
    assert source["pinned_for_me"] is False
    assert source["client"] is None
    assert source["access_policy"] == "private"
    assert source["owner_user_id"] == "dev"
    assert source["sync"] == {
        "status": "partial",
        "started_at": "2026-06-13T00:00:01+00:00",
        "finished_at": "2026-06-13T00:00:10+00:00",
        "docs_processed": 3,
        "docs_updated": 2,
        "docs_failed": 1,
        "memories_extracted": 4,
        "error_message": "one failed",
        "failed_docs": [{"doc_id": "doc-1", "error": "boom"}],
        "progress": {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 3, "unit": "page"},
            "counts": {"changed": 2, "failed": 1, "memories_created": 4},
        },
    }
    assert source["sync_schedule"] == {
        "enabled": True,
        "interval_minutes": 60,
        "next_run_at": "2026-06-13T01:00:00+00:00",
        "updated_at": "2026-06-13T00:00:00+00:00",
    }


def test_source_projects_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "confluence",
                "name": "Neutral Source",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "dev",
            }

        async def list_source_projects(
            self,
            source_id: str,
            *,
            include_private: bool = False,
            owner_user_id: str | None = None,
        ) -> list[dict]:
            assert source_id == "src-neutral"
            assert include_private is True
            assert owner_user_id == "dev"
            return [
                {
                    "project": "PAY",
                    "document_count": 3,
                    "memory_count": 7,
                    "last_observed_at": "2026-06-13T00:00:00+00:00",
                }
            ]

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            raise AssertionError("not used by source projects")

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/sources/src-neutral/projects")

    assert response.status_code == 200
    assert response.json() == {
        "source_id": "src-neutral",
        "projects": [
            {
                "project": "PAY",
                "document_count": 3,
                "memory_count": 7,
                "last_observed_at": "2026-06-13T00:00:00+00:00",
            }
        ],
    }


def test_source_schedule_routes_use_storage_neutral_store(tmp_path):
    class FakeSourceReader:
        def __init__(self) -> None:
            self.updated: tuple[str, bool, int] | None = None

        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "confluence",
                "name": "Neutral Source",
                "created_by_user_id": "user-a",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "user-a",
                "sync_schedule": {
                    "enabled": False,
                    "interval_minutes": 1440,
                    "next_run_at": None,
                    "updated_at": None,
                },
            }

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            self.updated = (source_id, enabled, interval_minutes)

    reader = FakeSourceReader()
    app = create_admin_app(
        db=reader,
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )

    with TestClient(app) as client:
        get_response = client.get("/api/sources/src-neutral/schedule")
        put_response = client.put(
            "/api/sources/src-neutral/schedule",
            json={"enabled": True, "interval_minutes": 60},
        )

    assert get_response.status_code == 200, get_response.text
    assert get_response.json() == {
        "enabled": False,
        "interval_minutes": 1440,
        "next_run_at": None,
        "updated_at": None,
    }
    assert put_response.status_code == 200, put_response.text
    assert reader.updated == ("src-neutral", True, 60)
