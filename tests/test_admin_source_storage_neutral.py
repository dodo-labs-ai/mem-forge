from __future__ import annotations

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.server.admin_api import create_admin_app


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def test_source_list_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def list_sources(self) -> list[dict]:
            return [
                {
                    "id": "src-neutral",
                    "type": "confluence",
                    "name": "Neutral Source",
                    "config": {"base_url": "https://wiki.example.test", "pat": "secret"},
                    "status": "active",
                    "last_sync": None,
                    "doc_count": 99,
                    "created_at": "2026-06-13T00:00:00+00:00",
                    "updated_at": "2026-06-13T00:00:00+00:00",
                }
            ]

        async def count_source_memories(self, source_id: str) -> int:
            assert source_id == "src-neutral"
            return 7

        async def count_documents(self, source: str | None = None) -> int:
            assert source == "src-neutral"
            return 3

        async def get_sync_history(
            self, source: str | None = None, limit: int = 20
        ) -> list[dict]:
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

        async def is_source_enabled_for_user(self, source_id: str, user_id: str) -> bool:
            assert source_id == "src-neutral"
            return True

        async def set_source_subscription(
            self, source_id: str, user_id: str, enabled: bool
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
    assert source["client"] is None
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
    }


def test_source_projects_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {"id": source_id, "type": "confluence", "name": "Neutral Source"}

        async def list_source_projects(self, source_id: str) -> list[dict]:
            assert source_id == "src-neutral"
            return [
                {
                    "project": "PAY",
                    "document_count": 3,
                    "memory_count": 7,
                    "last_observed_at": "2026-06-13T00:00:00+00:00",
                }
            ]

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
