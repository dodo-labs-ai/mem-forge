from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Memory, content_hash
from memforge.server.admin_api import create_admin_app
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.admin_memory import (
    MemoryAdminListFilters,
    MemoryAdminQueryPage,
)


def _memory(memory_id: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        memory_type="fact",
        content="Storage-neutral admin memory row.",
        content_hash=content_hash(memory_id),
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
    )


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def test_memory_list_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeAdminReader:
        def __init__(self) -> None:
            self.calls = []

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

        async def query_memory_admin_page(
            self,
            *,
            scope: AccessScope,
            filters: MemoryAdminListFilters,
            limit: int,
            offset: int,
        ) -> MemoryAdminQueryPage:
            self.calls.append((scope, filters, limit, offset))
            return MemoryAdminQueryPage(memories=[_memory("mem-neutral")], total=7)

        async def get_origin_source_pairs(self, memory_ids: list[str]):
            assert memory_ids == ["mem-neutral"]
            return {
                "mem-neutral": [
                    ("jira", "corroborated", None),
                    ("confluence", "extracted", None),
                ],
            }

    reader = FakeAdminReader()
    app = create_admin_app(db=reader, config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get(
            "/api/memories",
            params={
                "type": "fact",
                "status": "active",
                "source": "src-a",
                "project": "PAY",
                "search": "Payroll",
                "include_private": "true",
                "limit": 3,
                "offset": 2,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 7
    assert payload["limit"] == 3
    assert payload["offset"] == 2
    assert payload["data"][0]["id"] == "mem-neutral"
    assert payload["data"][0]["origin_source_type"] == "confluence"

    scope, filters, limit, offset = reader.calls[0]
    assert scope.user_id == LOCAL_DEV_USER_ID
    assert scope.include_private is True
    assert scope.allowed_statuses == ("active",)
    assert filters == MemoryAdminListFilters(
        memory_type="fact",
        status="active",
        source="src-a",
        project="PAY",
        search="Payroll",
    )
    assert limit == 3
    assert offset == 2


def test_memory_list_route_uses_injected_principal_resolver(tmp_path):
    class FakeAdminReader:
        def __init__(self) -> None:
            self.scope: AccessScope | None = None

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

        async def query_memory_admin_page(
            self,
            *,
            scope: AccessScope,
            filters: MemoryAdminListFilters,
            limit: int,
            offset: int,
        ) -> MemoryAdminQueryPage:
            self.scope = scope
            return MemoryAdminQueryPage(memories=[], total=0)

        async def get_origin_source_pairs(self, memory_ids: list[str]):
            return {}

    reader = FakeAdminReader()
    app = create_admin_app(
        db=reader,
        config=_config(tmp_path),
        principal_resolver=lambda request: request.headers["x-cloud-user-id"],
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/memories",
            headers={"x-cloud-user-id": "cloud-user-1"},
            params={"include_private": "true"},
        )

    assert response.status_code == 200, response.text
    assert reader.scope is not None
    assert reader.scope.user_id == "cloud-user-1"
