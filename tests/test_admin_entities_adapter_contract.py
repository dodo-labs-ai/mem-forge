from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Entity, EntityAlias, Memory, content_hash


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


class AdapterOnlyEntityDb:
    def __init__(self) -> None:
        self.entity = Entity(
            id=1,
            canonical_name="Payroll Area",
            tags=["payroll"],
            display_name="Payroll Area",
            created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
        )
        self.aliases = [
            EntityAlias(
                alias="Payroll Zone",
                alias_normalized="payroll zone",
                canonical_id=1,
                source="llm_extracted",
                created_at=datetime(2026, 6, 11, tzinfo=timezone.utc),
            )
        ]
        self.inserted_aliases: list[tuple[str, str, int, str]] = []
        self.removed_aliases: list[tuple[int, str]] = []
        self.memory = Memory(
            id="mem-1",
            memory_type="fact",
            content="Payroll cutoff is the 25th.",
            content_hash=content_hash("Payroll cutoff is the 25th."),
            tags=["payroll"],
            confidence=0.9,
            status="active",
            created_at=datetime(2026, 6, 10, tzinfo=timezone.utc),
            updated_at=datetime(2026, 6, 11, tzinfo=timezone.utc),
        )

    async def list_entities(
        self,
        *,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Entity], int]:
        return [self.entity], 1

    async def get_entity(self, entity_id: int) -> Entity | None:
        return self.entity if entity_id == self.entity.id else None

    async def count_memories_for_entity(self, entity_id: int) -> int:
        return 2 if entity_id == self.entity.id else 0

    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]:
        return list(self.aliases) if entity_id == self.entity.id else []

    async def insert_alias(
        self,
        alias: str,
        alias_normalized: str,
        canonical_id: int,
        source: str,
    ) -> None:
        self.inserted_aliases.append((alias, alias_normalized, canonical_id, source))

    async def remove_entity_alias(self, *, entity_id: int, alias_normalized: str) -> bool:
        self.removed_aliases.append((entity_id, alias_normalized))
        return True

    async def merge_entities(self, *, source_id: int, target_id: int) -> dict:
        return {
            "source_id": source_id,
            "source_name": "Legacy Payroll Area",
            "target_id": target_id,
            "target_name": self.entity.canonical_name,
        }

    async def get_schedule_config(self) -> dict:
        return {"enabled": False}

    async def get_memory(self, memory_id: str) -> Memory | None:
        return self.memory if memory_id == self.memory.id else None

    async def filter_visible_ids(self, ids: list[str], scope) -> set[str]:
        return {memory_id for memory_id in ids if memory_id == self.memory.id}

    async def get_memory_sources(self, memory_id: str) -> list:
        return []

    async def get_memory_entity_names(self, memory_id: str) -> list[str]:
        return [self.entity.canonical_name] if memory_id == self.memory.id else []

    async def get_origin_source_pairs(
        self, memory_ids: list[str]
    ) -> dict[str, list[tuple[str, str | None, str | None]]]:
        return {}


def test_entity_admin_routes_use_adapter_methods_without_sqlite_db(tmp_path: Path) -> None:
    from memforge.server.admin_api import create_admin_app

    database = AdapterOnlyEntityDb()
    app = create_admin_app(db=database, config=_config(tmp_path))

    with TestClient(app) as client:
        list_response = client.get("/api/entities")
        detail_response = client.get("/api/entities/1")
        aliases_response = client.get("/api/entities/1/aliases")
        add_alias_response = client.post("/api/entities/1/aliases", json={"alias": "Payroll Org"})
        remove_alias_response = client.delete("/api/entities/1/aliases/Payroll%20Zone")
        merge_response = client.post(
            "/api/entities/merge",
            json={"source_id": 2, "target_id": 1},
        )
        memory_detail_response = client.get("/api/memories/mem-1")

    assert list_response.status_code == 200, list_response.text
    assert list_response.json()["total"] == 1
    assert detail_response.status_code == 200, detail_response.text
    assert detail_response.json()["linked_memory_count"] == 2
    assert aliases_response.status_code == 200, aliases_response.text
    assert aliases_response.json()["data"][0]["alias"] == "Payroll Zone"
    assert add_alias_response.status_code == 200, add_alias_response.text
    assert database.inserted_aliases == [
        ("Payroll Org", "payroll org", 1, "admin_manual")
    ]
    assert remove_alias_response.status_code == 200, remove_alias_response.text
    assert database.removed_aliases == [(1, "payroll zone")]
    assert merge_response.status_code == 200, merge_response.text
    assert merge_response.json()["merged"]["target_name"] == "Payroll Area"
    assert memory_detail_response.status_code == 200, memory_detail_response.text
    assert memory_detail_response.json()["entity_refs"] == ["Payroll Area"]
