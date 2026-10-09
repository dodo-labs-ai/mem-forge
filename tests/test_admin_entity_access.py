"""Entity admin routes follow Memory visibility and workspace curation authority.

An Entity is discoverable only through a Memory the caller may query, so its
listing, detail, alias list, linked Memory count, and aggregate count never
reveal names learned from another user's private Memory. Manual aliases and
merges rewrite the workspace-wide entity graph and require workspace
administration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.models import Memory, Visibility, content_hash
from memforge.storage.database import Database

OWNER = "alice"
OTHER = "bob"
USER_HEADER = "x-test-user"
ROLE_HEADER = "x-test-workspace-role"


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _principal(request: Request) -> str:
    return request.headers.get(USER_HEADER, OWNER)


def _workspace_role(request: Request) -> str:
    return request.headers.get(ROLE_HEADER, "member")


def _as(user: str, role: str = "member") -> dict[str, str]:
    return {USER_HEADER: user, ROLE_HEADER: role}


def _memory(memory_id: str, *, owner: str | None = None) -> Memory:
    content = f"{memory_id} content"
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        visibility=Visibility.PRIVATE.value if owner else Visibility.WORKSPACE.value,
        owner_user_id=owner,
        created_at=now,
        updated_at=now,
    )


@dataclass(frozen=True)
class _Graph:
    shared: int
    private: int
    mixed: int
    orphan: int


async def _seed(database: Database) -> _Graph:
    """Seed one Entity per visibility shape.

    ``shared`` is linked to a workspace Memory, ``private`` only to Alice's
    private Memory, ``mixed`` to one workspace and one private Memory, and
    ``orphan`` to no Memory at all.
    """

    await database.insert_memory(_memory("mem-shared"))
    await database.insert_memory(_memory("mem-private", owner=OWNER))
    shared = await database.upsert_entity("payroll area", "Payroll Area")
    private = await database.upsert_entity("secret launch", "Secret Launch")
    mixed = await database.upsert_entity("cutoff calendar", "Cutoff Calendar")
    orphan = await database.upsert_entity("stale name", "Stale Name")
    await database.link_memory_entity("mem-shared", shared)
    await database.link_memory_entity("mem-private", private)
    await database.link_memory_entity("mem-shared", mixed)
    await database.link_memory_entity("mem-private", mixed)
    return _Graph(shared=shared, private=private, mixed=mixed, orphan=orphan)


@pytest.fixture
def seeded(tmp_path: Path):
    from memforge.server.admin_api import create_admin_app

    database = Database(str(tmp_path / "entities.db"))
    asyncio.run(database.connect())
    graph = asyncio.run(_seed(database))
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=_principal,
        workspace_role_resolver=_workspace_role,
    )
    try:
        with TestClient(app) as client:
            yield client, database, graph
    finally:
        asyncio.run(database.close())


def _listed_ids(client: TestClient, user: str, **params) -> set[int]:
    response = client.get("/api/v1/entities", params=params, headers=_as(user))
    assert response.status_code == 200, response.text
    payload = response.json()
    ids = {row["id"] for row in payload["data"]}
    assert payload["total"] == len(ids)
    return ids


def test_entity_list_admits_only_entities_linked_to_visible_memories(seeded) -> None:
    client, _database, graph = seeded

    assert _listed_ids(client, OWNER) == {graph.shared, graph.private, graph.mixed}
    assert _listed_ids(client, OTHER) == {graph.shared, graph.mixed}
    assert _listed_ids(client, OTHER, search="Secret") == set()


def test_entity_detail_and_aliases_hide_entities_without_visible_memories(seeded) -> None:
    client, _database, graph = seeded

    assert client.get(f"/api/v1/entities/{graph.private}", headers=_as(OWNER)).status_code == 200
    for entity_id in (graph.private, graph.orphan):
        assert client.get(f"/api/v1/entities/{entity_id}", headers=_as(OTHER)).status_code == 404
        assert client.get(f"/api/v1/entities/{entity_id}/aliases", headers=_as(OTHER)).status_code == 404


def test_linked_memory_count_excludes_memories_the_caller_cannot_query(seeded) -> None:
    client, _database, graph = seeded

    owner_view = client.get(f"/api/v1/entities/{graph.mixed}", headers=_as(OWNER)).json()
    other_view = client.get(f"/api/v1/entities/{graph.mixed}", headers=_as(OTHER)).json()

    assert owner_view["linked_memory_count"] == 2
    assert other_view["linked_memory_count"] == 1


def test_stats_entity_total_counts_only_visible_entities(seeded) -> None:
    client, _database, _graph = seeded

    assert client.get("/api/v1/stats", headers=_as(OWNER)).json()["total_entities"] == 3
    assert client.get("/api/v1/stats", headers=_as(OTHER)).json()["total_entities"] == 2


def test_stats_source_total_excludes_other_users_only_me_sources(seeded) -> None:
    client, _database, _graph = seeded
    created = client.post(
        "/api/v1/sources",
        json={
            "type": "confluence",
            "name": "Alice notes",
            "access_policy": "private",
            "config": {
                "base_url": "https://wiki.example.test/wiki/spaces/ARCH/pages/12345/Home",
                "pat": "test-token",
                "sync_mode": "page_tree",
                "page_tree_root": "12345",
                "include_children": True,
            },
        },
        headers=_as(OWNER),
    )
    assert created.status_code in {200, 201}, created.text

    assert client.get("/api/v1/stats", headers=_as(OWNER)).json()["total_sources"] == 1
    assert client.get("/api/v1/stats", headers=_as(OTHER)).json()["total_sources"] == 0


@pytest.mark.parametrize("role", ["member", "viewer"])
def test_entity_curation_requires_workspace_administration(seeded, role: str) -> None:
    client, database, graph = seeded
    headers = _as(OTHER, role)

    detail = client.get(f"/api/v1/entities/{graph.shared}", headers=headers)
    added = client.post(f"/api/v1/entities/{graph.shared}/aliases", json={"alias": "Pay Zone"}, headers=headers)
    removed = client.delete(f"/api/v1/entities/{graph.shared}/aliases/payroll area", headers=headers)
    merged = client.post(
        "/api/v1/entities/merge",
        json={"source_id": graph.mixed, "target_id": graph.shared},
        headers=headers,
    )

    assert detail.json()["can_curate"] is False
    for response in (added, removed, merged):
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["error"] == "entity_curation_forbidden"
    assert asyncio.run(database.get_aliases_for_entity(graph.shared)) == []
    assert asyncio.run(database.get_entity(graph.mixed)) is not None


@pytest.mark.parametrize("role", ["workspace_admin", "owner"])
def test_workspace_administrators_curate_visible_entities(seeded, role: str) -> None:
    client, database, graph = seeded
    headers = _as(OTHER, role)

    detail = client.get(f"/api/v1/entities/{graph.shared}", headers=headers)
    added = client.post(f"/api/v1/entities/{graph.shared}/aliases", json={"alias": "Pay Zone"}, headers=headers)
    removed = client.delete(f"/api/v1/entities/{graph.shared}/aliases/Pay Zone", headers=headers)
    merged = client.post(
        "/api/v1/entities/merge",
        json={"source_id": graph.mixed, "target_id": graph.shared},
        headers=headers,
    )

    assert detail.json()["can_curate"] is True
    assert added.status_code == 200, added.text
    assert removed.status_code == 200, removed.text
    assert merged.status_code == 200, merged.text
    assert asyncio.run(database.get_entity(graph.mixed)) is None


def test_curation_cannot_reach_entities_the_administrator_cannot_see(seeded) -> None:
    client, database, graph = seeded
    headers = _as(OTHER, "workspace_admin")

    added = client.post(f"/api/v1/entities/{graph.private}/aliases", json={"alias": "Launch"}, headers=headers)
    removed = client.delete(f"/api/v1/entities/{graph.private}/aliases/secret launch", headers=headers)
    merged_into_hidden = client.post(
        "/api/v1/entities/merge",
        json={"source_id": graph.shared, "target_id": graph.private},
        headers=headers,
    )
    merged_from_hidden = client.post(
        "/api/v1/entities/merge",
        json={"source_id": graph.orphan, "target_id": graph.shared},
        headers=headers,
    )

    assert added.status_code == 404
    assert removed.status_code == 404
    assert merged_into_hidden.status_code == 404
    assert merged_into_hidden.json()["error"] == "Target entity not found"
    assert merged_from_hidden.status_code == 404
    assert merged_from_hidden.json()["error"] == "Source entity not found"
    assert asyncio.run(database.get_aliases_for_entity(graph.private)) == []
    assert asyncio.run(database.get_entity(graph.shared)) is not None
    assert asyncio.run(database.get_entity(graph.orphan)) is not None
