"""HTTP coverage for the `/api/projects` CRUD surface.

The wire model exposes `kind: 'normal' | 'shared'` over the storage
`is_shared` column. Creating, changing, and deleting projects requires
workspace admin authority. Reserved keys (SHARED, UNSORTED) refuse to be
created, renamed, re-kinded, or deleted. A real project's delete
rebuckets its memories to UNSORTED across both the relational row and
the vector metadata before removing the row.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import re

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.lifecycle_plan import LifecycleGateState
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


_WORKSPACE_ROLE_HEADER = "x-test-workspace-role"


def _workspace_role(request: Request) -> str:
    return request.headers.get(_WORKSPACE_ROLE_HEADER, "member")


def _make_app(tmp_path: Path, **app_options):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = Database(str(tmp_path / "api.db"))

    async def _setup():
        await database.connect()

    asyncio.run(_setup())
    app = create_admin_app(db=database, config=cfg, **app_options)
    return app, database


def _make_role_app(tmp_path: Path):
    return _make_app(
        tmp_path,
        principal_resolver=lambda _request: "workspace-user",
        workspace_role_resolver=_workspace_role,
    )


def _as_role(role: str) -> dict[str, str]:
    return {_WORKSPACE_ROLE_HEADER: role}


def _projects_by_key(client: TestClient) -> dict[str, dict]:
    return {p["key"]: p for p in client.get("/api/v1/projects").json()["data"]}


def test_create_list_update_round_trip(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/v1/projects",
                json={"name": "Payroll", "kind": "normal"},
            )
            assert create.status_code == 201, create.text
            created = create.json()
            assert created["key"] == "PAYROLL"
            assert created["kind"] == "normal"
            assert created["name"] == "Payroll"

            listed = client.get("/api/v1/projects").json()["data"]
            keys = {p["key"] for p in listed}
            assert {SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY, "PAYROLL"} <= keys

            patch = client.patch(
                f"/api/v1/projects/{created['id']}",
                json={"name": "Pay", "kind": "shared"},
            )
            assert patch.status_code == 200, patch.text
            updated = patch.json()
            assert updated["name"] == "Pay"
            assert updated["kind"] == "shared"
    finally:
        asyncio.run(database.close())


def test_list_reports_the_memories_each_project_shows_the_caller(tmp_path):
    app, database = _make_role_app(tmp_path)

    def _memory(memory_id: str, project_key: str, **fields) -> Memory:
        return Memory(
            id=memory_id,
            memory_type="fact",
            content=memory_id,
            content_hash=content_hash(memory_id),
            project_key=project_key,
            **fields,
        )

    async def _seed():
        for memory in (
            _memory("m-pay-1", "PAY"),
            _memory("m-pay-2", "PAY"),
            _memory("m-pay-mine", "PAY", visibility=Visibility.PRIVATE.value, owner_user_id="workspace-user"),
            _memory("m-pay-theirs", "PAY", visibility=Visibility.PRIVATE.value, owner_user_id="someone-else"),
            _memory("m-pay-retired", "PAY", status="retired"),
            _memory("m-shared", SHARED_PROJECT_KEY),
        ):
            await database.insert_memory(memory)

    asyncio.run(_seed())

    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/v1/projects", json={"name": "Pay", "key": "PAY"}, headers=_as_role("workspace_admin")
            )
            assert create.status_code == 201, create.text
            create = client.post(
                "/api/v1/projects", json={"name": "Risk", "key": "RISK"}, headers=_as_role("workspace_admin")
            )
            assert create.status_code == 201, create.text

            counts = {key: project["memory_count"] for key, project in _projects_by_key(client).items()}
            assert counts == {"PAY": 3, "RISK": 0, SHARED_PROJECT_KEY: 1, UNSORTED_PROJECT_KEY: 0}
    finally:
        asyncio.run(database.close())


def test_create_with_explicit_key(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/v1/projects",
                json={"name": "Risk Engine", "key": "RISK"},
            )
            assert create.status_code == 201, create.text
            assert create.json()["key"] == "RISK"
    finally:
        asyncio.run(database.close())


def test_project_created_at_is_timezone_qualified(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/v1/projects",
                json={"name": "Risk Engine", "key": "RISK"},
            )
            assert create.status_code == 201, create.text
            created_at = create.json()["created_at"]
            assert re.search(r"(Z|[+-]\d{2}:\d{2})$", created_at), created_at
    finally:
        asyncio.run(database.close())


def test_duplicate_key_returns_conflict(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            first = client.post("/api/v1/projects", json={"name": "Pay"})
            assert first.status_code == 201
            second = client.post("/api/v1/projects", json={"name": "Pay"})
            assert second.status_code == 409
            assert second.json()["detail"] == "A project with the code PAY already exists. Pick another name or code."
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("reserved", [SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY])
def test_reserved_projects_refuse_rename_kind_change_and_delete(tmp_path, reserved):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            before = _projects_by_key(client)[reserved]
            attempts = (
                client.patch(f"/api/v1/projects/{before['id']}", json={"name": "Renamed"}),
                client.patch(f"/api/v1/projects/{before['id']}", json={"kind": "normal"}),
                client.patch(f"/api/v1/projects/{before['id']}", json={"kind": "shared"}),
                client.delete(f"/api/v1/projects/{before['id']}"),
            )
            for resp in attempts:
                assert resp.status_code == 409, resp.text
                detail = resp.json()["detail"]
                assert detail["error"] == "reserved_project"
                assert reserved in detail["message"]
            assert _projects_by_key(client)[reserved] == before
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Team", "key": SHARED_PROJECT_KEY},
        {"name": "Backlog", "key": UNSORTED_PROJECT_KEY.lower()},
        {"name": "Unsorted"},
    ],
)
def test_create_refuses_reserved_keys(tmp_path, payload):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            before = _projects_by_key(client)
            resp = client.post("/api/v1/projects", json=payload)
            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"]["error"] == "reserved_project"
            assert _projects_by_key(client) == before
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("role", ["viewer", "member"])
def test_non_admin_roles_cannot_change_projects(tmp_path, role):
    app, database = _make_role_app(tmp_path)
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/v1/projects",
                json={"name": "Pay", "key": "PAY"},
                headers=_as_role("workspace_admin"),
            )
            assert created.status_code == 201, created.text
            project_id = created.json()["id"]
            shared_id = _projects_by_key(client)[SHARED_PROJECT_KEY]["id"]
            before = _projects_by_key(client)

            attempts = (
                client.post("/api/v1/projects", json={"name": "Risk"}, headers=_as_role(role)),
                client.patch(
                    f"/api/v1/projects/{project_id}",
                    json={"name": "Renamed"},
                    headers=_as_role(role),
                ),
                client.patch(
                    f"/api/v1/projects/{shared_id}",
                    json={"kind": "normal"},
                    headers=_as_role(role),
                ),
                client.delete(f"/api/v1/projects/{project_id}", headers=_as_role(role)),
            )
            for resp in attempts:
                assert resp.status_code == 403, resp.text
                assert resp.json()["detail"]["error"] == "project_management_forbidden"

            listed = client.get("/api/v1/projects", headers=_as_role(role))
            assert listed.status_code == 200
            assert {p["key"]: p for p in listed.json()["data"]} == before
    finally:
        asyncio.run(database.close())


def test_workspace_admin_can_create_change_and_delete_projects(tmp_path):
    app, database = _make_role_app(tmp_path)
    admin = _as_role("workspace_admin")
    try:
        with TestClient(app) as client:
            created = client.post("/api/v1/projects", json={"name": "Pay"}, headers=admin)
            assert created.status_code == 201, created.text
            project_id = created.json()["id"]

            renamed = client.patch(
                f"/api/v1/projects/{project_id}",
                json={"name": "Payroll"},
                headers=admin,
            )
            assert renamed.status_code == 200, renamed.text
            assert renamed.json()["name"] == "Payroll"

            deleted = client.delete(f"/api/v1/projects/{project_id}", headers=admin)
            assert deleted.status_code == 200, deleted.text
            assert "PAY" not in _projects_by_key(client)
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("role", ["owner", "workspace_admin", "member", "viewer"])
def test_list_reports_the_project_authority_the_write_routes_enforce(tmp_path, role):
    app, database = _make_role_app(tmp_path)
    try:
        with TestClient(app) as client:
            listed = client.get("/api/v1/projects", headers=_as_role(role))
            assert listed.status_code == 200, listed.text

            created = client.post("/api/v1/projects", json={"name": "Pay"}, headers=_as_role(role))
            assert created.status_code in {201, 403}, created.text

            assert listed.json()["can_manage"] is (created.status_code == 201)
    finally:
        asyncio.run(database.close())


async def _seed_bound_sources(database: Database) -> None:
    bindings = {
        "src-fixed-pay": {"mode": "fixed", "project_key": "PAY"},
        "src-fixed-risk": {"mode": "fixed", "project_key": "RISK"},
        "src-field-pay": {
            "mode": "by_field",
            "field": "repo",
            "map": {"payroll": "PAY", "risk-engine": "RISK"},
            "default": "PAY",
        },
        "src-field-risk": {"mode": "by_field", "field": "repo", "map": {"risk-engine": "RISK"}, "default": "UNSORTED"},
        "src-retired-pay": {"mode": "fixed", "project_key": "PAY"},
        "src-unbound": None,
    }
    for source_id, binding in bindings.items():
        await database.upsert_source(
            id=source_id,
            type="local_markdown",
            name=source_id,
            config_json="{}",
            access_policy="private",
            # Another user's private Source is released like any other.
            owner_user_id="someone-else" if source_id == "src-field-pay" else "workspace-user",
            # A retired Source writes no memories, so deletion leaves its binding as is.
            status="retired" if source_id == "src-retired-pay" else None,
            project_binding=binding,
        )


def test_delete_releases_every_source_binding_that_names_the_project(tmp_path):
    app, database = _make_role_app(tmp_path)
    admin = _as_role("workspace_admin")

    async def _seed():
        await _seed_bound_sources(database)
        await database.insert_memory(
            Memory(
                id="m-pay-retired",
                memory_type="fact",
                content="retired payroll fact",
                content_hash=content_hash("retired payroll fact"),
                project_key="PAY",
                status="retired",
            )
        )

    asyncio.run(_seed())

    try:
        with TestClient(app) as client:
            project_id = client.post("/api/v1/projects", json={"name": "Pay", "key": "PAY"}, headers=admin).json()["id"]

            impact = client.get(f"/api/v1/projects/{project_id}/deletion-impact", headers=admin)
            assert impact.status_code == 200, impact.text
            assert impact.json() == {"memory_count": 1, "source_count": 2}

            deleted = client.delete(f"/api/v1/projects/{project_id}", headers=admin)
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["released_source_count"] == 2
            assert deleted.json()["rebucketed_memory_ids"] == ["m-pay-retired"]

        async def _bindings():
            return {source["id"]: source["project_binding"] for source in await database.list_sources()}

        assert asyncio.run(_bindings()) == {
            "src-fixed-pay": None,
            "src-fixed-risk": {"mode": "fixed", "project_key": "RISK"},
            "src-field-pay": {
                "mode": "by_field",
                "field": "repo",
                "map": {"risk-engine": "RISK"},
                "default": UNSORTED_PROJECT_KEY,
            },
            "src-field-risk": {
                "mode": "by_field",
                "field": "repo",
                "map": {"risk-engine": "RISK"},
                "default": "UNSORTED",
            },
            "src-unbound": None,
        }
        retired_source = asyncio.run(database.get_source("src-retired-pay"))
        assert retired_source is not None
        assert retired_source["project_binding"] == {"mode": "fixed", "project_key": "PAY"}
        stored = asyncio.run(database.get_memory("m-pay-retired"))
        assert stored is not None
        assert stored.project_key == UNSORTED_PROJECT_KEY
        assert stored.status == "retired"
    finally:
        asyncio.run(database.close())


def test_a_failed_relational_commit_keeps_the_sources_bound(tmp_path):
    app, database = _make_role_app(tmp_path)
    admin = _as_role("workspace_admin")
    asyncio.run(_seed_bound_sources(database))

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            project_id = client.post("/api/v1/projects", json={"name": "Pay", "key": "PAY"}, headers=admin).json()["id"]
            original_execute = database.db.execute

            def _refuse_project_drop(sql, *args, **kwargs):
                if sql.startswith("DELETE FROM projects"):
                    raise RuntimeError("database offline")
                return original_execute(sql, *args, **kwargs)

            database.db.execute = _refuse_project_drop  # type: ignore[method-assign]
            try:
                assert client.delete(f"/api/v1/projects/{project_id}", headers=admin).status_code >= 500
            finally:
                database.db.execute = original_execute  # type: ignore[method-assign]

            assert "PAY" in _projects_by_key(client)

        source = asyncio.run(database.get_source("src-fixed-pay"))
        assert source is not None
        assert source["project_binding"] == {"mode": "fixed", "project_key": "PAY"}
    finally:
        asyncio.run(database.close())


def test_delete_leaves_a_binding_that_is_not_a_json_object_as_stored(tmp_path):
    app, database = _make_role_app(tmp_path)
    admin = _as_role("workspace_admin")
    unreadable = {"src-truncated": '{"mode": "fixed", "project_key": "PA', "src-list": '["PAY"]'}

    async def _seed():
        await _seed_bound_sources(database)
        for source_id, stored in unreadable.items():
            await database.upsert_source(
                id=source_id,
                type="local_markdown",
                name=source_id,
                config_json="{}",
                access_policy="private",
                owner_user_id="workspace-user",
            )
            await database.db.execute("UPDATE sources SET project_binding = ? WHERE id = ?", (stored, source_id))
        await database.db.commit()

    async def _stored_bindings() -> dict[str, str | None]:
        async with database.db.execute("SELECT id, project_binding FROM sources") as cursor:
            return {row["id"]: row["project_binding"] for row in await cursor.fetchall()}

    asyncio.run(_seed())

    try:
        with TestClient(app) as client:
            project_id = client.post("/api/v1/projects", json={"name": "Pay", "key": "PAY"}, headers=admin).json()["id"]

            impact = client.get(f"/api/v1/projects/{project_id}/deletion-impact", headers=admin)
            deleted = client.delete(f"/api/v1/projects/{project_id}", headers=admin)

            assert impact.status_code == 200, impact.text
            assert impact.json()["source_count"] == 2
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["released_source_count"] == 2
            assert "PAY" not in _projects_by_key(client)

        stored = asyncio.run(_stored_bindings())
        assert {source_id: stored[source_id] for source_id in unreadable} == unreadable
        assert stored["src-fixed-pay"] is None
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("role", ["member", "viewer"])
def test_deletion_impact_needs_project_authority(tmp_path, role):
    app, database = _make_role_app(tmp_path)
    try:
        with TestClient(app) as client:
            project_id = client.post(
                "/api/v1/projects", json={"name": "Pay"}, headers=_as_role("workspace_admin")
            ).json()["id"]
            resp = client.get(f"/api/v1/projects/{project_id}/deletion-impact", headers=_as_role(role))
            assert resp.status_code == 403, resp.text
            assert resp.json()["detail"]["error"] == "project_management_forbidden"
    finally:
        asyncio.run(database.close())


def test_deletion_impact_refuses_built_in_projects(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            shared_id = _projects_by_key(client)[SHARED_PROJECT_KEY]["id"]
            resp = client.get(f"/api/v1/projects/{shared_id}/deletion-impact")
            assert resp.status_code == 409, resp.text
            assert resp.json()["detail"]["error"] == "reserved_project"
    finally:
        asyncio.run(database.close())


def test_delete_real_project_rebuckets_to_unsorted(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        # Seed a memory under PAY so the delete has something to rebucket.
        async def _seed():
            await database.insert_memory(
                Memory(
                    id="m-pay",
                    memory_type="fact",
                    content="payroll fact",
                    content_hash=content_hash("payroll fact"),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key="PAY",
                )
            )

        asyncio.run(_seed())

        with TestClient(app) as client:
            create = client.post(
                "/api/v1/projects",
                json={"name": "Pay", "key": "PAY"},
            )
            assert create.status_code == 201, create.text
            project_id = create.json()["id"]

            delete = client.delete(f"/api/v1/projects/{project_id}")
            assert delete.status_code == 200, delete.text
            body = delete.json()
            assert body["id"] == project_id
            assert body["rebucketed_count"] == 1
            assert body["rebucketed_memory_ids"] == ["m-pay"]

            # The project row is gone.
            assert client.get("/api/v1/projects").status_code == 200
            keys = set(_projects_by_key(client))
            assert "PAY" not in keys

        async def _verify_rebucket():
            stored = await database.get_memory("m-pay")
            assert stored is not None
            assert stored.project_key == UNSORTED_PROJECT_KEY

        asyncio.run(_verify_rebucket())
    finally:
        asyncio.run(database.close())


def test_delete_unknown_project_returns_404(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            resp = client.delete("/api/v1/projects/proj-does-not-exist")
            assert resp.status_code == 404
    finally:
        asyncio.run(database.close())


def test_derive_project_key_collapses_punctuation():
    from memforge.server.admin_api import _derive_project_key

    assert _derive_project_key("Pay & Risk") == "PAY_RISK"
    assert _derive_project_key("   ") == "PROJECT"
    assert _derive_project_key("a" * 100) == "A" * 32


def _local_markdown_source_payload(
    *,
    name: str = "Engineering Notes",
    root: str = "/tmp/memforge-engineering-notes",
    project_binding: dict | None = None,
) -> dict:
    payload = {
        "type": "local_markdown",
        "name": name,
        "access_policy": "private",
        "config": {
            "root": root,
            "vault_id": "engineering-notes",
        },
    }
    if project_binding is not None:
        payload["project_binding"] = project_binding
    return payload


def test_create_source_round_trips_project_binding(tmp_path):
    """A binding sent on POST /api/sources is persisted and surfaces on GET."""
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            binding = {
                "mode": "by_field",
                "field": "repo",
                "map": {"my-app": "APP"},
                "default": UNSORTED_PROJECT_KEY,
            }
            resp = client.post(
                "/api/v1/sources",
                json=_local_markdown_source_payload(project_binding=binding),
            )
            assert resp.status_code == 200, resp.text
            source_id = resp.json()["id"]

        async def _read():
            return (
                await database.get_source(source_id),
                await database.get_lifecycle_gate(source_id),
            )

        stored, gate = asyncio.run(_read())
        assert stored is not None
        assert stored["project_binding"] == binding
        assert gate.state is LifecycleGateState.ENABLED
    finally:
        asyncio.run(database.close())


def test_create_source_without_project_binding_lands_in_unmapped(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/sources",
                json=_local_markdown_source_payload(),
            )
            assert resp.status_code == 200, resp.text
            source_id = resp.json()["id"]

        async def _read():
            return await database.get_source(source_id)

        stored = asyncio.run(_read())
        assert stored is not None
        assert stored["project_binding"] is None
    finally:
        asyncio.run(database.close())


def test_update_source_replaces_and_preserves_project_binding(tmp_path):
    """PUT /api/sources/{id} accepts a new binding; PUT without one preserves it."""
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            initial = {"mode": "fixed", "project_key": "PAY"}
            create = client.post(
                "/api/v1/sources",
                json=_local_markdown_source_payload(project_binding=initial),
            )
            source_id = create.json()["id"]

            replacement = {"mode": "fixed", "project_key": "RISK"}
            put = client.put(
                f"/api/v1/sources/{source_id}",
                json={"project_binding": replacement},
            )
            assert put.status_code == 200, put.text

        async def _read():
            return await database.get_source(source_id)

        after_replace = asyncio.run(_read())
        assert after_replace["project_binding"] == replacement

        with TestClient(app) as client:
            # A PUT that does not mention the binding must not clear it.
            put_noop = client.put(
                f"/api/v1/sources/{source_id}",
                json={"name": "renamed"},
            )
            assert put_noop.status_code == 200, put_noop.text

        after_noop = asyncio.run(_read())
        assert after_noop["project_binding"] == replacement
    finally:
        asyncio.run(database.close())


def test_update_source_can_clear_project_binding_to_unmapped(tmp_path):
    app, database = _make_app(tmp_path)
    try:
        with TestClient(app) as client:
            create = client.post(
                "/api/v1/sources",
                json=_local_markdown_source_payload(project_binding={"mode": "fixed", "project_key": "PAY"}),
            )
            source_id = create.json()["id"]

            resp = client.put(
                f"/api/v1/sources/{source_id}",
                json={"project_binding": None},
            )
            assert resp.status_code == 200, resp.text

        async def _read():
            return await database.get_source(source_id)

        stored = asyncio.run(_read())
        assert stored["project_binding"] is None
    finally:
        asyncio.run(database.close())


def test_rebucket_partial_vector_failure_rolls_back_already_applied():
    """If a per-id vector upsert fails partway through the batch, every
    record this call already moved must be restored to its original
    metadata. The relational rebucket has not run yet, so a clean
    rollback returns the system to exactly the pre-call state."""
    from memforge.memory.store import MemoryStore

    captured_upserts: list[tuple[str, str]] = []

    class _FlakyVector:
        def __init__(self) -> None:
            self.records: dict[str, dict] = {
                "m1": {
                    "id": "m1",
                    "embedding": [0.1, 0.2],
                    "metadata": {"project_key": "PAY", "visibility": "workspace"},
                },
                "m2": {
                    "id": "m2",
                    "embedding": [0.3, 0.4],
                    "metadata": {"project_key": "PAY", "visibility": "workspace"},
                },
            }
            self._calls = 0

        async def get_record(self, memory_id: str):
            return self.records.get(memory_id)

        async def upsert(self, *, ids, embeddings, metadatas):
            assert len(ids) == 1
            self._calls += 1
            captured_upserts.append((ids[0], metadatas[0]["project_key"]))
            if ids[0] == "m2" and self._calls == 2:
                raise RuntimeError("vector store failed mid-batch")
            self.records[ids[0]] = {
                "id": ids[0],
                "embedding": embeddings[0],
                "metadata": dict(metadatas[0]),
            }

    store = MemoryStore.__new__(MemoryStore)
    store.vector = _FlakyVector()  # type: ignore[attr-defined]

    with __import__("pytest").raises(RuntimeError, match="vector store failed mid-batch"):
        asyncio.run(store.rebucket_project_memories(["m1", "m2"], UNSORTED_PROJECT_KEY))

    # m1 was upserted to UNSORTED then rolled back to PAY; m2 never moved.
    assert store.vector.records["m1"]["metadata"]["project_key"] == "PAY"  # type: ignore[attr-defined]
    assert store.vector.records["m2"]["metadata"]["project_key"] == "PAY"  # type: ignore[attr-defined]
    # The captured sequence proves rollback ran: forward(m1) -> forward(m2 fails) -> rollback(m1).
    assert captured_upserts == [
        ("m1", UNSORTED_PROJECT_KEY),
        ("m2", UNSORTED_PROJECT_KEY),
        ("m1", "PAY"),
    ]


def test_delete_orders_vector_before_relational_commit(tmp_path):
    """If the vector channel raises during rebucket, the project row and
    the relational rebucket must NOT be applied. SQLite and Chroma both
    keep pointing at the original project until the operation can be
    re-run."""
    app, database = _make_app(tmp_path)

    async def _seed():
        await database.insert_memory(
            Memory(
                id="m-fail",
                memory_type="fact",
                content="will not move",
                content_hash=content_hash("will not move"),
                visibility=Visibility.WORKSPACE.value,
                owner_user_id=None,
                project_key="PAY",
            )
        )

    asyncio.run(_seed())

    try:
        # raise_server_exceptions=False mirrors a real client: the
        # uncaught error becomes a 500 instead of bubbling to pytest.
        with TestClient(app, raise_server_exceptions=False) as client:
            create = client.post(
                "/api/v1/projects",
                json={"name": "Pay", "key": "PAY"},
            )
            project_id = create.json()["id"]

            from memforge.memory.store import MemoryStore

            original = MemoryStore.rebucket_project_memories

            async def _explode(self, *args, **kwargs):
                raise RuntimeError("vector store offline")

            MemoryStore.rebucket_project_memories = _explode  # type: ignore[assignment]
            try:
                resp = client.delete(f"/api/v1/projects/{project_id}")
                assert resp.status_code >= 500
            finally:
                MemoryStore.rebucket_project_memories = original  # type: ignore[assignment]

            # The project row still exists; the memory still points to PAY.
            keys = set(_projects_by_key(client))
            assert "PAY" in keys

        async def _verify_unmoved():
            stored = await database.get_memory("m-fail")
            assert stored is not None
            assert stored.project_key == "PAY"

        asyncio.run(_verify_unmoved())
    finally:
        asyncio.run(database.close())


def test_resolved_projects_endpoint_groups_memories_by_resolved_key(tmp_path):
    """GET /api/sources/{id}/projects/resolved reports the resolver's
    verdict on memories from this source, distinct from the raw
    `documents.space_or_project` view served by /projects."""
    from datetime import datetime, timezone

    from memforge.models import DocumentRecord

    app, database = _make_app(tmp_path)

    async def _seed():
        await database.upsert_source(
            id="src-doc",
            type="agent_session",
            name="codex sessions",
            config_json="{}",
            access_policy="private",
            owner_user_id="dev",
        )
        ts = datetime.now(tz=timezone.utc)
        for doc_id in ("doc-1", "doc-2"):
            await database.upsert_document(
                DocumentRecord(
                    doc_id=doc_id,
                    source="src-doc",
                    source_url="",
                    title=doc_id,
                    space_or_project="ignored-raw-value",
                    author=None,
                    last_modified=ts,
                    labels=[],
                    version="1",
                    content_hash=f"h-{doc_id}",
                    token_count=None,
                    last_synced=ts,
                )
            )
        for mid, doc, key in (
            ("m-pay", "doc-1", "PAY"),
            ("m-other", "doc-2", "RISK"),
        ):
            await database.insert_memory(
                Memory(
                    id=mid,
                    memory_type="fact",
                    content=mid,
                    content_hash=content_hash(mid),
                    visibility=Visibility.WORKSPACE.value,
                    owner_user_id=None,
                    project_key=key,
                )
            )
            await database.add_memory_source(
                memory_id=mid,
                doc_id=doc,
                source_type="agent_session",
                excerpt=None,
                source_updated_at=None,
            )

    asyncio.run(_seed())

    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/sources/src-doc/projects/resolved")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["source_id"] == "src-doc"
            by_key = {p["project_key"]: p["memory_count"] for p in body["projects"]}
            assert by_key == {"PAY": 1, "RISK": 1}
    finally:
        asyncio.run(database.close())
