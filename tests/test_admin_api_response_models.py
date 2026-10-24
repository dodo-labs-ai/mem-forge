"""The routes the admin UI calls answer bodies that validate against their declared response models.

Each test drives a route at the boundary on SQLite and validates the body
against the model the OpenAPI document publishes for it, so the generated
admin UI types describe what the route returns.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, TypeAdapter

from memforge.server.admin_api import (
    EntityAliasAddedResponse,
    EntityAliasListResponse,
    EntityMergeResponse,
    LocalCollectionReceiptResponse,
    OkResponse,
    SourceAccessTransitionOperationResponse,
    SourceAccessTransitionQueuedResponse,
    SourceAccessTransitionResponse,
    SourceCreatedResponse,
    SourceCurrentAccessTransitionResponse,
    SourcePinResponse,
    SourceSubscriptionChangeResponse,
    SourceSyncRunReceiptResponse,
    SourceSyncRunStateResponse,
    SourceUpdatedResponse,
)
from tests.test_source_authority_api import _app, _confluence_payload, _connect_database, _teams_payload
from tests.test_startup_runtime import TEST_SOURCE_KEY

OWNER = {"x-test-user": "owner-user", "x-test-workspace-role": "member"}
ADMIN = {"x-test-user": "admin-user", "x-test-workspace-role": "workspace_admin"}

# (method, path, success status, models the document names for that status)
DECLARED = [
    ("post", "/api/v1/sources", "200", {"SourceCreatedResponse"}),
    ("put", "/api/v1/sources/{source_id}", "200", {"SourceUpdatedResponse"}),
    ("put", "/api/v1/sources/{source_id}/pin", "200", {"SourcePinResponse"}),
    ("delete", "/api/v1/sources/{source_id}/pin", "200", {"SourcePinResponse"}),
    ("put", "/api/v1/sources/{source_id}/subscription", "200", {"SourceSubscriptionChangeResponse"}),
    ("post", "/api/v1/sources/{source_id}/access-transitions", "202", {"SourceAccessTransitionResponse"}),
    ("get", "/api/v1/sources/{source_id}/access-transition", "200", {"SourceCurrentAccessTransitionResponse"}),
    (
        "get",
        "/api/v1/sources/{source_id}/access-transitions/{operation_id}",
        "200",
        {"SourceAccessTransitionOperationResponse"},
    ),
    (
        "post",
        "/api/v1/sources/{source_id}/access-transitions/{operation_id}/retry",
        "200",
        {"SourceAccessTransitionQueuedResponse"},
    ),
    (
        "post",
        "/api/v1/sources/{source_id}/access-transitions/{operation_id}/revert",
        "200",
        {"SourceAccessTransitionQueuedResponse"},
    ),
    ("get", "/api/v1/sources/{source_id}/sync-runs/{run_id}", "200", {"SourceSyncRunStateResponse"}),
    (
        "post",
        "/api/v1/sources/{source_id}/force-resync",
        "202",
        {"SourceSyncRunReceiptResponse", "LocalCollectionReceiptResponse"},
    ),
    ("post", "/api/v1/entities/merge", "200", {"EntityMergeResponse"}),
    ("get", "/api/v1/entities/{entity_id}/aliases", "200", {"EntityAliasListResponse"}),
    ("post", "/api/v1/entities/{entity_id}/aliases", "200", {"EntityAliasAddedResponse"}),
    ("delete", "/api/v1/entities/{entity_id}/aliases/{alias}", "200", {"OkResponse"}),
]


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)


@pytest.fixture
def database(tmp_path):
    db = _connect_database(tmp_path)
    yield db
    asyncio.run(db.close())


@pytest.fixture
def client(tmp_path, database):
    with TestClient(_app(tmp_path, database)) as test_client:
        yield test_client


def _valid(model: type[BaseModel], response, status: int = 200) -> BaseModel:
    assert response.status_code == status, response.text
    return model.model_validate(response.json(), strict=False)


def _create(client: TestClient, payload: dict | None = None) -> str:
    created = client.post("/api/v1/sources", json=payload or _confluence_payload(access_policy="private"), headers=OWNER)
    return _valid(SourceCreatedResponse, created).id


def _schema_names(schema: dict) -> set[str]:
    if "$ref" in schema:
        return {schema["$ref"].rsplit("/", 1)[-1]}
    return {name for option in schema.get("anyOf", []) for name in _schema_names(option)}


@pytest.mark.parametrize(("method", "path", "status", "models"), DECLARED, ids=[f"{m} {p}" for m, p, _, _ in DECLARED])
def test_the_document_names_the_response_model(client, method, path, status, models):
    operation = client.get("/openapi.json").json()["paths"][path][method]

    schema = operation["responses"][status]["content"]["application/json"]["schema"]

    assert _schema_names(schema) == models


def test_source_create_and_update_answer_their_models(client):
    source_id = _create(client)

    updated = client.put(f"/api/v1/sources/{source_id}", json={"name": "Renamed Wiki"}, headers=OWNER)

    assert _valid(SourceUpdatedResponse, updated).id == source_id


def test_pin_and_subscription_answer_their_models(client):
    source_id = _create(client)

    pinned = client.put(f"/api/v1/sources/{source_id}/pin", headers=OWNER)
    unpinned = client.delete(f"/api/v1/sources/{source_id}/pin", headers=OWNER)
    subscribed = client.put(f"/api/v1/sources/{source_id}/subscription", json={"enabled": False}, headers=OWNER)

    assert _valid(SourcePinResponse, pinned).pinned is True
    assert _valid(SourcePinResponse, unpinned).pinned is False
    assert _valid(SourceSubscriptionChangeResponse, subscribed).subscription.enabled is False


def test_access_transition_routes_answer_their_models(client):
    source_id = _create(client)

    started = client.post(
        f"/api/v1/sources/{source_id}/access-transitions",
        json={"target_policy": "workspace"},
        headers={**OWNER, "Idempotency-Key": "share-once"},
    )
    operation_id = _valid(SourceAccessTransitionResponse, started, 202).operation_id
    operation = client.get(f"/api/v1/sources/{source_id}/access-transitions/{operation_id}", headers=OWNER)
    current = client.get(f"/api/v1/sources/{source_id}/access-transition", headers=OWNER)

    assert _valid(SourceAccessTransitionOperationResponse, operation).transition.status == "completed"
    assert _valid(SourceCurrentAccessTransitionResponse, current).transition is None


@pytest.mark.parametrize("action", ["retry", "revert"])
def test_a_failed_access_transition_reads_and_requeues_through_its_models(client, database, action):
    source_id = _create(client)
    transition = asyncio.run(
        database.create_source_access_transition(
            operation_id=f"sat-{action}",
            source_id=source_id,
            idempotency_key=action,
            actor_user_id="owner-user",
            target_policy="workspace",
        )
    )
    asyncio.run(
        database.mark_source_access_transition_failed(
            transition["operation_id"],
            error_code="source_access_reconciliation_failed",
            error_message="RuntimeError: vector store unavailable",
        )
    )

    current = client.get(f"/api/v1/sources/{source_id}/access-transition", headers=OWNER)
    queued = client.post(f"/api/v1/sources/{source_id}/access-transitions/sat-{action}/{action}", headers=OWNER)

    failed = _valid(SourceCurrentAccessTransitionResponse, current).transition
    assert failed is not None and failed.status == "failed"
    assert failed.error_code == "source_access_reconciliation_failed"
    assert _valid(SourceAccessTransitionQueuedResponse, queued).operation_id == f"sat-{action}"


def test_sync_run_read_answers_its_model(client, database):
    source_id = _create(client)
    run = asyncio.run(database.enqueue_source_sync_run(source_id=source_id))

    read = client.get(f"/api/v1/sources/{source_id}/sync-runs/{run.run_id}", headers=OWNER)

    assert _valid(SourceSyncRunStateResponse, read).status == "pending"


def test_force_resync_answers_a_server_receipt_or_a_local_collection_receipt(client):
    server_source = _create(client)
    local_source = _create(client, _teams_payload())
    receipt = TypeAdapter(SourceSyncRunReceiptResponse | LocalCollectionReceiptResponse)

    server = client.post(f"/api/v1/sources/{server_source}/force-resync", headers=OWNER)
    local = client.post(f"/api/v1/sources/{local_source}/force-resync", headers=OWNER)

    assert server.status_code == 202, server.text
    assert local.status_code == 202, local.text
    assert isinstance(receipt.validate_python(server.json()), SourceSyncRunReceiptResponse)
    assert isinstance(receipt.validate_python(local.json()), LocalCollectionReceiptResponse)


def test_entity_merge_and_alias_routes_answer_their_models(client, database):
    async def seed() -> tuple[int, int]:
        from tests.test_admin_entity_access import _memory

        await database.insert_memory(_memory("mem-shared"))
        kept = await database.upsert_entity("payroll area", "Payroll Area")
        merged = await database.upsert_entity("pay area", "Pay Area")
        await database.link_memory_entity("mem-shared", kept)
        await database.link_memory_entity("mem-shared", merged)
        return kept, merged

    kept, merged = asyncio.run(seed())

    added = client.post(f"/api/v1/entities/{kept}/aliases", json={"alias": "Pay Zone"}, headers=ADMIN)
    listed = client.get(f"/api/v1/entities/{kept}/aliases", headers=ADMIN)
    removed = client.delete(f"/api/v1/entities/{kept}/aliases/Pay Zone", headers=ADMIN)
    joined = client.post("/api/v1/entities/merge", json={"source_id": merged, "target_id": kept}, headers=ADMIN)

    assert _valid(EntityAliasAddedResponse, added).alias_normalized == "pay zone"
    assert [alias.alias for alias in _valid(EntityAliasListResponse, listed).data] == ["Pay Zone"]
    assert _valid(OkResponse, removed).ok is True
    assert _valid(EntityMergeResponse, joined).merged.target_name == "payroll area"
