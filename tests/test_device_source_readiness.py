"""The readiness a device Source exposes in the source list (design §10.7).

``execution.kind`` stays ``local_agent`` for the device executor, and
``execution.device`` names the execution owner's device that runs the Source,
its state and versions. A device Source that no device report names needs its
credential stored: its connection is ``action_required`` for
``authentication``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import memforge.server.admin_api as admin_api
from memforge.config import AppConfig
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.local_agent.readiness import LOCAL_AGENT_DEVICE_STALE_SECONDS
from memforge.local_agent.source_contract import DOCUMENT_ACQUISITION_OPERATION
from memforge.release import installed_version
from memforge.storage.database import Database
from tests.device_report_fixture import device_report

PRINCIPAL_HEADER = "X-Test-Principal"
CREATOR = "owner-a"
EXECUTION_OWNER = "member-b"
TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
CONFLUENCE = {
    "base_url": "https://wiki.example.com",
    "sync_mode": "space",
    "spaces": ["ACME"],
    "pat": "server-held-token",
}


def _principal(request: Request) -> str:
    return request.headers.get(PRINCIPAL_HEADER, CREATOR)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)


@pytest.fixture
def database(tmp_path):
    database = Database(str(tmp_path / "sources.db"))
    asyncio.run(database.connect())
    try:
        yield database
    finally:
        asyncio.run(database.close())


@pytest.fixture
def client(tmp_path, database):
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    app = admin_api.create_admin_app(
        db=database,
        config=config,
        principal_resolver=_principal,
        workspace_role_resolver=lambda request: "workspace_admin",
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def device_source(client) -> str:
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Team wiki", "config": CONFLUENCE, "access_policy": "workspace"},
    )
    assert created.status_code == 200, created.text
    source_id = created.json()["id"]
    moved = client.put(
        f"/api/v1/sources/{source_id}/executor",
        json={"executor": "device"},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert moved.status_code == 200, moved.text
    return source_id


def _report(database: Database, report, *, seen: datetime | None = None) -> None:
    asyncio.run(
        database.lease_local_agent_jobs(
            user_id=EXECUTION_OWNER,
            device=report,
            acquisition_contracts=declared_acquisition_contracts(),
            limit=1,
            lease_seconds=60,
            now=seen or datetime.now(timezone.utc),
        )
    )


def _row(client, source_id: str, *, viewer: str = EXECUTION_OWNER) -> dict:
    listed = client.get("/api/v1/sources", headers={PRINCIPAL_HEADER: viewer})
    assert listed.status_code == 200, listed.text
    [row] = [row for row in listed.json()["data"] if row["id"] == source_id]
    return row


def test_a_server_source_has_no_device_and_no_device_connection(client):
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Team wiki", "config": CONFLUENCE, "access_policy": "workspace"},
    )
    row = _row(client, created.json()["id"])

    assert row["execution"] == {"kind": "server", "operation": None, "immutable_config_fields": []}
    assert "connection_status" not in row


def test_a_device_source_named_by_its_online_device_is_ready(client, database, device_source):
    _report(database, device_report([device_source], name="Alex's MacBook", version=installed_version()))

    row = _row(client, device_source)

    assert row["execution"] == {
        "kind": "local_agent",
        "operation": DOCUMENT_ACQUISITION_OPERATION,
        "immutable_config_fields": [],
        "device": {
            "name": "Alex's MacBook",
            "state": "online",
            "installed_version": installed_version(),
            "required_version": installed_version(),
            "credential": "secret",
        },
    }
    assert row["connection_status"] == {"state": "ready", "reason": None}


def test_a_device_source_no_report_names_needs_its_credential(client, database, device_source):
    _report(database, device_report([], name="Alex's MacBook"))

    row = _row(client, device_source)

    assert row["execution"]["device"]["state"] == "online"
    assert row["execution"]["device"]["name"] == "Alex's MacBook"
    assert row["connection_status"] == {"state": "action_required", "reason": "authentication"}


def test_without_any_device_the_source_is_offline_and_needs_its_credential(client, device_source):
    row = _row(client, device_source)

    assert row["execution"]["device"] == {
        "name": None,
        "state": "offline",
        "installed_version": None,
        "required_version": installed_version(),
        "credential": "secret",
    }
    assert row["connection_status"] == {"state": "action_required", "reason": "authentication"}


@pytest.mark.parametrize("difference", [-1, 1], ids=["behind", "ahead"])
def test_a_device_whose_contract_for_the_type_differs_needs_an_update(client, database, device_source, difference):
    contracts = declared_acquisition_contracts()
    contracts["confluence"] += difference
    _report(database, device_report([device_source], version="0.2.0b8", acquisition_contracts=contracts))

    row = _row(client, device_source)

    assert row["execution"]["device"] == {
        "name": "Laptop",
        "state": "update_required",
        "installed_version": "0.2.0b8",
        "required_version": installed_version(),
        "credential": "secret",
    }
    assert row["connection_status"] == {"state": "ready", "reason": None}


def test_a_device_without_a_recent_report_is_offline(client, database, device_source):
    stale = datetime.now(timezone.utc) - timedelta(seconds=LOCAL_AGENT_DEVICE_STALE_SECONDS + 5)
    _report(database, device_report([device_source]), seen=stale)

    assert _row(client, device_source)["execution"]["device"]["state"] == "offline"


def test_the_device_that_holds_the_credential_runs_the_source_over_another_online_device(
    client, database, device_source
):
    stale = datetime.now(timezone.utc) - timedelta(seconds=LOCAL_AGENT_DEVICE_STALE_SECONDS + 5)
    _report(database, device_report([device_source], device_id="dev-laptop", name="Laptop"), seen=stale)
    _report(database, device_report([], device_id="dev-desktop", name="Desktop"))

    row = _row(client, device_source)

    assert (row["execution"]["device"]["name"], row["execution"]["device"]["state"]) == ("Laptop", "offline")
    assert row["connection_status"] == {"state": "ready", "reason": None}


def test_an_online_device_that_can_take_the_job_comes_before_one_that_needs_an_update(
    client, database, device_source
):
    contracts = declared_acquisition_contracts()
    contracts["confluence"] += 1
    _report(database, device_report([device_source], device_id="dev-laptop", name="Laptop"))
    _report(
        database,
        device_report([device_source], device_id="dev-desktop", name="Desktop", acquisition_contracts=contracts),
    )

    device = _row(client, device_source)["execution"]["device"]

    assert (device["name"], device["state"]) == ("Laptop", "online")


def test_another_member_sees_the_state_without_the_devices_name(client, database, device_source):
    contracts = declared_acquisition_contracts()
    contracts["confluence"] += 1
    _report(database, device_report([device_source], acquisition_contracts=contracts))

    device = _row(client, device_source, viewer=CREATOR)["execution"]["device"]

    assert device["name"] is None
    assert device["state"] == "update_required"


def test_a_token_source_tells_its_owner_to_store_the_secret_on_the_device(client, database, device_source):
    owner_row = _row(client, device_source)
    member_row = _row(client, device_source, viewer=CREATOR)

    assert owner_row["execution"]["device"]["credential"] == "secret"
    assert owner_row["ownership"]["viewer_is_execution_owner"] is True
    assert member_row["ownership"]["viewer_is_execution_owner"] is False


def test_a_browser_session_source_signs_in_on_the_device(client):
    created = client.post(
        "/api/v1/sources",
        json={
            "type": "jira",
            "name": "Issues",
            "config": {
                "base_url": "https://jira.example.com",
                "sync_mode": "cloud",
                "auth_mode": "browser_cookie",
                "projects": ["ACME"],
            },
            "access_policy": "workspace",
        },
    )
    assert created.status_code == 200, created.text
    source_id = created.json()["id"]
    moved = client.put(
        f"/api/v1/sources/{source_id}/executor",
        json={"executor": "device"},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert moved.status_code == 200, moved.text

    assert _row(client, source_id)["execution"]["device"]["credential"] == "browser_session"


def test_a_source_created_on_my_computer_needs_no_secret_and_waits_for_its_credential(client):
    config = {key: value for key, value in CONFLUENCE.items() if key != "pat"}
    checked = client.post(
        "/api/v1/genes/confluence/config-check",
        json={"config": config, "executor": "device"},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert checked.status_code == 200, checked.text
    assert checked.json()["issues"] == []

    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Wiki", "config": config, "access_policy": "workspace", "executor": "device"},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert created.status_code == 200, created.text

    row = _row(client, created.json()["id"])
    assert row["execution"]["operation"] == DOCUMENT_ACQUISITION_OPERATION
    assert row["ownership"]["execution_owner_user_id"] == EXECUTION_OWNER
    assert row["ownership"]["viewer_is_execution_owner"] is True
    assert row["connection_status"] == {"state": "action_required", "reason": "authentication"}


def test_a_source_created_on_my_computer_refuses_a_secret(client):
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Wiki", "config": CONFLUENCE, "access_policy": "workspace", "executor": "device"},
    )

    assert created.status_code == 400, created.text


def test_a_type_without_the_device_executor_is_not_created_on_my_computer(client):
    created = client.post(
        "/api/v1/sources",
        json={
            "type": "teams",
            "name": "Chats",
            "config": {"region": "emea"},
            "access_policy": "private",
            "executor": "device",
        },
    )

    assert created.status_code == 400, created.text
    assert "runs on" in created.text


def test_a_server_source_has_no_execution_owner_viewer(client):
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Team wiki", "config": CONFLUENCE, "access_policy": "workspace"},
    )

    assert _row(client, created.json()["id"], viewer=CREATOR)["ownership"]["viewer_is_execution_owner"] is False
