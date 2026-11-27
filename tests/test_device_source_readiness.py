"""The readiness a device Source exposes in the source list (design §10.7).

``execution.kind`` stays ``local_agent`` for the device executor, and
``execution.device`` describes the Source's execution device: its id, its
state and versions, and, for its execution owner, its name and what its
latest report says it lacks. A Source without an execution device has none
and shows Finish setup (``configuration``); a missing credential shows Sign
in required (``authentication``) and a missing device location Finish setup.
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
from memforge.local_agent.device_report import DeviceSourceReadiness, NotReadyReason
from tests.device_report_fixture import DEVICE_ID, connect_device, device_report

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
def device_source(client, database) -> str:
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Team wiki", "config": CONFLUENCE, "access_policy": "workspace"},
    )
    assert created.status_code == 200, created.text
    source_id = created.json()["id"]
    connect_device(database, EXECUTION_OWNER)
    moved = client.put(
        f"/api/v1/sources/{source_id}/executor",
        json={"executor": "device", "execution_device_id": DEVICE_ID},
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


def test_a_device_source_its_online_device_reports_ready_is_ready(client, database, device_source):
    _report(database, device_report([device_source], name="Alex's MacBook", version=installed_version()))

    row = _row(client, device_source)

    assert row["execution"] == {
        "kind": "local_agent",
        "operation": DOCUMENT_ACQUISITION_OPERATION,
        "immutable_config_fields": [],
        "device": {
            "id": DEVICE_ID,
            "name": "Alex's MacBook",
            "state": "online",
            "installed_version": installed_version(),
            "required_version": installed_version(),
            "credential": "secret",
            "not_ready": [],
        },
    }
    assert row["connection_status"] == {"state": "ready", "reason": None}


def test_a_missing_credential_on_the_execution_device_needs_a_sign_in(client, database, device_source):
    missing = DeviceSourceReadiness(reasons=(NotReadyReason(kind="credential"),))
    _report(database, device_report(not_ready={device_source: missing}, name="Alex's MacBook"))

    row = _row(client, device_source)

    assert row["execution"]["device"]["state"] == "online"
    assert row["execution"]["device"]["not_ready"] == [{"kind": "credential"}]
    assert row["connection_status"] == {"state": "action_required", "reason": "authentication"}


def test_a_missing_folder_on_the_execution_device_needs_its_setup_finished_with_the_path(
    client, database, device_source
):
    missing = DeviceSourceReadiness(reasons=(NotReadyReason(kind="location", field="root", path="/Users/alex/notes"),))
    _report(database, device_report(not_ready={device_source: missing}))

    owner_row = _row(client, device_source)
    member_row = _row(client, device_source, viewer=CREATOR)

    assert owner_row["connection_status"] == {"state": "action_required", "reason": "configuration"}
    assert owner_row["execution"]["device"]["not_ready"] == [
        {"kind": "location", "field": "root", "path": "/Users/alex/notes"}
    ]
    # Another member sees Finish setup, without the owner's path or device name.
    assert member_row["connection_status"] == {"state": "action_required", "reason": "configuration"}
    assert member_row["execution"]["device"]["not_ready"] == []
    assert member_row["execution"]["device"]["name"] is None


def test_a_device_source_without_an_execution_device_needs_its_setup_finished(client, database):
    created = client.post(
        "/api/v1/sources",
        json={"type": "confluence", "name": "Team wiki", "config": CONFLUENCE, "access_policy": "workspace"},
    )
    source_id = created.json()["id"]
    connect_device(database, EXECUTION_OWNER)
    client.put(
        f"/api/v1/sources/{source_id}/executor",
        json={"executor": "device", "execution_device_id": DEVICE_ID},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    asyncio.run(database.remove_local_agent_device(EXECUTION_OWNER, DEVICE_ID))

    row = _row(client, source_id)

    assert row["execution"]["device"] is None
    assert row["connection_status"] == {"state": "action_required", "reason": "configuration"}


def test_a_device_whose_report_does_not_name_the_source_yet_shows_no_connection(client, database, device_source):
    _report(database, device_report([]))

    row = _row(client, device_source)

    assert row["execution"]["device"]["state"] == "online"
    assert "connection_status" not in row or row["connection_status"] is None


@pytest.mark.parametrize("difference", [-1, 1], ids=["behind", "ahead"])
def test_a_device_whose_contract_for_the_type_differs_needs_an_update(client, database, device_source, difference):
    contracts = declared_acquisition_contracts()
    contracts["confluence"] += difference
    _report(database, device_report([device_source], version="0.2.0b8", acquisition_contracts=contracts))

    row = _row(client, device_source)

    assert row["execution"]["device"] == {
        "id": DEVICE_ID,
        "name": "Laptop",
        "state": "update_required",
        "installed_version": "0.2.0b8",
        "required_version": installed_version(),
        "credential": "secret",
        "not_ready": [],
    }
    assert row["connection_status"] == {"state": "ready", "reason": None}


def test_a_device_without_a_recent_report_is_offline(client, database, device_source):
    stale = datetime.now(timezone.utc) - timedelta(seconds=LOCAL_AGENT_DEVICE_STALE_SECONDS + 5)
    _report(database, device_report([device_source]), seen=stale)

    assert _row(client, device_source)["execution"]["device"]["state"] == "offline"


def test_the_execution_device_stands_for_the_source_whatever_the_owners_other_devices_report(
    client, database, device_source
):
    stale = datetime.now(timezone.utc) - timedelta(seconds=LOCAL_AGENT_DEVICE_STALE_SECONDS + 5)
    _report(database, device_report([device_source], name="Laptop"), seen=stale)
    _report(database, device_report([device_source], device_id="dev-desktop", name="Desktop"))

    device = _row(client, device_source)["execution"]["device"]

    assert (device["id"], device["name"], device["state"]) == (DEVICE_ID, "Laptop", "offline")


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


def test_a_browser_session_source_signs_in_on_the_device(client, database):
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
    connect_device(database, EXECUTION_OWNER)
    moved = client.put(
        f"/api/v1/sources/{source_id}/executor",
        json={"executor": "device", "execution_device_id": DEVICE_ID},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert moved.status_code == 200, moved.text

    assert _row(client, source_id)["execution"]["device"]["credential"] == "browser_session"


def test_a_source_created_on_my_computer_needs_no_secret_and_runs_on_the_chosen_device(client, database):
    config = {key: value for key, value in CONFLUENCE.items() if key != "pat"}
    checked = client.post(
        "/api/v1/genes/confluence/config-check",
        json={"config": config, "executor": "device"},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert checked.status_code == 200, checked.text
    assert checked.json()["issues"] == []

    body = {"type": "confluence", "name": "Wiki", "config": config, "access_policy": "workspace", "executor": "device"}
    unchosen = client.post("/api/v1/sources", json=body, headers={PRINCIPAL_HEADER: EXECUTION_OWNER})
    not_theirs = client.post(
        "/api/v1/sources", json={**body, "execution_device_id": DEVICE_ID}, headers={PRINCIPAL_HEADER: EXECUTION_OWNER}
    )
    connect_device(database, EXECUTION_OWNER)
    created = client.post(
        "/api/v1/sources", json={**body, "execution_device_id": DEVICE_ID}, headers={PRINCIPAL_HEADER: EXECUTION_OWNER}
    )
    assert created.status_code == 200, created.text

    assert unchosen.status_code == not_theirs.status_code == 400
    assert unchosen.json()["detail"]["code"] == not_theirs.json()["detail"]["code"] == "execution_device_invalid"
    row = _row(client, created.json()["id"])
    assert row["execution"]["operation"] == DOCUMENT_ACQUISITION_OPERATION
    assert row["execution"]["device"]["id"] == DEVICE_ID
    assert row["ownership"]["execution_owner_user_id"] == EXECUTION_OWNER
    assert row["ownership"]["viewer_is_execution_owner"] is True


def test_a_source_created_on_my_computer_refuses_a_secret(client):
    created = client.post(
        "/api/v1/sources",
        json={
            "type": "confluence",
            "name": "Wiki",
            "config": CONFLUENCE,
            "access_policy": "workspace",
            "executor": "device",
            "execution_device_id": DEVICE_ID,
        },
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


def _check_stored(client, source_id: str, config: dict, **request) -> list[dict]:
    checked = client.post(
        "/api/v1/genes/confluence/config-check",
        json={"config": config, "source_id": source_id, **request},
        headers={PRINCIPAL_HEADER: EXECUTION_OWNER},
    )
    assert checked.status_code == 200, checked.text
    return checked.json()["issues"]


def test_a_device_source_moving_to_the_server_is_checked_with_the_secret_the_server_needs(client, device_source):
    without_secret = {key: value for key, value in CONFLUENCE.items() if key != "pat"}

    assert _check_stored(client, device_source, CONFLUENCE, executor="server") == []
    [issue] = _check_stored(client, device_source, without_secret, executor="server")
    assert issue["key"] == "pat"
    assert "on the device" not in issue["message"]


def test_a_stored_device_source_without_an_executor_is_checked_on_its_device(client, device_source):
    without_secret = {key: value for key, value in CONFLUENCE.items() if key != "pat"}

    assert _check_stored(client, device_source, without_secret) == []
    [issue] = _check_stored(client, device_source, CONFLUENCE)
    assert issue["key"] == "pat"
    assert "on the device" in issue["message"]
