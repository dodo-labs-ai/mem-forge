"""What the server does for a device Source, whose provider only its owner's device reaches (ADR 0058).

The server holds no credential for a device Source and never reaches its
provider: it does not run it without a device acquisition, does not preview
it, and its browser sessions serve server Sources only. A device's lease
answer names the Sources that device executes, so it reports its readiness
for them and deletes the credentials of any other, and the status answer
names the owner's device Sources with their execution devices for
``memforge daemon status``. A ``credential_check`` job checks a device
Source's credential on its execution device, or captures a server Source's
browser session.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import memforge.server.admin_api as admin_api
from memforge.auth.jira_auth import JiraAuthSessionService
from memforge.config import AppConfig
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.release import installed_version
from memforge.runtime import SourceSyncBoundaryError, run_source_sync
from memforge.storage.database import Database
from tests.device_report_fixture import DEVICE_ID, device_report

PRINCIPAL_HEADER = "X-Test-Principal"
OWNER = "owner-a"
JIRA_COOKIE_CONFIG = {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie", "jql": "project = ACME"}
JIRA_TOKEN_CONFIG = {"base_url": "https://jira.example.test", "auth_mode": "pat", "jql": "project = ACME"}


def _principal(request: Request) -> str:
    return request.headers.get(PRINCIPAL_HEADER, OWNER)


def _config(tmp_path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


@pytest.fixture
def database(tmp_path):
    database = Database(str(tmp_path / "sources.db"))
    asyncio.run(database.connect())
    try:
        yield database
    finally:
        asyncio.run(database.close())


def _source(
    database: Database,
    source_id: str,
    *,
    executor: str | None,
    owner: str = OWNER,
    config: dict | None = None,
    source_type: str = "jira",
    device_id: str | None = DEVICE_ID,
) -> None:
    asyncio.run(
        database.upsert_source(
            id=source_id,
            type=source_type,
            name=source_id,
            config_json=json.dumps(config or JIRA_TOKEN_CONFIG),
            access_policy="workspace",
            owner_user_id=owner,
            execution_owner_user_id=owner,
            executor=executor,
            execution_device_id=device_id if executor == "device" else None,
        )
    )


def _app(tmp_path, database):
    return admin_api.create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=_principal,
        workspace_role_resolver=lambda request: "member",
    )


def test_a_source_keeps_its_executor_until_one_is_given(database) -> None:
    _source(database, "src-new", executor=None)
    _source(database, "src-device", executor="device")
    _source(database, "src-device", executor=None, config={**JIRA_TOKEN_CONFIG, "jql": "project = BETA"})

    assert asyncio.run(database.get_source("src-new"))["executor"] == "server"
    assert asyncio.run(database.get_source("src-device"))["executor"] == "device"
    assert [source["id"] for source in asyncio.run(database.list_device_sources(OWNER))] == ["src-device"]


def test_the_lease_answer_names_the_sources_the_device_executes_without_their_secrets(tmp_path, database) -> None:
    _source(database, "src-device", executor="device", config={**JIRA_TOKEN_CONFIG, "pat": "server-never-holds"})
    _source(database, "src-desktop", executor="device", device_id="dev-desktop")
    _source(database, "src-server", executor="server")
    _source(database, "src-other-owner", executor="device", owner="owner-b")

    with TestClient(_app(tmp_path, database)) as client:
        laptop = client.post(
            "/api/cloud/local-agent/jobs/lease",
            headers={PRINCIPAL_HEADER: OWNER},
            json={"limit": 1, "device": device_report(["src-server", "src-desktop"]).to_payload()},
        )
        desktop = client.post(
            "/api/cloud/local-agent/jobs/lease",
            headers={PRINCIPAL_HEADER: OWNER},
            json={"limit": 1, "device": device_report(device_id="dev-desktop").to_payload()},
        )

    assert laptop.status_code == 200, laptop.text
    assert laptop.json()["sources"] == [
        {"id": "src-device", "type": "jira", "config": JIRA_TOKEN_CONFIG, "execution_device_id": DEVICE_ID}
    ]
    assert [source["id"] for source in desktop.json()["sources"]] == ["src-desktop"]


def test_the_status_names_the_callers_device_sources_without_their_secrets(tmp_path, database) -> None:
    _source(database, "src-device", executor="device", config={**JIRA_TOKEN_CONFIG, "pat": "server-never-holds"})
    _source(database, "src-server", executor="server")
    _source(database, "src-other-owner", executor="device", owner="owner-b")

    with TestClient(_app(tmp_path, database)) as client:
        body = client.get("/api/cloud/local-agent/status", headers={PRINCIPAL_HEADER: OWNER}).json()

    assert body["sources"] == [
        {
            "id": "src-device",
            "name": "src-device",
            "type": "jira",
            "config": JIRA_TOKEN_CONFIG,
            "execution_device_id": DEVICE_ID,
        }
    ]
    assert body["server"] == {"version": installed_version(), "acquisition_contracts": declared_acquisition_contracts()}


def test_the_server_does_not_preview_a_device_configuration_or_a_device_source(tmp_path, database) -> None:
    _source(database, "src-device", executor="device")

    with TestClient(_app(tmp_path, database)) as client:
        new = client.post(
            "/api/v1/genes/jira/preview-discovery", json={"config": JIRA_TOKEN_CONFIG, "executor": "device"}
        )
        existing = client.post(
            "/api/v1/genes/jira/preview-discovery", json={"source_id": "src-device", "config": {}}
        )

    assert (new.status_code, new.json()["detail"]) == (409, "device_preview_unavailable")
    assert (existing.status_code, existing.json()["detail"]) == (409, "device_preview_unavailable")


def test_a_device_source_moving_to_the_server_previews_with_the_typed_credential(
    tmp_path, database, monkeypatch
) -> None:
    wiki = {"base_url": "https://wiki.example.test", "sync_mode": "space", "spaces": ["ACME"]}
    _source(database, "src-device-wiki", executor="device", source_type="confluence", config=wiki)
    previewed: list[dict] = []

    class _PreviewGene:
        def __init__(self, config: dict) -> None:
            previewed.append(config)

        async def authenticate(self) -> None:
            return None

        async def discover(self, since):
            yield SimpleNamespace(
                item_id="page-1", title="Runbook", source_url="https://wiki.example.test/1", last_modified=None
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(admin_api, "create_gene", lambda name, config, source_id: _PreviewGene(config))

    with TestClient(_app(tmp_path, database)) as client:
        answer = client.post(
            "/api/v1/genes/confluence/preview-discovery",
            json={"source_id": "src-device-wiki", "config": {"pat": "typed-token"}, "executor": "server", "limit": 5},
        )

    assert answer.status_code == 200, answer.text
    assert [item["item_id"] for item in answer.json()["items"]] == ["page-1"]
    assert previewed[0]["pat"] == "typed-token"


def test_the_server_does_not_browse_a_device_sources_repository(tmp_path, database) -> None:
    _source(
        database,
        "src-device-repo",
        executor="device",
        source_type="github_repo",
        config={"repo_url": "https://github.example.com/acme/handbook"},
    )

    with TestClient(_app(tmp_path, database)) as client:
        answer = client.post(
            "/api/v1/genes/github_repo/browse-tree", json={"source_id": "src-device-repo", "config": {}, "limit": 50}
        )

    assert (answer.status_code, answer.json()["detail"]) == (409, "device_preview_unavailable")


def test_the_server_never_runs_a_device_source_without_its_devices_acquisition(database) -> None:
    _source(database, "src-device", executor="device")
    source = asyncio.run(database.get_source("src-device"))

    with pytest.raises(SourceSyncBoundaryError, match="runs only on what its device acquired"):
        asyncio.run(run_source_sync(database, AppConfig(), source))


def test_the_servers_browser_sessions_serve_server_sources_only(database) -> None:
    _source(database, "src-device", executor="device", config=JIRA_COOKIE_CONFIG)
    _source(database, "src-server", executor="server", config=JIRA_COOKIE_CONFIG)

    reset = asyncio.run(
        JiraAuthSessionService(database)._source_ids_for_browser_session_origin("https://jira.example.test")
    )

    assert reset == ["src-server"]


def _credential_check(client: TestClient, source_id: str, *, user: str = OWNER):
    return client.post(
        "/api/cloud/local-agent/jobs",
        headers={PRINCIPAL_HEADER: user},
        json={"source_id": source_id, "source_type": "jira", "operation": "credential_check", "payload": {}},
    )


def test_a_device_sources_credential_check_runs_on_its_execution_device(tmp_path, database) -> None:
    _source(database, "src-device", executor="device", config={**JIRA_TOKEN_CONFIG, "pat": "never-sent"})
    _source(database, "src-unplaced", executor="device", device_id=None)

    with TestClient(_app(tmp_path, database)) as client:
        other = _credential_check(client, "src-device", user="owner-b")
        unplaced = _credential_check(client, "src-unplaced")
        created = _credential_check(client, "src-device")
        desktop = client.post(
            "/api/cloud/local-agent/jobs/lease",
            headers={PRINCIPAL_HEADER: OWNER},
            json={"limit": 1, "device": device_report(device_id="dev-desktop").to_payload()},
        ).json()["jobs"]
        [job] = client.post(
            "/api/cloud/local-agent/jobs/lease",
            headers={PRINCIPAL_HEADER: OWNER},
            json={"limit": 1, "device": device_report().to_payload()},
        ).json()["jobs"]

    assert (other.status_code, other.json()["detail"]) == (403, "credential_check_execution_owner_forbidden")
    assert (unplaced.status_code, unplaced.json()["detail"]["code"]) == (409, "execution_device_required")
    assert created.status_code == 201, created.text
    assert desktop == []
    assert job["operation"] == "credential_check"
    assert job["payload"] == {
        "source_id": "src-device",
        "source_type": "jira",
        "executor": "device",
        "config": JIRA_TOKEN_CONFIG,
    }


def test_a_server_sources_credential_check_only_captures_a_browser_session(tmp_path, database) -> None:
    _source(database, "src-token", executor="server")
    _source(database, "src-cookie", executor="server", config=JIRA_COOKIE_CONFIG)

    with TestClient(_app(tmp_path, database)) as client:
        token = _credential_check(client, "src-token")
        cookie = _credential_check(client, "src-cookie", user="member-b")
        missing = _credential_check(client, "src-missing")

    assert (token.status_code, token.json()["detail"]) == (400, "credential_check_requires_browser_session")
    assert cookie.status_code == 201, cookie.text
    assert missing.status_code == 404
