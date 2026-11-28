"""The routes a daemon calls, answered once in front of every workspace it serves.

The router reads jobs and devices through the app's broker and each
workspace's device Sources through its directory of served workspaces, as a
deployment with several workspaces mounts it. Here the broker is a real
database and the two workspaces are directories of their own.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.local_agent.broker import DeviceSourceDirectory
from memforge.server.admin_api import create_admin_app
from memforge.server.local_agent_host_api import build_local_agent_host_router
from memforge.storage.database import Database
from tests.device_report_fixture import DEVICE_ID, device_report

OWNER = "owner-user"
OTHER = "other-user"


class _Workspace:
    """One workspace's device Sources, as its database answers them."""

    def __init__(self, *sources: dict[str, Any]) -> None:
        self.sources = list(sources)

    async def list_device_sources(
        self, execution_owner_user_id: str, *, device_id: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            source
            for source in self.sources
            if source["execution_owner_user_id"] == execution_owner_user_id
            and (device_id is None or source["execution_device_id"] == device_id)
        ]

    async def release_execution_device(self, execution_owner_user_id: str, device_id: str) -> list[str]:
        released = [
            source
            for source in self.sources
            if source["execution_owner_user_id"] == execution_owner_user_id
            and source["execution_device_id"] == device_id
        ]
        for source in released:
            source["execution_device_id"] = None
        return [source["id"] for source in released]


class _Workspaces:
    def __init__(self, served: dict[str, dict[str, DeviceSourceDirectory]]) -> None:
        self.served = served

    async def served_workspaces(self, user_id: str) -> dict[str, DeviceSourceDirectory]:
        return self.served.get(user_id, {})


def _wiki(source_id: str, *, device_id: str | None = DEVICE_ID) -> dict[str, Any]:
    return {
        "id": source_id,
        "name": f"Wiki {source_id}",
        "type": "confluence",
        "config": {"base_url": "https://wiki.example.test", "space_keys": ["ENG"]},
        "execution_owner_user_id": OWNER,
        "execution_device_id": device_id,
    }


@pytest.fixture
def broker(tmp_path: Path):
    db = Database(str(tmp_path / "jobs.db"))
    asyncio.run(db.connect())
    yield db
    asyncio.run(db.close())


def _host(broker: Database, workspaces: _Workspaces) -> TestClient:
    app = FastAPI()
    app.state.principal_resolver = lambda request: request.headers.get("X-Test-Principal", OWNER)
    app.state.local_agent_broker = broker
    app.state.daemon_workspaces = workspaces
    app.include_router(build_local_agent_host_router())
    return TestClient(app)


def _lease(client: TestClient, ready: tuple[str, ...] = ()) -> dict[str, Any]:
    response = client.post(
        "/api/cloud/local-agent/jobs/lease",
        json={"limit": 5, "lease_seconds": 60, "device": device_report(ready).to_payload()},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_daemon_reads_and_releases_the_device_sources_of_every_workspace_it_serves(broker):
    alpha, beta = (
        _Workspace(_wiki("src-alpha")),
        _Workspace(_wiki("src-beta"), _wiki("src-elsewhere", device_id="dev-desk")),
    )
    client = _host(broker, _Workspaces({OWNER: {"team_alpha": alpha, "team_beta": beta}}))

    lease = _lease(client)
    status = client.get("/api/cloud/local-agent/status").json()
    removed = client.delete(f"/api/cloud/local-agent/devices/{DEVICE_ID}")

    assert {source["id"] for source in lease["sources"]} == {"src-alpha", "src-beta"}
    assert {source["id"] for source in status["sources"]} == {"src-alpha", "src-beta", "src-elsewhere"}
    assert [device["device_id"] for device in status["devices"]] == [DEVICE_ID]
    assert removed.status_code == 200, removed.text
    assert sorted(removed.json()["source_ids"]) == ["src-alpha", "src-beta"]
    assert [source["execution_device_id"] for source in beta.sources] == [None, "dev-desk"]
    assert asyncio.run(broker.list_local_agent_devices(OWNER)) == []


def test_a_job_is_read_only_by_its_people_while_its_workspace_is_served(broker):
    asyncio.run(
        broker.insert_local_agent_job(
            job_id="laj-pick",
            workspace_id="team_alpha",
            source_id="",
            source_type="local_markdown",
            operation="local_markdown_pick_root",
            payload={},
            created_by_user_id=OWNER,
            execution_owner_user_id=OWNER,
        )
    )
    served = _Workspaces({OWNER: {"team_alpha": _Workspace()}, OTHER: {"team_alpha": _Workspace()}})
    client = _host(broker, served)

    assert client.get("/api/cloud/local-agent/jobs/laj-pick").status_code == 200
    assert client.get("/api/cloud/local-agent/jobs/laj-pick", headers={"X-Test-Principal": OTHER}).status_code == 404
    served.served[OWNER] = {}
    assert client.get("/api/cloud/local-agent/jobs/laj-pick").status_code == 404


@pytest.mark.parametrize(
    ("payload", "result", "expected"),
    [
        ({}, {"source_sync_run_id": "run-1"}, 200),
        ({}, {}, 409),
        ({"dry_run": True}, {"plan": {}}, 200),
    ],
)
def test_a_sync_job_succeeds_only_naming_the_sync_run_it_started(broker, payload, result, expected):
    asyncio.run(
        broker.insert_local_agent_job(
            job_id="laj-sync",
            workspace_id="team_alpha",
            source_id="src-chats",
            source_type="teams",
            operation="teams_sync",
            payload=payload,
            created_by_user_id=OWNER,
            execution_owner_user_id=OWNER,
        )
    )
    client = _host(broker, _Workspaces({OWNER: {"team_alpha": _Workspace()}}))
    [job] = _lease(client)["jobs"]

    def complete(result: dict[str, Any]):
        return client.post(
            "/api/cloud/local-agent/jobs/laj-sync/complete",
            json={
                "status": "succeeded",
                "attempt_count": job["attempt_count"],
                "device_id": DEVICE_ID,
                "result": result,
            },
        )

    response = complete(result)

    assert response.status_code == expected, response.text
    if expected == 200 and not payload:
        assert complete(result).json()["status"] == "succeeded"
        assert complete({"source_sync_run_id": "run-2"}).status_code == 409


def test_an_app_that_does_not_serve_daemons_answers_no_daemon_route(tmp_path):
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    app = create_admin_app(config=config, serves_daemons=False, principal_resolver=lambda request: OWNER)

    with TestClient(app) as client:
        assert client.get("/api/cloud/local-agent/status").status_code == 404
        assert client.get("/api/v1/local-agent/jobs/current").status_code == 200

    paths = {route.path for route in app.routes}
    assert not any(path.startswith("/api/cloud/local-agent") for path in paths)
