"""HTTP contract tests for the self-hosted local-daemon status boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient
import pytest

from memforge.config import AppConfig
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.release import installed_version
from memforge.server import admin_api
from tests.device_report_fixture import device_report


STATUS_PATH = "/api/cloud/local-agent/status"
LEASE_PATH = "/api/cloud/local-agent/jobs/lease"
PRINCIPAL_HEADER = "X-Test-Principal"
SERVER = {"version": installed_version(), "acquisition_contracts": declared_acquisition_contracts()}


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


def _principal(request: Request) -> str:
    return request.headers.get(PRINCIPAL_HEADER, "local-owner")


def _lease(client: TestClient, principal: str, **device) -> None:
    response = client.post(
        LEASE_PATH,
        headers={PRINCIPAL_HEADER: principal},
        json={"limit": 1, "lease_seconds": 60, "wait_seconds": 0, "device": device_report(**device).to_payload()},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "jobs": [],
        "server": SERVER,
        "released_source_ids": [],
    }


def test_self_hosted_status_observes_public_lease_heartbeat_for_only_its_principal(
    tmp_path: Path,
) -> None:
    app = admin_api.create_admin_app(
        config=_config(tmp_path),
        principal_resolver=_principal,
    )

    with TestClient(app) as client:
        initial = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"})
        assert initial.status_code == 200, initial.text
        initial_body = initial.json()
        assert set(initial_body) == {
            "status",
            "last_seen_at",
            "devices",
            "sources",
            "server",
            "checked_at",
            "stale_after_seconds",
        }
        assert initial_body["devices"] == []
        assert initial_body["sources"] == []
        assert initial_body["server"] == SERVER
        assert initial_body["status"] == "offline"
        assert initial_body["last_seen_at"] is None
        assert initial_body["stale_after_seconds"] == 90
        datetime.fromisoformat(initial_body["checked_at"])

        _lease(client, "owner-a")

        online = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"})
        other_principal = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-b"})

    assert online.status_code == 200, online.text
    assert online.json()["status"] == "online"
    assert datetime.fromisoformat(online.json()["last_seen_at"]).tzinfo is not None
    assert online.json()["devices"] == [
        {
            "device_id": "dev-laptop",
            "name": "Laptop",
            "version": "0.2.0",
            "status": "online",
            "last_seen_at": online.json()["last_seen_at"],
        }
    ]
    assert other_principal.status_code == 200, other_principal.text
    assert other_principal.json()["status"] == "offline"
    assert other_principal.json()["last_seen_at"] is None
    assert other_principal.json()["devices"] == []


def _clock(at: datetime) -> type[datetime]:
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001
            return at.astimezone(tz or timezone.utc)

    return Clock


def test_status_reports_each_device_of_the_caller_by_its_latest_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from memforge.storage import database

    app = admin_api.create_admin_app(config=_config(tmp_path), principal_resolver=_principal)
    first = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)

    with TestClient(app) as client:
        monkeypatch.setattr(database, "datetime", _clock(first))
        _lease(client, "owner-a", device_id="dev-desktop", name="Desktop", version="0.2.0")
        monkeypatch.setattr(database, "datetime", _clock(first + timedelta(seconds=100)))
        _lease(client, "owner-a", device_id="dev-laptop", name="Laptop", version="0.2.0")
        _lease(client, "owner-a", device_id="dev-laptop", name="Work laptop", version="0.3.0")
        monkeypatch.setattr(admin_api, "datetime", _clock(first + timedelta(seconds=150)))
        body = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"}).json()

    assert body["status"] == "online"
    assert body["last_seen_at"] == (first + timedelta(seconds=100)).isoformat()
    assert body["devices"] == [
        {
            "device_id": "dev-laptop",
            "name": "Work laptop",
            "version": "0.3.0",
            "status": "online",
            "last_seen_at": (first + timedelta(seconds=100)).isoformat(),
        },
        {
            "device_id": "dev-desktop",
            "name": "Desktop",
            "version": "0.2.0",
            "status": "offline",
            "last_seen_at": first.isoformat(),
        },
    ]


def test_a_lease_without_a_well_formed_device_report_is_refused(tmp_path: Path) -> None:
    app = admin_api.create_admin_app(config=_config(tmp_path), principal_resolver=_principal)
    report = device_report().to_payload()

    with TestClient(app) as client:
        missing = client.post(LEASE_PATH, json={"limit": 1})
        nameless = client.post(LEASE_PATH, json={"limit": 1, "device": {**report, "name": " "}})
        status = client.get(STATUS_PATH)

    assert missing.status_code == 422
    assert nameless.status_code == 422
    assert status.json()["devices"] == []


def test_self_hosted_status_expires_at_the_declared_stale_cutoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = admin_api.create_admin_app(
        config=_config(tmp_path),
        principal_resolver=_principal,
    )

    with TestClient(app) as client:
        _lease(client, "owner-a")
        online = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"})
        last_seen_at = datetime.fromisoformat(online.json()["last_seen_at"])
        at_cutoff = last_seen_at + timedelta(seconds=90)
        after_cutoff = last_seen_at + timedelta(seconds=91)

        class FrozenDateTime(datetime):
            current = at_cutoff

            @classmethod
            def now(cls, tz=None):  # noqa: ANN001
                return cls.current.astimezone(tz or timezone.utc)

        monkeypatch.setattr(admin_api, "datetime", FrozenDateTime)
        still_online = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"})
        FrozenDateTime.current = after_cutoff
        stale = client.get(STATUS_PATH, headers={PRINCIPAL_HEADER: "owner-a"})

    assert still_online.status_code == 200, still_online.text
    assert still_online.json()["status"] == "online"
    assert still_online.json()["checked_at"] == at_cutoff.isoformat()
    assert stale.status_code == 200, stale.text
    assert stale.json() == {
        "status": "offline",
        "last_seen_at": online.json()["last_seen_at"],
        "devices": [{**online.json()["devices"][0], "status": "offline"}],
        "sources": [],
        "server": SERVER,
        "checked_at": after_cutoff.isoformat(),
        "stale_after_seconds": 90,
    }


def test_a_lease_or_heartbeat_without_a_device_report_names_the_update(tmp_path: Path) -> None:
    app = admin_api.create_admin_app(config=_config(tmp_path), principal_resolver=_principal)

    with TestClient(app) as client:
        lease = client.post(LEASE_PATH, json={"limit": 1, "lease_seconds": 60, "wait_seconds": 0})
        heartbeat = client.post(
            "/api/cloud/local-agent/jobs/job-1/heartbeat", json={"attempt_count": 1, "lease_seconds": 60}
        )
        status = client.get(STATUS_PATH)

    for answer in (lease, heartbeat):
        assert answer.status_code == 422, answer.text
        detail = answer.json()["detail"]
        assert detail["error"] == "device_report_required"
        assert f"Install MemForge {installed_version()}" in detail["message"]
    assert status.json()["devices"] == []
