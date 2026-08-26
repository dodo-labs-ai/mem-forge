"""HTTP contract tests for the self-hosted local-daemon status boundary."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Request
from fastapi.testclient import TestClient
import pytest

from memforge.config import AppConfig
from memforge.server import admin_api


STATUS_PATH = "/api/cloud/local-agent/status"
LEASE_PATH = "/api/cloud/local-agent/jobs/lease"
PRINCIPAL_HEADER = "X-Test-Principal"


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    return config


def _principal(request: Request) -> str:
    return request.headers.get(PRINCIPAL_HEADER, "local-owner")


def _lease(client: TestClient, principal: str) -> None:
    response = client.post(
        LEASE_PATH,
        headers={PRINCIPAL_HEADER: principal},
        json={"limit": 1, "lease_seconds": 60, "wait_seconds": 0},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"jobs": []}


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
            "checked_at",
            "stale_after_seconds",
        }
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
    assert set(online.json()) == {
        "status",
        "last_seen_at",
        "checked_at",
        "stale_after_seconds",
    }
    assert other_principal.status_code == 200, other_principal.text
    assert other_principal.json()["status"] == "offline"
    assert other_principal.json()["last_seen_at"] is None


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
        "checked_at": after_cutoff.isoformat(),
        "stale_after_seconds": 90,
    }
