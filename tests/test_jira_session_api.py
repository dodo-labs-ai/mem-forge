from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from memforge.auth import jira_auth
from memforge.config import AppConfig
from memforge.server.admin_api import _require_secure_or_loopback

TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Encryption key for storing the cookie at rest.
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)

    # Force the authoritative validator to a fixed principal map (no network).
    async def fake_validate(origin, cookie_header, tls_config=None):
        principals = {
            "SESSION=good": {"accountId": "user-123", "displayName": "Ann"},
            "SESSION=other": {"accountId": "user-999", "displayName": "Bo"},
        }
        if cookie_header in principals:
            return principals[cookie_header]
        raise jira_auth.JiraAuthSessionMissingError("not accepted")

    monkeypatch.setattr(jira_auth, "validate_jira_cookie_session", fake_validate)

    from memforge.server.admin_api import create_admin_app

    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.sync.worker_enabled = False
    app = create_admin_app(config=cfg)
    with TestClient(app) as c:
        yield c


def test_upload_then_status_then_forget(client):
    up = client.post(
        "/api/auth/jira-session",
        json={"base_url": "https://jira.example.test", "cookie_header": "SESSION=good", "browser": "Chrome"},
    )
    assert up.status_code == 200, up.text
    assert up.json()["status"] == "active"

    st = client.get("/api/auth/jira-session", params={"base_url": "https://jira.example.test"})
    assert st.json()["status"] == "active"

    origins = client.get("/api/auth/jira-origins")
    assert any(o["origin"] == "https://jira.example.test" for o in origins.json()["origins"])

    gone = client.request("DELETE", "/api/auth/jira-session", params={"base_url": "https://jira.example.test"})
    assert gone.json()["forgotten"] is True


def test_upload_rejects_dead_cookie(client):
    resp = client.post(
        "/api/auth/jira-session",
        json={"base_url": "https://jira.example.test", "cookie_header": "SESSION=dead"},
    )
    assert resp.status_code == 400


def test_upload_principal_change_returns_409(client):
    client.post(
        "/api/auth/jira-session",
        json={"base_url": "https://jira.example.test", "cookie_header": "SESSION=good"},
    )
    resp = client.post(
        "/api/auth/jira-session",
        json={"base_url": "https://jira.example.test", "cookie_header": "SESSION=other"},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"]["new_principal_id"] == "user-999"


def test_expire_marks_status(client):
    client.post(
        "/api/auth/jira-session",
        json={"base_url": "https://jira.example.test", "cookie_header": "SESSION=good"},
    )
    client.post("/api/auth/jira-session/expire", json={"base_url": "https://jira.example.test", "error": "logged out"})
    st = client.get("/api/auth/jira-session", params={"base_url": "https://jira.example.test"})
    assert st.json()["status"] == "expired"


def _fake_request(host, scheme, headers=None):
    return SimpleNamespace(
        client=SimpleNamespace(host=host),
        headers=Headers(headers or {}),
        url=SimpleNamespace(scheme=scheme),
    )


def test_gate_rejects_plaintext_remote():
    with pytest.raises(HTTPException) as ei:
        _require_secure_or_loopback(_fake_request("203.0.113.5", "http"))
    assert ei.value.status_code == 400


def test_gate_allows_https_and_loopback():
    # No exception means allowed.
    _require_secure_or_loopback(_fake_request("203.0.113.5", "https"))
    _require_secure_or_loopback(_fake_request("203.0.113.5", "http", {"x-forwarded-proto": "https"}))
    _require_secure_or_loopback(_fake_request("127.0.0.1", "http"))


def test_gate_allows_plaintext_when_trusted_flag_set(monkeypatch):
    # Behind Docker the client IP can be the host's own public address, so a trusted
    # local/dev deployment opts in explicitly. Default (flag unset) still rejects.
    request = _fake_request("130.214.163.68", "http")
    with pytest.raises(HTTPException):
        _require_secure_or_loopback(request)
    monkeypatch.setenv("MEMFORGE_ALLOW_PLAINTEXT_SESSION_UPLOAD", "1")
    _require_secure_or_loopback(request)  # no exception now
