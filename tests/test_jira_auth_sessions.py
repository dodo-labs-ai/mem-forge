from __future__ import annotations

import json
from http.cookiejar import Cookie, CookieJar
from pathlib import Path

import pytest

from meminception.config import AppConfig
from meminception.models import SyncState
from meminception.storage.database import Database

TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "auth-sessions.db"))
    await database.connect()
    yield database
    await database.close()


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    return cfg


def _cookie(
    name: str,
    value: str,
    domain: str,
    path: str = "/",
    secure: bool = True,
    domain_specified: bool = True,
) -> Cookie:
    return Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=domain_specified,
        domain_initial_dot=domain.startswith("."),
        path=path,
        path_specified=True,
        secure=secure,
        expires=None,
        discard=True,
        comment=None,
        comment_url=None,
        rest={},
        rfc2109=False,
    )


def test_cookie_header_post_filters_domain_path_and_secure_scope():
    from meminception.auth.jira_auth import _cookie_header_from_jar

    jar = CookieJar()
    jar.set_cookie(_cookie("root", "ok", ".example.test", "/", secure=False))
    jar.set_cookie(_cookie("rest", "ok", "jira.example.test", "/rest", secure=False))
    jar.set_cookie(_cookie("wrong-host", "bad", "other.example.test", "/"))
    jar.set_cookie(_cookie("wrong-path", "bad", "jira.example.test", "/secure-only"))
    jar.set_cookie(_cookie("wrong-prefix-path", "bad", "jira.example.test", "/rest/api/2/my"))
    jar.set_cookie(_cookie("host-only-parent", "bad", "example.test", "/", domain_specified=False))
    jar.set_cookie(_cookie("secure-on-http", "bad", "jira.example.test", "/", secure=True))

    https_header = _cookie_header_from_jar(
        jar,
        hostname="jira.example.test",
        request_path="/rest/api/2/myself",
        is_https=True,
    )
    http_header = _cookie_header_from_jar(
        jar,
        hostname="jira.example.test",
        request_path="/rest/api/2/myself",
        is_https=False,
    )

    assert https_header == "root=ok; rest=ok; secure-on-http=bad"
    assert http_header == "root=ok; rest=ok"


@pytest.mark.asyncio
async def test_jira_auth_session_is_encrypted_redacted_and_shared_by_origin(db, monkeypatch):
    from meminception.auth.jira_auth import JiraAuthSessionService
    from meminception.source_secrets import decrypt_secret

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    service = JiraAuthSessionService(db, session_validator=lambda origin, cookie, tls_config=None: {"accountId": "user-123"})

    await service.store_validated_session(
        base_url="https://jira.example.test/projects/PAY",
        cookie_header="JSESSIONID=session; atlassian.xsrf.token=token",
        principal={
            "accountId": "user-123",
            "displayName": "Codex User",
            "emailAddress": "codex@example.com",
        },
        browser="Chrome",
    )

    stored = await db.get_auth_session("jira", "https://jira.example.test")
    assert stored is not None
    assert stored["secret_encrypted"] != "JSESSIONID=session; atlassian.xsrf.token=token"
    assert decrypt_secret(stored["secret_encrypted"]) == "JSESSIONID=session; atlassian.xsrf.token=token"

    status = await service.get_status("https://jira.example.test/wiki")
    assert status == {
        "provider": "jira",
        "origin": "https://jira.example.test",
        "status": "active",
        "principal_id": "user-123",
        "principal_name": "Codex User",
        "principal_email": "codex@example.com",
        "browser": "Chrome",
        "captured_at": stored["captured_at"],
        "validated_at": stored["validated_at"],
        "last_error": None,
    }
    assert "secret_encrypted" not in status
    assert await service.cookie_header_for_sync("https://jira.example.test") == (
        "JSESSIONID=session; atlassian.xsrf.token=token"
    )


@pytest.mark.asyncio
async def test_jira_auth_session_missing_does_not_implicitly_read_browser_when_sync_disallows_refresh(db):
    from meminception.auth.jira_auth import JiraAuthSessionMissingError, JiraAuthSessionService

    def fail_extractor(origin, browser):
        raise AssertionError("sync should not read browser cookies implicitly")

    service = JiraAuthSessionService(db, browser_extractor=fail_extractor)

    with pytest.raises(JiraAuthSessionMissingError):
        await service.cookie_header_for_sync("https://jira.example.test", allow_browser_refresh=False)


@pytest.mark.asyncio
async def test_jira_auth_session_refresh_failure_is_persisted_for_missing_session(db, monkeypatch):
    from meminception.auth.jira_auth import JiraAuthSessionError, JiraAuthSessionService

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)

    def fail_extractor(origin, browser):
        raise RuntimeError("Chrome profile is locked")

    service = JiraAuthSessionService(db, browser_extractor=fail_extractor)

    with pytest.raises(JiraAuthSessionError):
        await service.refresh_from_browser(base_url="https://jira.example.test")

    status = await service.get_status("https://jira.example.test")
    assert status["status"] == "failed"
    assert status["last_error"] == "Chrome profile is locked"


@pytest.mark.asyncio
async def test_jira_auth_session_refresh_resets_matching_sources_only_after_principal_confirmation(
    db,
    tmp_path,
    monkeypatch,
):
    from meminception.auth.jira_auth import JiraAuthSessionService, JiraPrincipalChangedError

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    service = JiraAuthSessionService(db, session_validator=lambda origin, cookie, tls_config=None: {"accountId": "old-user"})
    await service.store_validated_session(
        base_url="https://jira.example.test",
        cookie_header="JSESSIONID=old",
        principal={"accountId": "old-user", "displayName": "Old User"},
        browser="Chrome",
    )
    for source_id, base_url in {
        "src-a": "https://jira.example.test",
        "src-b": "https://jira.example.test/projects/ABC",
        "src-c": "https://other-jira.example.test",
        "src-pat": "https://jira.example.test",
    }.items():
        await db.upsert_source(
            id=source_id,
            type="jira",
            name=source_id,
            config_json=json.dumps({
                "base_url": base_url,
                "projects": ["PAY"],
                "auth_mode": "pat" if source_id == "src-pat" else "browser_cookie",
                "pat_encrypted": "enc:v1:not-real" if source_id == "src-pat" else None,
            }),
        )
        await db.upsert_sync_state(SyncState(source=source_id, last_sync_status="success"))

    with pytest.raises(JiraPrincipalChangedError):
        await service.store_validated_session(
            base_url="https://jira.example.test",
            cookie_header="JSESSIONID=new",
            principal={"accountId": "new-user", "displayName": "New User"},
            browser="Chrome",
            confirm_principal_change=False,
        )

    assert await db.get_sync_state("src-a") is not None
    assert await db.get_sync_state("src-b") is not None

    result = await service.store_validated_session(
        base_url="https://jira.example.test",
        cookie_header="JSESSIONID=new",
        principal={"accountId": "new-user", "displayName": "New User"},
        browser="Chrome",
        confirm_principal_change=True,
    )

    assert result["principal_changed"] is True
    assert result["sources_reset"] == ["src-a", "src-b"]
    assert await db.get_sync_state("src-a") is None
    assert await db.get_sync_state("src-b") is None
    assert await db.get_sync_state("src-c") is not None
    assert await db.get_sync_state("src-pat") is not None


def test_admin_allows_jira_browser_session_source_without_source_cookie(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "projects": ["PAY"],
                    "auth_mode": "browser_cookie",
                },
            },
        )
        assert response.status_code == 200, response.text
        source_id = response.json()["id"]
        sources = client.get("/api/sources").json()["data"]

    stored = next(source for source in sources if source["id"] == source_id)
    assert "jira_cookie" not in stored["config"]
    assert "jira_cookie_configured" not in stored["config"]
    assert stored["config"]["auth_mode"] == "browser_cookie"


def test_admin_rejects_source_owned_jira_cookie(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from meminception.server.admin_api import create_admin_app

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    app = create_admin_app(config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.post(
            "/api/sources",
            json={
                "type": "jira",
                "name": "Enterprise Jira",
                "config": {
                    "base_url": "https://jira.example.test",
                    "projects": ["PAY"],
                    "auth_mode": "browser_cookie",
                    "jira_cookie": "JSESSIONID=source-owned",
                },
            },
        )

    assert response.status_code == 400
    assert "shared auth sessions" in response.json()["detail"]


@pytest.mark.asyncio
async def test_runtime_resolves_jira_browser_session_without_persisting_cookie(db, tmp_path, monkeypatch):
    from meminception.auth.jira_auth import JiraAuthSessionService
    from meminception import runtime

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    await JiraAuthSessionService(db).store_validated_session(
        base_url="https://jira.example.test",
        cookie_header="JSESSIONID=runtime",
        principal={"accountId": "user-123"},
        browser="Chrome",
    )
    source = {
        "id": "src-jira",
        "type": "jira",
        "name": "Runtime Jira",
        "config": {
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
        },
    }

    captured = {}

    class FakeOrchestrator:
        async def sync_gene(self, *, gene, source_name, source_id, progress_callback=None, force_full_sync=False):
            captured["gene_config"] = gene.config
            return SyncState(source=source_id, last_sync_status="success")

    class FakeRuntime:
        def orchestrator(self):
            return FakeOrchestrator()

    class FakeJiraAuthSessionService:
        def __init__(self, database):
            self.database = database

        async def cookie_header_for_sync(self, base_url, *, allow_browser_refresh=True, tls_config=None):
            return await JiraAuthSessionService(
                self.database,
                session_validator=lambda origin, cookie, tls_config=None: {"accountId": "user-123"},
            ).cookie_header_for_sync(base_url, allow_browser_refresh=allow_browser_refresh, tls_config=tls_config)

    monkeypatch.setattr(runtime, "JiraAuthSessionService", FakeJiraAuthSessionService)

    result = await runtime.run_source_sync(
        db=db,
        config=_config(tmp_path),
        source=source,
        runtime=FakeRuntime(),
    )

    assert result.last_sync_status == "success"
    assert captured["gene_config"]["jira_cookie"] == "JSESSIONID=runtime"
    assert "jira_cookie" not in source["config"]


@pytest.mark.asyncio
async def test_runtime_keeps_legacy_jira_pat_source_in_pat_mode(db, tmp_path, monkeypatch):
    from meminception import runtime
    from meminception.source_secrets import prepare_source_config_for_storage

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    source = {
        "id": "src-legacy-pat",
        "type": "jira",
        "name": "Legacy PAT Jira",
        "config": prepare_source_config_for_storage(
            {
                "base_url": "https://jira.example.test",
                "projects": ["PAY"],
                "pat": "legacy-pat",
            },
            secret_fields=("pat", "jira_cookie"),
        ),
    }

    captured = {}

    class FailingJiraAuthSessionService:
        def __init__(self, database):
            self.database = database

        async def cookie_header_for_sync(self, *args, **kwargs):
            raise AssertionError("legacy PAT source should not resolve a browser session")

    class FakeOrchestrator:
        async def sync_gene(self, *, gene, source_name, source_id, progress_callback=None, force_full_sync=False):
            captured["gene_config"] = gene.config
            return SyncState(source=source_id, last_sync_status="success")

    class FakeRuntime:
        def orchestrator(self):
            return FakeOrchestrator()

    monkeypatch.setattr(runtime, "JiraAuthSessionService", FailingJiraAuthSessionService)

    await runtime.run_source_sync(
        db=db,
        config=_config(tmp_path),
        source=source,
        runtime=FakeRuntime(),
    )

    assert captured["gene_config"]["pat"] == "legacy-pat"
    assert "jira_cookie" not in captured["gene_config"]


@pytest.mark.asyncio
async def test_sync_service_marks_shared_jira_session_expired_when_sync_reports_browser_session_failure(
    db,
    tmp_path,
    monkeypatch,
):
    from meminception.auth.jira_auth import JiraAuthSessionService
    from meminception import runtime

    monkeypatch.setenv("MEMINCEPTION_SECRET_KEY", TEST_SOURCE_KEY)
    await db.upsert_source(
        id="src-expired",
        type="jira",
        name="Expired Jira",
        config_json=json.dumps({
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
        }),
    )
    await JiraAuthSessionService(db).store_validated_session(
        base_url="https://jira.example.test",
        cookie_header="JSESSIONID=expired",
        principal={"accountId": "user-123"},
        browser="Chrome",
    )

    async def fail_sync(**kwargs):
        raise RuntimeError("Jira browser session cookie expired or is not accepted. Refresh the session.")

    monkeypatch.setattr(runtime, "run_source_sync", fail_sync)
    service = runtime.SyncService(db, _config(tmp_path))

    await service._run_source_task("src-expired")

    status = await JiraAuthSessionService(db).get_status("https://jira.example.test")
    sync_state = await db.get_sync_state("src-expired")
    assert status["status"] == "expired"
    assert "Refresh the session" in status["last_error"]
    assert sync_state is not None
    assert sync_state.last_sync_status == "failed"
