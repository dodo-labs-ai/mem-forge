"""Tests for the provider-agnostic browser-session layer using a fake provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from memforge.auth import browser_session as bs
from memforge.storage.database import Database


class _FakeService:
    def __init__(self, db):
        self.db = db

    async def get_status(self, base_url):
        return {"provider": "demo", "origin": base_url, "status": "active"}

    async def cookie_header_for_sync(self, base_url, *, tls_config=None):
        return "DEMO-COOKIE=abc"

    async def store_uploaded_session(
        self, *, base_url, cookie_header, browser=None, tls_config=None, confirm_principal_change=False
    ):
        return {
            "provider": "demo",
            "origin": base_url,
            "status": "active",
            "cookie_header": cookie_header,
            "browser": browser,
        }


def _demo_provider() -> bs.BrowserSessionProvider:
    return bs.BrowserSessionProvider(
        provider="demo",
        source_type="demo",
        label="Demo",
        cookie_config_key="demo_cookie",
        canonical_origin=lambda url: url.rstrip("/"),
        service_factory=_FakeService,
        uses_browser_session=lambda cfg: str(cfg.get("auth_mode") or "browser_cookie") == "browser_cookie",
    )


def _db(tmp_path: Path) -> Database:
    db = Database(str(tmp_path / "db.sqlite"))
    asyncio.run(db.connect())
    return db


def test_register_and_lookup():
    bs.register_provider(_demo_provider())
    assert bs.get_provider("demo").label == "Demo"
    assert bs.provider_for_source_type("demo").provider == "demo"
    assert bs.provider_for_source_type("nope") is None


def test_list_origins_merges_sessions_and_sources(tmp_path):
    bs.register_provider(_demo_provider())
    db = _db(tmp_path)
    try:
        asyncio.run(
            db.upsert_auth_session(
                provider="demo",
                origin="https://demo.one",
                secret_encrypted="SECRET",
                principal_id="u1",
                principal_name="Alice",
                principal_email=None,
                browser="Chrome",
                status="active",
                captured_at="t",
                validated_at="t",
                last_error=None,
            )
        )
        asyncio.run(
            db.upsert_source(
                id="src-demo1",
                type="demo",
                name="Demo Two",
                config_json=json.dumps({"base_url": "https://demo.two/"}),
                access_policy="workspace",
                owner_user_id="dev",
            )
        )
        origins = {o["origin"]: o for o in asyncio.run(bs.list_origins(db, "demo"))}
    finally:
        asyncio.run(db.close())
    assert origins["https://demo.one"]["status"] == "active"
    assert origins["https://demo.one"]["principal_name"] == "Alice"
    assert origins["https://demo.two"]["configured"] is True  # trailing slash canonicalized off
    assert "secret_encrypted" not in json.dumps(origins)


def test_forget_deletes(tmp_path):
    bs.register_provider(_demo_provider())
    db = _db(tmp_path)
    try:
        asyncio.run(
            db.upsert_auth_session(
                provider="demo",
                origin="https://demo.one",
                secret_encrypted="SECRET",
                principal_id=None,
                principal_name=None,
                principal_email=None,
                browser=None,
                status="active",
                captured_at="t",
                validated_at="t",
                last_error=None,
            )
        )
        result = asyncio.run(bs.forget(db, "demo", "https://demo.one/"))
        remaining = asyncio.run(db.list_auth_sessions("demo"))
    finally:
        asyncio.run(db.close())
    assert result["forgotten"] is True
    assert result["origin"] == "https://demo.one"
    assert remaining == []


def test_inject_cookie_for_source(tmp_path):
    bs.register_provider(_demo_provider())
    db = _db(tmp_path)
    try:
        config = {"base_url": "https://demo.one", "auth_mode": "browser_cookie"}
        injected = asyncio.run(bs.inject_cookie_for_source(db, "demo", config))
        # A non-browser-session type is a no-op.
        other = {"base_url": "x"}
        skipped = asyncio.run(bs.inject_cookie_for_source(db, "unknown_type", other))
    finally:
        asyncio.run(db.close())
    assert injected is True
    assert config["demo_cookie"] == "DEMO-COOKIE=abc"
    assert skipped is False
    assert "demo_cookie" not in other


def test_store_uploaded_dispatches_to_service(tmp_path):
    bs.register_provider(_demo_provider())
    db = _db(tmp_path)
    try:
        result = asyncio.run(
            bs.store_uploaded(
                db,
                "demo",
                base_url="https://demo.one",
                cookie_header="SESSION=x",
                browser="Chrome",
            )
        )
    finally:
        asyncio.run(db.close())
    assert result["status"] == "active"
    assert result["cookie_header"] == "SESSION=x"
    assert result["browser"] == "Chrome"


def test_unknown_provider_raises():
    try:
        bs.get_provider("does-not-exist")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown provider")
