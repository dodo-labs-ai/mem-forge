"""Browser boundary for acquiring Teams access tokens."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Protocol

import httpx


logger = logging.getLogger(__name__)

CHAT_API_AUDIENCE = "https://ic3.teams.office.com"
GRAPH_API_AUDIENCE = "https://graph.microsoft.com"
CAPTURED_AUDIENCES = frozenset({CHAT_API_AUDIENCE, GRAPH_API_AUDIENCE})
TEAMS_WEB_CLIENT_ID = "5e3ce6c0-2b1f-4285-8d4b-75ee78787346"
ENTRA_TOKEN_URL = "https://login.microsoftonline.com/common/oauth2/v2.0/token"
MINIMUM_CAPTURE_VALIDITY_SECONDS = 300
TEAMS_WEB_URL = "https://teams.microsoft.com/v2/"
DEFAULT_PROFILE_DIR = Path.home() / ".memforge" / "browser-profiles" / "teams"


class TeamsBrowserCaptureStatus(StrEnum):
    CAPTURED = "captured"
    INTERACTION_REQUIRED = "interaction_required"
    FAILED = "failed"


@dataclass(frozen=True)
class TeamsBrowserCapture:
    status: TeamsBrowserCaptureStatus
    tokens: dict[str, dict] | None = None
    detail: str | None = None

    @classmethod
    def captured(cls, tokens: dict[str, dict]) -> TeamsBrowserCapture:
        return cls(status=TeamsBrowserCaptureStatus.CAPTURED, tokens=tokens)

    @classmethod
    def interaction_required(cls, detail: str | None = None) -> TeamsBrowserCapture:
        return cls(status=TeamsBrowserCaptureStatus.INTERACTION_REQUIRED, detail=detail)

    @classmethod
    def failed(cls, detail: str) -> TeamsBrowserCapture:
        return cls(status=TeamsBrowserCaptureStatus.FAILED, detail=detail)


@dataclass(frozen=True)
class TeamsTokenRefresh:
    access_token: str
    refresh_token: str | None = None


class TeamsBrowserSessionProtocol(Protocol):
    def capture(
        self,
        *,
        interactive: bool,
        timeout_seconds: int,
        poll_interval_seconds: float,
        rejected_token_hashes: set[str],
    ) -> TeamsBrowserCapture: ...


class BrowserLauncherProtocol(Protocol):
    def launch_persistent_context(self, profile_dir: Path, *, headless: bool) -> Any: ...


class TeamsTokenClientProtocol(Protocol):
    def refresh(self, refresh_token: str) -> TeamsTokenRefresh | None: ...


class TeamsBrowserSession:
    def __init__(
        self,
        *,
        profile_dir: Path = DEFAULT_PROFILE_DIR,
        browser_launcher: BrowserLauncherProtocol | None = None,
        token_client: TeamsTokenClientProtocol | None = None,
    ) -> None:
        self._profile_dir = profile_dir
        self._browser_launcher = browser_launcher or PlaywrightBrowserLauncher()
        self._token_client = token_client or EntraTeamsTokenClient()

    def capture(
        self,
        *,
        interactive: bool,
        timeout_seconds: int,
        poll_interval_seconds: float,
        rejected_token_hashes: set[str],
    ) -> TeamsBrowserCapture:
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._profile_dir.chmod(0o700)

        try:
            context = self._browser_launcher.launch_persistent_context(
                self._profile_dir,
                headless=not interactive,
            )
        except Exception as exc:
            logger.warning("Unable to launch the Teams browser session: %s", exc)
            return TeamsBrowserCapture.failed("Unable to launch the Teams browser session")

        tokens: dict[str, dict] = {}
        try:
            page = context.new_page()

            def capture_request(request: Any) -> None:
                headers = request.headers() if callable(request.headers) else request.headers
                authorization = headers.get("authorization", "")
                raw = authorization[7:] if authorization.startswith("Bearer ") else ""
                self._capture_token(tokens, raw, rejected_token_hashes)

            page.on("request", capture_request)
            try:
                page.goto(TEAMS_WEB_URL, wait_until="domcontentloaded", timeout=60_000)
            except Exception:
                return TeamsBrowserCapture.failed("Unable to load Teams Web for session renewal")

            if CHAT_API_AUDIENCE in tokens:
                return TeamsBrowserCapture.captured(tokens)
            if not interactive and self._renew_from_teams_web_session(page, tokens, rejected_token_hashes):
                return TeamsBrowserCapture.captured(tokens)

            deadline = time.monotonic() + max(int(timeout_seconds), 0)
            while True:
                self._capture_local_storage_tokens(page, tokens, rejected_token_hashes)
                if CHAT_API_AUDIENCE in tokens:
                    return TeamsBrowserCapture.captured(tokens)
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(max(int(poll_interval_seconds * 1000), 100))

            if not interactive:
                return TeamsBrowserCapture.interaction_required(
                    "The Teams Browser Session requires Interactive Reauthentication"
                )
            return TeamsBrowserCapture.failed("Teams sign-in was not completed before the authentication window timed out")
        finally:
            context.close()

    @staticmethod
    def _capture_local_storage_tokens(
        page: Any,
        tokens: dict[str, dict],
        rejected_token_hashes: set[str],
    ) -> None:
        try:
            values = page.evaluate("Object.values(window.localStorage)")
        except Exception:
            return
        if not isinstance(values, list):
            return
        for value in values:
            for raw in _token_candidates(value):
                TeamsBrowserSession._capture_token(tokens, raw, rejected_token_hashes)

    def _renew_from_teams_web_session(
        self,
        page: Any,
        tokens: dict[str, dict],
        rejected_token_hashes: set[str],
    ) -> bool:
        try:
            entries = page.evaluate("Object.entries(window.localStorage)")
        except Exception:
            return False
        if not isinstance(entries, list):
            return False

        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
                continue
            try:
                record = json.loads(entry[1])
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            if record.get("credentialType") != "RefreshToken" or record.get("clientId") != TEAMS_WEB_CLIENT_ID:
                continue
            refresh_token = record.get("secret")
            if not isinstance(refresh_token, str) or not refresh_token:
                continue

            refreshed = self._token_client.refresh(refresh_token)
            if refreshed is None:
                return False
            if refreshed.refresh_token and refreshed.refresh_token != refresh_token:
                record["secret"] = refreshed.refresh_token
                try:
                    page.evaluate(
                        "entry => window.localStorage.setItem(entry.key, JSON.stringify(entry.value))",
                        {"key": entry[0], "value": record},
                    )
                except Exception:
                    return False
            self._capture_token(tokens, refreshed.access_token, rejected_token_hashes)
            return CHAT_API_AUDIENCE in tokens
        return False

    @staticmethod
    def _capture_token(tokens: dict[str, dict], raw: str, rejected_token_hashes: set[str]) -> None:
        if not raw.startswith("eyJ"):
            return
        if hashlib.sha256(raw.encode("utf-8")).hexdigest() in rejected_token_hashes:
            return
        claims = _decode_jwt_claims(raw)
        audience = claims.get("aud")
        expires_at = claims.get("exp")
        if audience not in CAPTURED_AUDIENCES:
            return
        if not isinstance(expires_at, (int, float)):
            return
        if expires_at <= time.time() + MINIMUM_CAPTURE_VALIDITY_SECONDS:
            return
        tokens[audience] = {
            "token": raw,
            "expiresAt": int(expires_at),
            "scopes": str(claims.get("scp") or ""),
        }


class PlaywrightBrowserLauncher:
    def launch_persistent_context(self, profile_dir: Path, *, headless: bool) -> Any:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is required for Teams browser authentication") from exc

        playwright = sync_playwright().start()
        try:
            context = playwright.chromium.launch_persistent_context(
                str(profile_dir),
                channel="chrome",
                headless=headless,
                args=["--disable-blink-features=AutomationControlled"],
            )
        except Exception:
            playwright.stop()
            raise
        return _PlaywrightContext(context, playwright)


class EntraTeamsTokenClient:
    def refresh(self, refresh_token: str) -> TeamsTokenRefresh | None:
        try:
            response = httpx.post(
                ENTRA_TOKEN_URL,
                headers={"Origin": "https://teams.microsoft.com"},
                data={
                    "client_id": TEAMS_WEB_CLIENT_ID,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "scope": f"{CHAT_API_AUDIENCE}/.default offline_access",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        access_token = payload.get("access_token") if isinstance(payload, dict) else None
        rotated_refresh_token = payload.get("refresh_token") if isinstance(payload, dict) else None
        if not isinstance(access_token, str) or not access_token:
            return None
        return TeamsTokenRefresh(
            access_token=access_token,
            refresh_token=rotated_refresh_token if isinstance(rotated_refresh_token, str) else None,
        )


class _PlaywrightContext:
    def __init__(self, context: Any, playwright: Any) -> None:
        self._context = context
        self._playwright = playwright

    def new_page(self) -> Any:
        return self._context.new_page()

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._playwright.stop()


def _token_candidates(value: Any) -> list[str]:
    if isinstance(value, str):
        if value.startswith("eyJ"):
            return [value]
        try:
            return _token_candidates(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(value, dict):
        candidates: list[str] = []
        for nested in value.values():
            candidates.extend(_token_candidates(nested))
        return candidates
    if isinstance(value, list):
        candidates = []
        for nested in value:
            candidates.extend(_token_candidates(nested))
        return candidates
    return []


def _decode_jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload))
        return decoded if isinstance(decoded, dict) else {}
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}
