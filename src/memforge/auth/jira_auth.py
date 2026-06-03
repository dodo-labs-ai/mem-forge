"""Shared Jira browser-session authentication.

Jira browser sessions are credentials for a Jira origin, not for one source.
This module owns discovery, validation, encrypted storage, and runtime header
resolution for those sessions.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from memforge.auth.browser_session import (
    BrowserSessionError,
    BrowserSessionMissingError,
    BrowserSessionPrincipalChangedError,
    BrowserSessionProvider,
    register_provider,
)
from memforge.genes.atlassian_auth import require_https_base_url, tls_verify
from memforge.source_secrets import decrypt_secret, encrypt_secret
from memforge.storage.database import Database

logger = logging.getLogger(__name__)

JIRA_AUTH_PROVIDER = "jira"
JIRA_SESSION_ACTIVE = "active"
JIRA_SESSION_EXPIRED = "expired"
JIRA_SESSION_MISSING = "missing"
JIRA_SESSION_FAILED = "failed"

# A container's path to a corporate Jira is often intermittently flaky (cold
# connections, dual-stack IPv6 races, VPN or DNS warmup), so transient connect or
# read failures are retried a few times before giving up, with a generous read
# budget for slow first responses.
JIRA_VALIDATE_CONNECT_TIMEOUT_SECONDS = 10.0
JIRA_VALIDATE_READ_TIMEOUT_SECONDS = 60.0
JIRA_VALIDATE_ATTEMPTS = 4
JIRA_VALIDATE_RETRY_DELAY_SECONDS = 0.5

SessionValidator = Callable[[str, str, dict[str, Any] | None], Any]


class JiraAuthSessionError(BrowserSessionError):
    """Base class for Jira browser-session failures."""


class JiraAuthSessionMissingError(JiraAuthSessionError, BrowserSessionMissingError):
    """Raised when no usable Jira browser session exists for an origin."""


class JiraPrincipalChangedError(JiraAuthSessionError, BrowserSessionPrincipalChangedError):
    """Raised when a refreshed browser session belongs to a different user."""

    def __init__(self, origin: str, old_principal_id: str | None, new_principal_id: str | None) -> None:
        BrowserSessionError.__init__(
            self,
            f"Jira browser session for {origin} belongs to {new_principal_id}; "
            f"existing session belongs to {old_principal_id}. Confirm principal change to continue.",
        )
        self.origin = origin
        self.old_principal_id = old_principal_id
        self.new_principal_id = new_principal_id


@dataclass(frozen=True)
class JiraValidatedSession:
    origin: str
    cookie_header: str
    principal: dict[str, Any]
    browser: str | None = None


def canonical_jira_origin(base_url: str) -> str:
    """Return the exact HTTPS origin used to scope shared Jira auth."""
    raw = str(base_url or "").strip().rstrip("/")
    require_https_base_url(raw, "Jira")
    parsed = urlsplit(raw)
    if not parsed.hostname:
        raise ValueError("Jira base_url must include a host")
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    port = parsed.port
    if port and not (scheme == "https" and port == 443):
        return f"{scheme}://{host}:{port}"
    return f"{scheme}://{host}"


def effective_jira_auth_mode(config: dict[str, Any]) -> str:
    """Return Jira auth mode with legacy PAT configs inferred consistently."""
    configured = config.get("auth_mode")
    if configured is None and (config.get("pat") or config.get("pat_encrypted")):
        return "pat"
    return str(configured or "browser_cookie").strip().lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _principal_id(principal: dict[str, Any]) -> str | None:
    value = principal.get("accountId") or principal.get("key") or principal.get("name")
    return str(value) if value else None


def _principal_name(principal: dict[str, Any]) -> str | None:
    value = principal.get("displayName") or principal.get("name")
    return str(value) if value else None


def _principal_email(principal: dict[str, Any]) -> str | None:
    value = principal.get("emailAddress") or principal.get("email")
    return str(value) if value else None


class JiraAuthSessionService:
    """Owns shared Jira browser sessions keyed by provider and canonical origin."""

    def __init__(
        self,
        db: Database,
        *,
        session_validator: SessionValidator | None = None,
    ) -> None:
        self.db = db
        self._session_validator = session_validator or validate_jira_cookie_session

    async def get_status(self, base_url: str) -> dict[str, Any]:
        origin = canonical_jira_origin(base_url)
        stored = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
        if not stored:
            return {
                "provider": JIRA_AUTH_PROVIDER,
                "origin": origin,
                "status": JIRA_SESSION_MISSING,
                "principal_id": None,
                "principal_name": None,
                "principal_email": None,
                "browser": None,
                "captured_at": None,
                "validated_at": None,
                "last_error": None,
            }
        return _redacted_status(stored)

    async def store_validated_session(
        self,
        *,
        base_url: str,
        cookie_header: str,
        principal: dict[str, Any],
        browser: str | None,
        confirm_principal_change: bool = False,
    ) -> dict[str, Any]:
        origin = canonical_jira_origin(base_url)
        existing = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
        old_principal_id = existing.get("principal_id") if existing else None
        new_principal_id = _principal_id(principal)
        principal_changed = bool(old_principal_id and new_principal_id and old_principal_id != new_principal_id)
        if principal_changed and not confirm_principal_change:
            raise JiraPrincipalChangedError(origin, old_principal_id, new_principal_id)

        timestamp = _now_iso()
        sources_reset = await self._source_ids_for_browser_session_origin(origin) if principal_changed else []
        await self.db.upsert_auth_session_and_reset_sources(
            provider=JIRA_AUTH_PROVIDER,
            origin=origin,
            secret_encrypted=encrypt_secret(cookie_header.strip()),
            principal_id=new_principal_id,
            principal_name=_principal_name(principal),
            principal_email=_principal_email(principal),
            browser=browser,
            status=JIRA_SESSION_ACTIVE,
            captured_at=timestamp,
            validated_at=timestamp,
            last_error=None,
            reset_source_ids=sources_reset,
        )

        status = await self.get_status(origin)
        return {
            **status,
            "principal_changed": principal_changed,
            "sources_reset": sources_reset,
        }

    async def store_uploaded_session(
        self,
        *,
        base_url: str,
        cookie_header: str,
        browser: str | None = None,
        tls_config: dict[str, Any] | None = None,
        confirm_principal_change: bool = False,
    ) -> dict[str, Any]:
        """Validate a client-uploaded cookie authoritatively, then store it."""
        origin = canonical_jira_origin(base_url)
        try:
            principal = await _maybe_await(self._session_validator(origin, cookie_header, tls_config))
        except JiraPrincipalChangedError:
            raise
        except Exception as exc:
            await self._record_failure(origin, str(exc), status=JIRA_SESSION_EXPIRED)
            raise JiraAuthSessionError(str(exc)) from exc
        return await self.store_validated_session(
            base_url=origin,
            cookie_header=cookie_header,
            principal=principal,
            browser=browser,
            confirm_principal_change=confirm_principal_change,
        )

    async def cookie_header_for_sync(
        self,
        base_url: str,
        *,
        tls_config: dict[str, Any] | None = None,
    ) -> str:
        origin = canonical_jira_origin(base_url)
        stored = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
        if not stored:
            raise JiraAuthSessionMissingError(
                f"No Jira browser session is available for {origin}. "
                "Sign in to Jira in your browser, then run `memforge adapter auth jira refresh`."
            )

        try:
            cookie_header = decrypt_secret(stored["secret_encrypted"])
            principal = await _maybe_await(self._session_validator(origin, cookie_header, tls_config))
        except Exception as exc:
            await self.db.mark_auth_session_status(
                provider=JIRA_AUTH_PROVIDER,
                origin=origin,
                status=JIRA_SESSION_EXPIRED,
                last_error=str(exc),
            )
            raise JiraAuthSessionMissingError(
                f"Jira browser session expired for {origin}. "
                "Sign in to Jira in your browser, then run `memforge adapter auth jira refresh`."
            ) from exc

        if stored.get("principal_id") and _principal_id(principal) != stored.get("principal_id"):
            await self.db.mark_auth_session_status(
                provider=JIRA_AUTH_PROVIDER,
                origin=origin,
                status=JIRA_SESSION_FAILED,
                last_error="Validated Jira principal changed during sync",
            )
            raise JiraPrincipalChangedError(origin, stored.get("principal_id"), _principal_id(principal))

        await self.db.mark_auth_session_status(
            provider=JIRA_AUTH_PROVIDER,
            origin=origin,
            status=JIRA_SESSION_ACTIVE,
            last_error=None,
        )
        return cookie_header

    async def mark_expired(self, base_url: str, error: str) -> None:
        origin = canonical_jira_origin(base_url)
        if await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin):
            await self.db.mark_auth_session_status(
                provider=JIRA_AUTH_PROVIDER,
                origin=origin,
                status=JIRA_SESSION_EXPIRED,
                last_error=error,
            )
        else:
            await self._record_failure(origin, error, status=JIRA_SESSION_EXPIRED)

    async def _record_failure(self, origin: str, error: str, status: str = JIRA_SESSION_FAILED) -> None:
        existing = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
        encrypted = existing.get("secret_encrypted") if existing else encrypt_secret("")
        await self.db.upsert_auth_session(
            provider=JIRA_AUTH_PROVIDER,
            origin=origin,
            secret_encrypted=encrypted,
            principal_id=existing.get("principal_id") if existing else None,
            principal_name=existing.get("principal_name") if existing else None,
            principal_email=existing.get("principal_email") if existing else None,
            browser=existing.get("browser") if existing else None,
            status=status,
            captured_at=existing.get("captured_at") if existing else _now_iso(),
            validated_at=existing.get("validated_at") if existing else None,
            last_error=error,
        )

    async def _source_ids_for_browser_session_origin(self, origin: str) -> list[str]:
        source_ids: list[str] = []
        for source in await self.db.list_sources():
            if source.get("type") != "jira":
                continue
            source_config = source.get("config", {})
            if effective_jira_auth_mode(source_config) != "browser_cookie":
                continue
            try:
                if canonical_jira_origin(source_config.get("base_url", "")) != origin:
                    continue
            except ValueError:
                continue
            source_ids.append(source["id"])
        return source_ids


def _redacted_status(stored: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": stored["provider"],
        "origin": stored["origin"],
        "status": stored["status"],
        "principal_id": stored.get("principal_id"),
        "principal_name": stored.get("principal_name"),
        "principal_email": stored.get("principal_email"),
        "browser": stored.get("browser"),
        "captured_at": stored.get("captured_at"),
        "validated_at": stored.get("validated_at"),
        "last_error": stored.get("last_error"),
    }


async def _probe_jira_myself(client: httpx.AsyncClient, origin: str) -> httpx.Response:
    """GET ``/myself``, retrying transient transport failures before giving up.

    A connect or read failure to a corporate Jira from inside a container is often
    intermittent (a dual-stack IPv6 race, a cold connection, VPN or DNS warmup), so
    a few quick retries usually succeed where the first attempt did not. After the
    attempts are exhausted it raises a clear, non-blank message.
    """
    last_exc: Exception | None = None
    for attempt in range(JIRA_VALIDATE_ATTEMPTS):
        try:
            return await client.get("/rest/api/2/myself")
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt + 1 < JIRA_VALIDATE_ATTEMPTS:
                await asyncio.sleep(JIRA_VALIDATE_RETRY_DELAY_SECONDS)
    detail = str(last_exc) or type(last_exc).__name__
    raise JiraAuthSessionError(
        f"Could not reach Jira at {canonical_jira_origin(origin)} to validate the session ({detail})."
    ) from last_exc


async def validate_jira_cookie_session(
    origin: str,
    cookie_header: str,
    tls_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a browser Cookie header against Jira and return the principal.

    A live session returns a JSON principal from ``/rest/api/2/myself``. An
    expired SSO session typically answers with a 200 HTML login page or a
    redirect to the IdP, so a non-JSON body is treated as "session not accepted"
    rather than surfacing as an opaque parse error.
    """
    headers = {
        "Accept": "application/json",
        "Cookie": cookie_header,
    }
    async with httpx.AsyncClient(
        base_url=canonical_jira_origin(origin),
        headers=headers,
        timeout=httpx.Timeout(JIRA_VALIDATE_READ_TIMEOUT_SECONDS, connect=JIRA_VALIDATE_CONNECT_TIMEOUT_SECONDS),
        follow_redirects=True,
        verify=tls_verify(tls_config or {}),
    ) as client:
        response = await _probe_jira_myself(client, origin)
        if response.status_code == 401:
            raise JiraAuthSessionMissingError("Jira browser session is expired or not accepted")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            raise JiraAuthSessionMissingError(
                "Jira returned a non-JSON response (likely a login page); the browser session is not accepted"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise JiraAuthSessionMissingError(
                "Jira /myself did not return JSON; the browser session is not accepted"
            ) from exc
        if not isinstance(data, dict) or not _principal_id(data):
            raise JiraAuthSessionError("Jira /myself response did not contain a stable principal")
        return data


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


# Register Jira as a browser-session provider so the generic CLI / sync layer can
# manage its sessions without any Jira-specific branching.
register_provider(
    BrowserSessionProvider(
        provider=JIRA_AUTH_PROVIDER,
        source_type="jira",
        label="Jira",
        cookie_config_key="jira_cookie",
        canonical_origin=canonical_jira_origin,
        service_factory=lambda db: JiraAuthSessionService(db),
        uses_browser_session=lambda config: effective_jira_auth_mode(config) == "browser_cookie",
    )
)
