"""Shared Jira browser-session authentication.

Jira browser sessions are credentials for a Jira origin, not for one source.
This module owns discovery, validation, encrypted storage, and runtime header
resolution for those sessions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from typing import Any
from urllib.parse import urlsplit

import httpx

from memforge.genes.atlassian_auth import require_https_base_url, tls_verify
from memforge.source_secrets import decrypt_secret, encrypt_secret
from memforge.storage.database import Database

logger = logging.getLogger(__name__)

JIRA_AUTH_PROVIDER = "jira"
JIRA_SESSION_ACTIVE = "active"
JIRA_SESSION_EXPIRED = "expired"
JIRA_SESSION_MISSING = "missing"
JIRA_SESSION_FAILED = "failed"

BrowserExtractor = Callable[[str, str | None], tuple[str, str]]
SessionValidator = Callable[[str, str, dict[str, Any] | None], Any]


class JiraAuthSessionError(RuntimeError):
    """Base class for Jira browser-session failures."""


class JiraAuthSessionMissingError(JiraAuthSessionError):
    """Raised when no usable Jira browser session exists for an origin."""


class JiraPrincipalChangedError(JiraAuthSessionError):
    """Raised when a refreshed browser session belongs to a different user."""

    def __init__(self, origin: str, old_principal_id: str | None, new_principal_id: str | None) -> None:
        super().__init__(
            f"Jira browser session for {origin} belongs to {new_principal_id}; "
            f"existing session belongs to {old_principal_id}. Confirm principal change to continue."
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
        browser_extractor: BrowserExtractor | None = None,
        session_validator: SessionValidator | None = None,
    ) -> None:
        self.db = db
        self._browser_extractor = browser_extractor or extract_browser_cookie_header
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

    async def refresh_from_browser(
        self,
        *,
        base_url: str,
        browser: str | None = None,
        tls_config: dict[str, Any] | None = None,
        confirm_principal_change: bool = False,
    ) -> dict[str, Any]:
        origin = canonical_jira_origin(base_url)
        try:
            cookie_header, browser_name = self._browser_extractor(origin, browser)
            principal = await _maybe_await(self._session_validator(origin, cookie_header, tls_config))
            return await self.store_validated_session(
                base_url=origin,
                cookie_header=cookie_header,
                principal=principal,
                browser=browser_name,
                confirm_principal_change=confirm_principal_change,
            )
        except JiraPrincipalChangedError:
            raise
        except Exception as exc:
            await self._record_failure(origin, str(exc))
            raise JiraAuthSessionError(str(exc)) from exc

    async def cookie_header_for_sync(
        self,
        base_url: str,
        *,
        allow_browser_refresh: bool = True,
        tls_config: dict[str, Any] | None = None,
    ) -> str:
        origin = canonical_jira_origin(base_url)
        stored = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
        if not stored and allow_browser_refresh:
            await self.refresh_from_browser(base_url=origin, tls_config=tls_config)
            stored = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)

        if not stored:
            raise JiraAuthSessionMissingError(
                f"No Jira browser session is available for {origin}. "
                "Sign in to Jira in your browser, then refresh the browser session."
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
            if allow_browser_refresh:
                await self.refresh_from_browser(base_url=origin, tls_config=tls_config)
                refreshed = await self.db.get_auth_session(JIRA_AUTH_PROVIDER, origin)
                if refreshed:
                    return decrypt_secret(refreshed["secret_encrypted"])
            raise JiraAuthSessionMissingError(
                f"Jira browser session expired for {origin}. "
                "Sign in to Jira in your browser, then refresh the browser session."
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


def extract_browser_cookie_header(origin: str, browser: str | None = None) -> tuple[str, str]:
    """Extract a Cookie header for exactly one Jira origin from local browser storage."""
    try:
        import browser_cookie3
    except ImportError as exc:
        raise JiraAuthSessionMissingError(
            "browser_cookie3 is not installed. Install the sso extra before using browser-session auth."
        ) from exc

    parsed_origin = urlsplit(canonical_jira_origin(origin))
    hostname = parsed_origin.hostname
    if not hostname:
        raise ValueError("Jira origin must include a host")

    browser_loaders = _browser_loaders(browser_cookie3, browser)
    failures: list[str] = []
    for browser_name, loader in browser_loaders:
        try:
            cookie_header = _cookie_header_from_jar(
                loader(domain_name=hostname),
                hostname=hostname,
                request_path="/rest/api/2/myself",
                is_https=parsed_origin.scheme == "https",
            )
            if cookie_header:
                return cookie_header, browser_name
            dotted_header = _cookie_header_from_jar(
                loader(domain_name=f".{hostname}"),
                hostname=hostname,
                request_path="/rest/api/2/myself",
                is_https=parsed_origin.scheme == "https",
            )
            if dotted_header:
                return dotted_header, browser_name
        except Exception as exc:
            failures.append(f"{browser_name}: {exc}")

    detail = "; ".join(failures) if failures else "no matching cookies found"
    raise JiraAuthSessionMissingError(
        f"No active Jira browser session cookies were found for {hostname}. {detail}"
    )


def _browser_loaders(browser_cookie3: Any, browser: str | None) -> list[tuple[str, Callable[..., CookieJar]]]:
    candidates = [
        ("Chrome", "chrome"),
        ("Edge", "edge"),
        ("Firefox", "firefox"),
        ("Safari", "safari"),
        ("Brave", "brave"),
    ]
    if browser:
        wanted = browser.strip().lower()
        candidates = [candidate for candidate in candidates if candidate[1] == wanted]
        if not candidates:
            raise ValueError(f"Unsupported browser for Jira session extraction: {browser}")
    loaders: list[tuple[str, Callable[..., CookieJar]]] = []
    for display_name, attr_name in candidates:
        loader = getattr(browser_cookie3, attr_name, None)
        if loader:
            loaders.append((display_name, loader))
    return loaders


def _cookie_header_from_jar(
    cookie_jar: CookieJar,
    *,
    hostname: str,
    request_path: str,
    is_https: bool,
) -> str:
    now = datetime.now(timezone.utc).timestamp()
    pairs: list[str] = []
    for cookie in cookie_jar:
        if not cookie.name or not cookie.value:
            continue
        if cookie.expires is not None and cookie.expires <= now:
            continue
        if cookie.secure and not is_https:
            continue
        if not _cookie_domain_matches(
            hostname,
            cookie.domain,
            domain_specified=bool(cookie.domain_specified),
        ):
            continue
        if not _cookie_path_matches(request_path, cookie.path or "/"):
            continue
        pairs.append(f"{cookie.name}={cookie.value}")
    return "; ".join(dict.fromkeys(pairs))


def _cookie_domain_matches(hostname: str, cookie_domain: str, *, domain_specified: bool) -> bool:
    domain = (cookie_domain or "").lower().lstrip(".")
    host = hostname.lower()
    if not domain_specified:
        return host == domain
    return host == domain or host.endswith(f".{domain}")


def _cookie_path_matches(request_path: str, cookie_path: str) -> bool:
    normalized_cookie_path = cookie_path if cookie_path.startswith("/") else f"/{cookie_path}"
    normalized_request_path = request_path if request_path.startswith("/") else f"/{request_path}"
    if normalized_cookie_path == "/":
        return True
    if normalized_request_path == normalized_cookie_path:
        return True
    if not normalized_request_path.startswith(normalized_cookie_path):
        return False
    return normalized_cookie_path.endswith("/") or normalized_request_path[len(normalized_cookie_path)] == "/"


async def validate_jira_cookie_session(
    origin: str,
    cookie_header: str,
    tls_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a browser Cookie header against Jira and return the principal."""
    headers = {
        "Accept": "application/json",
        "Cookie": cookie_header,
    }
    async with httpx.AsyncClient(
        base_url=canonical_jira_origin(origin),
        headers=headers,
        timeout=30.0,
        follow_redirects=True,
        verify=tls_verify(tls_config or {}),
    ) as client:
        response = await client.get("/rest/api/2/myself")
        if response.status_code == 401:
            raise JiraAuthSessionMissingError("Jira browser session is expired or not accepted")
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not _principal_id(data):
            raise JiraAuthSessionError("Jira /myself response did not contain a stable principal")
        return data


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
