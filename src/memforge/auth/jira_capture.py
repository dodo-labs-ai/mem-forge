"""Client-side Jira browser-session capture.

This module runs on the user's machine (the CLI), where the signed-in browser
lives. It reads the local browser cookie store and produces a Cookie header for
exactly one Jira origin. The server never imports this module, so it never
depends on ``browser_cookie3``.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from typing import Any
from urllib.parse import urlsplit

from memforge.auth.jira_auth import (
    JiraAuthSessionMissingError,
    canonical_jira_origin,
    validate_jira_cookie_session,
)
from memforge.auth.jira_browser_session import (
    JiraBrowserCaptureStatus,
    JiraBrowserSession,
    JiraBrowserSessionProtocol,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JiraCaptureResult:
    origin: str
    cookie_header: str
    browser: str | None
    principal: dict[str, Any]


async def capture_and_prevalidate(
    base_url: str,
    *,
    browser: str | None = None,
    tls_config: dict[str, Any] | None = None,
    extractor: Callable[[str, str | None], tuple[str, str]] | None = None,
    validator: Callable[..., Any] | None = None,
    interactive: bool = False,
    browser_session: JiraBrowserSessionProtocol | None = None,
    silent_timeout_seconds: int = 3,
    interactive_timeout_seconds: int = 300,
    poll_interval_seconds: float = 0.5,
) -> JiraCaptureResult:
    """Acquire and validate one Jira session, preferring silent renewal.

    The persistent MemForge browser profile is tried headlessly first. A signed-in
    system browser is the second silent path. Only an explicit interactive caller
    may open the persistent profile for visible SSO reauthentication.
    """
    origin = canonical_jira_origin(base_url)
    extract = extractor or extract_browser_cookie_header
    validate = validator or validate_jira_cookie_session
    session = browser_session or JiraBrowserSession()
    rejected_cookie_hashes: set[str] = set()
    missing_details: list[str] = []

    async def validated_result(cookie_header: str, browser_name: str | None) -> JiraCaptureResult:
        result = validate(origin, cookie_header, tls_config)
        principal = await result if inspect.isawaitable(result) else result
        return JiraCaptureResult(
            origin=origin,
            cookie_header=cookie_header,
            browser=browser_name,
            principal=principal,
        )

    silent = await asyncio.to_thread(
        session.capture,
        origin=origin,
        interactive=False,
        timeout_seconds=silent_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        rejected_cookie_hashes=rejected_cookie_hashes,
    )
    if silent.status is JiraBrowserCaptureStatus.CAPTURED and silent.cookie_header:
        try:
            return await validated_result(silent.cookie_header, silent.browser)
        except JiraAuthSessionMissingError as exc:
            rejected_cookie_hashes.add(_cookie_hash(silent.cookie_header))
            missing_details.append(str(exc))
    elif silent.detail:
        missing_details.append(silent.detail)

    try:
        cookie_header, browser_name = extract(origin, browser)
    except JiraAuthSessionMissingError as exc:
        missing_details.append(str(exc))
    else:
        try:
            captured = await validated_result(cookie_header, browser_name)
        except JiraAuthSessionMissingError as exc:
            rejected_cookie_hashes.add(_cookie_hash(cookie_header))
            missing_details.append(str(exc))
        else:
            try:
                await asyncio.to_thread(session.store, origin=origin, cookie_header=cookie_header)
            except Exception as exc:
                logger.warning("Unable to persist the validated Jira session in the browser profile: %s", exc)
            return captured

    if interactive:
        visible = await asyncio.to_thread(
            session.capture,
            origin=origin,
            interactive=True,
            timeout_seconds=interactive_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            rejected_cookie_hashes=rejected_cookie_hashes,
        )
        if visible.status is JiraBrowserCaptureStatus.CAPTURED and visible.cookie_header:
            return await validated_result(visible.cookie_header, visible.browser)
        if visible.detail:
            missing_details.append(visible.detail)

    detail = next((value for value in reversed(missing_details) if value), "No active Jira session was found")
    raise JiraAuthSessionMissingError(detail)


def _cookie_hash(cookie_header: str) -> str:
    return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()


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
        f"No active Jira browser session cookies were found for {hostname}. {detail}. "
        "If your OS asked to allow keychain or keyring access to read the browser, approve it and retry."
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
