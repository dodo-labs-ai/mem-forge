"""Persistent browser boundary for silent-first Jira SSO renewal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Protocol
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

DEFAULT_PROFILE_ROOT = Path.home() / ".memforge" / "browser-profiles" / "jira"
JIRA_MYSELF_PATH = "/rest/api/2/myself"
JIRA_PROFILE_BROWSER = "MemForge Chrome profile"
JIRA_KEYCHAIN_SERVICE = "memforge-jira-browser-session"
_PROFILE_LOCKS: dict[Path, threading.Lock] = {}
_PROFILE_LOCKS_GUARD = threading.Lock()


class JiraBrowserCaptureStatus(StrEnum):
    CAPTURED = "captured"
    INTERACTION_REQUIRED = "interaction_required"
    FAILED = "failed"


@dataclass(frozen=True)
class JiraBrowserCapture:
    status: JiraBrowserCaptureStatus
    cookie_header: str | None = None
    browser: str | None = None
    detail: str | None = None

    @classmethod
    def captured(cls, cookie_header: str) -> JiraBrowserCapture:
        return cls(
            status=JiraBrowserCaptureStatus.CAPTURED,
            cookie_header=cookie_header,
            browser=JIRA_PROFILE_BROWSER,
        )

    @classmethod
    def interaction_required(cls, detail: str | None = None) -> JiraBrowserCapture:
        return cls(status=JiraBrowserCaptureStatus.INTERACTION_REQUIRED, detail=detail)

    @classmethod
    def failed(cls, detail: str) -> JiraBrowserCapture:
        return cls(status=JiraBrowserCaptureStatus.FAILED, detail=detail)


class JiraBrowserSessionProtocol(Protocol):
    def capture(
        self,
        *,
        origin: str,
        interactive: bool,
        timeout_seconds: int,
        poll_interval_seconds: float,
        rejected_cookie_hashes: set[str],
    ) -> JiraBrowserCapture: ...

    def store(self, *, origin: str, cookie_header: str) -> None: ...


class BrowserLauncherProtocol(Protocol):
    def launch_persistent_context(self, profile_dir: Path, *, headless: bool) -> Any: ...


class JiraCookieStoreProtocol(Protocol):
    def load(self, origin: str) -> str | None: ...

    def save(self, origin: str, cookie_header: str) -> None: ...

    def delete(self, origin: str) -> None: ...


class JiraBrowserSession:
    def __init__(
        self,
        *,
        profile_root: Path = DEFAULT_PROFILE_ROOT,
        browser_launcher: BrowserLauncherProtocol | None = None,
        cookie_store: JiraCookieStoreProtocol | None = None,
    ) -> None:
        self._profile_root = profile_root
        self._browser_launcher = browser_launcher or PlaywrightBrowserLauncher()
        self._cookie_store = cookie_store or KeyringJiraCookieStore()

    def capture(
        self,
        *,
        origin: str,
        interactive: bool,
        timeout_seconds: int,
        poll_interval_seconds: float,
        rejected_cookie_hashes: set[str],
    ) -> JiraBrowserCapture:
        profile_dir = jira_profile_dir(self._profile_root, origin)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_dir.chmod(0o700)

        with _profile_lock(profile_dir):
            return self._capture_locked(
                profile_dir=profile_dir,
                origin=origin,
                interactive=interactive,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                rejected_cookie_hashes=rejected_cookie_hashes,
            )

    def _capture_locked(
        self,
        *,
        profile_dir: Path,
        origin: str,
        interactive: bool,
        timeout_seconds: int,
        poll_interval_seconds: float,
        rejected_cookie_hashes: set[str],
    ) -> JiraBrowserCapture:

        try:
            context = self._browser_launcher.launch_persistent_context(
                profile_dir,
                headless=not interactive,
            )
        except Exception as exc:
            logger.warning("Unable to launch the Jira browser session: %s", exc)
            return JiraBrowserCapture.failed("Unable to launch the Jira browser session")

        request_url = f"{origin.rstrip('/')}{JIRA_MYSELF_PATH}"
        try:
            stored_cookie = self._cookie_store.load(origin)
            if stored_cookie:
                context.add_cookies(_cookies_from_header(origin, stored_cookie))
            page = context.new_page()
            try:
                page.goto(request_url, wait_until="domcontentloaded", timeout=60_000)
            except Exception as exc:
                logger.warning("Unable to load Jira for session renewal: %s", exc)
                return JiraBrowserCapture.failed("Unable to load Jira for session renewal")

            deadline = time.monotonic() + max(int(timeout_seconds), 0)
            while True:
                cookie_header = _cookie_header_from_context(context, request_url)
                if (
                    cookie_header
                    and _cookie_hash(cookie_header) not in rejected_cookie_hashes
                    and context.jira_session_is_active(request_url)
                ):
                    try:
                        self._cookie_store.save(origin, cookie_header)
                    except Exception as exc:
                        logger.warning("Unable to update the Jira browser session in Keychain: %s", exc)
                    return JiraBrowserCapture.captured(cookie_header)
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(max(int(poll_interval_seconds * 1000), 100))

            if not interactive:
                return JiraBrowserCapture.interaction_required(
                    "The Jira browser profile requires interactive reauthentication"
                )
            return JiraBrowserCapture.failed(
                "Jira sign-in was not completed before the authentication window timed out"
            )
        finally:
            context.close()

    def store(self, *, origin: str, cookie_header: str) -> None:
        self._cookie_store.save(origin, cookie_header)

    def forget(self, *, origin: str) -> None:
        self._cookie_store.delete(origin)


def _profile_lock(profile_dir: Path) -> threading.Lock:
    resolved = profile_dir.resolve()
    with _PROFILE_LOCKS_GUARD:
        return _PROFILE_LOCKS.setdefault(resolved, threading.Lock())


def jira_profile_dir(profile_root: Path, origin: str) -> Path:
    parsed = urlsplit(origin)
    host = parsed.hostname or "jira"
    safe_host = re.sub(r"[^a-zA-Z0-9._-]+", "-", host).strip("-") or "jira"
    origin_hash = hashlib.sha256(origin.encode("utf-8")).hexdigest()[:10]
    return profile_root / f"{safe_host}-{origin_hash}"


def _cookie_header_from_context(context: Any, request_url: str) -> str:
    now = time.time()
    pairs: list[str] = []
    for cookie in context.cookies([request_url]):
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        expires = cookie.get("expires")
        if not name or not value:
            continue
        if isinstance(expires, (int, float)) and expires > 0 and expires <= now:
            continue
        pairs.append(f"{name}={value}")
    return "; ".join(dict.fromkeys(pairs))


def _cookies_from_header(origin: str, cookie_header: str) -> list[dict[str, str]]:
    cookies: list[dict[str, str]] = []
    for pair in cookie_header.split(";"):
        name, separator, value = pair.strip().partition("=")
        if separator and name and value:
            cookies.append({"name": name, "value": value, "url": origin})
    return cookies


def _cookie_hash(cookie_header: str) -> str:
    return hashlib.sha256(cookie_header.encode("utf-8")).hexdigest()


class PlaywrightBrowserLauncher:
    def launch_persistent_context(self, profile_dir: Path, *, headless: bool) -> Any:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Playwright is required for Jira browser authentication") from exc

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


class KeyringJiraCookieStore:
    def load(self, origin: str) -> str | None:
        try:
            import keyring

            return keyring.get_password(JIRA_KEYCHAIN_SERVICE, origin)
        except Exception as exc:
            logger.warning("Unable to read the Jira browser session from Keychain: %s", exc)
            return None

    def save(self, origin: str, cookie_header: str) -> None:
        import keyring

        keyring.set_password(JIRA_KEYCHAIN_SERVICE, origin, cookie_header)

    def delete(self, origin: str) -> None:
        import keyring

        try:
            keyring.delete_password(JIRA_KEYCHAIN_SERVICE, origin)
        except keyring.errors.PasswordDeleteError:
            pass


class _PlaywrightContext:
    def __init__(self, context: Any, playwright: Any) -> None:
        self._context = context
        self._playwright = playwright

    def new_page(self) -> Any:
        return self._context.new_page()

    def cookies(self, urls: list[str]) -> list[dict[str, Any]]:
        return self._context.cookies(urls)

    def add_cookies(self, cookies: list[dict[str, str]]) -> None:
        self._context.add_cookies(cookies)

    def jira_session_is_active(self, request_url: str) -> bool:
        try:
            response = self._context.request.get(request_url, timeout=30_000)
            if response.status != 200:
                return False
            content_type = str(response.headers.get("content-type") or "").lower()
            if "application/json" not in content_type:
                return False
            payload = response.json()
        except Exception:
            return False
        return isinstance(payload, dict) and any(payload.get(key) for key in ("accountId", "name", "key"))

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            self._playwright.stop()
