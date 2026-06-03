"""Provider-agnostic local browser-session management.

Some sources can only be reached through the user's *local* browser session:
the service cannot authenticate as the user (no REST/PAT quota, SSO-gated, etc.).
The session is captured on the user's machine and stored in the shared
``auth_sessions`` table, keyed by ``(provider, origin)``.

This module owns the provider-agnostic lifecycle — list / status / forget /
refresh / sync-time cookie injection — so adding a new browser-session source
means registering one :class:`BrowserSessionProvider` descriptor, not
reimplementing session management. Each descriptor supplies the provider's
origin canonicaliser, the config key its gene reads the cookie from, and a
factory for a :class:`BrowserSessionService` that knows how to capture and
validate that provider's session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from memforge.storage.database import Database


class BrowserSessionError(RuntimeError):
    """A browser-session capture or validation failed."""


class BrowserSessionMissingError(BrowserSessionError):
    """No usable browser session is available for an origin."""


class BrowserSessionPrincipalChangedError(BrowserSessionError):
    """The validated principal for an origin differs from the stored one."""

    def __init__(self, origin: str, old_principal_id: str | None, new_principal_id: str | None) -> None:
        self.origin = origin
        self.old_principal_id = old_principal_id
        self.new_principal_id = new_principal_id
        super().__init__(f"A different principal is signed in for {origin}.")


@runtime_checkable
class BrowserSessionService(Protocol):
    """The provider-specific capture/validate surface the generic ops dispatch to."""

    async def get_status(self, base_url: str) -> dict[str, Any]: ...

    async def refresh_from_browser(
        self, *, base_url: str, browser: str | None = None, confirm_principal_change: bool = False
    ) -> dict[str, Any]: ...

    async def cookie_header_for_sync(
        self, base_url: str, *, allow_browser_refresh: bool = True, tls_config: dict | None = None
    ) -> str: ...

    async def mark_expired(self, base_url: str, error: str) -> None: ...


@dataclass(frozen=True)
class BrowserSessionProvider:
    """Everything the generic layer needs to manage one provider's sessions."""

    provider: str
    source_type: str
    label: str
    cookie_config_key: str
    canonical_origin: Callable[[str], str]
    service_factory: Callable[[Database], BrowserSessionService]
    uses_browser_session: Callable[[dict], bool]


_PROVIDERS: dict[str, BrowserSessionProvider] = {}


def register_provider(descriptor: BrowserSessionProvider) -> None:
    _PROVIDERS[descriptor.provider] = descriptor


def get_provider(provider: str) -> BrowserSessionProvider:
    try:
        return _PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown browser-session provider: {provider}") from exc


def provider_for_source_type(source_type: str) -> BrowserSessionProvider | None:
    for descriptor in _PROVIDERS.values():
        if descriptor.source_type == source_type:
            return descriptor
    return None


def registered_providers() -> list[BrowserSessionProvider]:
    return sorted(_PROVIDERS.values(), key=lambda descriptor: descriptor.provider)


def ensure_builtin_providers() -> None:
    """Import modules that register built-in providers (idempotent).

    Done lazily to avoid an import cycle: provider modules import this one.
    """
    import memforge.auth.jira_auth  # noqa: F401  (registers the jira provider)


async def list_origins(db: Database, provider: str) -> list[dict[str, Any]]:
    """Known origins for a provider: authenticated sessions + configured sources.

    Only safe fields are returned; the encrypted cookie is never included.
    """
    descriptor = get_provider(provider)
    sessions = await db.list_auth_sessions(provider)
    sources = [src for src in await db.list_sources() if src.get("type") == descriptor.source_type]

    entries: dict[str, dict[str, Any]] = {}
    for session in sessions:
        origin = str(session.get("origin") or "").strip()
        if not origin:
            continue
        entries[origin] = {
            "origin": origin,
            "status": session.get("status"),
            "principal_name": session.get("principal_name"),
            "configured": False,
            "source_name": None,
        }
    for source in sources:
        base_url = str((source.get("config") or {}).get("base_url") or "").strip()
        if not base_url:
            continue
        try:
            origin = descriptor.canonical_origin(base_url)
        except ValueError:
            continue
        entry = entries.setdefault(
            origin,
            {"origin": origin, "status": None, "principal_name": None, "configured": False, "source_name": None},
        )
        entry["configured"] = True
        entry["source_name"] = source.get("name")
    return sorted(entries.values(), key=lambda item: item["origin"])


async def status(db: Database, provider: str, base_url: str) -> dict[str, Any]:
    descriptor = get_provider(provider)
    return await descriptor.service_factory(db).get_status(base_url)


async def forget(db: Database, provider: str, base_url: str) -> dict[str, Any]:
    descriptor = get_provider(provider)
    origin = descriptor.canonical_origin(base_url)
    removed = await db.delete_auth_session(provider, origin)
    return {"ok": True, "provider": provider, "origin": origin, "forgotten": removed}


async def refresh(
    db: Database,
    provider: str,
    *,
    base_url: str,
    browser: str | None = None,
    confirm_principal_change: bool = False,
) -> dict[str, Any]:
    descriptor = get_provider(provider)
    return await descriptor.service_factory(db).refresh_from_browser(
        base_url=base_url,
        browser=browser,
        confirm_principal_change=confirm_principal_change,
    )


async def inject_cookie_for_source(db: Database, source_type: str, config: dict[str, Any]) -> bool:
    """Inject a captured session cookie into a source config before sync.

    No-op for source types that do not use a browser session, or whose config
    selects a non-browser auth mode. Returns True when a cookie was injected.
    """
    descriptor = provider_for_source_type(source_type)
    if descriptor is None or not descriptor.uses_browser_session(config):
        return False
    service = descriptor.service_factory(db)
    config[descriptor.cookie_config_key] = await service.cookie_header_for_sync(
        str(config.get("base_url") or ""),
        tls_config=config,
        allow_browser_refresh=False,
    )
    return True


async def mark_expired_for_source(db: Database, source_type: str, base_url: str, error: str) -> bool:
    """Mark a source's browser session expired after a sync failure. No-op for non-providers."""
    descriptor = provider_for_source_type(source_type)
    if descriptor is None:
        return False
    await descriptor.service_factory(db).mark_expired(base_url, error)
    return True
