from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading


def test_expired_teams_access_token_is_renewed_without_visible_sign_in(monkeypatch, tmp_path: Path):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserCapture

    captured_modes: list[bool] = []
    stored: list[dict] = []
    fresh_tokens = {
        CHAT_API_AUDIENCE: {
            "token": "fresh-token",
            "expiresAt": 4_102_444_800,
            "scopes": "Teams.AccessAsUser.All",
        }
    }

    class BrowserSession:
        def capture(self, *, interactive: bool, **_options):
            captured_modes.append(interactive)
            return TeamsBrowserCapture.captured(fresh_tokens)

    monkeypatch.setattr(
        TeamsAuthenticator,
        "_load_keychain_token_data",
        staticmethod(
            lambda: {
                "tokens": {
                    CHAT_API_AUDIENCE: {
                        "token": "expired-token",
                        "expiresAt": 1,
                        "scopes": "Teams.AccessAsUser.All",
                    }
                }
            }
        ),
    )
    monkeypatch.setattr(
        TeamsAuthenticator,
        "_save_keychain_token_data",
        staticmethod(lambda token_data: stored.append(token_data) is None),
    )

    token_data = TeamsAuthenticator(browser_session=BrowserSession()).authenticate(
        region="emea",
        wait_seconds=90,
    )

    assert token_data["tokens"] == fresh_tokens
    assert captured_modes == [False]
    assert stored == [token_data]


def test_valid_keychain_token_is_reused_without_starting_browser_session(monkeypatch):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator

    keychain_token_data = {
        "version": 1,
        "captured_at": "2026-07-12T00:00:00+00:00",
        "region": "emea",
        "tokens": {
            CHAT_API_AUDIENCE: {
                "token": "keychain-token",
                "expiresAt": 4_102_444_800,
                "scopes": "Teams.AccessAsUser.All",
            }
        },
    }

    class BrowserSession:
        @staticmethod
        def capture(**_options):
            raise AssertionError("A valid Keychain token must not start the Teams Browser Session")

    monkeypatch.setattr(
        TeamsAuthenticator,
        "_load_keychain_token_data",
        staticmethod(lambda: keychain_token_data),
    )

    authenticator = TeamsAuthenticator(browser_session=BrowserSession())
    token_data = authenticator.authenticate()

    assert token_data == keychain_token_data
    assert authenticator.keychain_session_available is True


def test_keychain_token_without_expiry_is_renewed_instead_of_treated_as_permanent(monkeypatch):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserCapture

    renewals: list[bool] = []
    fresh_tokens = {
        CHAT_API_AUDIENCE: {
            "token": "fresh-token",
            "expiresAt": 4_102_444_800,
            "scopes": "Teams.AccessAsUser.All",
        }
    }

    class BrowserSession:
        def capture(self, *, interactive: bool, **_options):
            renewals.append(interactive)
            return TeamsBrowserCapture.captured(fresh_tokens)

    monkeypatch.setattr(
        TeamsAuthenticator,
        "_load_keychain_token_data",
        staticmethod(lambda: {
            "tokens": {
                CHAT_API_AUDIENCE: {
                    "token": "token-without-expiry",
                    "expiresAt": 0,
                    "scopes": "Teams.AccessAsUser.All",
                }
            }
        }),
    )
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: True))

    token_data = TeamsAuthenticator(browser_session=BrowserSession()).authenticate()

    assert renewals == [False]
    assert token_data["tokens"] == fresh_tokens


def test_interactive_reauthentication_starts_only_when_silent_renewal_requires_it(monkeypatch):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserCapture

    captured_modes: list[bool] = []
    fresh_tokens = {
        CHAT_API_AUDIENCE: {
            "token": "fresh-token",
            "expiresAt": 4_102_444_800,
            "scopes": "Teams.AccessAsUser.All",
        }
    }

    class BrowserSession:
        def capture(self, *, interactive: bool, **_options):
            captured_modes.append(interactive)
            if not interactive:
                return TeamsBrowserCapture.interaction_required("SAP SSO requires interaction")
            return TeamsBrowserCapture.captured(fresh_tokens)

    monkeypatch.setattr(TeamsAuthenticator, "_load_keychain_token_data", staticmethod(lambda: None))
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: True))

    token_data = TeamsAuthenticator(browser_session=BrowserSession()).authenticate(
        region="emea",
        wait_seconds=90,
    )

    assert token_data["tokens"] == fresh_tokens
    assert captured_modes == [False, True]


def test_concurrent_authentication_shares_one_session_renewal(monkeypatch):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserCapture

    first_read = threading.Barrier(2)
    thread_reads: set[int] = set()
    state_lock = threading.Lock()
    stored: dict | None = None
    capture_count = 0
    fresh_tokens = {
        CHAT_API_AUDIENCE: {
            "token": "fresh-token",
            "expiresAt": 4_102_444_800,
            "scopes": "Teams.AccessAsUser.All",
        }
    }

    def load_keychain():
        thread_id = threading.get_ident()
        with state_lock:
            first_for_thread = thread_id not in thread_reads
            thread_reads.add(thread_id)
            current = stored
        if first_for_thread:
            first_read.wait(timeout=2)
        return current

    def save_keychain(token_data):
        nonlocal stored
        with state_lock:
            stored = token_data
        return True

    class BrowserSession:
        def capture(self, **_options):
            nonlocal capture_count
            with state_lock:
                capture_count += 1
            return TeamsBrowserCapture.captured(fresh_tokens)

    monkeypatch.setattr(TeamsAuthenticator, "_load_keychain_token_data", staticmethod(load_keychain))
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(save_keychain))

    results: list[dict] = []
    errors: list[BaseException] = []

    def authenticate():
        try:
            results.append(TeamsAuthenticator(browser_session=BrowserSession()).authenticate())
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=authenticate) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert errors == []
    assert len(results) == 2
    assert capture_count == 1
    assert results[0]["tokens"] == results[1]["tokens"] == fresh_tokens


def test_silent_session_renewal_captures_ic3_token_from_teams_request(monkeypatch, tmp_path: Path):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserSession

    token = _jwt({
        "aud": CHAT_API_AUDIENCE,
        "exp": 4_102_444_800,
        "scp": "Teams.AccessAsUser.All",
    })

    class Request:
        @staticmethod
        def headers():
            return {"authorization": f"Bearer {token}"}

    class Page:
        url = "https://teams.microsoft.com/v2/"

        def on(self, event, callback):
            assert event == "request"
            self.on_request = callback

        def goto(self, _url, **_options):
            self.on_request(Request())

        @staticmethod
        def evaluate(_script):
            return []

        @staticmethod
        def wait_for_timeout(_milliseconds):
            return None

    class Context:
        def new_page(self):
            return Page()

        @staticmethod
        def close():
            return None

    class BrowserLauncher:
        def launch_persistent_context(self, profile_dir, *, headless):
            assert profile_dir == tmp_path / "teams-browser-profile"
            assert headless is True
            return Context()

    monkeypatch.setattr(TeamsAuthenticator, "_load_keychain_token_data", staticmethod(lambda: None))
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: True))

    browser_session = TeamsBrowserSession(
        profile_dir=tmp_path / "teams-browser-profile",
        browser_launcher=BrowserLauncher(),
    )
    token_data = TeamsAuthenticator(browser_session=browser_session).authenticate()

    assert token_data["tokens"][CHAT_API_AUDIENCE] == {
        "token": token,
        "expiresAt": 4_102_444_800,
        "scopes": "Teams.AccessAsUser.All",
    }


def test_teams_access_token_near_expiry_is_renewed_before_collection(monkeypatch):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserCapture

    renewals: list[bool] = []
    near_expiry = int(datetime.now(timezone.utc).timestamp()) + 120
    fresh_tokens = {
        CHAT_API_AUDIENCE: {
            "token": "fresh-token",
            "expiresAt": 4_102_444_800,
            "scopes": "Teams.AccessAsUser.All",
        }
    }

    class BrowserSession:
        def capture(self, *, interactive: bool, **_options):
            renewals.append(interactive)
            return TeamsBrowserCapture.captured(fresh_tokens)

    monkeypatch.setattr(
        TeamsAuthenticator,
        "_load_keychain_token_data",
        staticmethod(lambda: {
            "tokens": {
                CHAT_API_AUDIENCE: {
                    "token": "near-expiry-token",
                    "expiresAt": near_expiry,
                    "scopes": "Teams.AccessAsUser.All",
                }
            }
        }),
    )
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: True))

    token_data = TeamsAuthenticator(browser_session=BrowserSession()).authenticate()

    assert renewals == [False]
    assert token_data["tokens"] == fresh_tokens


def test_rejected_teams_access_token_is_redeemed_silently_with_teams_web_session(monkeypatch, tmp_path: Path):
    from memforge.auth.teams_auth import CHAT_API_AUDIENCE, TeamsAuthenticator
    from memforge.auth.teams_browser_session import TeamsBrowserSession, TeamsTokenRefresh

    stale_token = _jwt({
        "aud": CHAT_API_AUDIENCE,
        "exp": 4_102_444_800,
        "scp": "Teams.AccessAsUser.All",
    })
    fresh_token = _jwt({
        "aud": CHAT_API_AUDIENCE,
        "exp": 4_102_444_900,
        "scp": "Teams.AccessAsUser.All",
        "renewed": True,
    })
    persisted_refresh_tokens: list[str] = []

    class Request:
        @staticmethod
        def headers():
            return {"authorization": f"Bearer {stale_token}"}

    class Page:
        url = "https://teams.microsoft.com/v2/"

        def on(self, _event, callback):
            self.on_request = callback

        def goto(self, _url, **_options):
            self.on_request(Request())

        @staticmethod
        def wait_for_timeout(_milliseconds):
            return None

        def evaluate(self, script, argument=None):
            if "Object.entries" in script:
                return [[
                    "teams-refresh-token",
                    json.dumps({
                        "credentialType": "RefreshToken",
                        "clientId": "5e3ce6c0-2b1f-4285-8d4b-75ee78787346",
                        "secret": "refresh-token-before-rotation",
                    }),
                ]]
            if "setItem" in script:
                persisted_refresh_tokens.append(argument["value"]["secret"])
                return None
            if "Object.values" in script:
                return [json.dumps({"secret": stale_token})]
            raise AssertionError(f"Unexpected browser expression: {script}")

    class Context:
        @staticmethod
        def new_page():
            return Page()

        @staticmethod
        def close():
            return None

    class BrowserLauncher:
        @staticmethod
        def launch_persistent_context(_profile_dir, *, headless):
            assert headless is True
            return Context()

    class TokenClient:
        @staticmethod
        def refresh(refresh_token):
            assert refresh_token == "refresh-token-before-rotation"
            return TeamsTokenRefresh(
                access_token=fresh_token,
                refresh_token="refresh-token-after-rotation",
            )

    monkeypatch.setattr(
        TeamsAuthenticator,
        "_load_keychain_token_data",
        staticmethod(lambda: {
            "tokens": {
                CHAT_API_AUDIENCE: {
                    "token": stale_token,
                    "expiresAt": 4_102_444_800,
                    "scopes": "Teams.AccessAsUser.All",
                }
            }
        }),
    )
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: True))

    browser_session = TeamsBrowserSession(
        profile_dir=tmp_path / "teams-browser-profile",
        browser_launcher=BrowserLauncher(),
        token_client=TokenClient(),
    )
    token_data = TeamsAuthenticator(browser_session=browser_session).authenticate(
        rejected_token_hashes={hashlib.sha256(stale_token.encode()).hexdigest()},
    )

    assert persisted_refresh_tokens == ["refresh-token-after-rotation"]
    assert token_data["tokens"][CHAT_API_AUDIENCE]["token"] == fresh_token


def _jwt(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.signature"
