"""Teams access-token cache and renewal orchestration."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone

from memforge.auth.teams_browser_session import (
    CHAT_API_AUDIENCE,
    TeamsBrowserCaptureStatus,
    TeamsBrowserSession,
    TeamsBrowserSessionProtocol,
)

logger = logging.getLogger(__name__)

__all__ = ["TeamsAuthenticator"]

KEYCHAIN_SERVICE = "memforge-teams-session"
KEYCHAIN_USERNAME = "default"

RENEWAL_WINDOW_SECONDS = 300

_SESSION_RENEWAL_LOCK = threading.Lock()


class TeamsAuthenticator:
    """Reuse a cached Teams Access Token or renew it through Teams Web."""

    def __init__(self, browser_session: TeamsBrowserSessionProtocol | None = None) -> None:
        self.keychain_session_available = False
        self._browser_session = browser_session or TeamsBrowserSession()

    def authenticate(
        self,
        region: str = "emea",
        *,
        wait_seconds: int = 0,
        poll_interval_seconds: float = 2.0,
        rejected_token_hashes: set[str] | None = None,
    ) -> dict:
        """Reuse a cached token, then renew silently before requesting interaction.

        Returns:
            Token data dict with version, captured_at, region, tokens.
        """
        rejected_token_hashes = rejected_token_hashes or set()

        keychain_data = self._load_keychain_token_data()
        tokens = self._valid_tokens_from_data(keychain_data, minimum_validity_seconds=RENEWAL_WINDOW_SECONDS)
        if tokens and not self._has_rejected_token(tokens, rejected_token_hashes):
            self.keychain_session_available = True
            return keychain_data  # type: ignore[return-value]

        with _SESSION_RENEWAL_LOCK:
            keychain_data = self._load_keychain_token_data()
            tokens = self._valid_tokens_from_data(keychain_data, minimum_validity_seconds=RENEWAL_WINDOW_SECONDS)
            if tokens and not self._has_rejected_token(tokens, rejected_token_hashes):
                self.keychain_session_available = True
                return keychain_data  # type: ignore[return-value]

            capture = self._browser_session.capture(
                interactive=False,
                timeout_seconds=20,
                poll_interval_seconds=poll_interval_seconds,
                rejected_token_hashes=rejected_token_hashes,
            )
            if capture.status is TeamsBrowserCaptureStatus.INTERACTION_REQUIRED:
                capture = self._browser_session.capture(
                    interactive=True,
                    timeout_seconds=max(int(wait_seconds), 0),
                    poll_interval_seconds=poll_interval_seconds,
                    rejected_token_hashes=rejected_token_hashes,
                )
            if capture.status is not TeamsBrowserCaptureStatus.CAPTURED or not capture.tokens:
                raise RuntimeError(capture.detail or "Unable to renew the Teams session")
            cached_tokens = keychain_data.get("tokens") if isinstance(keychain_data, dict) else None
            tokens = self._preserve_usable_tokens(cached_tokens, rejected_token_hashes)
            tokens.update(capture.tokens)

            token_data = {
                "version": 1,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "region": region,
                "tokens": tokens,
            }

            self.keychain_session_available = self.save_tokens(token_data)
            if self.keychain_session_available:
                logger.info("Saved %d Teams access tokens to the OS keychain", len(tokens))
            else:
                logger.warning("Teams access tokens are available for this operation but were not persisted")
            return token_data

    @staticmethod
    def _preserve_usable_tokens(tokens: object, rejected_token_hashes: set[str]) -> dict[str, dict]:
        if not isinstance(tokens, dict):
            return {}
        validity = TeamsAuthenticator.check_token_expiry(tokens)
        return {
            audience: entry
            for audience, entry in tokens.items()
            if isinstance(audience, str)
            and isinstance(entry, dict)
            and validity.get(audience, False)
            and not TeamsAuthenticator._has_rejected_token({audience: entry}, rejected_token_hashes)
        }

    @staticmethod
    def _has_rejected_token(tokens: dict[str, dict], rejected_hashes: set[str]) -> bool:
        if not rejected_hashes:
            return False
        for entry in tokens.values():
            token = entry.get("token") if isinstance(entry, dict) else entry
            if isinstance(token, str) and hashlib.sha256(token.encode("utf-8")).hexdigest() in rejected_hashes:
                return True
        return False

    @staticmethod
    def save_tokens(token_data: dict) -> bool:
        """Save token data and return whether the OS keychain write succeeded."""
        return TeamsAuthenticator._save_keychain_token_data(token_data)

    @staticmethod
    def _load_keychain_token_data() -> dict | None:
        try:
            import keyring

            value = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_USERNAME)
            if not value:
                return None
            data = json.loads(value)
            return data if isinstance(data, dict) else None
        except Exception:
            logger.warning("Unable to read the Teams Access Token from the OS keychain")
            return None

    @staticmethod
    def _save_keychain_token_data(token_data: dict) -> bool:
        try:
            import keyring

            keyring.set_password(
                KEYCHAIN_SERVICE,
                KEYCHAIN_USERNAME,
                json.dumps(token_data, separators=(",", ":")),
            )
            return True
        except Exception:
            logger.warning("Unable to save the Teams Access Token to the OS keychain")
            return False

    @staticmethod
    def _valid_tokens_from_data(
        token_data: dict | None,
        *,
        minimum_validity_seconds: int = 0,
    ) -> dict | None:
        if not isinstance(token_data, dict):
            return None
        tokens = token_data.get("tokens")
        if not isinstance(tokens, dict) or not tokens:
            return None
        validity = TeamsAuthenticator.check_token_expiry(
            tokens,
            minimum_validity_seconds=minimum_validity_seconds,
        )
        return tokens if validity.get(CHAT_API_AUDIENCE, False) else None

    @staticmethod
    def load_tokens() -> dict | None:
        """Load tokens from available sources.

        This read-only helper never starts a browser session.
        """
        keychain_data = TeamsAuthenticator._load_keychain_token_data()
        tokens = TeamsAuthenticator._valid_tokens_from_data(keychain_data)
        if tokens:
            logger.debug("Loaded Teams session from the OS keychain")
            return tokens

        return None

    @staticmethod
    def get_token_for_audience(tokens: dict, audience: str) -> str | None:
        """Get a specific token by audience URL."""
        entry = tokens.get(audience)
        if entry:
            return entry.get("token") if isinstance(entry, dict) else entry
        return None

    @staticmethod
    def check_token_expiry(tokens: dict, *, minimum_validity_seconds: int = 0) -> dict[str, bool]:
        """Check which tokens are expired. Returns {audience: is_valid}."""
        now = datetime.now(timezone.utc).timestamp()
        result = {}
        for audience, entry in tokens.items():
            if isinstance(entry, dict):
                expires = entry.get("expiresAt", 0)
                result[audience] = (
                    isinstance(expires, (int, float))
                    and expires > now + max(int(minimum_validity_seconds), 0)
                )
            else:
                result[audience] = False
        return result
