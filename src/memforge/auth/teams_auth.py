"""Teams OAuth token capture — extracts tokens from Chrome browser cookies.

Primary approach: extract Bearer tokens from Chrome cookies for
teams.microsoft.com. This requires the user to have an active Teams session
in Chrome. If no valid tokens are found, opens the system browser to the
Teams login page for the user to authenticate.

Token location: ~/.memforge/tokens/teams.json
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)

__all__ = ["TeamsAuthenticator"]

TOKEN_DIR = Path.home() / ".memforge" / "tokens"
TOKEN_FILE = TOKEN_DIR / "teams.json"

CHAT_API_AUDIENCE = "https://ic3.teams.office.com"
_TEAMS_LOGIN_URL = "https://teams.microsoft.com/v2/"


def _open_teams_login() -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", _TEAMS_LOGIN_URL], check=False)
    else:
        import webbrowser

        webbrowser.open(_TEAMS_LOGIN_URL)


class TeamsAuthenticator:
    """Extracts Teams OAuth tokens from Chrome cookies or cached files."""

    def authenticate(
        self,
        region: str = "emea",
        *,
        wait_seconds: int = 0,
        poll_interval_seconds: float = 2.0,
        rejected_token_hashes: set[str] | None = None,
    ) -> dict:
        """Extract tokens from Chrome cookies. Opens browser if no session found.

        Returns:
            Token data dict with version, captured_at, region, tokens.
        """
        # Try extracting from Chrome cookies first
        rejected_token_hashes = rejected_token_hashes or set()
        tokens = self._extract_from_chrome()
        if tokens and self._has_rejected_token(tokens, rejected_token_hashes):
            tokens = None

        if not tokens:
            # No valid tokens — open browser for user to log in
            logger.info("No valid Teams session in Chrome. Opening browser...")
            _open_teams_login()
            deadline = time.monotonic() + max(int(wait_seconds), 0)
            while time.monotonic() < deadline:
                time.sleep(max(float(poll_interval_seconds), 0.1))
                tokens = self._extract_from_chrome()
                if tokens and not self._has_rejected_token(tokens, rejected_token_hashes):
                    break
                tokens = None
            if not tokens:
                raise RuntimeError(
                    "No active Teams session found in Chrome.\n"
                    "A browser window has been opened to Teams.\n"
                    "Log in, then try again."
                )

        token_data = {
            "version": 1,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "region": region,
            "tokens": tokens,
        }

        self.save_tokens(token_data)
        logger.info("Saved %d tokens to %s", len(tokens), TOKEN_FILE)
        return token_data

    def _extract_from_chrome(self) -> dict[str, dict] | None:
        """Extract Teams Bearer tokens from Chrome cookies."""
        try:
            import browser_cookie3
        except ImportError:
            logger.warning("browser_cookie3 not installed — cannot extract Chrome cookies")
            return None

        tokens: dict[str, dict] = {}

        try:
            cj = browser_cookie3.chrome(domain_name=".teams.microsoft.com")
            for cookie in cj:
                if not cookie.value:
                    continue

                # Look for Bearer token cookies
                raw = unquote(cookie.value)
                if raw.startswith("Bearer "):
                    raw = raw[7:]

                if not raw.startswith("eyJ"):
                    continue

                # Decode JWT to get audience and expiry
                aud = self._decode_jwt_field(raw, "aud")
                exp = self._decode_jwt_field(raw, "exp")
                scp = self._decode_jwt_field(raw, "scp")

                if not aud:
                    continue

                # Skip expired tokens
                if exp and int(exp) < datetime.now(timezone.utc).timestamp():
                    logger.debug("Skipping expired token for %s", aud)
                    continue

                # Deduplicate by audience (keep first seen)
                if aud not in tokens:
                    tokens[aud] = {
                        "token": raw,
                        "expiresAt": int(exp) if exp else 0,
                        "scopes": str(scp) if scp else "",
                    }
                    logger.info("Extracted token: %s (scopes: %s)", aud, scp)

        except Exception as e:
            logger.warning("Chrome cookie extraction failed: %s", e)
            return None

        return tokens if tokens else None

    @staticmethod
    def _decode_jwt_field(token: str, field: str):
        """Decode a single field from a JWT payload without verification."""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return None
            payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
            decoded = json.loads(base64.b64decode(payload))
            return decoded.get(field)
        except Exception:
            return None

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
    def save_tokens(token_data: dict) -> None:
        """Save token data to disk."""
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(token_data, indent=2))
        TOKEN_FILE.chmod(0o600)

    @staticmethod
    def load_tokens() -> dict | None:
        """Load tokens from available sources.

        Search order:
        1. ~/.memforge/tokens/teams.json (cached)
        2. Chrome cookies (live extraction)
        """
        # 1. Cached tokens
        if TOKEN_FILE.exists():
            try:
                data = json.loads(TOKEN_FILE.read_text())
                tokens = data.get("tokens", {})
                if tokens:
                    now = datetime.now(timezone.utc).timestamp()
                    has_valid = any(
                        (t.get("expiresAt", 0) == 0 or t.get("expiresAt", 0) > now)
                        for t in tokens.values()
                        if isinstance(t, dict)
                    )
                    if has_valid:
                        logger.debug("Loaded tokens from %s", TOKEN_FILE)
                        return tokens
                    logger.debug("Cached tokens expired, trying Chrome...")
            except Exception:
                logger.warning("Failed to read %s", TOKEN_FILE, exc_info=True)

        # 2. Live Chrome cookie extraction
        auth = TeamsAuthenticator()
        chrome_tokens = auth._extract_from_chrome()
        if chrome_tokens:
            auth.save_tokens({
                "version": 1,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "region": "emea",
                "tokens": chrome_tokens,
            })
            return chrome_tokens

        return None

    @staticmethod
    def get_token_for_audience(tokens: dict, audience: str) -> str | None:
        """Get a specific token by audience URL."""
        entry = tokens.get(audience)
        if entry:
            return entry.get("token") if isinstance(entry, dict) else entry
        return None

    @staticmethod
    def check_token_expiry(tokens: dict) -> dict[str, bool]:
        """Check which tokens are expired. Returns {audience: is_valid}."""
        now = datetime.now(timezone.utc).timestamp()
        result = {}
        for audience, entry in tokens.items():
            if isinstance(entry, dict):
                expires = entry.get("expiresAt", 0)
                result[audience] = expires == 0 or expires > now
            else:
                result[audience] = True
        return result
