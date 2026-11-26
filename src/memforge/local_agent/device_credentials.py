"""The provider credentials a device holds for its device Sources (ADR 0058).

Each secret lives in the operating-system keyring, under one entry per
Source and field. The keyring cannot be listed, so an index beside the daemon
configuration names each Source this device holds credentials for and the
fields it stored, never their values. The index is what the device report
names: a ``document_acquisition`` job of a Source goes only to a device that
holds its credentials. A Source whose type keeps no secret, or whose
configuration signs in with a browser session, is held with no fields, which
marks this device as the one that acquires it.

A browser session lives in the keyring per provider and origin, because one
sign-in serves every Source of that site. The device's Gene reads a Source's
secrets and session from here (:meth:`DeviceCredentialStore.gene_config`),
and the session is checked against the provider before each run
(:func:`check_browser_session`), on the device, never on the server.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memforge.local_agent.service import DaemonServicePaths, SecretStore

__all__ = [
    "DEVICE_CREDENTIALS_KEYRING_SERVICE",
    "DEVICE_SESSIONS_KEYRING_SERVICE",
    "DeviceCredentialStore",
    "DeviceCredentialsMissing",
    "check_browser_session",
    "credentials_command",
    "device_credentials_path",
]

DEVICE_CREDENTIALS_KEYRING_SERVICE = "memforge-device-source"
DEVICE_SESSIONS_KEYRING_SERVICE = "memforge-device-session"
DEVICE_CREDENTIALS_FILENAME = "device-credentials.json"
DEVICE_CREDENTIALS_VERSION = 1


def device_credentials_path() -> Path:
    """Where this installation keeps its credential index: beside the daemon service configuration."""

    return DaemonServicePaths.for_home(Path.home()).config.with_name(DEVICE_CREDENTIALS_FILENAME)


def credentials_command(source_id: str) -> str:
    """The command that stores a device Source's credentials on this device."""

    return f"memforge source credentials set {source_id}"


class DeviceCredentialsMissing(RuntimeError):
    """This device holds no credential the Source's configuration needs.

    It never passes by itself: the owner stores the credential or signs in again.
    """

    retryable = False


class DeviceCredentialStore:
    """The device's per-Source provider credentials: values in the keyring, their names in the index."""

    def __init__(self, path: Path, *, keyring_store: SecretStore | None = None) -> None:
        self.path = path
        self._keyring = keyring_store

    @property
    def keyring(self) -> SecretStore:
        if self._keyring is None:
            import keyring

            self._keyring = keyring
        return self._keyring

    @staticmethod
    def _account(source_id: str, field: str) -> str:
        return f"{source_id}/{field}"

    def _index(self) -> dict[str, list[str]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read the device credential index: {self.path}") from exc
        sources = payload.get("sources") if isinstance(payload, dict) else None
        if payload.get("version") != DEVICE_CREDENTIALS_VERSION or not isinstance(sources, dict):
            raise ValueError(f"Invalid device credential index: {self.path}")
        return {str(source_id): [str(field) for field in fields] for source_id, fields in sources.items()}

    def _write_index(self, index: Mapping[str, list[str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        body = {"version": DEVICE_CREDENTIALS_VERSION, "sources": {key: sorted(index[key]) for key in sorted(index)}}
        try:
            temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def source_ids(self) -> frozenset[str]:
        """The Sources this device holds credentials for."""

        return frozenset(self._index())

    def set(self, source_id: str, secrets: Mapping[str, str]) -> None:
        """Hold exactly ``secrets`` for the Source, replacing what it held before."""

        index = self._index()
        for field, value in secrets.items():
            self.keyring.set_password(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field), value)
        for field in set(index.get(source_id, ())) - set(secrets):
            self._delete_secret(source_id, field)
        index[source_id] = sorted(secrets)
        self._write_index(index)

    def get(self, source_id: str) -> dict[str, str] | None:
        """The Source's secrets by field, or ``None`` when this device holds none for it."""

        fields = self._index().get(source_id)
        if fields is None:
            return None
        secrets: dict[str, str] = {}
        for field in fields:
            value = self.keyring.get_password(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field))
            if value is None:
                raise ValueError(f"The keyring no longer holds {field} of Source {source_id}")
            secrets[field] = value
        return secrets

    def delete(self, source_id: str) -> None:
        """Forget the Source's credentials on this device."""

        index = self._index()
        for field in index.pop(source_id, ()):
            self._delete_secret(source_id, field)
        self._write_index(index)

    def _delete_secret(self, source_id: str, field: str) -> None:
        try:
            self.keyring.delete_password(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field))
        except Exception as exc:
            if exc.__class__.__name__ != "PasswordDeleteError":
                raise

    def session(self, provider: str, origin: str) -> str | None:
        """The browser session this device holds for the provider's origin."""

        return self.keyring.get_password(DEVICE_SESSIONS_KEYRING_SERVICE, f"{provider}:{origin}")

    def set_session(self, provider: str, origin: str, cookie_header: str) -> None:
        self.keyring.set_password(DEVICE_SESSIONS_KEYRING_SERVICE, f"{provider}:{origin}", cookie_header)

    def gene_config(self, source_id: str, source_type: str, config: Mapping[str, Any]) -> dict[str, Any]:
        """The configuration the device runs the Source's Gene with: the job's, plus what this device holds.

        That is the Source's secrets by field and, when the configuration
        signs in with a browser session, the session of its origin under the
        key the type's Gene reads it from. Raises
        :class:`DeviceCredentialsMissing` when this device holds neither.
        """

        from memforge.auth.browser_session import ensure_builtin_providers, provider_for_source_type

        secrets = self.get(source_id)
        if secrets is None:
            raise DeviceCredentialsMissing(
                f"This device holds no credentials for Source {source_id}; run `{credentials_command(source_id)}`"
            )
        gene_config = {**config, **secrets}
        ensure_builtin_providers()
        provider = provider_for_source_type(source_type)
        if provider is not None and provider.uses_browser_session(gene_config):
            origin = provider.canonical_origin(str(gene_config.get("base_url") or ""))
            cookie_header = self.session(provider.provider, origin)
            if cookie_header is None:
                raise DeviceCredentialsMissing(
                    f"This device holds no {provider.label} session for {origin}; sign in to the Source again"
                )
            gene_config[provider.cookie_config_key] = cookie_header
        return gene_config


async def check_browser_session(source_type: str, gene_config: Mapping[str, Any]) -> None:
    """Check the browser session a device Gene configuration carries against its provider.

    Does nothing for a configuration that signs in otherwise. A session the
    provider refuses raises :class:`DeviceCredentialsMissing`, because only a
    new sign-in renews it.
    """

    from memforge.auth.browser_session import ensure_builtin_providers, provider_for_source_type

    ensure_builtin_providers()
    provider = provider_for_source_type(source_type)
    if provider is None or not provider.uses_browser_session(dict(gene_config)):
        return
    origin = provider.canonical_origin(str(gene_config.get("base_url") or ""))
    try:
        await provider.validate_session(origin, str(gene_config[provider.cookie_config_key]), dict(gene_config))
    except Exception as exc:
        raise DeviceCredentialsMissing(
            f"The {provider.label} session for {origin} has expired; sign in to the Source again ({exc})"
        ) from exc
