"""The provider credentials a device holds for its device Sources (ADR 0058).

Each secret lives in the operating-system keyring, under one entry per
Source and field. The keyring cannot be listed, so an index beside the daemon
configuration names each Source this device holds credentials for and the
fields it stored, never their values. The index is what the device report
names: a ``document_acquisition`` job of a Source goes only to a device that
holds its credentials. A Source whose type keeps no secret, or whose
configuration signs in with a browser session, is held with no fields, which
marks this device as the one that acquires it.

A device can serve more than one MemForge server, each with its own Sources,
so everything it holds belongs to the server it holds it for: the index keeps
one entry per server's API origin, and every keyring account starts with that
origin. A store is opened for one server (:func:`device_credentials`); the
daemon reports, and releases on that server's word, only what it holds for the
server it leases from.

A browser session lives in the keyring per server, provider and origin,
because one sign-in serves every Source of that site. The index names the session each
Source signs in with, so the session is deleted together with the last Source
that uses it. The device's Gene reads a Source's secrets and session from
here (:meth:`DeviceCredentialStore.gene_config`), and the session is checked
against the provider before each run (:func:`check_browser_session`), on the
device, never on the server.

The daemon's lease loop, its ``credential_check`` jobs and the ``memforge
source credentials set`` command change the index from separate threads and
processes, so each change holds an exclusive lock on a file beside it.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
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
    "device_credentials",
    "device_credentials_path",
]

DEVICE_CREDENTIALS_KEYRING_SERVICE = "memforge-device-source"
DEVICE_SESSIONS_KEYRING_SERVICE = "memforge-device-session"
DEVICE_CREDENTIALS_FILENAME = "device-credentials.json"
DEVICE_CREDENTIALS_VERSION = 1


def device_credentials_path() -> Path:
    """Where this installation keeps its credential index: beside the daemon service configuration."""

    return DaemonServicePaths.for_home(Path.home()).config.with_name(DEVICE_CREDENTIALS_FILENAME)


def device_credentials(server: str) -> DeviceCredentialStore:
    """What this installation holds for the server at the API origin ``server``."""

    return DeviceCredentialStore(device_credentials_path(), server=server)


def credentials_command(source_id: str) -> str:
    """The command that stores a device Source's credentials on this device."""

    return f"memforge source credentials set {source_id}"


class DeviceCredentialsMissing(RuntimeError):
    """This device holds no credential the Source's configuration needs.

    It never passes by itself: the owner stores the credential or signs in again.
    """

    retryable = False


@dataclass(frozen=True, slots=True)
class _HeldSource:
    """What the index records for one Source: its secret fields and the browser session it signs in with."""

    fields: tuple[str, ...] = ()
    session: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"fields": sorted(self.fields)}
        if self.session is not None:
            payload["session"] = self.session
        return payload


class DeviceCredentialStore:
    """The provider credentials the device holds for one server's Sources: values in the keyring, names in the index."""

    def __init__(self, path: Path, *, server: str, keyring_store: SecretStore | None = None) -> None:
        self.path = path
        self.server = server
        self._keyring = keyring_store

    @property
    def keyring(self) -> SecretStore:
        if self._keyring is None:
            import keyring

            self._keyring = keyring
        return self._keyring

    def _account(self, source_id: str, field: str) -> str:
        return f"{self.server}#{source_id}/{field}"

    def _session_account(self, provider: str, origin: str) -> str:
        return f"{self.server}#{provider}:{origin}"

    @contextmanager
    def _changing(self) -> Iterator[dict[str, _HeldSource]]:
        """This server's Sources in the index, held exclusively while the caller changes them, then written back."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(self.path.with_name(f".{self.path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            servers = self._servers()
            index = servers.setdefault(self.server, {})
            yield index
            if not index:
                del servers[self.server]
            self._write_servers(servers)
        finally:
            os.close(handle)

    def _index(self) -> dict[str, _HeldSource]:
        return self._servers().get(self.server, {})

    def _servers(self) -> dict[str, dict[str, _HeldSource]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read the device credential index: {self.path}") from exc
        servers = payload.get("servers") if isinstance(payload, dict) else None
        if payload.get("version") != DEVICE_CREDENTIALS_VERSION or not isinstance(servers, dict):
            raise ValueError(f"Invalid device credential index: {self.path}")
        index: dict[str, dict[str, _HeldSource]] = {}
        for server, sources in servers.items():
            if not isinstance(sources, dict):
                raise ValueError(f"Invalid device credential index: {self.path}")
            held_sources: dict[str, _HeldSource] = {}
            for source_id, held in sources.items():
                if not isinstance(held, dict) or not isinstance(held.get("fields"), list):
                    raise ValueError(f"Invalid device credential index: {self.path}")
                session = held.get("session")
                held_sources[str(source_id)] = _HeldSource(
                    fields=tuple(str(field) for field in held["fields"]),
                    session=str(session) if session is not None else None,
                )
            index[str(server)] = held_sources
        return index

    def _write_servers(self, servers: Mapping[str, Mapping[str, _HeldSource]]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        body = {
            "version": DEVICE_CREDENTIALS_VERSION,
            "servers": {
                server: {key: sources[key].to_payload() for key in sorted(sources)} for server, sources in servers.items()
            },
        }
        try:
            temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def source_ids(self) -> frozenset[str]:
        """The Sources of this server this device holds credentials for."""

        return frozenset(self._index())

    def set(self, source_id: str, secrets: Mapping[str, str]) -> None:
        """Hold exactly ``secrets`` for the Source, replacing what it held before, its session included."""

        with self._changing() as index:
            previous = index.get(source_id, _HeldSource())
            for field, value in secrets.items():
                self.keyring.set_password(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field), value)
            for field in set(previous.fields) - set(secrets):
                self._delete_secret(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field))
            index[source_id] = _HeldSource(fields=tuple(secrets))
            self._release_unused_session(index, previous.session)

    def hold_session(self, source_id: str, provider: str, origin: str, cookie_header: str) -> None:
        """Keep the provider's session for ``origin`` and record that the Source signs in with it."""

        session = self._session_account(provider, origin)
        with self._changing() as index:
            previous = index.get(source_id, _HeldSource())
            self.keyring.set_password(DEVICE_SESSIONS_KEYRING_SERVICE, session, cookie_header)
            index[source_id] = _HeldSource(fields=previous.fields, session=session)
            self._release_unused_session(index, previous.session)

    def get(self, source_id: str) -> dict[str, str] | None:
        """The Source's secrets by field, or ``None`` when this device holds none for it."""

        held = self._index().get(source_id)
        if held is None:
            return None
        secrets: dict[str, str] = {}
        for field in held.fields:
            value = self.keyring.get_password(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field))
            if value is None:
                raise ValueError(f"The keyring no longer holds {field} of Source {source_id}")
            secrets[field] = value
        return secrets

    def delete(self, source_id: str) -> None:
        """Forget the Source's credentials on this device, and its session once no held Source uses it."""

        with self._changing() as index:
            held = index.pop(source_id, _HeldSource())
            for field in held.fields:
                self._delete_secret(DEVICE_CREDENTIALS_KEYRING_SERVICE, self._account(source_id, field))
            self._release_unused_session(index, held.session)

    def _release_unused_session(self, index: Mapping[str, _HeldSource], session: str | None) -> None:
        if session is not None and all(held.session != session for held in index.values()):
            self._delete_secret(DEVICE_SESSIONS_KEYRING_SERVICE, session)

    def _delete_secret(self, service: str, account: str) -> None:
        try:
            self.keyring.delete_password(service, account)
        except Exception as exc:
            if exc.__class__.__name__ != "PasswordDeleteError":
                raise

    def session(self, provider: str, origin: str) -> str | None:
        """The browser session this device holds for the provider's origin."""

        return self.keyring.get_password(DEVICE_SESSIONS_KEYRING_SERVICE, self._session_account(provider, origin))

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
