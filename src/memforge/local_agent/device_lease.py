"""What a daemon does with the server's answer to its lease (ADR 0058).

Besides the jobs, every lease answer names:

- ``sources``: the Sources this device executes, each with its type and its
  configuration without secrets. The device reports its readiness for each
  of them with its next lease and heartbeats, and deletes the credentials it
  holds, on the server it leases from, for every other Source: one deleted,
  moved to the server or moved to another device. It holds nothing for
  another server there.
- ``server``: the server's MemForge version and the acquisition contract of
  each document type. A device whose contracts differ takes none of their
  jobs; the daemon says so in one log line when its contracts stop matching
  the server's, and in another when they match again, not on every lease.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from memforge.local_agent.device_credentials import DeviceCredentialStore
from memforge.local_agent.device_report import declared_acquisition_contracts

__all__ = ["EXECUTED_SOURCES_FIELD", "UPGRADE_COMMAND", "DeviceLeaseAnswers"]

EXECUTED_SOURCES_FIELD = "sources"
UPGRADE_COMMAND = "memforge upgrade"


class DeviceLeaseAnswers:
    """Applies each lease answer to this device: keeps the Sources it executes, releases credentials, tracks contracts."""

    def __init__(self, credentials: DeviceCredentialStore, *, log: Callable[[str], None]) -> None:
        self._credentials = credentials
        self._log = log
        self._matching: bool | None = None
        self._sources: tuple[Mapping[str, Any], ...] = ()

    @property
    def sources(self) -> tuple[Mapping[str, Any], ...]:
        """The Sources the latest lease answer says this device executes; none before the first answer."""

        return self._sources

    def observe(self, answer: Mapping[str, Any]) -> None:
        executed = answer.get(EXECUTED_SOURCES_FIELD)
        if isinstance(executed, list):
            self._sources = tuple(source for source in executed if isinstance(source, Mapping) and source.get("id"))
            named = {str(source["id"]) for source in self._sources}
            for source_id in sorted(self._credentials.source_ids() - named):
                self._credentials.delete(source_id)
                self._log(f"Deleted this device's credentials for Source {source_id}: this device no longer runs it.")
        server = answer.get("server")
        if not isinstance(server, Mapping) or not isinstance(server.get("acquisition_contracts"), Mapping):
            return
        matching = dict(server["acquisition_contracts"]) == declared_acquisition_contracts()
        if matching == self._matching or (self._matching is None and matching):
            self._matching = matching
            return
        self._matching = matching
        if matching:
            self._log("This device's acquisition contracts match the server's again; its document Sources sync.")
        else:
            self._log(
                f"The server runs MemForge {server.get('version')} with other acquisition contracts; "
                f"document Sources on this device wait until `{UPGRADE_COMMAND}` installs that version."
            )
