"""What a daemon does with the server's answer to its lease (ADR 0058).

Besides the jobs, every lease answer names:

- ``released_source_ids``: the Sources the device report named that are no
  longer device Sources of this owner, because they were deleted, moved to
  the server or handed to another execution owner. The device deletes the
  credentials it holds for them on the server it leases from, and nothing it
  holds for another server.
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

__all__ = ["RELEASED_SOURCE_IDS_FIELD", "UPGRADE_COMMAND", "DeviceLeaseAnswers"]

RELEASED_SOURCE_IDS_FIELD = "released_source_ids"
UPGRADE_COMMAND = "memforge upgrade"


class DeviceLeaseAnswers:
    """Applies each lease answer to this device: releases credentials and tracks contract agreement."""

    def __init__(self, credentials: DeviceCredentialStore, *, log: Callable[[str], None]) -> None:
        self._credentials = credentials
        self._log = log
        self._matching: bool | None = None

    def observe(self, answer: Mapping[str, Any]) -> None:
        for source_id in answer.get(RELEASED_SOURCE_IDS_FIELD) or ():
            self._credentials.delete(str(source_id))
            self._log(f"Deleted this device's credentials for Source {source_id}: it is no longer one of its Sources.")
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
