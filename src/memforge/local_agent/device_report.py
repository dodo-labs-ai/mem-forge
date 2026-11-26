"""The device report: what a daemon tells the server about its device with every lease and job heartbeat.

Each installation of the daemon is one **device**, with an id and a name that
``memforge setup`` gives it (:mod:`memforge.local_agent.device_identity`).
The report also names the MemForge version the device runs, the acquisition
contract of each document type its code declares, and the device Sources
whose credentials it holds (ADR 0058). The server keeps the latest report of
each device and routes a ``document_acquisition`` job only to a device of the
Source's execution owner whose report names the Source and whose contract for
the type equals the server's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

__all__ = [
    "DEVICE_ID_MAX_CHARS",
    "DEVICE_NAME_MAX_CHARS",
    "DEVICE_REPORT_MAX_SOURCES",
    "DEVICE_VERSION_MAX_CHARS",
    "DeviceReport",
    "LocalAgentDevice",
    "declared_acquisition_contracts",
    "this_device_report",
]

DEVICE_ID_MAX_CHARS = 128
DEVICE_NAME_MAX_CHARS = 200
DEVICE_VERSION_MAX_CHARS = 64
DEVICE_REPORT_MAX_SOURCES = 1_000
"""The most device Sources one report names."""


def declared_acquisition_contracts() -> dict[str, int]:
    """The acquisition contract of each registered document type, as this installation's code declares it.

    The server's are the contracts it interprets under; a device's are the
    contracts it acquires under.
    """

    from memforge.source_types import registered_source_types

    return {
        descriptor.source_type: descriptor.acquisition_contract
        for descriptor in registered_source_types()
        if descriptor.acquisition_contract is not None
    }


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """One device's report: who it is, what it runs, and which device Sources it can acquire.

    ``acquisition_contracts`` maps each document type to the contract the
    device's code declares for it, and ``source_ids`` names the device Sources
    whose credentials the device holds.
    """

    device_id: str
    name: str
    version: str
    acquisition_contracts: Mapping[str, int]
    source_ids: frozenset[str]

    def __post_init__(self) -> None:
        for field, value, bound in (
            ("device_id", self.device_id, DEVICE_ID_MAX_CHARS),
            ("name", self.name, DEVICE_NAME_MAX_CHARS),
            ("version", self.version, DEVICE_VERSION_MAX_CHARS),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > bound:
                raise ValueError(f"a device report's {field} is a non-empty string of at most {bound} characters")
        contracts = dict(self.acquisition_contracts)
        if not all(
            isinstance(source_type, str) and isinstance(contract, int) and not isinstance(contract, bool)
            for source_type, contract in contracts.items()
        ):
            raise ValueError("a device report maps each document type to an integer acquisition contract")
        if len(self.source_ids) > DEVICE_REPORT_MAX_SOURCES or not all(
            isinstance(source_id, str) and source_id for source_id in self.source_ids
        ):
            raise ValueError(f"a device report names at most {DEVICE_REPORT_MAX_SOURCES} Sources by id")
        object.__setattr__(self, "acquisition_contracts", MappingProxyType(contracts))
        object.__setattr__(self, "source_ids", frozenset(self.source_ids))

    def acquires(self, source_type: str, server_contracts: Mapping[str, int]) -> bool:
        """Whether the device acquires this document type under the server's contract for it."""

        contract = self.acquisition_contracts.get(source_type)
        return contract is not None and contract == server_contracts.get(source_type)

    def to_payload(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "version": self.version,
            "acquisition_contracts": dict(sorted(self.acquisition_contracts.items())),
            "source_ids": sorted(self.source_ids),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DeviceReport:
        contracts = payload.get("acquisition_contracts")
        source_ids = payload.get("source_ids")
        if not isinstance(contracts, Mapping) or not isinstance(source_ids, list | tuple):
            raise ValueError("a device report holds its acquisition contracts and the Sources it names")
        return cls(
            device_id=payload.get("device_id"),  # type: ignore[arg-type]
            name=payload.get("name"),  # type: ignore[arg-type]
            version=payload.get("version"),  # type: ignore[arg-type]
            acquisition_contracts=dict(contracts),
            source_ids=frozenset(source_ids),
        )


def this_device_report(device_id: str, name: str, source_ids: frozenset[str]) -> DeviceReport:
    """This installation's report: its identity, its MemForge version and the contracts its code declares."""

    from memforge.release import installed_version

    return DeviceReport(
        device_id=device_id,
        name=name,
        version=installed_version(),
        acquisition_contracts=declared_acquisition_contracts(),
        source_ids=source_ids,
    )


@dataclass(frozen=True, slots=True)
class LocalAgentDevice:
    """A device of one user, as its latest report left it, and when the server received that report."""

    user_id: str
    report: DeviceReport
    last_seen_at: datetime
