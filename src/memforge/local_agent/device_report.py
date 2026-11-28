"""The device report: what a daemon tells the server about its device with every lease and job heartbeat.

Each installation of the daemon is one **device**, with an id and a name that
``memforge setup`` gives it (:mod:`memforge.local_agent.device_identity`).
The report also names the MemForge version the device runs, the acquisition
contract of each document type its code declares, and, for each Source the
device executes, whether it is ready to run it (ADR 0058): it holds the
Source's credentials and finds each of its type's device locations. The
server keeps the latest report of each device and routes a
``document_acquisition`` job only to the Source's execution device, while its
report names the Source ready and its contract for the type equals the
server's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "DEVICE_ID_MAX_CHARS",
    "DEVICE_LOCATION_PATH_MAX_CHARS",
    "DEVICE_NAME_MAX_CHARS",
    "DEVICE_REPORT_MAX_SOURCES",
    "DEVICE_VERSION_MAX_CHARS",
    "DeviceReport",
    "DeviceSourceReadiness",
    "LocalAgentDevice",
    "NOT_READY_CREDENTIAL",
    "NOT_READY_LOCATION",
    "READY",
    "NotReadyReason",
    "declared_acquisition_contracts",
    "this_device_report",
]

DEVICE_ID_MAX_CHARS = 128
DEVICE_NAME_MAX_CHARS = 200
DEVICE_VERSION_MAX_CHARS = 64
DEVICE_REPORT_MAX_SOURCES = 1_000
"""The most device Sources one report names."""
DEVICE_LOCATION_PATH_MAX_CHARS = 4_096
NOT_READY_CREDENTIAL = "credential"
NOT_READY_LOCATION = "location"


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
class NotReadyReason:
    """Why a device cannot run a Source yet.

    ``credential``: it holds no value for a secret field the configuration
    shows, or no browser session for a configuration that signs in with one.
    ``location``: the folder at the type's device location ``field`` is
    missing or unreadable at ``path``.
    """

    kind: Literal["credential", "location"]
    field: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind == NOT_READY_CREDENTIAL:
            if self.field is not None or self.path is not None:
                raise ValueError("a missing credential names no field or path")
        elif self.kind == NOT_READY_LOCATION:
            if not isinstance(self.field, str) or not self.field or not isinstance(self.path, str):
                raise ValueError("a missing device location names its field and the path the device did not find")
            if len(self.field) > DEVICE_NAME_MAX_CHARS or len(self.path) > DEVICE_LOCATION_PATH_MAX_CHARS:
                raise ValueError("a missing device location's field or path is too long")
        else:
            raise ValueError("a Source is not ready for a missing credential or a missing device location")

    def to_payload(self) -> dict[str, Any]:
        if self.kind == NOT_READY_CREDENTIAL:
            return {"kind": self.kind}
        return {"kind": self.kind, "field": self.field, "path": self.path}

    @classmethod
    def from_payload(cls, payload: object) -> NotReadyReason:
        if not isinstance(payload, Mapping):
            raise ValueError("a not-ready reason is an object")
        return cls(kind=payload.get("kind"), field=payload.get("field"), path=payload.get("path"))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DeviceSourceReadiness:
    """Whether a device is ready to run one Source it executes: ready without reasons, else not ready with them."""

    reasons: tuple[NotReadyReason, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.reasons

    def to_payload(self) -> dict[str, Any]:
        return {"ready": self.ready, "reasons": [reason.to_payload() for reason in self.reasons]}

    @classmethod
    def from_payload(cls, payload: object) -> DeviceSourceReadiness:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("reasons"), list | tuple):
            raise ValueError("a Source's readiness on a device holds its reasons")
        readiness = cls(reasons=tuple(NotReadyReason.from_payload(reason) for reason in payload["reasons"]))
        if payload.get("ready") is not readiness.ready:
            raise ValueError("a Source is ready on a device exactly when no reason holds it back")
        return readiness


READY = DeviceSourceReadiness()


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """One device's report: who it is, what it runs, and whether it is ready for each Source it executes.

    ``acquisition_contracts`` maps each document type to the contract the
    device's code declares for it, and ``sources`` maps each Source the device
    executes, as its latest lease answer named them, to its readiness there.
    """

    device_id: str
    name: str
    version: str
    acquisition_contracts: Mapping[str, int]
    sources: Mapping[str, DeviceSourceReadiness]

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
        sources = dict(self.sources)
        if len(sources) > DEVICE_REPORT_MAX_SOURCES or not all(
            isinstance(source_id, str) and source_id and isinstance(readiness, DeviceSourceReadiness)
            for source_id, readiness in sources.items()
        ):
            raise ValueError(f"a device report names at most {DEVICE_REPORT_MAX_SOURCES} Sources by id")
        object.__setattr__(self, "acquisition_contracts", MappingProxyType(contracts))
        object.__setattr__(self, "sources", MappingProxyType(sources))

    def acquires(self, source_type: str, server_contracts: Mapping[str, int]) -> bool:
        """Whether the device acquires this document type under the server's contract for it."""

        contract = self.acquisition_contracts.get(source_type)
        return contract is not None and contract == server_contracts.get(source_type)

    def ready_source_ids(self) -> frozenset[str]:
        """The Sources the device reports ready to run."""

        return frozenset(source_id for source_id, readiness in self.sources.items() if readiness.ready)

    def to_payload(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "version": self.version,
            "acquisition_contracts": dict(sorted(self.acquisition_contracts.items())),
            "sources": {source_id: self.sources[source_id].to_payload() for source_id in sorted(self.sources)},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DeviceReport:
        contracts = payload.get("acquisition_contracts")
        sources = payload.get("sources")
        if not isinstance(contracts, Mapping) or not isinstance(sources, Mapping):
            raise ValueError("a device report holds its acquisition contracts and its readiness per Source")
        if len(sources) > DEVICE_REPORT_MAX_SOURCES:
            raise ValueError(f"a device report names at most {DEVICE_REPORT_MAX_SOURCES} Sources by id")
        return cls(
            device_id=payload.get("device_id"),  # type: ignore[arg-type]
            name=payload.get("name"),  # type: ignore[arg-type]
            version=payload.get("version"),  # type: ignore[arg-type]
            acquisition_contracts=dict(contracts),
            sources={
                str(source_id): DeviceSourceReadiness.from_payload(readiness) for source_id, readiness in sources.items()
            },
        )


def this_device_report(device_id: str, name: str, sources: Mapping[str, DeviceSourceReadiness]) -> DeviceReport:
    """This installation's report: its identity, its MemForge version, the contracts its code declares and ``sources``."""

    from memforge.release import installed_version

    return DeviceReport(
        device_id=device_id,
        name=name,
        version=installed_version(),
        acquisition_contracts=declared_acquisition_contracts(),
        sources=sources,
    )


@dataclass(frozen=True, slots=True)
class LocalAgentDevice:
    """A device of one user, as its latest report left it, and when the server received that report."""

    user_id: str
    report: DeviceReport
    last_seen_at: datetime
