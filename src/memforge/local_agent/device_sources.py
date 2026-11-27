"""The Sources a device executes, and whether it is ready to run each (ADR 0058).

The lease answer names the Sources this device executes, with their type and
configuration without secrets. :func:`source_readiness_on_device` tells, for
one of them, what this machine holds of what the type needs: a stored value
for each required secret field the configuration shows, a browser session
for a configuration that signs in with one, and an existing, readable folder
at each of the type's device locations. A type that needs none of these is
ready as soon as the device runs. The device report carries that readiness to
the server with every lease and job heartbeat.

``GET /local-agent/status`` names the caller's device Sources with the device
that executes each; :func:`device_source_states` tells, per Source this
device executes, whether it is ready, waits for an update, waits for a
credential or misses its folder, with what moves it on, and
:func:`credential_fields` names the secret fields ``memforge source
credentials set`` asks for.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from memforge.local_agent.credential_check import signs_in_with_browser_session
from memforge.local_agent.device_credentials import DeviceCredentialStore, credentials_command
from memforge.local_agent.device_lease import UPGRADE_COMMAND
from memforge.local_agent.device_report import (
    NOT_READY_CREDENTIAL,
    NOT_READY_LOCATION,
    DeviceSourceReadiness,
    NotReadyReason,
    declared_acquisition_contracts,
)
from memforge.models import ConfigField, ConfigFieldType

__all__ = [
    "DEVICE_SOURCE_MISSES_LOCATION",
    "DEVICE_SOURCE_READY",
    "DEVICE_SOURCE_WAITS_FOR_CREDENTIAL",
    "DEVICE_SOURCE_WAITS_FOR_UPDATE",
    "SIGN_IN_INSTRUCTION",
    "credential_fields",
    "device_source_states",
    "readiness_of_sources",
    "source_readiness_on_device",
]

DEVICE_SOURCE_READY = "ready"
DEVICE_SOURCE_WAITS_FOR_UPDATE = "waits_for_update"
DEVICE_SOURCE_WAITS_FOR_CREDENTIAL = "waits_for_credential"
DEVICE_SOURCE_MISSES_LOCATION = "misses_location"
SIGN_IN_INSTRUCTION = "Sign in from the Source's Configure page in MemForge"


def credential_fields(source_type: str, config: Mapping[str, Any]) -> list[ConfigField]:
    """The secret fields of the type's form that the configuration shows, in form order."""

    from memforge.genes import GENE_REGISTRY
    from memforge.source_types import source_type_descriptor
    from memforge.source_types.form import visible_fields

    gene_cls = GENE_REGISTRY.get(source_type)
    descriptor = source_type_descriptor(source_type)
    if gene_cls is None or descriptor is None:
        raise ValueError(f"Unknown source type: {source_type}")
    return [
        field
        for field in visible_fields(gene_cls.config_schema(), config, descriptor)
        if field.field_type == ConfigFieldType.SECRET
    ]


def _readable_folder(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value).expanduser()
    return path.is_dir() and os.access(path, os.R_OK | os.X_OK)


def source_readiness_on_device(
    source_id: str, source_type: str, config: Mapping[str, Any], credentials: DeviceCredentialStore
) -> DeviceSourceReadiness:
    """Whether this device holds what the Source's type needs on this machine, else why not."""

    from memforge.source_types import source_type_descriptor

    reasons: list[NotReadyReason] = []
    if not credentials.holds_credentials(source_id, source_type, config):
        reasons.append(NotReadyReason(kind=NOT_READY_CREDENTIAL))
    descriptor = source_type_descriptor(source_type)
    for field in descriptor.device_location_fields if descriptor is not None else ():
        value = config.get(field)
        if not _readable_folder(value):
            reasons.append(NotReadyReason(kind=NOT_READY_LOCATION, field=field, path=str(value or "")))
    return DeviceSourceReadiness(reasons=tuple(reasons))


def readiness_of_sources(
    sources: Iterable[Mapping[str, Any]], credentials: DeviceCredentialStore
) -> dict[str, DeviceSourceReadiness]:
    """This device's readiness for each Source a lease answer names, by id."""

    return {
        str(source["id"]): source_readiness_on_device(
            str(source["id"]), str(source["type"]), source.get("config") or {}, credentials
        )
        for source in sources
    }


def device_source_states(
    status: Mapping[str, Any], credentials: DeviceCredentialStore, *, device_id: str
) -> list[dict[str, Any]]:
    """Each Source this device executes, by the status answer: ready, or what it waits for.

    A device whose acquisition contract for the type differs from the
    server's cannot run the Source at all, so an update comes first; then a
    missing credential, with the command or the sign-in that stores it; then
    each missing folder, with the path this device did not find.
    """

    ours = declared_acquisition_contracts()
    server = status.get("server") if isinstance(status.get("server"), Mapping) else {}
    server_contracts = server.get("acquisition_contracts") if isinstance(server, Mapping) else {}
    states: list[dict[str, Any]] = []
    for source in status.get("sources") or ():
        if source.get("execution_device_id") != device_id:
            continue
        source_id = str(source["id"])
        source_type = str(source["type"])
        config = source.get("config") or {}
        entry: dict[str, Any] = {"id": source_id, "name": source.get("name"), "type": source_type}
        readiness = source_readiness_on_device(source_id, source_type, config, credentials)
        locations = [reason for reason in readiness.reasons if reason.kind == NOT_READY_LOCATION]
        if ours.get(source_type) is None or ours.get(source_type) != (server_contracts or {}).get(source_type):
            entry.update(state=DEVICE_SOURCE_WAITS_FOR_UPDATE, command=UPGRADE_COMMAND)
        elif any(reason.kind == NOT_READY_CREDENTIAL for reason in readiness.reasons):
            if signs_in_with_browser_session(source_type, config):
                entry.update(state=DEVICE_SOURCE_WAITS_FOR_CREDENTIAL, instruction=SIGN_IN_INSTRUCTION)
            else:
                entry.update(state=DEVICE_SOURCE_WAITS_FOR_CREDENTIAL, command=credentials_command(source_id))
        elif locations:
            entry.update(
                state=DEVICE_SOURCE_MISSES_LOCATION,
                missing=[{"field": reason.field, "path": reason.path} for reason in locations],
            )
        else:
            entry["state"] = DEVICE_SOURCE_READY
        states.append(entry)
    return states
