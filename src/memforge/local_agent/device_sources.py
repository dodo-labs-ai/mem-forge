"""The owner's device Sources as this device sees them (ADR 0058).

``GET /local-agent/status`` names the caller's device Sources, with their
configuration without secrets, and the server's acquisition contracts. From
that and what this device holds, :func:`device_source_states` tells per
Source whether it is ready, waits for an update or waits for a credential,
with the command that moves it on, and :func:`credential_fields` names the
secret fields ``memforge source credentials set`` asks for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.local_agent.credential_check import signs_in_with_browser_session
from memforge.local_agent.device_credentials import DeviceCredentialStore, credentials_command
from memforge.local_agent.device_lease import UPGRADE_COMMAND
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.models import ConfigField, ConfigFieldType

__all__ = [
    "DEVICE_SOURCE_READY",
    "DEVICE_SOURCE_WAITS_FOR_CREDENTIAL",
    "DEVICE_SOURCE_WAITS_FOR_UPDATE",
    "SIGN_IN_INSTRUCTION",
    "credential_fields",
    "device_source_states",
]

DEVICE_SOURCE_READY = "ready"
DEVICE_SOURCE_WAITS_FOR_UPDATE = "waits_for_update"
DEVICE_SOURCE_WAITS_FOR_CREDENTIAL = "waits_for_credential"
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


def device_source_states(status: Mapping[str, Any], credentials: DeviceCredentialStore) -> list[dict[str, Any]]:
    """Each device Source of the status answer: ``ready``, ``waits_for_update`` or ``waits_for_credential``.

    A device whose acquisition contract for the type differs from the
    server's cannot run the Source at all, so an update comes before a
    credential.
    """

    held = credentials.source_ids()
    ours = declared_acquisition_contracts()
    server = status.get("server") if isinstance(status.get("server"), Mapping) else {}
    server_contracts = server.get("acquisition_contracts") if isinstance(server, Mapping) else {}
    states: list[dict[str, Any]] = []
    for source in status.get("sources") or ():
        source_id = str(source["id"])
        source_type = str(source["type"])
        entry: dict[str, Any] = {"id": source_id, "name": source.get("name"), "type": source_type}
        if ours.get(source_type) is None or ours.get(source_type) != (server_contracts or {}).get(source_type):
            entry.update(state=DEVICE_SOURCE_WAITS_FOR_UPDATE, command=UPGRADE_COMMAND)
        elif source_id not in held:
            if signs_in_with_browser_session(source_type, source.get("config") or {}):
                entry.update(state=DEVICE_SOURCE_WAITS_FOR_CREDENTIAL, instruction=SIGN_IN_INSTRUCTION)
            else:
                entry.update(state=DEVICE_SOURCE_WAITS_FOR_CREDENTIAL, command=credentials_command(source_id))
        else:
            entry["state"] = DEVICE_SOURCE_READY
        states.append(entry)
    return states
