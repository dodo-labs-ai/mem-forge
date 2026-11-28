"""Provider-neutral readiness exposed for local source connections."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, NotRequired, TypedDict

from memforge.local_agent.device_report import NOT_READY_CREDENTIAL, LocalAgentDevice


class BrowserSessionStatus(TypedDict, total=False):
    status: str
    principal_changed: bool


SourceConnectionState = Literal["ready", "action_required"]
SourceConnectionReason = Literal["authentication", "configuration", "identity_conflict"]


class SourceConnectionStatus(TypedDict):
    state: SourceConnectionState
    reason: NotRequired[SourceConnectionReason | None]


def connection_status_from_browser_session(
    session: BrowserSessionStatus,
) -> SourceConnectionStatus:
    """Translate provider-specific browser auth into the shared source contract."""
    if session.get("principal_changed"):
        return {"state": "action_required", "reason": "identity_conflict"}
    if session.get("status") == "active":
        return {"state": "ready", "reason": None}
    return {"state": "action_required", "reason": "authentication"}


LOCAL_AGENT_DEVICE_STALE_SECONDS = 90
"""How long a device stays online after its latest report; the daemon reports with every lease and heartbeat."""

DeviceState = Literal["online", "offline", "update_required"]
DeviceCredential = Literal["secret", "browser_session"]
"""How a device Source's credential reaches its device: a secret ``memforge source credentials set`` stores, or a
browser session a ``credential_check`` job captures."""


class SourceNotReadyReason(TypedDict):
    """Why the execution device cannot run the Source yet: a missing credential, or a missing device location."""

    kind: Literal["credential", "location"]
    field: NotRequired[str]
    path: NotRequired[str]


class SourceExecutionDevice(TypedDict):
    """The execution device of a device Source, as the source API exposes it (design §10.7).

    ``name`` and ``not_ready`` are shown only to the Source's execution
    owner, and are ``None`` and empty for any other viewer.
    ``installed_version`` is the MemForge version of the device's latest
    report, and ``required_version`` the server's, which ``memforge upgrade``
    installs. ``credential`` says how the Source's credential reaches the
    device, and ``not_ready`` what the device's latest report says it lacks.
    """

    id: str
    name: str | None
    state: DeviceState
    installed_version: str
    required_version: str
    credential: DeviceCredential
    not_ready: list[SourceNotReadyReason]


def device_source_readiness(
    *,
    source_id: str,
    source_type: str,
    execution_device_id: str | None,
    devices: Sequence[LocalAgentDevice],
    server_contracts: Mapping[str, int],
    server_version: str,
    now: datetime,
    reveal_device: bool,
    credential: DeviceCredential,
) -> tuple[SourceExecutionDevice | None, SourceConnectionStatus | None]:
    """Device and connection readiness of a device Source from its execution device's latest report.

    A Source without an execution device, or whose device its owner no
    longer has, has no device and needs its owner to choose one
    (``configuration``). Otherwise the device is offline without a recent
    report, ``update_required`` while its acquisition contract for the type
    differs from the server's, and online else. Its report says whether it is
    ready for the Source: a missing credential needs a sign-in
    (``authentication``), a missing device location a finished setup
    (``configuration``). Until its report names the Source at all, the
    connection is not observed yet and is ``None``.
    """

    device = next((device for device in devices if device.report.device_id == execution_device_id), None)
    if device is None:
        return None, {"state": "action_required", "reason": "configuration"}
    if (now - device.last_seen_at).total_seconds() > LOCAL_AGENT_DEVICE_STALE_SECONDS:
        state: DeviceState = "offline"
    elif not device.report.acquires(source_type, server_contracts):
        state = "update_required"
    else:
        state = "online"
    readiness = device.report.sources.get(source_id)
    connection: SourceConnectionStatus | None
    if readiness is None:
        connection = None
    elif readiness.ready:
        connection = {"state": "ready", "reason": None}
    elif any(reason.kind == NOT_READY_CREDENTIAL for reason in readiness.reasons):
        connection = {"state": "action_required", "reason": "authentication"}
    else:
        connection = {"state": "action_required", "reason": "configuration"}
    not_ready: list[SourceNotReadyReason] = []
    if reveal_device and readiness is not None:
        for reason in readiness.reasons:
            entry: SourceNotReadyReason = {"kind": reason.kind}
            if reason.field is not None:
                entry["field"] = reason.field
            if reason.path is not None:
                entry["path"] = reason.path
            not_ready.append(entry)
    return (
        {
            "id": device.report.device_id,
            "name": device.report.name if reveal_device else None,
            "state": state,
            "installed_version": device.report.version,
            "required_version": server_version,
            "credential": credential,
            "not_ready": not_ready,
        },
        connection,
    )
