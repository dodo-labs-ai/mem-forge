"""Provider-neutral readiness exposed for local source connections."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal, NotRequired, TypedDict

from memforge.local_agent.device_report import LocalAgentDevice


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
_STATE_RANK: dict[str, int] = {"online": 0, "update_required": 1, "offline": 2}


class SourceExecutionDevice(TypedDict):
    """The device a device Source runs on, as the source API exposes it (design §10.7).

    ``name`` is shown only to the Source's execution owner, and is ``None``
    for any other viewer or when the owner has no device. ``installed_version``
    is the MemForge version of the device's latest report, and
    ``required_version`` the server's, which ``memforge upgrade`` installs.
    """

    name: str | None
    state: DeviceState
    installed_version: str | None
    required_version: str


def device_source_readiness(
    *,
    source_id: str,
    source_type: str,
    devices: Sequence[LocalAgentDevice],
    server_contracts: Mapping[str, int],
    server_version: str,
    now: datetime,
    reveal_device: bool,
) -> tuple[SourceExecutionDevice, SourceConnectionStatus]:
    """Device and connection readiness of a device Source from its execution owner's device reports.

    The Source runs on a device whose report names it, because only such a
    device holds its credential. Until one does, the Source needs its
    credential stored (``action_required``, ``authentication``) and its
    device is the owner's device that would store it. Among the candidates,
    a device that can take the Source's job now comes first, then one that
    needs an update, then the most recently seen; a device acquires the type
    only under the server's contract for it.
    """

    naming = [device for device in devices if source_id in device.report.source_ids]
    candidates = naming or list(devices)

    def state(device: LocalAgentDevice) -> DeviceState:
        if (now - device.last_seen_at).total_seconds() > LOCAL_AGENT_DEVICE_STALE_SECONDS:
            return "offline"
        if not device.report.acquires(source_type, server_contracts):
            return "update_required"
        return "online"

    connection: SourceConnectionStatus = (
        {"state": "ready", "reason": None} if naming else {"state": "action_required", "reason": "authentication"}
    )
    if not candidates:
        return (
            {"name": None, "state": "offline", "installed_version": None, "required_version": server_version},
            connection,
        )
    device = min(candidates, key=lambda candidate: (_STATE_RANK[state(candidate)], -candidate.last_seen_at.timestamp()))
    return (
        {
            "name": device.report.name if reveal_device else None,
            "state": state(device),
            "installed_version": device.report.version,
            "required_version": server_version,
        },
        connection,
    )
