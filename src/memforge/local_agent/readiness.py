"""Provider-neutral readiness exposed for local source connections."""

from typing import Literal, NotRequired, TypedDict


class BrowserSessionStatus(TypedDict, total=False):
    status: str
    principal_changed: bool


class SourceConnectionStatus(TypedDict):
    state: Literal["ready", "action_required"]
    reason: NotRequired[Literal["authentication", "configuration", "identity_conflict"] | None]


def connection_status_from_browser_session(
    session: BrowserSessionStatus,
) -> SourceConnectionStatus:
    """Translate provider-specific browser auth into the shared source contract."""
    if session.get("principal_changed"):
        return {"state": "action_required", "reason": "identity_conflict"}
    if session.get("status") == "active":
        return {"state": "ready", "reason": None}
    return {"state": "action_required", "reason": "authentication"}
