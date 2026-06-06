"""The single place the default visibility for a new memory is decided.

Agent-session sources default to private with the uploader as owner. Every other
source defaults to workspace with no owner. Until a server-derived principal is
wired everywhere, an absent uploader hint falls back to LOCAL_DEV_USER_ID so a
private memory always carries a non-empty owner.
"""

from __future__ import annotations

from memforge.models import Visibility
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID


def default_visibility(
    source_type: str | None,
    *,
    user_id: str | None = None,
) -> tuple[str, str | None]:
    """Return the (visibility, owner_user_id) a new memory from this source gets."""
    if source_type == "agent_session":
        return (Visibility.PRIVATE.value, user_id or LOCAL_DEV_USER_ID)
    return (Visibility.WORKSPACE.value, None)
