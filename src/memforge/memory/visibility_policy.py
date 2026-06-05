"""The single place the default visibility for a new memory is decided.

Every source currently writes only workspace memories: the read-side access
predicate that makes a private row safe to store is not in place yet, so every
source defaults to team-visible here. The agent-session private default and a
per-source override arrive once that predicate exists, and they will change
only this function.
"""

from __future__ import annotations

from memforge.models import Visibility


def default_visibility(source_type: str | None) -> tuple[str, str | None]:
    """Return the (visibility, owner_user_id) a new memory from this source gets."""
    return (Visibility.WORKSPACE.value, None)
