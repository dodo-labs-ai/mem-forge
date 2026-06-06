"""The per-request caller context consumed by the storage adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# The implicit caller for the single local datastore. In a hosted deployment
# the verified principal replaces this; locally there is exactly one user.
LOCAL_DEV_USER_ID = "dev"


ScopeMode = Literal["project", "project-first", "workspace"]
_VALID_SCOPE_MODES: frozenset[str] = frozenset(("project", "project-first", "workspace"))


@dataclass(frozen=True)
class AccessScope:
    """The filter a caller is allowed to see, built once per request.

    Three concerns ride together, kept explicit:
      - ACCESS    : user_id and include_private gate what the caller may see
                    within the bound datastore.
      - LIFECYCLE : allowed_statuses is the existing include_superseded knob,
                    not an access dimension.
      - RELEVANCE : active_project and scope_mode shape ranking. ``scope_mode``
                    is a three-valued switch: ``project-first`` (default)
                    weights cross-project hits with a penalty but keeps them
                    visible; ``workspace`` weights everything equally; and
                    ``project`` is the only mode that gates upstream, pruning
                    every key that is not the active project or SHARED at the
                    predicate (UNSORTED and other projects are excluded, not
                    just down-weighted).

    The store is bound to one datastore at construction; this object is the
    per-request caller context (who is asking, in which project), not a
    storage-location identity.
    """

    user_id: str
    include_private: bool
    allowed_statuses: tuple[str, ...]
    active_project: str | None
    scope_mode: ScopeMode

    def __post_init__(self) -> None:
        # The Literal type alias is a static hint; this guard catches any
        # non-HTTP caller that bypasses the request layer with a stray string.
        if self.scope_mode not in _VALID_SCOPE_MODES:
            raise ValueError(
                f"AccessScope.scope_mode must be one of {sorted(_VALID_SCOPE_MODES)}, "
                f"got {self.scope_mode!r}"
            )
