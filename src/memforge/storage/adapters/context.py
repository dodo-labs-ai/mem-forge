"""The per-request caller context consumed by the storage adapters."""

from __future__ import annotations

from dataclasses import dataclass

# The implicit caller for the single local datastore. In a hosted deployment
# the verified principal replaces this; locally there is exactly one user.
LOCAL_DEV_USER_ID = "dev"


@dataclass(frozen=True)
class AccessScope:
    """The filter a caller is allowed to see, built once per request.

    Three concerns ride together, kept explicit:
      - ACCESS    : user_id, open_projects, member_projects, include_private
                    gate what the caller may see within the bound datastore.
      - LIFECYCLE : allowed_statuses is the existing include_superseded knob,
                    not an access dimension.
      - RELEVANCE : active_project and scope_mode only weight results, they
                    never gate them.

    The store is bound to one datastore at construction; this object is the
    per-request caller context (who is asking, in which project), not a
    storage-location identity.
    """

    user_id: str
    open_projects: frozenset[str]
    member_projects: frozenset[str]
    include_private: bool
    allowed_statuses: tuple[str, ...]
    active_project: str | None
    scope_mode: str
