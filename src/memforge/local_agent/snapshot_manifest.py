"""The bounds and excluded-scope listings of a device's acquisition manifest."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

SNAPSHOT_MANIFEST_MAX_ITEMS = 10_000
"""The most items one device manifest, and each of its excluded listings, may name."""


def validate_excluded_listings_for_job(
    listings: Sequence[ExcludedScopeListing],
    carried_scope_keys: Iterable[str],
) -> None:
    """Check that a manifest lists only the excluded scopes its job carried, each at most once.

    A scope the job carried may be left out, as when the device could not
    reach it; the transition then stays open until a manifest lists it.
    """
    carried = set(carried_scope_keys)
    scope_keys = [listing.scope_key for listing in listings]
    if len(set(scope_keys)) != len(scope_keys):
        raise ValueError("snapshot excluded listings contain duplicate scope_key")
    unknown = sorted(set(scope_keys) - carried)
    if unknown:
        raise ValueError(f"snapshot excluded listing names a scope the job did not carry: {unknown[0]}")


@dataclass(frozen=True)
class ExcludedScopeListing:
    """What a device listed in one excluded scope of a transition, or that the scope was rejected.

    ``scope_key`` names the excluded scope. A listing holds the Document ids
    the scope lists, possibly none. A definitive rejection, such as a JQL the
    provider refuses, carries its cause in ``rejection`` and lists nothing.
    """

    scope_key: str
    doc_ids: frozenset[str] = frozenset()
    rejection: str | None = None

    def __post_init__(self) -> None:
        if not self.scope_key:
            raise ValueError("excluded scope listing requires scope_key")
        if self.rejection is not None and (not self.rejection or self.doc_ids):
            raise ValueError("a rejected excluded scope carries a cause and lists nothing")
