"""Plan complete local-agent snapshots without transferring unchanged bodies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memforge.source_time import SOURCE_UPDATED_AT_KEY

SNAPSHOT_MANIFEST_MAX_ITEMS = 10_000
"""The most items one device manifest, and each of its excluded listings, may name."""
COMPLETE_SNAPSHOT = "complete_snapshot"
"""The coverage every device manifest declares: the complete item set of its scope (ADR 0011)."""
UPSERT = "upsert"
"""The change kind of every manifest item: the item as the device read it."""
# Only new manifests are held to these values. Stored manifests keep what they
# were written with, including ``bounded_delta`` and ``tombstone``, and every
# reader loads them as stored.


@dataclass(frozen=True)
class SnapshotManifestItem:
    doc_id: str
    revision: str


def snapshot_manifest_sha256(items: Iterable[SnapshotManifestItem]) -> str:
    """The digest that binds a device manifest: its coverage and its items.

    Items are in Document id order. Coverage and change kind are constants,
    and both stay in the digest, so a manifest hashes as it always has.
    """
    payload: dict[str, object] = {
        "coverage": COMPLETE_SNAPSHOT,
        "items": sorted(
            ({"doc_id": item.doc_id, "revision": item.revision, "change_kind": UPSERT} for item in items),
            key=lambda item: item["doc_id"],
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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


def validate_snapshot_manifest_values(coverage: str, manifest_items: Sequence[tuple[str, str, str]]) -> None:
    """Check that a manifest declares complete coverage and that every item is an upsert."""
    if coverage != COMPLETE_SNAPSHOT:
        raise ValueError("snapshot manifest coverage is invalid")
    if any(change_kind != UPSERT for _doc_id, _revision, change_kind in manifest_items):
        raise ValueError("snapshot manifest change_kind is invalid")


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


@dataclass(frozen=True)
class SnapshotManifestPlan:
    required_doc_ids: tuple[str, ...]
    reused_memberships: tuple[tuple[str, str], ...]


def plan_snapshot_manifest(
    items: Sequence[SnapshotManifestItem],
    retained_inputs: Iterable[Any],
    *,
    require_source_time: bool = False,
) -> SnapshotManifestPlan:
    """Match exact provider revisions to attested immutable package inputs.

    The manifest is complete source membership for one fenced collection
    attempt. An item is reusable only when both its stable document identity
    and opaque provider revision match an artifact whose own bytes were
    attested when retained. With ``require_source_time`` a retained package is
    reusable only when it also carries the content's source time; the agent
    uploads the others again with their time.
    """

    declared: dict[str, str] = {}
    for item in items:
        doc_id = str(item.doc_id or "").strip()
        revision = str(item.revision or "").strip()
        if not doc_id or not revision:
            raise ValueError("snapshot manifest items require doc_id and revision")
        if doc_id in declared:
            raise ValueError(f"snapshot manifest contains duplicate doc_id: {doc_id}")
        declared[doc_id] = revision

    retained_by_identity: dict[tuple[str, str], str] = {}
    ordered_inputs = sorted(
        retained_inputs,
        key=lambda value: int(getattr(value, "input_generation", 0)),
        reverse=True,
    )
    for retained in ordered_inputs:
        metadata = getattr(retained, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        entry = metadata.get("manifest_entry")
        if not isinstance(entry, Mapping):
            continue
        package_hash = str(metadata.get("package_sha256") or "").strip()
        entry_package_hash = str(entry.get("package_sha256") or "").strip()
        if not package_hash or package_hash != entry_package_hash:
            continue
        # A manifest written before every item was an upsert may hold a
        # tombstone; it names no body to reuse.
        if str(entry.get("change_kind") or UPSERT) != UPSERT:
            continue
        doc_id = str(entry.get("doc_id") or "").strip()
        revision = str(entry.get("provider_revision") or entry.get("version") or "").strip()
        input_id = str(getattr(retained, "input_id", "") or "").strip()
        if not doc_id or not revision or not input_id:
            continue
        if require_source_time and not str(entry.get(SOURCE_UPDATED_AT_KEY) or "").strip():
            continue
        retained_by_identity.setdefault((doc_id, revision), input_id)

    required: list[str] = []
    reused: list[tuple[str, str]] = []
    for doc_id, revision in declared.items():
        input_id = retained_by_identity.get((doc_id, revision))
        if input_id is None:
            required.append(doc_id)
        else:
            reused.append((doc_id, input_id))
    return SnapshotManifestPlan(
        required_doc_ids=tuple(required),
        reused_memberships=tuple(reused),
    )


