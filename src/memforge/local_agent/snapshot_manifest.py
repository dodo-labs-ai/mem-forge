"""Plan complete local-agent snapshots without transferring unchanged bodies."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CollectionCoverage(str, Enum):
    COMPLETE_SNAPSHOT = "complete_snapshot"
    BOUNDED_DELTA = "bounded_delta"
    PARTIAL = "partial"


class CollectionChangeKind(str, Enum):
    UPSERT = "upsert"
    TOMBSTONE = "tombstone"


@dataclass(frozen=True)
class SnapshotManifestItem:
    doc_id: str
    revision: str
    change_kind: CollectionChangeKind


@dataclass(frozen=True)
class SnapshotManifestPlan:
    required_doc_ids: tuple[str, ...]
    reused_memberships: tuple[tuple[str, str], ...]


def plan_snapshot_manifest(
    items: Sequence[SnapshotManifestItem],
    retained_inputs: Iterable[Any],
    *,
    coverage: CollectionCoverage,
) -> SnapshotManifestPlan:
    """Match exact provider revisions to attested immutable package inputs.

    The manifest is complete source membership for one fenced collection
    attempt. An item is reusable only when both its stable document identity
    and opaque provider revision match an artifact whose own bytes were
    attested when retained.
    """

    if coverage is CollectionCoverage.PARTIAL:
        raise ValueError("partial collection cannot be planned")

    declared: dict[str, tuple[str, CollectionChangeKind]] = {}
    for item in items:
        doc_id = str(item.doc_id or "").strip()
        revision = str(item.revision or "").strip()
        if not doc_id or not revision:
            raise ValueError("snapshot manifest items require doc_id and revision")
        if not isinstance(item.change_kind, CollectionChangeKind):
            raise ValueError("snapshot manifest items require a valid change_kind")
        if doc_id in declared:
            raise ValueError(f"snapshot manifest contains duplicate doc_id: {doc_id}")
        declared[doc_id] = (revision, item.change_kind)

    retained_by_identity: dict[tuple[str, str, CollectionChangeKind], str] = {}
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
        doc_id = str(entry.get("doc_id") or "").strip()
        revision = str(entry.get("provider_revision") or entry.get("version") or "").strip()
        try:
            change_kind = CollectionChangeKind(str(entry.get("change_kind") or "upsert"))
        except ValueError:
            continue
        input_id = str(getattr(retained, "input_id", "") or "").strip()
        if not doc_id or not revision or not input_id:
            continue
        retained_by_identity.setdefault((doc_id, revision, change_kind), input_id)

    required: list[str] = []
    reused: list[tuple[str, str]] = []
    for doc_id, (revision, change_kind) in declared.items():
        input_id = retained_by_identity.get((doc_id, revision, change_kind))
        if input_id is None:
            required.append(doc_id)
        else:
            reused.append((doc_id, input_id))
    return SnapshotManifestPlan(
        required_doc_ids=tuple(required),
        reused_memberships=tuple(reused),
    )
