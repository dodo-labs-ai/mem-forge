"""Plan complete local-agent snapshots without transferring unchanged bodies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from memforge.source_activity import SourceActivityConflict
from memforge.source_time import SOURCE_UPDATED_AT_KEY

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
    """The digest that binds a device manifest: its coverage and its items, in Document id order.

    Coverage and change kind are constants, and both stay in the digest, so a
    manifest hashes as it always has.
    """
    payload = {
        "coverage": COMPLETE_SNAPSHOT,
        "items": sorted(
            ({"doc_id": item.doc_id, "revision": item.revision, "change_kind": UPSERT} for item in items),
            key=lambda item: item["doc_id"],
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_snapshot_manifest_values(coverage: str, manifest_items: Sequence[tuple[str, str, str]]) -> None:
    """Check that a manifest declares complete coverage and that every item is an upsert."""
    if coverage != COMPLETE_SNAPSHOT:
        raise ValueError("snapshot manifest coverage is invalid")
    if any(change_kind != UPSERT for _doc_id, _revision, change_kind in manifest_items):
        raise ValueError("snapshot manifest change_kind is invalid")


class SnapshotConfirmationAnswer(str, Enum):
    """A device's answer for one item the server asked it to confirm.

    ``absent``, ``exists`` and ``hidden`` are ``confirm_absent``'s answers for
    the item. ``failed`` means ``confirm_absent`` raised; every issued item
    then carries it with the cause.
    """

    ABSENT = "absent"
    EXISTS = "exists"
    HIDDEN = "hidden"
    FAILED = "failed"


@dataclass(frozen=True)
class SnapshotConfirmation:
    """One unlisted item a snapshot asks its device to confirm, and the answer once stored.

    The server issues it with the manifest: ``doc_id`` is the Document id and
    ``provider_key`` the provider's key of the item, by which the device
    confirms it (for Jira, the issue key, such as ``PAY-7``). ``answer`` is
    ``None`` until the device answers; ``cause`` accompanies only ``failed``.
    """

    doc_id: str
    provider_key: str
    answer: SnapshotConfirmationAnswer | None = None
    cause: str | None = None

    def __post_init__(self) -> None:
        if not self.doc_id or not self.provider_key:
            raise ValueError("snapshot confirmation requires doc_id and provider_key")
        if (self.answer is SnapshotConfirmationAnswer.FAILED) != bool(self.cause):
            raise ValueError("snapshot confirmation carries a cause exactly when it failed")


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


def validate_snapshot_manifest_extras(
    manifest_items: Sequence[tuple[str, str, str]],
    confirm: Sequence[SnapshotConfirmation],
    excluded_listings: Sequence[ExcludedScopeListing],
) -> None:
    """Check a manifest's issued confirmations and excluded listings before they are stored."""
    confirm_doc_ids = [item.doc_id for item in confirm]
    if len(set(confirm_doc_ids)) != len(confirm_doc_ids):
        raise ValueError("snapshot confirmations contain duplicate doc_id")
    if any(item.answer is not None for item in confirm):
        raise ValueError("snapshot confirmations are issued unanswered")
    if set(confirm_doc_ids) & {doc_id for doc_id, _revision, _change_kind in manifest_items}:
        raise ValueError("snapshot confirmations name only items the manifest does not list")
    scope_keys = [listing.scope_key for listing in excluded_listings]
    if len(set(scope_keys)) != len(scope_keys):
        raise ValueError("snapshot excluded listings contain duplicate scope_key")


def excluded_listing_rows(
    listings: Sequence[ExcludedScopeListing],
) -> list[tuple[str, str | None, str | None]]:
    """The stored rows of excluded listings: one scope row each, then one row per listed Document."""
    rows: list[tuple[str, str | None, str | None]] = []
    for listing in listings:
        rows.append((listing.scope_key, None, listing.rejection))
        rows.extend((listing.scope_key, doc_id, None) for doc_id in sorted(listing.doc_ids))
    return rows


def excluded_listings_from_rows(
    rows: Iterable[tuple[str, str | None, str | None]],
) -> tuple[ExcludedScopeListing, ...]:
    rejections: dict[str, str | None] = {}
    doc_ids: dict[str, set[str]] = {}
    for scope_key, doc_id, rejection in rows:
        if doc_id is None:
            rejections[scope_key] = rejection
        else:
            doc_ids.setdefault(scope_key, set()).add(doc_id)
    return tuple(
        ExcludedScopeListing(
            scope_key=scope_key,
            doc_ids=frozenset(doc_ids.get(scope_key, ())),
            rejection=rejections[scope_key],
        )
        for scope_key in sorted(rejections)
    )


def snapshot_confirmation(
    doc_id: object,
    provider_key: object,
    answer: object,
    cause: object,
) -> SnapshotConfirmation:
    return SnapshotConfirmation(
        doc_id=str(doc_id),
        provider_key=str(provider_key),
        answer=SnapshotConfirmationAnswer(str(answer)) if answer is not None else None,
        cause=str(cause) if cause is not None else None,
    )


def validate_snapshot_confirmation_submission(
    answers: Sequence[SnapshotConfirmation],
    failure_cause: str | None,
) -> None:
    """Check the shape of a device's confirmation answers, before the issued set is read."""
    if failure_cause is not None:
        if not failure_cause or answers:
            raise ValueError("a failed confirmation carries its cause and no answers")
        return
    if any(
        item.answer not in {
            SnapshotConfirmationAnswer.ABSENT,
            SnapshotConfirmationAnswer.EXISTS,
            SnapshotConfirmationAnswer.HIDDEN,
        }
        for item in answers
    ):
        raise ValueError("each confirmation answer is absent, exists or hidden")
    doc_ids = [item.doc_id for item in answers]
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("snapshot confirmations answer an item twice")


def snapshot_confirmation_updates(
    *,
    manifest: tuple[str, str, int] | None,
    submitted: tuple[str, str, int],
    issued: Sequence[SnapshotConfirmation],
    answers: Sequence[SnapshotConfirmation],
    failure_cause: str | None,
) -> tuple[SnapshotConfirmation, ...]:
    """The issued confirmations a valid submission answers now.

    ``manifest`` and ``submitted`` are (digest, job id, attempt count). The
    result is empty for an identical replay. Raises for a missing manifest,
    another digest or attempt, an unissued or missing item, and answers that
    differ from the stored ones.
    """
    if manifest is None:
        raise ValueError("source snapshot manifest not found")
    if manifest[0] != submitted[0]:
        raise SourceActivityConflict("snapshot confirmations answer another manifest digest")
    if manifest[1:] != submitted[1:]:
        raise SourceActivityConflict("snapshot confirmations belong to another collection attempt")
    issued_keys = {item.doc_id: item.provider_key for item in issued}
    if failure_cause is not None:
        wanted = {
            item.doc_id: replace(item, answer=SnapshotConfirmationAnswer.FAILED, cause=failure_cause)
            for item in issued
        }
    else:
        for item in answers:
            if issued_keys.get(item.doc_id) != item.provider_key:
                raise ValueError(f"snapshot confirmation was not issued: {item.doc_id}")
        wanted = {item.doc_id: item for item in answers}
        if set(wanted) != set(issued_keys):
            raise ValueError("snapshot confirmations must answer every issued item")
    pending: list[SnapshotConfirmation] = []
    for item in issued:
        answer = wanted[item.doc_id]
        if item.answer is None:
            pending.append(answer)
        elif (item.answer, item.cause) != (answer.answer, answer.cause):
            raise SourceActivityConflict("snapshot confirmations were already answered differently")
    return tuple(pending)
