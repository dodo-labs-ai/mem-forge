"""Evaluation snapshots: replay what a source type's collection hands the pipeline.

The public quality cohort cannot reach real provider accounts, so every
registered source type keeps a synthetic snapshot instead: what its
collection hands the rest of the pipeline for one item at one revision.
Evaluation owns the acquisition that replays it, and nothing else in the
product uses it.

- A document type's snapshot is the stored input of one Unit revision: the
  item its Gene discovered, the raw bytes and their content type, the PDF
  export when the type exports one, and the artifacts. A snapshot Gene
  (:func:`snapshot_gene_class`) subclasses the type's real Gene and replaces
  only how it acquires items: ``authenticate``, ``discover``, ``rediscover``,
  ``fetch``, ``fetch_pdf`` and ``open_source_artifact``, with the listing and
  the absence answers a discovery implies. Normalization, ``project_native``,
  readings, derivation and Support are the type's own.
- A conversation type's snapshot is the provider's messages for one
  conversation, in the type's canonical record schema. A
  :class:`SnapshotConversationProvider` implements the conversation provider
  port over them, and the real device executor runs it, so the Collection
  Manifest, proven intervals and absence rules stay the conversation layer's.

A Source reads snapshots when its configuration names a set under
:data:`SNAPSHOT_CONFIG_FIELD`::

    {"evaluation_snapshot": {"set": "jira", "revision": 1, "replayed_at": "2026-10-30T09:00:00+00:00"}}

``set`` is a directory below the deployment's snapshot root, holding one
directory per item and one numbered directory per revision of it::

    <root>/<set>/<item>/<revision>/item.json

An item at revision ``N`` is its highest revision directory not above ``N``,
so a revision directory holds only what changed. Item and revision
directories are the set's own, never links. ``replayed_at``
is when the Source began replaying, stamped when it is created unless the
configuration gives it; a set's message times move forward by whole days so
that its newest message lands in the day before it (:func:`replay_offset`),
which keeps every revision inside the Source's Initial History.

The acquisition is off by default. A deployment turns it on by naming the
snapshot root in ``MEMFORGE_EVALUATION_SNAPSHOT_ROOT``; while it is unset,
creating or syncing a Source that names a set is refused, and a set must
resolve below the root. The construction seams choose the acquisition from
the Source (:func:`memforge.genes.gene_class`, the conversation provider
reference of the type's descriptor, and the executor choice of
:func:`memforge.local_agent.source_contract.local_agent_sync_operation`), so
the sync pipeline never branches on it.
"""

from __future__ import annotations

import functools
import json
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memforge.conversations.port import (
    ArtifactBody,
    ConversationAccess,
    ConversationLabel,
    ConversationListing,
    ConversationPoll,
    ConversationRef,
    ConversationStatus,
    Interval,
    MessageArtifact,
    PollOutcome,
    ProviderMessage,
    ThreadingModel,
)
from memforge.genes.base import SourceConfigurationError
from memforge.genes.document_gene import DocumentGene
from memforge.models import (
    AbsenceAnswer,
    ContentItem,
    RawContent,
    ScopeListing,
    ScopeListingKind,
)
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactDownload
from memforge.source_time import parse_source_time

if TYPE_CHECKING:
    from memforge.source_adapters.contracts import CanonicalRecordSchema

__all__ = [
    "SNAPSHOT_CONFIG_FIELD",
    "SNAPSHOT_ROOT_VARIABLE",
    "ConversationSnapshot",
    "DocumentSnapshot",
    "SnapshotConversationProvider",
    "SnapshotReference",
    "SnapshotsDisabled",
    "complete_snapshot_config",
    "conversation_snapshots",
    "document_snapshots",
    "item_dirs",
    "open_snapshot_provider",
    "replay_offset",
    "revision_dirs",
    "snapshot_gene_class",
    "snapshot_reference",
    "snapshot_root",
    "snapshot_set_dir",
]

SNAPSHOT_ROOT_VARIABLE = "MEMFORGE_EVALUATION_SNAPSHOT_ROOT"
"""The deployment setting that turns snapshot replay on by naming the directory every set lies below."""

SNAPSHOT_CONFIG_FIELD = "evaluation_snapshot"
"""The Source configuration field that names the set a Source replays and its revision."""

ITEM_FILE = "item.json"
MESSAGES_FILE = "messages.json"

# A replayed conversation's newest message lands at least this long before the Source's replay began.
REPLAY_LEAD = timedelta(hours=1)
_REVISION_NAME = re.compile(r"[1-9][0-9]*")
# A poll reaches its start: the snapshot holds the conversation's whole history.
_BEFORE_START = timedelta(milliseconds=1)


class SnapshotsDisabled(SourceConfigurationError):
    """A Source names an evaluation snapshot set, and the deployment does not replay them."""


class SnapshotUnavailable(SourceConfigurationError):
    """A Source names an evaluation snapshot set or revision that cannot be read."""


@dataclass(frozen=True, slots=True)
class SnapshotReference:
    """The set one Source replays, the revision it replays now, and when its replay began."""

    set: str
    revision: int
    replayed_at: datetime | None = None

    def to_config(self) -> dict[str, object]:
        value: dict[str, object] = {"set": self.set, "revision": self.revision}
        if self.replayed_at is not None:
            value["replayed_at"] = self.replayed_at.isoformat()
        return value


def snapshot_reference(config: Mapping[str, Any] | None) -> SnapshotReference | None:
    """The snapshot set a Source configuration names, or ``None`` for a Source that reads its provider.

    A Source names a set by holding :data:`SNAPSHOT_CONFIG_FIELD` at all, so
    a malformed reference, ``null`` included, raises
    :class:`SnapshotUnavailable`.
    """

    config = config or {}
    if SNAPSHOT_CONFIG_FIELD not in config:
        return None
    value = config[SNAPSHOT_CONFIG_FIELD]
    if not isinstance(value, Mapping):
        raise SnapshotUnavailable(f"{SNAPSHOT_CONFIG_FIELD} must name a set and a revision")
    set_name = str(value.get("set") or "").strip()
    revision = value.get("revision")
    if not set_name:
        raise SnapshotUnavailable(f"{SNAPSHOT_CONFIG_FIELD} must name a set")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise SnapshotUnavailable(f"{SNAPSHOT_CONFIG_FIELD} revision must be a whole number from 1")
    replayed_at = value.get("replayed_at")
    parsed = parse_source_time(replayed_at) if replayed_at is not None else None
    if replayed_at is not None and parsed is None:
        raise SnapshotUnavailable(f"{SNAPSHOT_CONFIG_FIELD} replayed_at must be an offset-aware time")
    return SnapshotReference(set=set_name, revision=revision, replayed_at=parsed)


def snapshot_root() -> Path | None:
    """The deployment's snapshot root, or ``None`` while replay is off."""

    value = os.environ.get(SNAPSHOT_ROOT_VARIABLE, "").strip()
    return Path(value).expanduser().resolve() if value else None


def snapshot_set_dir(reference: SnapshotReference) -> Path:
    """The directory of the set a Source names, refusing it while replay is off or outside the root."""

    root = snapshot_root()
    if root is None:
        raise SnapshotsDisabled(
            f"this Source replays the evaluation snapshot set {reference.set!r}, and this deployment does not "
            f"replay evaluation snapshots: set {SNAPSHOT_ROOT_VARIABLE} to enable them"
        )
    set_dir = (root / reference.set).resolve()
    if not set_dir.is_relative_to(root) or set_dir == root:
        raise SnapshotUnavailable(f"the evaluation snapshot set {reference.set!r} is not below the snapshot root")
    if not set_dir.is_dir():
        raise SnapshotUnavailable(f"the evaluation snapshot set {reference.set!r} does not exist")
    return set_dir


def complete_snapshot_config(
    config: dict[str, Any], existing_config: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Check the set a configuration names before it is stored, and stamp when its replay began.

    A Source replays a set from its creation or never, and always the same
    set; only its revision changes. The set must exist and hold the revision,
    which requires replay to be on. Raises ``ValueError`` with the reason.
    """

    try:
        existing = snapshot_reference(existing_config) if existing_config is not None else None
        reference = snapshot_reference(config)
        if existing_config is not None and (reference is None) != (existing is None):
            raise ValueError("a Source replays an evaluation snapshot set from its creation or never")
        if reference is None:
            return config
        if existing is not None and existing.set != reference.set:
            raise ValueError("a Source keeps the evaluation snapshot set it was created with")
        set_dir = snapshot_set_dir(reference)
    except SourceConfigurationError as exc:
        raise ValueError(str(exc)) from exc
    if not _set_holds_revision(set_dir, reference.revision):
        raise ValueError(f"the evaluation snapshot set {reference.set!r} has no revision {reference.revision}")
    replayed_at = (existing.replayed_at if existing is not None else None) or reference.replayed_at
    config[SNAPSHOT_CONFIG_FIELD] = replace(
        reference, replayed_at=replayed_at or datetime.now(timezone.utc)
    ).to_config()
    return config


# -- Reading a set ------------------------------------------------------------------------


def _own_directories(parent: Path) -> list[Path]:
    """The visible directories of ``parent``; a link is refused, since a set holds its own files."""

    directories = []
    for path in parent.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_symlink():
            raise SnapshotUnavailable(f"evaluation snapshot directory {path} is a link")
        if path.is_dir():
            directories.append(path)
    return directories


def revision_dirs(item_dir: Path) -> list[tuple[int, Path]]:
    """Each numbered revision directory of one item, in revision order; any other directory is refused."""

    revisions = []
    for path in _own_directories(item_dir):
        if not _REVISION_NAME.fullmatch(path.name):
            raise SnapshotUnavailable(
                f"evaluation snapshot directory {path} is not a revision: name it 1, 2 and so on"
            )
        revisions.append((int(path.name), path))
    return sorted(revisions)


def _item_revision_dir(item_dir: Path, revision: int) -> Path | None:
    """The directory holding an item at ``revision``: its highest revision not above it."""

    held = [path for number, path in revision_dirs(item_dir) if number <= revision]
    return held[-1] if held else None


def item_dirs(set_dir: Path) -> list[Path]:
    """Each item directory of a set, in name order."""

    return sorted(_own_directories(set_dir))


def _set_holds_revision(set_dir: Path, revision: int) -> bool:
    return any(number == revision for item_dir in item_dirs(set_dir) for number, _ in revision_dirs(item_dir))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotUnavailable(f"evaluation snapshot file {path} cannot be read: {exc}") from exc


def _contained(base: Path, relative: object) -> Path:
    """A file a snapshot names, which must resolve, through any link, to a file inside ``base``."""

    path = (base / str(relative)).resolve()
    if not path.is_relative_to(base.resolve()) or not path.is_file():
        raise SnapshotUnavailable(f"evaluation snapshot file {relative!r} is not a file in {base}")
    return path


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    """One artifact of a snapshot: its provider descriptor and the file that holds its bytes."""

    descriptor: RawSourceArtifact
    path: Path


@dataclass(frozen=True, slots=True)
class DocumentSnapshot:
    """The stored input of one document item at one revision."""

    item: ContentItem
    body: bytes
    content_type: str
    pdf: bytes | None
    artifacts: tuple[SnapshotArtifact, ...]
    directory: Path


def read_document_snapshot(revision_dir: Path) -> DocumentSnapshot:
    """One document item's snapshot, from the revision directory that holds it."""

    spec = _read_json(revision_dir / ITEM_FILE)
    try:
        item = ContentItem.from_payload(dict(spec["item"]))
        raw = spec["raw"]
        raw_path = _contained(revision_dir, raw["file"])
        content_type = str(raw["content_type"])
        pdf = _contained(revision_dir, spec["pdf"]).read_bytes() if spec.get("pdf") else None
        artifacts = []
        for entry in spec.get("artifacts") or ():
            path = _contained(revision_dir, entry["file"])
            artifacts.append(
                SnapshotArtifact(
                    descriptor=RawSourceArtifact(
                        provider_key=str(entry["provider_key"]),
                        parent_observation_type=str(entry["parent_observation_type"]),
                        parent_provider_key=str(entry["parent_provider_key"]),
                        provider_revision=str(entry["provider_revision"]),
                        filename=str(entry.get("filename") or path.name),
                        media_type=str(entry["media_type"]),
                        declared_size_bytes=path.stat().st_size,
                        locator={"snapshot_file": str(path.relative_to(revision_dir.parent.parent))},
                    ),
                    path=path,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotUnavailable(f"evaluation snapshot {revision_dir / ITEM_FILE} is malformed: {exc}") from exc
    return DocumentSnapshot(
        item=item,
        body=raw_path.read_bytes(),
        content_type=content_type,
        pdf=pdf,
        artifacts=tuple(artifacts),
        directory=revision_dir,
    )


def document_snapshots(set_dir: Path, revision: int) -> tuple[DocumentSnapshot, ...]:
    """Every document item of a set as it stands at ``revision``, ordered by item directory."""

    snapshots = []
    for item_dir in item_dirs(set_dir):
        revision_dir = _item_revision_dir(item_dir, revision)
        if revision_dir is not None:
            snapshots.append(read_document_snapshot(revision_dir))
    item_ids = [snapshot.item.item_id for snapshot in snapshots]
    if len(set(item_ids)) != len(item_ids):
        raise SnapshotUnavailable(f"evaluation snapshot set {set_dir} names one item id in two item directories")
    return tuple(snapshots)


@dataclass(frozen=True, slots=True)
class SnapshotMessage:
    """One provider message of a conversation snapshot, with the files of its artifacts."""

    message: ProviderMessage
    artifacts: Mapping[MessageArtifact, tuple[Path, str, str]]


@dataclass(frozen=True, slots=True)
class ConversationSnapshot:
    """The provider's messages for one conversation at one revision."""

    ref: ConversationRef
    messages: tuple[SnapshotMessage, ...]
    directory: Path


def _conversation_ref(spec: Mapping[str, Any]) -> ConversationRef:
    label = spec["conversation"]
    return ConversationRef(
        label=ConversationLabel(
            conversation_id=str(label["conversation_id"]),
            kind=str(label["kind"]),
            name=label.get("name"),
            title_kind=str(label.get("title_kind") or "Conversation"),
            title_fields=tuple((str(key), str(value)) for key, value in label.get("title_fields") or ()),
            message_link=label.get("message_link"),
        ),
        threading=ThreadingModel(spec.get("threading") or ThreadingModel.INLINE.value),
        poll_includes_replies=bool(spec.get("poll_includes_replies", True)),
    )


def _shifted(value: object, offset: timedelta) -> object:
    parsed = parse_source_time(value)
    return (parsed + offset).isoformat() if parsed is not None else value


def read_conversation_snapshot(revision_dir: Path, *, offset: timedelta = timedelta(0)) -> ConversationSnapshot:
    """One conversation's snapshot, its message times moved by ``offset``.

    The move applies to each message's creation time and to the record's
    ``/time`` and ``/edited_time``, the times the conversation layer reads
    from every message record.
    """

    spec = _read_json(revision_dir / ITEM_FILE)
    entries = _read_json(revision_dir / MESSAGES_FILE)
    try:
        ref = _conversation_ref(spec)
        messages = []
        for entry in entries:
            record = dict(entry["record"])
            for key in ("time", "edited_time"):
                if record.get(key):
                    record[key] = _shifted(record[key], offset)
            created_at = parse_source_time(entry["created_at"])
            if created_at is None:
                raise ValueError(f"message {entry.get('message_id')!r} needs an offset-aware created_at")
            artifacts: dict[MessageArtifact, tuple[Path, str, str]] = {}
            for artifact in entry.get("artifacts") or ():
                path = _contained(revision_dir, artifact["file"])
                key = MessageArtifact(str(artifact["provider_key"]), str(artifact["provider_revision"]))
                artifacts[key] = (path, str(artifact["media_type"]), str(artifact.get("filename") or path.name))
            messages.append(
                SnapshotMessage(
                    message=ProviderMessage(
                        message_id=str(entry["message_id"]),
                        created_at=created_at + offset,
                        record=record,
                        thread_root_id=entry.get("thread_root_id"),
                        artifacts=tuple(sorted(artifacts)),
                    ),
                    artifacts=artifacts,
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotUnavailable(f"evaluation snapshot {revision_dir} is malformed: {exc}") from exc
    return ConversationSnapshot(ref=ref, messages=tuple(messages), directory=revision_dir)


def _newest_message_time(set_dir: Path) -> datetime:
    """The newest message creation time across every conversation and revision of a set."""

    times = [
        parse_source_time(entry.get("created_at"))
        for item_dir in item_dirs(set_dir)
        for _, revision_dir in revision_dirs(item_dir)
        for entry in _read_json(revision_dir / MESSAGES_FILE)
    ]
    known = [time for time in times if time is not None]
    if not known:
        raise SnapshotUnavailable(f"evaluation snapshot set {set_dir} holds no message")
    return max(known)


def replay_offset(newest: datetime, replayed_at: datetime) -> timedelta:
    """How far a set's message times move: the whole days that bring its newest message closest before replay.

    The newest message across all conversations and revisions lands within a day before
    :data:`REPLAY_LEAD` ahead of ``replayed_at``. Whole days keep each
    message's time of day, so a Unit's title, which names its first
    message's time, changes only its date.
    """

    return timedelta(days=(replayed_at - REPLAY_LEAD - newest) // timedelta(days=1))


def conversation_snapshots(
    set_dir: Path, revision: int, *, replayed_at: datetime | None = None
) -> tuple[ConversationSnapshot, ...]:
    """Every conversation of a set as it stands at ``revision``.

    With ``replayed_at``, every time in the set moves by one
    :func:`replay_offset`, whatever the conversation or revision, so a
    message keeps its revision and a Unit's title its time of day.
    """

    offset = replay_offset(_newest_message_time(set_dir), replayed_at) if replayed_at is not None else timedelta(0)
    snapshots = []
    for item_dir in item_dirs(set_dir):
        revision_dir = _item_revision_dir(item_dir, revision)
        if revision_dir is not None:
            snapshots.append(read_conversation_snapshot(revision_dir, offset=offset))
    conversation_ids = [snapshot.ref.conversation_id for snapshot in snapshots]
    if len(set(conversation_ids)) != len(conversation_ids):
        raise SnapshotUnavailable(f"evaluation snapshot set {set_dir} names one conversation in two item directories")
    return tuple(snapshots)


# -- Document family: the snapshot Gene ------------------------------------------------------


@functools.cache
def snapshot_gene_class(gene_cls: type[DocumentGene]) -> type[DocumentGene]:
    """The snapshot Gene of a document type: its real Gene, acquiring items from a snapshot set.

    Everything but acquisition is inherited, so normalization, the PDF
    requirement and the stored-input reading are the type's own.
    """

    class SnapshotGene(gene_cls):  # type: ignore[valid-type, misc]
        __doc__ = f"{gene_cls.__name__} replaying an evaluation snapshot set."

        def __init__(self, config: dict, source_id: str) -> None:
            super().__init__(config, source_id)
            reference = snapshot_reference(config)
            assert reference is not None, "a snapshot Gene is created only for a Source that names a set"
            self._snapshot_reference = reference
            self._snapshot_set_dir = snapshot_set_dir(reference)
            self._snapshots: dict[str, DocumentSnapshot] | None = None

        def _current(self) -> dict[str, DocumentSnapshot]:
            if self._snapshots is None:
                self._snapshots = {
                    snapshot.item.item_id: snapshot
                    for snapshot in document_snapshots(self._snapshot_set_dir, self._snapshot_reference.revision)
                }
            return self._snapshots

        def _snapshot(self, item: ContentItem) -> DocumentSnapshot:
            snapshot = self._current().get(item.item_id)
            if snapshot is None:
                raise SnapshotUnavailable(
                    f"item {item.item_id} is not in revision {self._snapshot_reference.revision} of the "
                    f"evaluation snapshot set {self._snapshot_reference.set!r}"
                )
            return snapshot

        @classmethod
        def lists_scope(cls, config: Mapping[str, Any]) -> bool:
            """A snapshot set lists every item it holds."""
            return True

        async def rediscover(self, item: ContentItem) -> ContentItem | None:
            """The set's current item, for whenever the type's own ``rediscovers_documents`` lets the pipeline ask."""
            snapshot = self._current().get(item.item_id)
            return _discovered(snapshot) if snapshot is not None else None

        async def authenticate(self) -> None:
            """Nothing to authenticate: the set is read from the snapshot root."""
            self._snapshots = None
            self._current()

        async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
            """Every item of the set at the Source's revision; an unchanged one is not extracted again."""
            snapshots = self._current()
            for snapshot in snapshots.values():
                yield _discovered(snapshot)
            self.record_scope_listing(_listing_kind(gene_cls), frozenset(snapshots))

        async def list_scope(self) -> ScopeListing:
            return self._discovered_scope or ScopeListing(
                kind=_listing_kind(gene_cls), doc_ids=frozenset(self._current())
            )

        async def confirm_absent(self, items: Sequence[ContentItem]) -> Mapping[str, AbsenceAnswer]:
            held = self._current()
            return {
                item.item_id: AbsenceAnswer.EXISTS if item.item_id in held else AbsenceAnswer.ABSENT for item in items
            }

        async def fetch(self, item: ContentItem) -> RawContent:
            snapshot = self._snapshot(item)
            raw = self.raw_from_stored_input(item, snapshot.body, snapshot.content_type)
            return replace(raw, artifacts=tuple(artifact.descriptor for artifact in snapshot.artifacts))

        async def fetch_pdf(self, item: ContentItem) -> bytes | None:
            return self._snapshot(item).pdf

        @asynccontextmanager
        async def open_source_artifact(self, artifact: RawSourceArtifact) -> AsyncIterator[SourceArtifactDownload]:
            path = _contained(self._snapshot_set_dir, artifact.locator.get("snapshot_file"))
            yield SourceArtifactDownload(
                chunks=_chunks(path), media_type=artifact.media_type, content_length=path.stat().st_size
            )

        async def health_check(self) -> dict:
            return {
                "healthy": True,
                "evaluation_snapshot": self._snapshot_reference.set,
                "revision": self._snapshot_reference.revision,
                "items": len(self._current()),
            }

    SnapshotGene.__name__ = SnapshotGene.__qualname__ = f"Snapshot{gene_cls.__name__}"
    return SnapshotGene


def _discovered(snapshot: DocumentSnapshot) -> ContentItem:
    """A snapshot's item as discovery yields it: a copy the pipeline may change."""

    return replace(snapshot.item, extra=dict(snapshot.item.extra), labels=list(snapshot.item.labels))


def _listing_kind(gene_cls: type[DocumentGene]) -> ScopeListingKind:
    """The kind of listing a set is: existence when the real Gene declares it, since a set names all it holds."""

    if ScopeListingKind.EXISTENCE in gene_cls.listing_kinds or not gene_cls.listing_kinds:
        return ScopeListingKind.EXISTENCE
    return ScopeListingKind.QUERY


async def _chunks(path: Path) -> AsyncIterator[bytes]:
    yield path.read_bytes()


# -- Conversation family: the snapshot provider ----------------------------------------------


class SnapshotConversationProvider:
    """The conversation provider port over a snapshot set.

    The set holds each conversation's whole history at each revision, so a
    poll returns the messages created in its interval, newest first, and
    always reaches its start.
    """

    def __init__(self, record_schema: CanonicalRecordSchema, conversations: Sequence[ConversationSnapshot]) -> None:
        self.record_schema = record_schema
        self._conversations = {snapshot.ref.conversation_id: snapshot for snapshot in conversations}

    async def list_conversations(self) -> ConversationListing:
        refs = tuple(snapshot.ref for snapshot in self._conversations.values())
        return ConversationListing(
            conversations=refs,
            statuses={
                ref.conversation_id: ConversationStatus(ConversationAccess.READABLE, "in the evaluation snapshot")
                for ref in refs
            },
        )

    async def get_conversation(self, conversation_id: str) -> tuple[ConversationStatus, ConversationRef | None]:
        return ConversationStatus(ConversationAccess.GONE, "not in the evaluation snapshot"), None

    async def poll(self, conversation: ConversationRef, interval: Interval) -> ConversationPoll:
        snapshot = self._conversations[conversation.conversation_id]
        messages = sorted(
            (
                entry.message
                for entry in snapshot.messages
                if interval.start <= entry.message.created_at
                and (interval.end is None or entry.message.created_at < interval.end)
            ),
            key=lambda message: (message.created_at, message.message_id),
            reverse=True,
        )
        return ConversationPoll(
            messages=tuple(messages),
            outcome=PollOutcome.REACHED_START,
            earliest_complete_page_time=interval.start - _BEFORE_START,
        )

    async def read_artifact(
        self, conversation: ConversationRef, message: ProviderMessage, artifact: MessageArtifact
    ) -> ArtifactBody:
        snapshot = self._conversations[conversation.conversation_id]
        for entry in snapshot.messages:
            if entry.message.message_id == message.message_id and artifact in entry.artifacts:
                path, media_type, filename = entry.artifacts[artifact]
                return ArtifactBody(content=path.read_bytes(), media_type=media_type, filename=filename)
        raise SnapshotUnavailable(f"artifact {artifact.provider_key} of message {message.message_id} is not in the snapshot")


@asynccontextmanager
async def open_snapshot_provider(
    provider_cls: type, reference: SnapshotReference
) -> AsyncIterator[SnapshotConversationProvider]:
    """Open the snapshot provider of one device collection, in the record schema the type's provider collects."""

    if reference.replayed_at is None:
        raise SnapshotUnavailable(f"{SNAPSHOT_CONFIG_FIELD} must record when the Source's replay began")
    set_dir = snapshot_set_dir(reference)
    yield SnapshotConversationProvider(
        provider_cls.record_schema,
        conversation_snapshots(set_dir, reference.revision, replayed_at=reference.replayed_at),
    )

