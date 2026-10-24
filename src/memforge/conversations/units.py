"""Thread and window Source Units, and their members.

A thread Unit holds a root and all its replies, because a reply is read with
the message it answers from the same Unit. A window Unit holds top-level
messages outside any thread, cut by the Window Policy. A Unit's provider key
is computed once, when the Unit is created; afterwards the Unit is found by
its members' identities, never by recomputing a key, so Units another key
scheme created keep their keys.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from memforge.conversations.records import MessageIdentity
from memforge.conversations.undated import is_undated


class UnitKind(str, Enum):
    """The Unit Type of a conversation Source Unit."""

    THREAD = "conversation_thread"
    WINDOW = "conversation_window"


CONVERSATION_UNIT_TYPES = frozenset(kind.value for kind in UnitKind)


def thread_unit_key(*, source_id: str, conversation_id: str, thread_root_id: str) -> str:
    """The provider key of a new thread Unit."""

    return "conv-thread:v1:" + _canonical_token(
        {"source_id": source_id, "conversation_id": conversation_id, "thread_ref": thread_root_id}
    )


def window_unit_key(*, source_id: str, conversation_id: str, anchor_message_id: str) -> str:
    """The provider key of a new window Unit, named by the message that opened it."""

    return "conv-window:v1:" + _canonical_token(
        {"source_id": source_id, "conversation_id": conversation_id, "anchor_message_id": anchor_message_id}
    )


def _canonical_token(value: Mapping[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


@dataclass(frozen=True, slots=True)
class MemberArtifact:
    """A stored binary of one member, such as an inline image, with the bytes that hold it."""

    provider_key: str
    provider_revision: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    uri: str


@dataclass(frozen=True, slots=True)
class Member:
    """One message of a Unit: its identity, creation time, canonical record and stored artifacts.

    ``undated`` says whether ``record`` carries no creation time of its own
    (:mod:`~memforge.conversations.undated`); it is derived from the record
    when the member is built.
    """

    identity: MessageIdentity
    created_at: datetime
    record: Mapping[str, object]
    # The revision of ``record``, as the manifest names it.
    revision: str
    artifacts: tuple[MemberArtifact, ...] = ()
    undated: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "undated", is_undated(self.record))

    @property
    def content_identity(self) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        """What a Unit revision depends on for this member: its record and the bytes of its artifacts."""

        return self.revision, tuple(
            sorted((artifact.provider_key, artifact.provider_revision, artifact.sha256) for artifact in self.artifacts)
        )


@dataclass(frozen=True, slots=True)
class UnitRef:
    """One current conversation Unit, without its members.

    ``thread_root_id`` is set for a thread Unit.
    """

    unit_id: str
    document_id: str
    provider_key: str
    kind: UnitKind
    conversation_id: str
    thread_root_id: str | None = None

    def __post_init__(self) -> None:
        if (self.kind is UnitKind.THREAD) != (self.thread_root_id is not None):
            raise ValueError("a thread Unit, and only a thread Unit, names its root")


@dataclass(frozen=True, slots=True)
class StoredUnit:
    """One current conversation Unit with its complete current membership."""

    ref: UnitRef
    members: tuple[Member, ...]

    def __post_init__(self) -> None:
        if any(member.identity.conversation_id != self.ref.conversation_id for member in self.members):
            raise ValueError("every member of a Unit belongs to its conversation")

    @property
    def document_id(self) -> str:
        return self.ref.document_id

    @property
    def provider_key(self) -> str:
        return self.ref.provider_key

    @property
    def kind(self) -> UnitKind:
        return self.ref.kind

    @property
    def thread_root_id(self) -> str | None:
        return self.ref.thread_root_id


@dataclass(frozen=True, slots=True)
class Span:
    """The earliest and latest creation time over a Unit's members."""

    earliest: datetime
    latest: datetime

    @classmethod
    def of(cls, members: tuple[Member, ...] | list[Member]) -> Span | None:
        if not members:
            return None
        times = [member.created_at for member in members]
        return cls(earliest=min(times), latest=max(times))
