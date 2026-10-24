"""Message identity and the fields read from a message record.

A message is identified by its conversation and its provider message id
together: a Teams message id is a millisecond timestamp unique only within a
chat. Its Observation's provider key is the message id, and its locator
records the conversation.

Each provider declares the canonical record schema of its messages. Every
schema states the creation time at ``/time`` and the edit time at
``/edited_time``; the conversation layer reads only the creation time and
treats the rest of a record as opaque content, and projection reads the
reply referent, sender and content to present it. Nothing rewrites a stored
record, so a member carried into a Unit's next revision keeps its Observation
revision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from memforge.source_time import parse_source_time

MESSAGE_OBSERVATION_TYPE = "message"


@dataclass(frozen=True, slots=True, order=True)
class MessageIdentity:
    """One message: its conversation and its provider message id."""

    conversation_id: str
    message_id: str

    def __post_init__(self) -> None:
        if not self.conversation_id or not self.message_id:
            raise ValueError("a message identity needs its conversation and message id")


def canonical_record_json(record: Mapping[str, object]) -> str:
    """The canonical JSON of a record: sorted keys, no whitespace, UTF-8 text."""

    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def record_revision(record: Mapping[str, object]) -> str:
    """The revision of a mapped record: the SHA-256 of its canonical JSON.

    A provider change and a mapping change both produce a new revision.
    """

    return hashlib.sha256(canonical_record_json(record).encode("utf-8")).hexdigest()


def record_created_at(record: Mapping[str, object]) -> datetime:
    """A record's creation time, from ``/time``; a record without one is invalid."""

    created_at = parse_source_time(record.get("time"))
    if created_at is None:
        raise ValueError("a message record needs its creation time")
    return created_at


def record_source_time(record: Mapping[str, object]) -> object:
    """A record's source time: its edit time, else its creation time."""

    return record.get("edited_time") or record.get("time")
