"""Teams-owned immutable record fields and comparison semantics."""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from typing import Mapping

from memforge.conversations.units import UnitKind
from memforge.source_projection import provider_item_key
from memforge.source_adapters.contracts import CanonicalRecordField, CanonicalRecordSchema, canonical_record_profile


CANONICAL_RECORD_SCHEMAS: Mapping[tuple[str, int], CanonicalRecordSchema] = {
    ("teams-message", 1): CanonicalRecordSchema(
        name="teams-message",
        version=1,
        fields=(CanonicalRecordField("/content", nested_profile="markdown-structural"),),
        tombstone_pointer="/deleted",
    ),
}


MESSAGE_SCHEMA = replace(
    CANONICAL_RECORD_SCHEMAS[("teams-message", 1)], version=2,
    fields=(CanonicalRecordField("/content", nested_profile="markdown-structural"),
            *(CanonicalRecordField(pointer, contextual=True) for pointer in
              ("/from", "/time", "/edited_time", "/reply_to_id", "/conversation_id", "/conversation_name"))),
    framing_fields=(("/from", "Reported sender"), ("/time", "Message created"),
                    ("/edited_time", "Message edited"), ("/reply_to_id", "Reply referent"),
                    ("/conversation_id", "Conversation ID"), ("/conversation_name", "Conversation")),
    model_interpretation="Sender and message times are supplied source framing. Missing sender or a reported Unknown sender does not establish identity. Message time is not a rule effective date. An authenticated reply referent establishes a relationship; chronological adjacency alone does not.",
)
CANONICAL_RECORD_SCHEMAS = {**CANONICAL_RECORD_SCHEMAS, (MESSAGE_SCHEMA.name, MESSAGE_SCHEMA.version): MESSAGE_SCHEMA}
CURRENT_OBSERVATION_PROFILES = {"message": canonical_record_profile(MESSAGE_SCHEMA.name, MESSAGE_SCHEMA.version)}
LEGACY_OBSERVATION_PROFILES = {"message": canonical_record_profile(MESSAGE_SCHEMA.name, 1)}


def teams_unit_shape(provider_key: str) -> tuple[UnitKind, str | None]:
    """The Unit Type of a Teams Unit by its provider key, and a thread's root message id.

    Teams keys a thread ``teams-thread:v1:`` and a window ``teams-block:v1:``,
    each followed by the base64url JSON of its source, conversation and root
    or anchor message. A Unit that came back after its removal carries its
    incarnation (:func:`~memforge.source_projection.provider_item_key`).
    """

    prefix, _, token = provider_item_key(provider_key).partition(":v1:")
    if prefix == "teams-block":
        return UnitKind.WINDOW, None
    if prefix != "teams-thread" or not token:
        raise ValueError(f"not a Teams Unit key: {provider_key!r}")
    identity = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    return UnitKind.THREAD, str(identity["root_or_anchor_message_id"])


def reply_referent(message: Mapping[str, object]) -> object:
    """Retain a supplied immediate referent; never infer a reply from ordering."""
    referent = (message.get("reply_to_id") or message.get("replyToId")
                or message.get("parentMessageId") or message.get("rootMessageId"))
    return referent if referent != message.get("id") else None


def message_record(
    message: Mapping[str, object], *, conversation_id: str, conversation_name: object
) -> dict[str, object]:
    """Retain only supplied message framing and scoped reply identity."""
    return {
        "content": message.get("content"), "attachments": message.get("attachments"),
        "deleted": message.get("deletedDateTime") or message.get("deleted_at"),
        "from": message.get("from"), "time": message.get("time"),
        "edited_time": message.get("edited_time"),
        "reply_to_id": reply_referent(message),
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "representation": MESSAGE_SCHEMA.qualified_name,
    }


