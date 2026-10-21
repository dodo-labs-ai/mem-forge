"""Conversation Unit payloads for tests that project Teams messages directly."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from memforge.conversations.membership import PlannedUnit
from memforge.conversations.projection import unit_payload
from memforge.conversations.records import MessageIdentity, record_created_at, record_revision
from memforge.conversations.units import Member, UnitKind
from memforge.genes.teams_provider import teams_conversation_label
from memforge.models import ContentItem, RawContent
from memforge.source_adapters.teams import MESSAGE_SCHEMA, message_record

TEAMS_RECORD_SCHEMA = f"{MESSAGE_SCHEMA.name}:{MESSAGE_SCHEMA.version}"


def teams_unit_payload(
    messages: Sequence[Mapping[str, object]],
    *,
    conversation_id: str = "conv-1",
    conversation_name: str | None = None,
    conversation_type: str = "group_chat",
    provider_key: str = "window-1",
    kind: UnitKind = UnitKind.WINDOW,
    thread_root_id: str | None = None,
) -> dict[str, object]:
    """The conversation Unit payload of Teams window messages, each recorded as Teams records it."""

    conversation = teams_conversation_label(
        {"id": conversation_id, "topic": conversation_name, "type": conversation_type}
    )
    members = []
    for message in messages:
        record = message_record(message, conversation_id=conversation_id, conversation_name=conversation_name)
        members.append(
            Member(
                identity=MessageIdentity(conversation_id, str(message["id"])),
                created_at=record_created_at(record),
                record=record,
                revision=record_revision(record),
            )
        )
    unit = PlannedUnit(
        kind=kind,
        provider_key=provider_key,
        conversation=conversation,
        members=tuple(members),
        thread_root_id=thread_root_id if kind is UnitKind.THREAD else None,
    )
    return unit_payload(unit, record_schema=TEAMS_RECORD_SCHEMA)


def teams_unit_raw(item: ContentItem, messages: Sequence[Mapping[str, object]], **unit) -> RawContent:
    payload = teams_unit_payload(messages, **unit)
    return RawContent(item=item, body=json.dumps(payload).encode("utf-8"), content_type="application/json")
