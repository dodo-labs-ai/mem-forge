"""A planned conversation Unit as the sync run receives it: its payload, item and readable text.

Each thread and window Unit is one Document. Its stored input is the
conversation Unit payload: the Unit's kind, key and conversation label, and
the record and artifacts of every member in creation order. Projection
(:mod:`memforge.source_adapters.conversation`) reads only that payload, so
reprocessing never calls the provider.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from memforge.conversations.membership import PlannedUnit
from memforge.conversations.port import ConversationLabel
from memforge.conversations.records import MESSAGE_OBSERVATION_TYPE, record_source_time
from memforge.conversations.units import UnitKind
from memforge.models import ContentItem, NormalizedContent, RawContent, slugify
from memforge.source_artifacts import RawSourceArtifact
from memforge.source_time import SOURCE_UPDATED_AT_KEY, latest_source_time, parse_source_time

PAYLOAD_KIND = "conversation_unit"
PAYLOAD_VERSION = 1
CONTENT_TYPE = "application/json"


def conversation_document_id(source_id: str, provider_key: str) -> str:
    """The Document id of a new conversation Unit, from its provider key."""

    digest = hashlib.sha256(f"{source_id}|{provider_key}".encode("utf-8")).hexdigest()[:12]
    return "-".join(["conv", slugify(source_id)[:30], digest])


def unit_payload(unit: PlannedUnit, *, record_schema: str) -> dict[str, Any]:
    """The stored input of one planned Unit."""

    conversation = unit.conversation
    return {
        "payload_kind": PAYLOAD_KIND,
        "payload_version": PAYLOAD_VERSION,
        "record_schema": record_schema,
        "unit": {
            "kind": unit.kind.value,
            "provider_key": unit.provider_key,
            "thread_root_id": unit.thread_root_id,
        },
        "conversation": {
            "conversation_id": conversation.conversation_id,
            "kind": conversation.kind,
            "name": conversation.name,
            "title_kind": conversation.title_kind,
            "title_fields": [list(pair) for pair in conversation.title_fields],
            "message_link": conversation.message_link,
        },
        "members": [
            {
                "message_id": member.identity.message_id,
                "record": dict(member.record),
                "artifacts": [
                    {
                        "provider_key": artifact.provider_key,
                        "provider_revision": artifact.provider_revision,
                        "filename": artifact.filename,
                        "media_type": artifact.media_type,
                        "size_bytes": artifact.size_bytes,
                        "sha256": artifact.sha256,
                        "uri": artifact.uri,
                    }
                    for artifact in member.artifacts
                ],
            }
            for member in unit.members
        ],
    }


def unit_content_item(unit: PlannedUnit, *, source_id: str) -> ContentItem:
    """The ContentItem the sync run processes for one planned Unit."""

    span = unit.span
    document_id = unit.stored.document_id if unit.stored is not None else None
    membership = hashlib.sha256(
        json.dumps(
            [[member.identity.message_id, *member.content_identity] for member in unit.members],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    last_modified = parse_source_time(latest_source_time(record_source_time(member.record) for member in unit.members))
    return ContentItem(
        item_id=document_id or conversation_document_id(source_id, unit.provider_key),
        title=_title_text(unit.conversation, unit.kind, span.earliest),
        source_url=_message_link(unit.conversation, unit.members[0].identity.message_id),
        last_modified=last_modified or span.latest,
        content_type=CONTENT_TYPE,
        version=membership,
        extra={
            "conversation_id": unit.conversation.conversation_id,
            "unit_kind": unit.kind.value,
            "provider_key": unit.provider_key,
        },
    )


def unit_raw_content(item: ContentItem, payload: Mapping[str, Any], *, with_artifacts: bool) -> RawContent:
    """The raw content of one Unit payload; ``with_artifacts`` names each member's stored binaries to project.

    Reprocessing reads a Unit's artifacts from its committed revision, so the
    raw content rebuilt from stored input names none.
    """

    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    artifacts = (
        tuple(
            RawSourceArtifact(
                provider_key=str(artifact["provider_key"]),
                parent_observation_type=MESSAGE_OBSERVATION_TYPE,
                parent_provider_key=str(member["message_id"]),
                provider_revision=str(artifact["provider_revision"]),
                filename=str(artifact["filename"]),
                media_type=str(artifact["media_type"]),
                declared_size_bytes=int(artifact["size_bytes"]),
                locator={"input_uri": str(artifact["uri"]), "content_sha256": str(artifact["sha256"])},
            )
            for member in payload.get("members") or ()
            for artifact in member.get("artifacts") or ()
        )
        if with_artifacts
        else ()
    )
    return RawContent(item=item, body=body, content_type=CONTENT_TYPE, artifacts=artifacts)


def is_unit_payload(native: object) -> bool:
    return isinstance(native, Mapping) and native.get("payload_kind") == PAYLOAD_KIND


def payload_label(native: Mapping[str, Any]) -> ConversationLabel:
    """The conversation label a Unit payload keeps."""

    conversation = native["conversation"]
    name = conversation.get("name")
    return ConversationLabel(
        conversation_id=str(conversation["conversation_id"]),
        kind=str(conversation["kind"]),
        name=str(name) if name is not None else None,
        title_kind=str(conversation.get("title_kind") or "Conversation"),
        title_fields=tuple((str(field_name), str(text)) for field_name, text in conversation.get("title_fields") or ()),
        message_link=conversation.get("message_link"),
    )


def normalize_unit(raw: RawContent) -> NormalizedContent:
    """The Unit as readable markdown: a header, then each message with its sender and time."""

    native = json.loads(raw.body.decode("utf-8"))
    conversation = payload_label(native)
    records = [member["record"] for member in native.get("members") or []]
    root_id = native["unit"].get("thread_root_id")
    participants = sorted({str(record["from"]) for record in records if record.get("from")})
    times = [str(record.get("time") or "") for record in records]
    header = [f"# {raw.item.title}", ""]
    if conversation.name:
        header.append(f"**{conversation.title_kind}**: {conversation.name}")
    if participants:
        header.append(f"**Participants**: {', '.join(participants)}")
    if times:
        header.append(f"**Date range**: {min(times)[:16]} – {max(times)[:16]}")
    header.extend([f"**Messages**: {len(records)}", "", "---", ""])
    body: list[str] = []
    for index, (member, record) in enumerate(zip(native.get("members") or [], records, strict=True)):
        content = str(record.get("content") or "").strip()
        if not content:
            continue
        author = str(record.get("from") or "Unknown")
        heading = f"**{author}** ({str(record.get('time') or '')[:16]}):"
        leads = member["message_id"] == root_id if root_id else index == 0 or not record.get("reply_to_id")
        if leads:
            body.extend([heading, content, ""])
        else:
            body.append(f"> {heading}")
            body.extend(f"> {line}" for line in content.split("\n"))
            body.append("")
    return NormalizedContent(
        item=raw.item,
        markdown_body="\n".join(header) + "\n\n" + "\n".join(body),
        source_semantics={
            "conversation_id": conversation.conversation_id,
            "unit_kind": native["unit"]["kind"],
            "participants": participants,
            "message_count": len(records),
            # The Unit's time is its latest message edit or post.
            SOURCE_UPDATED_AT_KEY: latest_source_time(record_source_time(record) for record in records),
        },
    )


def _message_link(label: ConversationLabel, message_id: str) -> str:
    return label.message_link.format(message_id=message_id) if label.message_link else ""


def _title_text(label: ConversationLabel, kind: UnitKind, start: datetime) -> str:
    noun = "thread" if kind is UnitKind.THREAD else "messages"
    return f"{label.name or label.title_kind} {noun} from {start.isoformat()[:16]}"
