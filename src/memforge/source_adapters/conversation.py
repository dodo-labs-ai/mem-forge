"""Conversation Units as Source Projection: each member message is one Observation.

A message Observation's content is its record's canonical JSON, so a member
carried unchanged keeps its Observation revision. A reply relates to the
message it answers; any other message follows the one before it. The
conversation layer decided every member, so a Unit is a complete snapshot of
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memforge.conversations.projection import PAYLOAD_VERSION, payload_label
from memforge.conversations.records import MESSAGE_OBSERVATION_TYPE, canonical_record_json
from memforge.conversations.undated import payload_member_times
from memforge.conversations.units import UnitKind
from memforge.models import ContentItem
from memforge.source_adapters.contracts import _NativeProjection, _ObservationInput, _unit_title
from memforge.source_projection import ProjectionCoverage, SourceRelationType


def project_native(*, item: ContentItem, native: Mapping[str, Any]) -> _NativeProjection:
    """Provider identity, Observations, relations and locator of one conversation Unit payload."""

    if native.get("payload_version") != PAYLOAD_VERSION:
        raise ValueError(f"unsupported conversation Unit payload version: {native.get('payload_version')!r}")
    unit = native["unit"]
    label = payload_label(native)
    conversation_id = label.conversation_id
    kind = UnitKind(unit["kind"])
    members = native.get("members") or []
    if not members:
        raise ValueError("a conversation Unit payload has members")
    inputs = []
    relations = []
    previous_id: str | None = None
    times = []
    for member in members:
        message_id = str(member["message_id"])
        record = member["record"]
        created_at, source_time, _shown = payload_member_times(native, member)
        times.append(created_at)
        inputs.append(
            _ObservationInput(
                MESSAGE_OBSERVATION_TYPE,
                message_id,
                canonical_record_json(record),
                record,
                {"conversation_id": conversation_id},
                source_time,
                {"claim_evidence_scope": "atomic"},
            )
        )
        reply_to = record.get("reply_to_id")
        if reply_to and reply_to != message_id:
            relations.append((SourceRelationType.REPLIES_TO, message_id, str(reply_to), None, {}))
        elif previous_id:
            relations.append((SourceRelationType.PRECEDES, previous_id, message_id, None, {}))
        previous_id = message_id
    observed_from = min(times).isoformat()
    locator: dict[str, object] = {
        "conversation_id": conversation_id,
        "observed_from": observed_from,
        "observed_to": max(times).isoformat(),
        "url": item.source_url or None,
    }
    if kind is UnitKind.THREAD:
        locator["thread_root_id"] = unit["thread_root_id"]
    return _NativeProjection(
        unit_type=kind.value,
        provider_key=str(unit["provider_key"]),
        observations=tuple(inputs),
        relations=tuple(relations),
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator=locator,
        title=_unit_title(
            label.title_kind,
            *label.title_fields,
            ("Conversation", label.name),
            ("From", observed_from),
        ),
    )
