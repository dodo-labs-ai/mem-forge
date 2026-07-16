"""Shared evidence contract for Teams collection, upload, and replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib

from memforge.pipeline.normalizer_utils import html_to_markdown


MEMORY_MESSAGE_TYPES = frozenset(
    {
        "Text",
        "RichText/Html",
        "RichText",
        "RichText/Media_GenericCard",
    }
)


class TeamsMessageEvidenceError(ValueError):
    """Raised when a potentially memory-bearing message is ambiguous."""


def teams_scope_attestation_window_id(*, source_id: str, conversation_id: str) -> str:
    identity = hashlib.sha256(f"{source_id}\n{conversation_id}".encode("utf-8")).hexdigest()
    return f"teams-scope:v1:{identity}"


def parse_teams_source_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def validate_teams_provider_message(message: Mapping[str, object]) -> bool:
    """Validate one raw Chatsvc record and return whether it has semantic text.

    An explicit unsupported message type is a provider control/system record.
    Missing type is treated conservatively as a possible text record. Every
    possible text record, including an explicitly emptied one, needs identity
    and source time so its lifecycle can be reconciled deterministically.
    """

    message_type = str(message.get("messagetype") or message.get("messageType") or "").strip()
    if message_type and message_type not in MEMORY_MESSAGE_TYPES:
        return False
    if "content" not in message or message.get("content") is None:
        raise TeamsMessageEvidenceError("Teams text message is missing content")
    content = message.get("content")
    if not isinstance(content, str):
        raise TeamsMessageEvidenceError("Teams text message has invalid content")
    message_id = str(message.get("id") or message.get("amsreferencesid") or "").strip()
    timestamp = parse_teams_source_timestamp(message.get("composetime") or message.get("originalarrivaltime"))
    if not message_id or timestamp is None:
        raise TeamsMessageEvidenceError("Teams text message is missing a stable id or source timestamp")
    normalized = html_to_markdown(content).strip() if "<" in content else content.strip()
    return bool(normalized)


def validate_teams_canonical_messages(value: object) -> tuple[Mapping[str, object], ...]:
    """Validate canonical messages persisted in a local-agent package."""

    if not isinstance(value, list) or not value:
        raise TeamsMessageEvidenceError("Teams window messages must be a non-empty list")
    result: list[Mapping[str, object]] = []
    for message in value:
        if not isinstance(message, Mapping):
            raise TeamsMessageEvidenceError("Teams window message must be an object")
        if "content" not in message or not isinstance(message.get("content"), str):
            raise TeamsMessageEvidenceError("Teams window message has invalid content")
        if not str(message.get("id") or "").strip():
            raise TeamsMessageEvidenceError("Teams window message is missing a stable id")
        if parse_teams_source_timestamp(message.get("time")) is None:
            raise TeamsMessageEvidenceError("Teams window message is missing a source timestamp")
        result.append(message)
    return tuple(result)


def validate_teams_window_payload(
    payload: Mapping[str, object],
    *,
    conversation_id: str,
    window_id: str,
    source_id: str | None = None,
    root_message_id: str | None = None,
    window_type: str | None = None,
    tombstone_reasons: frozenset[str],
) -> str:
    """Return ``snapshot`` or ``tombstone`` after exact identity validation."""

    from memforge.local_agent.source_contract import is_direct_teams_conversation_id

    if not is_direct_teams_conversation_id(conversation_id):
        raise TeamsMessageEvidenceError("Teams conversation identity is invalid")
    if str(payload.get("conversation_id") or "").strip() != conversation_id:
        raise TeamsMessageEvidenceError("Teams payload conversation identity mismatch")
    if str(payload.get("window_id") or "").strip() != window_id:
        raise TeamsMessageEvidenceError("Teams payload window identity mismatch")
    messages = payload.get("messages")
    if payload.get("_scope_attestation") is True:
        target_conversations = payload.get("target_conversation_ids")
        poll = payload.get("poll")
        if (
            messages != []
            or not str(payload.get("transition_id") or "").strip()
            or not str(payload.get("target_scope_fingerprint") or "").strip()
            or not str(payload.get("collection_attempt_id") or "").strip()
            or not isinstance(target_conversations, list)
            or not target_conversations
            or not all(is_direct_teams_conversation_id(value) for value in target_conversations)
            or len(set(target_conversations)) != len(target_conversations)
            or not isinstance(poll, Mapping)
            or str(poll.get("raw_conversation_id") or "").strip() != conversation_id
            or (
                source_id is not None
                and window_id
                != teams_scope_attestation_window_id(
                    source_id=source_id,
                    conversation_id=conversation_id,
                )
            )
            or str(window_type or "").strip() != "scope_attestation"
            or bool(str(root_message_id or "").strip())
        ):
            raise TeamsMessageEvidenceError("Teams scope attestation is invalid")
        return "scope_attestation"
    from memforge.local_agent.teams_ledger import (
        build_teams_window_id,
        decode_teams_window_id,
    )

    normalized_root_id = str(root_message_id or "").strip()
    normalized_window_type = str(window_type or "").strip()
    if not normalized_root_id or normalized_window_type not in {"thread", "time_block"}:
        raise TeamsMessageEvidenceError("Teams window locator is incomplete")
    try:
        decoded = decode_teams_window_id(window_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise TeamsMessageEvidenceError("Teams window identity is invalid") from exc
    if (
        decoded["conversation_id"] != conversation_id
        or decoded["root_or_anchor_message_id"] != normalized_root_id
        or decoded["window_type"] != normalized_window_type
        or (source_id is not None and decoded["source_id"] != source_id)
        or build_teams_window_id(
            source_id=decoded["source_id"],
            conversation_id=conversation_id,
            root_or_anchor_message_id=normalized_root_id,
            window_type=normalized_window_type,
        )
        != window_id
    ):
        raise TeamsMessageEvidenceError("Teams window locator does not match the package")
    if payload.get("_tombstone") is True:
        if (
            messages != []
            or payload.get("_authoritative_snapshot") is not True
            or str(payload.get("tombstone_reason") or "").strip() not in tombstone_reasons
        ):
            raise TeamsMessageEvidenceError("Teams tombstone evidence is invalid")
        return "tombstone"
    validate_teams_canonical_messages(messages)
    return "snapshot"
