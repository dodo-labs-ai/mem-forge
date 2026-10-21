"""Shared evidence contract for Teams collection, upload, and replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from memforge.pipeline.normalizer_utils import html_to_markdown


MEMORY_MESSAGE_TYPES = frozenset(
    {
        "Text",
        "RichText/Html",
        "RichText",
        "RichText/Media_GenericCard",
    }
)


MILLISECONDS_PER_SECOND = 1000


class TeamsMessageEvidenceError(ValueError):
    """Raised when a potentially memory-bearing message is ambiguous."""


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


def parse_teams_edit_time(message: Mapping[str, object]) -> datetime | None:
    """When the chat service last edited a message; None for a message never edited.

    Chatsvc reports an edit as ``properties.edittime`` in epoch milliseconds.
    """

    properties = message.get("properties")
    value = properties.get("edittime") if isinstance(properties, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        milliseconds = int(value)
    except ValueError:
        return None
    return datetime.fromtimestamp(milliseconds / MILLISECONDS_PER_SECOND, timezone.utc)


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
        if message.get("edited_time") is not None and parse_teams_source_timestamp(message.get("edited_time")) is None:
            raise TeamsMessageEvidenceError("Teams window message has an invalid edit time")
        result.append(message)
    return tuple(result)


