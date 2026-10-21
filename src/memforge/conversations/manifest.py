"""The collection request and the Collection Manifest.

The conversation layer builds one collection request per collection attempt
from the current Source Projection, so an executor never reads server state.
The executor answers with one immutable Collection Manifest: every message it
saw with its revision, each conversation's status, and the raw evidence of
each poll. The layer decides from the manifest alone; it derives proven
intervals from the raw evidence and never trusts an executor's conclusion.

A collection attempt is one lease attempt of one Source sync run, or for a
device one attempt of its collection job. Item retries within the attempt
reuse its manifest; a recovered run is a new attempt and collects again.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memforge.conversations.port import (
    ConversationAccess,
    ConversationLabel,
    ConversationRef,
    ConversationStatus,
    Interval,
    MessageArtifact,
    PollOutcome,
    ThreadingModel,
)
from memforge.conversations.records import MessageIdentity
from memforge.source_time import parse_source_time

MANIFEST_VERSION = 1


class ManifestFormatError(ValueError):
    """A manifest or request payload that does not have the shape its version defines."""


@dataclass(frozen=True, slots=True)
class RequestedConversation:
    """One conversation the executor collects, and the intervals it polls.

    ``last_known`` is the label the previous attempt recorded, which the
    executor keeps for a conversation its provider's listing leaves out.
    """

    conversation_id: str
    intervals: tuple[Interval, ...]
    last_known: ConversationLabel | None = None


@dataclass(frozen=True, slots=True)
class CollectionRequest:
    """Everything an executor needs from server state to collect one Source.

    The server keeps the request it issued to each collection attempt, and the
    attempt's manifest answers exactly that request.
    """

    conversations: tuple[RequestedConversation, ...]


@dataclass(frozen=True, slots=True)
class PollEvidence:
    """The raw evidence of one poll of one requested interval.

    ``started_at`` is the executor's clock when the poll began; the layer
    bounds what the poll proves by it and by the manifest's receipt time.
    """

    requested: Interval
    outcome: PollOutcome
    started_at: datetime
    earliest_returned_at: datetime | None = None
    earliest_complete_page_time: datetime | None = None
    stop_reason: str = ""


@dataclass(frozen=True, slots=True)
class ConversationEvidence:
    """One conversation's status and the evidence of every poll of it.

    ``ref`` is absent for a conversation the Source cannot read; such a
    conversation keeps, as ``last_known``, the label its request carried, so
    the next request carries it on.
    """

    conversation_id: str
    status: ConversationStatus
    ref: ConversationRef | None = None
    polls: tuple[PollEvidence, ...] = ()
    last_known: ConversationLabel | None = None

    def __post_init__(self) -> None:
        readable = self.status.access is ConversationAccess.READABLE
        if readable and self.ref is None:
            raise ValueError("a readable conversation needs its reference")
        if self.ref is not None and self.ref.conversation_id != self.conversation_id:
            raise ValueError("a conversation's reference names another conversation")
        if not readable and self.polls:
            raise ValueError("a conversation the Source cannot read has no poll evidence")
        if self.ref is not None and self.last_known is not None:
            raise ValueError("a conversation with its reference is labelled by it")

    @property
    def label(self) -> ConversationLabel | None:
        """The label this attempt knows the conversation by."""

        return self.ref.label if self.ref is not None else self.last_known


@dataclass(frozen=True, slots=True)
class ManifestMessage:
    """One message an executor saw: its identity, revision, thread root and artifacts."""

    identity: MessageIdentity
    revision: str
    created_at: datetime
    thread_root_id: str | None = None
    artifacts: tuple[MessageArtifact, ...] = ()


@dataclass(frozen=True, slots=True)
class CollectionManifest:
    """One collection attempt's immutable record of what the executor saw.

    ``received_at`` is the server's clock when it accepted the manifest.
    ``record_schema`` names the canonical record schema of every body.
    """

    source_id: str
    collection_attempt_id: str
    config_revision: str
    record_schema: str
    received_at: datetime
    conversations: tuple[ConversationEvidence, ...]
    messages: tuple[ManifestMessage, ...]

    def __post_init__(self) -> None:
        conversation_ids = [evidence.conversation_id for evidence in self.conversations]
        if len(set(conversation_ids)) != len(conversation_ids):
            raise ValueError("a manifest records each conversation once")
        identities = [message.identity for message in self.messages]
        if len(set(identities)) != len(identities):
            raise ValueError("a manifest records each message once")
        readable = {
            evidence.conversation_id
            for evidence in self.conversations
            if evidence.status.access is ConversationAccess.READABLE
        }
        if any(identity.conversation_id not in readable for identity in identities):
            raise ValueError("every manifest message belongs to a readable conversation it records")

    def conversation(self, conversation_id: str) -> ConversationEvidence | None:
        for evidence in self.conversations:
            if evidence.conversation_id == conversation_id:
                return evidence
        return None

    def messages_of(self, conversation_id: str) -> tuple[ManifestMessage, ...]:
        return tuple(message for message in self.messages if message.identity.conversation_id == conversation_id)


# ----------------------------------------------------------------------
# Wire and storage form
# ----------------------------------------------------------------------


def manifest_to_payload(manifest: CollectionManifest) -> dict[str, Any]:
    """The manifest as JSON-compatible data, for upload and storage."""

    return {
        "manifest_version": MANIFEST_VERSION,
        "source_id": manifest.source_id,
        "collection_attempt_id": manifest.collection_attempt_id,
        "config_revision": manifest.config_revision,
        "record_schema": manifest.record_schema,
        "received_at": manifest.received_at.isoformat(),
        "conversations": [_conversation_to_payload(evidence) for evidence in manifest.conversations],
        "messages": [_message_to_payload(message) for message in manifest.messages],
    }


def manifest_from_payload(payload: object, *, received_at: datetime | None = None) -> CollectionManifest:
    """Read a manifest; ``received_at`` replaces the stored receipt time when the server accepts an upload.

    A payload of another shape raises :class:`ManifestFormatError`.
    """

    try:
        value = _mapping(payload)
        if value.get("manifest_version") != MANIFEST_VERSION:
            raise ManifestFormatError(f"unsupported Collection Manifest version: {value.get('manifest_version')!r}")
        receipt = received_at if received_at is not None else _required_time(value.get("received_at"), "received_at")
        return CollectionManifest(
            source_id=_required_text(value.get("source_id"), "source_id"),
            collection_attempt_id=_required_text(value.get("collection_attempt_id"), "collection_attempt_id"),
            config_revision=_required_text(value.get("config_revision"), "config_revision"),
            record_schema=_required_text(value.get("record_schema"), "record_schema"),
            received_at=receipt,
            conversations=tuple(_conversation_from_payload(item) for item in _list(value.get("conversations"))),
            messages=tuple(_message_from_payload(item) for item in _list(value.get("messages"))),
        )
    except ManifestFormatError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestFormatError(f"malformed Collection Manifest: {exc}") from exc


def request_to_payload(request: CollectionRequest) -> dict[str, Any]:
    return {
        "conversations": [
            {
                "conversation_id": conversation.conversation_id,
                "intervals": [_interval_to_payload(interval) for interval in conversation.intervals],
                "last_known": _label_to_payload(conversation.last_known) if conversation.last_known else None,
            }
            for conversation in request.conversations
        ],
    }


def request_from_payload(payload: object) -> CollectionRequest:
    try:
        return CollectionRequest(
            conversations=tuple(
                RequestedConversation(
                    conversation_id=_required_text(item.get("conversation_id"), "conversation_id"),
                    intervals=tuple(_interval_from_payload(interval) for interval in _list(item.get("intervals"))),
                    last_known=_label_from_payload(item["last_known"]) if item.get("last_known") else None,
                )
                for item in (_mapping(value) for value in _list(_mapping(payload).get("conversations")))
            ),
        )
    except ManifestFormatError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestFormatError(f"malformed collection request: {exc}") from exc


def _label_to_payload(label: ConversationLabel) -> dict[str, Any]:
    return {
        "conversation_id": label.conversation_id,
        "kind": label.kind,
        "name": label.name,
        "title_kind": label.title_kind,
        "title_fields": [list(pair) for pair in label.title_fields],
        "message_link": label.message_link,
    }


def _label_from_payload(value: object) -> ConversationLabel:
    label = _mapping(value)
    name = label.get("name")
    return ConversationLabel(
        conversation_id=_required_text(label.get("conversation_id"), "conversation_id"),
        kind=_required_text(label.get("kind"), "kind"),
        name=str(name) if name is not None else None,
        title_kind=str(label.get("title_kind") or "Conversation"),
        title_fields=tuple((str(field_name), str(text)) for field_name, text in _list(label.get("title_fields"))),
        message_link=str(label["message_link"]) if label.get("message_link") else None,
    )


def _conversation_to_payload(evidence: ConversationEvidence) -> dict[str, Any]:
    ref = evidence.ref
    return {
        "conversation_id": evidence.conversation_id,
        "status": {"access": evidence.status.access.value, "provider_answer": evidence.status.provider_answer},
        "ref": None
        if ref is None
        else {
            "label": _label_to_payload(ref.label),
            "threading": ref.threading.value,
            "poll_includes_replies": ref.poll_includes_replies,
            "visible_from_certified": _time_or_none(ref.visible_from_certified),
        },
        "polls": [
            {
                "requested": _interval_to_payload(poll.requested),
                "outcome": poll.outcome.value,
                "started_at": poll.started_at.isoformat(),
                "earliest_returned_at": _time_or_none(poll.earliest_returned_at),
                "earliest_complete_page_time": _time_or_none(poll.earliest_complete_page_time),
                "stop_reason": poll.stop_reason,
            }
            for poll in evidence.polls
        ],
        "last_known": _label_to_payload(evidence.last_known) if evidence.last_known is not None else None,
    }


def _conversation_from_payload(value: object) -> ConversationEvidence:
    item = _mapping(value)
    status = _mapping(item.get("status"))
    ref_value = item.get("ref")
    ref = None
    if ref_value is not None:
        ref_item = _mapping(ref_value)
        ref = ConversationRef(
            label=_label_from_payload(ref_item.get("label")),
            threading=ThreadingModel(ref_item.get("threading")),
            poll_includes_replies=bool(ref_item.get("poll_includes_replies")),
            visible_from_certified=parse_source_time(ref_item.get("visible_from_certified")),
        )
    return ConversationEvidence(
        conversation_id=_required_text(item.get("conversation_id"), "conversation_id"),
        status=ConversationStatus(
            access=ConversationAccess(status.get("access")),
            provider_answer=str(status.get("provider_answer") or ""),
        ),
        ref=ref,
        last_known=_label_from_payload(item["last_known"]) if item.get("last_known") else None,
        polls=tuple(
            PollEvidence(
                requested=_interval_from_payload(poll.get("requested")),
                outcome=PollOutcome(poll.get("outcome")),
                started_at=_required_time(poll.get("started_at"), "started_at"),
                earliest_returned_at=parse_source_time(poll.get("earliest_returned_at")),
                earliest_complete_page_time=parse_source_time(poll.get("earliest_complete_page_time")),
                stop_reason=str(poll.get("stop_reason") or ""),
            )
            for poll in (_mapping(value) for value in _list(item.get("polls")))
        ),
    )


def _message_to_payload(message: ManifestMessage) -> dict[str, Any]:
    return {
        "conversation_id": message.identity.conversation_id,
        "message_id": message.identity.message_id,
        "revision": message.revision,
        "created_at": message.created_at.isoformat(),
        "thread_root_id": message.thread_root_id,
        "artifacts": [
            {"provider_key": artifact.provider_key, "provider_revision": artifact.provider_revision}
            for artifact in message.artifacts
        ],
    }


def _message_from_payload(value: object) -> ManifestMessage:
    item = _mapping(value)
    return ManifestMessage(
        identity=MessageIdentity(
            conversation_id=_required_text(item.get("conversation_id"), "conversation_id"),
            message_id=_required_text(item.get("message_id"), "message_id"),
        ),
        revision=_required_text(item.get("revision"), "revision"),
        created_at=_required_time(item.get("created_at"), "created_at"),
        thread_root_id=str(item["thread_root_id"]) if item.get("thread_root_id") else None,
        artifacts=tuple(
            MessageArtifact(
                provider_key=_required_text(artifact.get("provider_key"), "provider_key"),
                provider_revision=_required_text(artifact.get("provider_revision"), "provider_revision"),
            )
            for artifact in (_mapping(value) for value in _list(item.get("artifacts")))
        ),
    )


def _interval_to_payload(interval: Interval) -> dict[str, Any]:
    return {"start": interval.start.isoformat(), "end": _time_or_none(interval.end)}


def _interval_from_payload(value: object) -> Interval:
    item = _mapping(value)
    return Interval(start=_required_time(item.get("start"), "start"), end=parse_source_time(item.get("end")))


def _time_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _required_time(value: object, name: str) -> datetime:
    parsed = parse_source_time(value)
    if parsed is None:
        raise ManifestFormatError(f"Collection Manifest field {name} is required")
    return parsed


def _required_text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestFormatError(f"Collection Manifest field {name} is required")
    return text


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestFormatError("Collection Manifest objects must be JSON objects")
    return value


def _list(value: object) -> Iterable[Any]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ManifestFormatError("Collection Manifest lists must be JSON arrays")
    return value
