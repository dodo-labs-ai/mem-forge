"""Microsoft Teams as a conversation provider.

Teams implements the conversation provider port over the Chat Service client.
It maps each message to its ``teams-message`` record and each refused read to
a conversation status, and draws no conclusion of its own: windows, threads,
coverage, absence and retention belong to the conversation layer.

Every Teams conversation threads ``inline``: a message whose root or parent
is another message is a reply, and a post without replies stays in a window.
A poll returns replies with the posts, so threads need no fetch of their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

import httpx

from memforge.conversations.port import (
    ArtifactBody,
    ConversationAccess,
    ConversationLabel,
    ConversationListing,
    ConversationPoll,
    ConversationRef,
    ConversationStatus,
    ConversationUnavailable,
    Interval,
    MessageArtifact,
    ProviderMessage,
    ThreadingModel,
)
from memforge.genes.teams_client import TeamsChatClient, infer_conversation_type
from memforge.source_adapters.contracts import CanonicalRecordSchema
from memforge.source_adapters.teams import MESSAGE_SCHEMA, message_record

TEAMS_TITLE_KIND = "Teams conversation"
DEFAULT_POLL_BUDGET_SECONDS = 300

_IMAGE_EXTENSIONS = {"image/gif": ".gif", "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}

# The Chat Service answers these when the user can no longer read a
# conversation; neither proves the conversation was deleted.
_REFUSED_STATUSES = frozenset({403, 404})


class TeamsConversationProvider:
    """The conversation provider port over the Teams Chat Service."""

    record_schema: CanonicalRecordSchema = MESSAGE_SCHEMA

    def __init__(self, client: TeamsChatClient, *, poll_budget_seconds: float = DEFAULT_POLL_BUDGET_SECONDS) -> None:
        self._client = client
        self._poll_budget_seconds = poll_budget_seconds
        # A channel's hosted images are read through its team.
        self._team_ids: dict[str, str] = {}

    async def list_conversations(self) -> ConversationListing:
        """The user's recent conversations; a selected one older than the listing is read by id."""

        listed = await self._client.list_conversations()
        self._team_ids = {str(value["id"]): str(value["team_id"]) for value in listed if value.get("team_id")}
        conversations = tuple(teams_conversation_ref(value) for value in listed)
        return ConversationListing(
            conversations=conversations,
            statuses={
                ref.conversation_id: ConversationStatus(ConversationAccess.READABLE, "listed")
                for ref in conversations
            },
        )

    async def get_conversation(self, conversation_id: str) -> tuple[ConversationStatus, ConversationRef | None]:
        """A selected conversation the listing did not include.

        The listing holds only recent conversations, so one missing from it
        may still be readable; its poll decides.
        """

        ref = teams_conversation_ref({"id": conversation_id})
        return ConversationStatus(ConversationAccess.READABLE, "not in the recent conversation listing"), ref

    async def poll(self, conversation: ConversationRef, interval: Interval) -> ConversationPoll:
        try:
            result = await self._client.read_messages(
                conversation.conversation_id,
                start=interval.start,
                end=interval.end,
                budget_seconds=self._poll_budget_seconds,
            )
        except httpx.HTTPStatusError as error:
            if error.response.status_code in _REFUSED_STATUSES:
                raise ConversationUnavailable(
                    ConversationStatus(ConversationAccess.NO_ACCESS, f"HTTP {error.response.status_code}")
                ) from error
            raise
        return ConversationPoll(
            messages=tuple(teams_provider_message(message, conversation) for message in result.messages),
            outcome=result.outcome,
            earliest_complete_page_time=result.earliest_complete_page_time,
            stop_reason=result.stop_reason,
        )

    async def read_artifact(
        self, conversation: ConversationRef, message: ProviderMessage, artifact: MessageArtifact
    ) -> ArtifactBody:
        """One inline image of a message, read from Microsoft Graph."""

        hosted_content_id = artifact.provider_key.rpartition("/hostedContents/")[2]
        content, media_type = await self._client.get_hosted_content(
            conversation_id=conversation.conversation_id,
            message_id=message.message_id,
            hosted_content_id=hosted_content_id,
            conversation_type=conversation.label.kind,
            team_id=self._team_ids.get(conversation.conversation_id),
            root_message_id=message.thread_root_id or message.message_id,
        )
        return ArtifactBody(
            content=content,
            media_type=media_type,
            filename=f"teams-hosted-{hosted_content_id}{_IMAGE_EXTENSIONS.get(media_type, '')}",
        )


def teams_conversation_ref(conversation: Mapping[str, object]) -> ConversationRef:
    """A Teams conversation, as the Chat Service lists it."""

    return ConversationRef(
        label=teams_conversation_label(conversation),
        threading=ThreadingModel.INLINE,
        poll_includes_replies=True,
    )


def teams_conversation_label(conversation: Mapping[str, object]) -> ConversationLabel:
    """How a Teams conversation is named and shown; one without a listed topic is named generically."""

    conversation_id = str(conversation["id"])
    kind = str(conversation.get("type") or infer_conversation_type(conversation_id))
    team_name = conversation.get("team_name")
    title_fields = [("Conversation type", kind)]
    if team_name:
        title_fields.append(("Team", str(team_name)))
    return ConversationLabel(
        conversation_id=conversation_id,
        kind=kind,
        name=str(conversation.get("topic") or "Teams conversation"),
        title_kind=TEAMS_TITLE_KIND,
        title_fields=tuple(title_fields),
        message_link=f"https://teams.microsoft.com/l/message/{conversation_id}/{{message_id}}",
    )


def teams_provider_message(message: Mapping[str, object], conversation: ConversationRef) -> ProviderMessage:
    """One parsed Chat Service message as a provider message with its ``teams-message`` record."""

    message_id = str(message["id"])
    created_at = message["time"]
    assert isinstance(created_at, datetime)
    root = message.get("rootMessageId") or message.get("parentMessageId")
    return ProviderMessage(
        message_id=message_id,
        created_at=created_at,
        record=teams_message_record(message, conversation),
        thread_root_id=str(root) if root and str(root) != message_id else None,
        artifacts=tuple(
            MessageArtifact(
                provider_key=f"{message_id}/hostedContents/{hosted_content_id}",
                provider_revision=f"{message_id}:{hosted_content_id}",
            )
            for hosted_content_id in dict.fromkeys(str(value) for value in message.get("hosted_content_ids") or ())
            if hosted_content_id
        ),
    )


def teams_message_record(message: Mapping[str, object], conversation: ConversationRef) -> dict[str, object]:
    """The ``teams-message`` record of one parsed message, field for field as stored Units hold it."""

    edited_time = message.get("edited_time")
    fields = {
        "id": message["id"],
        "from": message.get("from"),
        "content": message.get("content"),
        "time": _iso(message["time"]),
        **({"edited_time": _iso(edited_time)} if edited_time else {}),
        "attachments": message.get("attachments", []),
        **({"parentMessageId": message["parentMessageId"]} if message.get("parentMessageId") else {}),
        **({"rootMessageId": message["rootMessageId"]} if message.get("rootMessageId") else {}),
    }
    return message_record(fields, conversation_id=conversation.conversation_id, conversation_name=conversation.label.name)


def _iso(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)
