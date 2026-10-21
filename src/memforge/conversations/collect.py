"""Running a provider for one collection request: what every executor shares.

An executor asks the provider exactly what the request names and records the
answers as they came: each conversation's status, each poll's raw evidence
and every message with its record. It draws no conclusion from them. A device
executor uploads the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    RequestedConversation,
)
from memforge.conversations.port import (
    ConversationAccess,
    ConversationProvider,
    ConversationRef,
    ConversationStatus,
    ConversationUnavailable,
    ProviderMessage,
)
from memforge.conversations.records import MessageIdentity, record_revision


@dataclass(slots=True)
class CollectedAttempt:
    """What one executor collected for one request, before the server receives it.

    ``provider_messages`` and ``refs`` keep each provider message and
    conversation, so the executor can read a message's artifacts once the
    server names them.
    """

    source_id: str
    collection_attempt_id: str
    config_revision: str
    record_schema: str
    conversations: list[ConversationEvidence] = field(default_factory=list)
    messages: dict[MessageIdentity, ManifestMessage] = field(default_factory=dict)
    bodies: dict[MessageIdentity, dict[str, object]] = field(default_factory=dict)
    provider_messages: dict[MessageIdentity, ProviderMessage] = field(default_factory=dict)
    refs: dict[str, ConversationRef] = field(default_factory=dict)

    def manifest(self, *, received_at: datetime) -> CollectionManifest:
        """The Collection Manifest as the server receives it at ``received_at``."""

        return CollectionManifest(
            source_id=self.source_id,
            collection_attempt_id=self.collection_attempt_id,
            config_revision=self.config_revision,
            record_schema=self.record_schema,
            received_at=received_at,
            conversations=tuple(self.conversations),
            messages=tuple(self.messages[identity] for identity in sorted(self.messages)),
        )

    def add(self, conversation_id: str, message: ProviderMessage) -> None:
        identity = MessageIdentity(conversation_id, message.message_id)
        if identity in self.messages:
            return
        record = dict(message.record)
        self.messages[identity] = ManifestMessage(
            identity=identity,
            revision=record_revision(record),
            created_at=message.created_at,
            thread_root_id=message.thread_root_id,
            artifacts=tuple(sorted(set(message.artifacts))),
        )
        self.bodies[identity] = record
        self.provider_messages[identity] = message


async def collect(
    provider: ConversationProvider,
    request: CollectionRequest,
    *,
    source_id: str,
    collection_attempt_id: str,
    config_revision: str,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CollectedAttempt:
    """Collect every conversation the request names.

    A conversation the provider refuses to show is recorded with that status
    and nothing it returned; any other provider error propagates and fails
    the attempt. A conversation the provider's listing leaves out keeps the
    label the request carries for it, so its records keep their names.
    """

    attempt = CollectedAttempt(
        source_id=source_id,
        collection_attempt_id=collection_attempt_id,
        config_revision=config_revision,
        record_schema=provider.record_schema.qualified_name,
    )
    listing = await provider.list_conversations()
    listed = {ref.conversation_id: ref for ref in listing.conversations}
    for requested in request.conversations:
        conversation_id = requested.conversation_id
        ref = listed.get(conversation_id)
        status = listing.statuses.get(conversation_id)
        if ref is None or status is None:
            status, ref = await provider.get_conversation(conversation_id)
            if ref is not None and requested.last_known is not None:
                ref = replace(ref, label=requested.last_known)
        if ref is None or status.access is not ConversationAccess.READABLE:
            attempt.conversations.append(
                ConversationEvidence(
                    conversation_id=conversation_id, status=_unreadable(status), last_known=requested.last_known
                )
            )
            continue
        try:
            evidence, messages = await _poll(provider, ref, requested, clock)
        except ConversationUnavailable as unavailable:
            attempt.conversations.append(
                ConversationEvidence(
                    conversation_id=conversation_id, status=unavailable.status, last_known=requested.last_known
                )
            )
            continue
        attempt.conversations.append(evidence)
        attempt.refs[conversation_id] = ref
        for message in messages:
            attempt.add(conversation_id, message)
    return attempt


def _unreadable(status: ConversationStatus) -> ConversationStatus:
    if status.access is ConversationAccess.READABLE:
        return ConversationStatus(ConversationAccess.NO_ACCESS, "readable without a conversation reference")
    return status


async def _poll(
    provider: ConversationProvider,
    ref: ConversationRef,
    requested: RequestedConversation,
    clock: Callable[[], datetime],
) -> tuple[ConversationEvidence, Sequence[ProviderMessage]]:
    polls: list[PollEvidence] = []
    messages: list[ProviderMessage] = []
    for interval in requested.intervals:
        started_at = clock()
        poll = await provider.poll(ref, interval)
        polls.append(
            PollEvidence(
                requested=interval,
                outcome=poll.outcome,
                started_at=started_at,
                earliest_returned_at=min((message.created_at for message in poll.messages), default=None),
                earliest_complete_page_time=poll.earliest_complete_page_time,
                stop_reason=poll.stop_reason,
            )
        )
        messages.extend(poll.messages)
    evidence = ConversationEvidence(
        conversation_id=ref.conversation_id,
        status=ConversationStatus(ConversationAccess.READABLE, "read"),
        ref=ref,
        polls=tuple(polls),
    )
    return evidence, messages
