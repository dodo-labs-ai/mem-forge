"""Receiving a Collection Manifest and its bodies from a device executor.

A device uploads the manifest first. The manifest must answer the request the
server issued to the attempt: every requested conversation, and for each one
it could read, exactly the requested intervals. The server stamps its receipt
time, keeps the manifest, and answers with what it lacks of each message: a body
that no current Unit holds at the manifest's revision, and artifact bytes no
current Unit holds at the manifest's artifact revision. The device uploads
the artifact bytes it can read, then the bodies, each checked against its
manifest revision and naming the uploads of its artifacts. Processing starts
once every message has a body; an artifact the device could not read is
named again by the next attempt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from memforge.conversations.collections import CollectedBody, StoredCollections
from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    ManifestMessage,
    manifest_from_payload,
)
from memforge.conversations.port import ConversationAccess, MessageArtifact
from memforge.conversations.reader import ConversationUnitReader
from memforge.conversations.records import MessageIdentity, record_created_at, record_revision


class CollectionRejected(ValueError):
    """An upload that does not belong to the attempt it names, or contradicts its manifest."""


@dataclass(frozen=True, slots=True)
class MissingMessage:
    """A manifest message the server lacks something of: its body, artifact bytes, or both."""

    identity: MessageIdentity
    body: bool
    artifacts: tuple[MessageArtifact, ...] = ()


class CollectionIntake:
    """Receives one Source's collection uploads, checked against the request and manifest of their attempt."""

    def __init__(self, *, reader: ConversationUnitReader, collections: StoredCollections) -> None:
        self._reader = reader
        self._collections = collections

    async def receive_manifest(
        self,
        payload: Mapping[str, Any],
        *,
        source_id: str,
        collection_attempt_id: str,
        config_revision: str,
        record_schema: str,
        received_at: datetime,
    ) -> tuple[MissingMessage, ...]:
        """Keep one attempt's manifest and name what the server lacks of its messages.

        ``record_schema`` is the schema the Source's connector reads its
        messages in; a manifest collected in another is refused.
        """

        manifest = manifest_from_payload(payload, received_at=received_at)
        if manifest.source_id != source_id or manifest.collection_attempt_id != collection_attempt_id:
            raise CollectionRejected("the manifest names another Source or collection attempt")
        if manifest.config_revision != config_revision:
            raise CollectionRejected("the manifest was collected under another configuration")
        if manifest.record_schema != record_schema:
            raise CollectionRejected(f"the manifest records messages as {manifest.record_schema}, not {record_schema}")
        request = await self._collections.request(source_id, collection_attempt_id)
        if request is None:
            raise CollectionRejected(f"collection attempt {collection_attempt_id} was issued no request")
        _require_answer(manifest, request)
        stored = await self._collections.manifest(source_id, collection_attempt_id)
        if stored is None:
            try:
                await self._collections.record_manifest(manifest)
            except ValueError as exc:
                # Another upload of this attempt recorded a different manifest first.
                raise CollectionRejected(str(exc)) from exc
        elif replace(stored, received_at=received_at) != manifest:
            raise CollectionRejected(f"collection attempt {collection_attempt_id} already has another manifest")
        else:
            # The device repeated its upload; the manifest keeps its first receipt.
            manifest = stored
        return await self.missing(manifest)

    async def receive_bodies(
        self,
        bodies: Mapping[MessageIdentity, CollectedBody],
        *,
        source_id: str,
        collection_attempt_id: str,
    ) -> None:
        """Keep message bodies of one attempt.

        Each body matches its manifest message's revision and creation time
        and names only that message's artifacts, each uploaded for it.
        """

        manifest = await self._require_manifest(source_id, collection_attempt_id)
        messages = {message.identity: message for message in manifest.messages}
        for identity, body in bodies.items():
            message = messages.get(identity)
            if message is None:
                raise CollectionRejected(f"message {identity} is not in the attempt's manifest")
            if record_revision(body.record) != message.revision:
                raise CollectionRejected(f"the body of message {identity} does not match its manifest revision")
            if _created_at(body.record) != message.created_at:
                raise CollectionRejected(f"the body of message {identity} does not record its manifest creation time")
            named = {(artifact.provider_key, artifact.provider_revision) for artifact in message.artifacts}
            if any((link.provider_key, link.provider_revision) not in named for link in body.artifacts):
                raise CollectionRejected(f"the body of message {identity} names an artifact its manifest does not")
        try:
            await self._collections.uploaded_artifacts(source_id, bodies)
            await self._collections.record_bodies(source_id, collection_attempt_id, bodies)
        except ValueError as exc:
            raise CollectionRejected(str(exc)) from exc

    async def missing_bodies_of(self, source_id: str, collection_attempt_id: str) -> tuple[MessageIdentity, ...]:
        """The manifest messages without a body; processing waits for them, not for artifact bytes."""

        manifest = await self._require_manifest(source_id, collection_attempt_id)
        return tuple(item.identity for item in await self.missing(manifest) if item.body)

    async def missing(self, manifest: CollectionManifest) -> tuple[MissingMessage, ...]:
        uploaded = await self._collections.bodies(manifest.source_id, manifest.collection_attempt_id)
        by_conversation: dict[str, list[ManifestMessage]] = {}
        for message in manifest.messages:
            by_conversation.setdefault(message.identity.conversation_id, []).append(message)
        missing = []
        for conversation_id, messages in by_conversation.items():
            holders = await self._reader.holders(
                manifest.source_id, conversation_id, {message.identity.message_id for message in messages}
            )
            units = await self._reader.read(tuple(unit for refs in holders.values() for unit in refs))
            held = {member.identity: member for unit in units for member in unit.members}
            for message in messages:
                body = uploaded.get(message.identity)
                stored = held.get(message.identity)
                has_body = body is not None or (stored is not None and stored.revision == message.revision)
                have = {(link.provider_key, link.provider_revision) for link in (body.artifacts if body else ())}
                have |= {(a.provider_key, a.provider_revision) for a in (stored.artifacts if stored else ())}
                lacking = tuple(
                    artifact
                    for artifact in message.artifacts
                    if (artifact.provider_key, artifact.provider_revision) not in have
                )
                if not has_body or lacking:
                    missing.append(MissingMessage(identity=message.identity, body=not has_body, artifacts=lacking))
        return tuple(sorted(missing, key=lambda item: item.identity))

    async def _require_manifest(self, source_id: str, collection_attempt_id: str) -> CollectionManifest:
        manifest = await self._collections.manifest(source_id, collection_attempt_id)
        if manifest is None:
            raise CollectionRejected(f"collection attempt {collection_attempt_id} has no manifest")
        return manifest


def _created_at(record: Mapping[str, object]) -> datetime | None:
    try:
        return record_created_at(record)
    except (KeyError, TypeError, ValueError):
        return None


def _require_answer(manifest: CollectionManifest, request: CollectionRequest) -> None:
    """Refuse a manifest that does not answer its attempt's request.

    It records every requested conversation and no other, a conversation it
    could read was polled over exactly the requested intervals, and one it
    could not read keeps the label the request carried.
    """

    requested = {conversation.conversation_id: conversation for conversation in request.conversations}
    recorded = {evidence.conversation_id: evidence for evidence in manifest.conversations}
    if recorded.keys() != requested.keys() or len(recorded) != len(manifest.conversations):
        raise CollectionRejected("the manifest does not record exactly the requested conversations")
    for conversation_id, evidence in recorded.items():
        polled = tuple(poll.requested for poll in evidence.polls)
        if evidence.status.access is ConversationAccess.READABLE and polled != requested[conversation_id].intervals:
            raise CollectionRejected(f"conversation {conversation_id} was not polled over the requested intervals")
        if evidence.ref is None and evidence.last_known != requested[conversation_id].last_known:
            raise CollectionRejected(f"conversation {conversation_id} does not keep the label its request carried")
