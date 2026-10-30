"""The device executor of a conversation Source.

A local agent collects a conversation Source the server cannot read itself,
such as Teams through the user's browser session. For one job attempt it asks
the server what to collect, refuses before polling when the server reads the
messages in another record schema than its provider collects, runs the
provider for that request, and uploads the result: the Collection Manifest
first, then only what the server reports it lacks, the bytes of message
artifacts such as inline images followed by message bodies naming those
uploads. It then hands the attempt over: the server processes it, or for a
dry run reports the plan it would apply. It holds no window, coverage,
absence or retention logic and reads nothing else from the server; the
conversation layer decides from the manifest once processing starts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Protocol

from memforge.conversations.collect import CollectedAttempt, collect
from memforge.conversations.manifest import (
    LOCAL_AGENT_UPGRADE_REQUIRED,
    SERVER_UPGRADE_REQUIRED,
    manifest_to_payload,
    predates_record_schema,
    request_from_payload,
)
from memforge.conversations.port import ConversationProvider, MessageArtifact
from memforge.conversations.records import MESSAGE_OBSERVATION_TYPE, MessageIdentity
from memforge.sync_progress import SyncProgressPhase

logger = logging.getLogger(__name__)

# Bodies per upload request; the server accepts at most 2,000.
BODY_BATCH_SIZE = 500


class ConversationUploadClient(Protocol):
    def conversation_collection_request(
        self, *, source_id: str, local_agent_job_id: str, local_agent_attempt_count: int
    ) -> dict[str, Any]: ...

    def submit_conversation_manifest(
        self, *, source_id: str, manifest: dict[str, Any], local_agent_job_id: str, local_agent_attempt_count: int
    ) -> dict[str, Any]: ...

    def submit_conversation_bodies(
        self, *, source_id: str, bodies: list[dict[str, Any]], local_agent_job_id: str, local_agent_attempt_count: int
    ) -> dict[str, Any]: ...

    def plan_conversation_collection(
        self, *, source_id: str, local_agent_job_id: str, local_agent_attempt_count: int
    ) -> dict[str, Any]: ...

    def start_source_processing(
        self,
        *,
        source_id: str,
        force_full_sync: bool,
        sync_snapshot_id: str,
        local_agent_job_id: str,
        local_agent_attempt_count: int,
    ) -> dict[str, Any]: ...

    def push_source_artifact(
        self,
        *,
        source_id: str,
        source_unit_key: str,
        provider_key: str,
        provider_revision: str,
        parent_observation_type: str,
        parent_provider_key: str,
        filename: str,
        media_type: str,
        content: bytes,
        local_agent_job_id: str | None = None,
        local_agent_attempt_count: int | None = None,
    ) -> dict[str, Any]: ...


class ConversationUploadError(RuntimeError):
    """The server refused part of the upload; ``response`` is its answer."""

    def __init__(self, step: str, response: dict[str, Any]) -> None:
        super().__init__(f"{step} failed: {response.get('detail') or response.get('error')}")
        self.response = response


class RecordSchemaMismatch(RuntimeError):
    """The server reads the Source's messages in another record schema than this agent's provider collects."""


@dataclass(frozen=True, slots=True)
class DeviceCollectionResult:
    """What one job attempt collected and uploaded, and the server's answer when it was handed over.

    ``plan`` is the server's plan for a dry run; ``processing`` is its receipt
    for the processing it started otherwise.
    """

    collection_attempt_id: str
    conversations: int
    messages: int
    bodies_uploaded: int
    artifacts_uploaded: int = 0
    # Artifacts the provider would not serve; the next attempt asks for them again.
    artifacts_unavailable: int = 0
    plan: dict[str, Any] | None = None
    processing: dict[str, Any] | None = None

    @property
    def counts(self) -> dict[str, int]:
        return {
            "conversations": self.conversations,
            "messages": self.messages,
            "bodies_uploaded": self.bodies_uploaded,
            "artifacts_uploaded": self.artifacts_uploaded,
            "artifacts_unavailable": self.artifacts_unavailable,
        }


async def run_device_collection(
    provider: ConversationProvider,
    client: ConversationUploadClient,
    *,
    source_id: str,
    local_agent_job_id: str,
    local_agent_attempt_count: int,
    config_revision: str,
    dry_run: bool = False,
    force_full_sync: bool = False,
    report: Callable[[SyncProgressPhase, int, int], None] = lambda phase, completed, total: None,
) -> DeviceCollectionResult:
    """Collect one job attempt of a conversation Source, upload it, and hand it over to the server."""

    attempt = {"local_agent_job_id": local_agent_job_id, "local_agent_attempt_count": local_agent_attempt_count}
    answer = _checked("collection request", client.conversation_collection_request(source_id=source_id, **attempt))
    _require_record_schema(provider, answer.get("record_schema"))
    collection_attempt_id = str(answer["collection_attempt_id"])
    request = request_from_payload(answer["request"])
    report("fetching", 0, len(request.conversations))
    collection = await collect(
        provider,
        request,
        source_id=source_id,
        collection_attempt_id=collection_attempt_id,
        config_revision=config_revision,
    )
    answer = _checked(
        "manifest upload",
        client.submit_conversation_manifest(source_id=source_id, manifest=_upload_payload(collection), **attempt),
    )
    missing = answer["missing"]
    bodies: list[dict[str, Any]] = []
    uploaded = unavailable = 0
    for index, item in enumerate(missing):
        report("uploading", index, len(missing))
        identity = MessageIdentity(str(item["conversation_id"]), str(item["message_id"]))
        links = []
        for wanted in item.get("artifacts") or ():
            artifact = MessageArtifact(str(wanted["provider_key"]), str(wanted["provider_revision"]))
            link = await _upload_artifact(provider, client, collection, identity, artifact, source_id, attempt)
            if link is None:
                unavailable += 1
            else:
                uploaded += 1
                links.append(link)
        if item.get("body") or links:
            bodies.append(
                {
                    "conversation_id": identity.conversation_id,
                    "message_id": identity.message_id,
                    "record": collection.bodies[identity],
                    "artifacts": links,
                }
            )
    for offset in range(0, len(bodies), BODY_BATCH_SIZE):
        _checked(
            "body upload",
            client.submit_conversation_bodies(
                source_id=source_id, bodies=bodies[offset:offset + BODY_BATCH_SIZE], **attempt
            ),
        )
    report("uploading", len(missing), len(missing))
    result = DeviceCollectionResult(
        collection_attempt_id=collection_attempt_id,
        conversations=len(collection.conversations),
        messages=len(collection.messages),
        bodies_uploaded=sum(1 for item in missing if item.get("body")),
        artifacts_uploaded=uploaded,
        artifacts_unavailable=unavailable,
    )
    if dry_run:
        return replace(
            result,
            plan=_checked("dry-run plan", client.plan_conversation_collection(source_id=source_id, **attempt)),
        )
    return replace(
        result,
        processing=_checked(
            "processing start",
            client.start_source_processing(
                source_id=source_id,
                force_full_sync=force_full_sync,
                sync_snapshot_id=collection_attempt_id,
                **attempt,
            ),
        ),
    )


async def _upload_artifact(
    provider: ConversationProvider,
    client: ConversationUploadClient,
    collection: CollectedAttempt,
    identity: MessageIdentity,
    artifact: MessageArtifact,
    source_id: str,
    attempt: dict[str, Any],
) -> dict[str, str] | None:
    """Read one artifact from the provider and upload it; ``None`` when the provider would not serve it."""

    try:
        body = await provider.read_artifact(
            collection.refs[identity.conversation_id], collection.provider_messages[identity], artifact
        )
    except Exception:
        logger.warning("Could not read artifact %s of message %s", artifact.provider_key, identity, exc_info=True)
        return None
    response = _checked(
        "artifact upload",
        client.push_source_artifact(
            source_id=source_id,
            # The upload is scoped by its conversation; the body that names it
            # places it in whichever Unit holds the message.
            source_unit_key=identity.conversation_id,
            provider_key=artifact.provider_key,
            provider_revision=artifact.provider_revision,
            parent_observation_type=MESSAGE_OBSERVATION_TYPE,
            parent_provider_key=identity.message_id,
            filename=body.filename,
            media_type=body.media_type,
            content=body.content,
            **attempt,
        ),
    )
    return {
        "provider_key": artifact.provider_key,
        "provider_revision": artifact.provider_revision,
        "input_sha256": str(response["input_sha256"]),
    }


def _upload_payload(collection: CollectedAttempt) -> dict[str, Any]:
    """The manifest as the device sends it: the server stamps its own receipt time."""

    payload = manifest_to_payload(collection.manifest(received_at=datetime.now(timezone.utc)))
    del payload["received_at"]
    return payload


def _require_record_schema(provider: ConversationProvider, read: object) -> None:
    """Refuse, before polling, a request whose messages the server reads in another record schema.

    A server that names no record schema leaves the check to manifest intake.
    """

    collected = provider.record_schema.qualified_name
    if read is None or read == collected:
        return
    read = str(read)
    if predates_record_schema(collected, read):
        raise RecordSchemaMismatch(
            f"{LOCAL_AGENT_UPGRADE_REQUIRED}: this local daemon collects messages as {collected} and the server "
            f"reads {read}; upgrade memforge-ai on this device, then run `memforge daemon restart`"
        )
    if predates_record_schema(read, collected):
        raise RecordSchemaMismatch(
            f"{SERVER_UPGRADE_REQUIRED}: this local daemon collects messages as {collected} and the server reads "
            f"the earlier {read}"
        )
    raise RecordSchemaMismatch(f"this local daemon collects messages as {collected} and the server reads {read}")


def _checked(step: str, response: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("error"):
        raise ConversationUploadError(step, response if isinstance(response, dict) else {"error": str(response)})
    return response
