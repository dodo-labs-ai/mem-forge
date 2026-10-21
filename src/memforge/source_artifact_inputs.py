"""Binary inputs an executor keeps for a Source, such as a message's inline images.

Each binary is kept once as an immutable Source Sync Input whose descriptor
names it as its provider does: the scope it belongs to, its provider key and
revision, its parent Observation and its filename, with its media type, size
and content hash. For a conversation Source the scope is the conversation,
and the parent is the message. A device executor uploads it; a server
executor keeps it directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memforge.models import SourceSyncInput

if TYPE_CHECKING:
    from memforge.storage.database import Database

DESCRIPTOR_KEY = "local_source_artifact"


@dataclass(frozen=True, slots=True)
class ArtifactInput:
    """One uploaded binary as its descriptor names it, with the stored bytes that hold it."""

    scope_key: str
    provider_key: str
    provider_revision: str
    parent_observation_type: str
    parent_provider_key: str
    filename: str
    media_type: str
    size_bytes: int
    content_sha256: str
    uri: str


def artifact_input(source_input: SourceSyncInput) -> ArtifactInput | None:
    """The binary one Source Sync Input holds, ``None`` for an input that is not an artifact upload."""

    descriptor = source_input.metadata.get(DESCRIPTOR_KEY)
    if not isinstance(descriptor, Mapping):
        return None
    return ArtifactInput(
        scope_key=str(descriptor["source_unit_key"]),
        provider_key=str(descriptor["provider_key"]),
        provider_revision=str(descriptor["provider_revision"]),
        parent_observation_type=str(descriptor["parent_observation_type"]),
        parent_provider_key=str(descriptor["parent_provider_key"]),
        filename=str(descriptor["filename"]),
        media_type=str(descriptor["media_type"]),
        size_bytes=int(descriptor["size_bytes"]),
        content_sha256=str(descriptor["content_sha256"]),
        uri=source_input.raw_uri,
    )


async def retain_artifact_input(
    db: Database,
    artifact_store: Any,
    *,
    source_id: str,
    workspace_id: str,
    identity: dict[str, str],
    media_type: str,
    content: Any,
    size_bytes: int,
    content_sha256: str,
) -> str:
    """Keep one uploaded binary as an immutable input of its Source and return the input's hash.

    ``identity`` names the binary as its provider does: ``source_unit_key``
    (the Source Unit, or for a conversation Source the conversation, it
    belongs to), its provider key and revision, its parent Observation and its
    filename. The same binary uploaded again is the same input.
    """

    descriptor = {
        **identity,
        "media_type": media_type,
        "size_bytes": size_bytes,
        "content_sha256": content_sha256,
    }
    input_sha256 = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if await db.get_source_sync_inputs_by_raw_hashes(
        source_id=source_id, workspace_id=workspace_id, raw_sha256s=(input_sha256,)
    ):
        return input_sha256
    # This write does not hold the Source activity lease, so the attempt
    # writes a key of its own: a cleanup task queued by an earlier failed
    # attempt can never name the object written here.
    artifact_uri = await asyncio.to_thread(
        artifact_store.store_source_artifact,
        source_id=source_id,
        artifact_id=f"local-input-{input_sha256[:24]}-{uuid.uuid4().hex}",
        filename=identity["filename"],
        content=content,
        content_type=media_type,
        size_bytes=size_bytes,
        sha256=content_sha256,
    )
    try:
        retained = await db.create_source_sync_input(
            source_id=source_id,
            workspace_id=workspace_id,
            raw_uri=artifact_uri,
            raw_sha256=input_sha256,
            raw_content_type=media_type,
            metadata={DESCRIPTOR_KEY: descriptor},
        )
    except Exception:
        raced = await db.get_source_sync_inputs_by_raw_hashes(
            source_id=source_id, workspace_id=workspace_id, raw_sha256s=(input_sha256,)
        )
        if not raced or raced[0].raw_uri != artifact_uri:
            await db.enqueue_source_artifact_cleanup_task(source_id=source_id, artifact_uri=artifact_uri)
        raise
    return retained.raw_sha256
