"""What each collection attempt was asked, what it answered, and the bodies it collected.

An attempt keeps the first request issued to it, and its Collection Manifest
is immutable once recorded. A body names, for each of its message's
artifacts, the uploaded input that holds the artifact's bytes; the upload must
have been made for that message. Once an attempt is processed, its bodies and
everything older go; its manifest stays for the next request to continue from.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    manifest_from_payload,
    manifest_to_payload,
    request_from_payload,
    request_to_payload,
)
from memforge.conversations.records import (
    MESSAGE_OBSERVATION_TYPE,
    MessageIdentity,
    canonical_record_json,
    record_revision,
)
from memforge.conversations.units import MemberArtifact
from memforge.models import SourceSyncInput
from memforge.source_artifact_inputs import artifact_input


@runtime_checkable
class CollectionStore(Protocol):
    async def record_conversation_collection_request(
        self, *, workspace_id: str, source_id: str, collection_attempt_id: str, request_json: str, issued_at: str
    ) -> str: ...

    async def get_conversation_collection_request(
        self, *, workspace_id: str, source_id: str, collection_attempt_id: str
    ) -> str | None: ...

    async def prune_conversation_collections(
        self, *, workspace_id: str, source_id: str, processed_attempt_id: str
    ) -> None: ...

    async def record_conversation_collection_manifest(
        self, *, workspace_id: str, source_id: str, collection_attempt_id: str, manifest_json: str, received_at: str
    ) -> None: ...

    async def get_conversation_collection_manifest(
        self, *, workspace_id: str, source_id: str, collection_attempt_id: str
    ) -> str | None: ...

    async def get_latest_conversation_collection_manifest(self, *, workspace_id: str, source_id: str) -> str | None: ...

    async def record_conversation_collection_bodies(
        self,
        *,
        workspace_id: str,
        source_id: str,
        collection_attempt_id: str,
        bodies: Sequence[tuple[str, str, str, str, str]],
    ) -> None: ...

    async def list_conversation_collection_bodies(
        self, *, workspace_id: str, source_id: str, collection_attempt_id: str
    ) -> list[tuple[str, str, str, str, str]]: ...

    async def get_source_sync_inputs_by_raw_hashes(
        self, *, source_id: str, workspace_id: str, raw_sha256s: Sequence[str]
    ) -> list[SourceSyncInput]: ...


@dataclass(frozen=True, slots=True)
class ArtifactLink:
    """The upload that holds one message artifact's bytes, named by its input hash."""

    provider_key: str
    provider_revision: str
    input_sha256: str


@dataclass(frozen=True, slots=True)
class CollectedBody:
    """One message body a collection attempt holds, with the uploads of its artifacts."""

    record: Mapping[str, object]
    artifacts: tuple[ArtifactLink, ...] = ()


class StoredCollections:
    """Collection Manifests and bodies of one workspace, read and written as domain values."""

    def __init__(self, store: CollectionStore, *, workspace_id: str) -> None:
        self._store = store
        self._workspace_id = workspace_id

    async def issue_request(
        self, source_id: str, collection_attempt_id: str, request: CollectionRequest, *, issued_at: datetime
    ) -> CollectionRequest:
        """Keep ``request`` as the attempt's request, unless one was issued to it already; return the attempt's request."""

        stored = await self._store.record_conversation_collection_request(
            workspace_id=self._workspace_id,
            source_id=source_id,
            collection_attempt_id=collection_attempt_id,
            request_json=_json(request_to_payload(request)),
            issued_at=issued_at.isoformat(),
        )
        return request_from_payload(json.loads(stored))

    async def request(self, source_id: str, collection_attempt_id: str) -> CollectionRequest | None:
        stored = await self._store.get_conversation_collection_request(
            workspace_id=self._workspace_id, source_id=source_id, collection_attempt_id=collection_attempt_id
        )
        return request_from_payload(json.loads(stored)) if stored is not None else None

    async def prune(self, source_id: str, processed_attempt_id: str) -> None:
        """Drop the bodies of a processed attempt and every older collection, keeping its manifest."""

        await self._store.prune_conversation_collections(
            workspace_id=self._workspace_id, source_id=source_id, processed_attempt_id=processed_attempt_id
        )

    async def record_manifest(self, manifest: CollectionManifest) -> None:
        await self._store.record_conversation_collection_manifest(
            workspace_id=self._workspace_id,
            source_id=manifest.source_id,
            collection_attempt_id=manifest.collection_attempt_id,
            manifest_json=_json(manifest_to_payload(manifest)),
            received_at=manifest.received_at.isoformat(),
        )

    async def manifest(self, source_id: str, collection_attempt_id: str) -> CollectionManifest | None:
        stored = await self._store.get_conversation_collection_manifest(
            workspace_id=self._workspace_id, source_id=source_id, collection_attempt_id=collection_attempt_id
        )
        return manifest_from_payload(json.loads(stored)) if stored is not None else None

    async def latest_manifest(self, source_id: str) -> CollectionManifest | None:
        stored = await self._store.get_latest_conversation_collection_manifest(
            workspace_id=self._workspace_id, source_id=source_id
        )
        return manifest_from_payload(json.loads(stored)) if stored is not None else None

    async def record_bodies(
        self, source_id: str, collection_attempt_id: str, bodies: Mapping[MessageIdentity, CollectedBody]
    ) -> None:
        await self._store.record_conversation_collection_bodies(
            workspace_id=self._workspace_id,
            source_id=source_id,
            collection_attempt_id=collection_attempt_id,
            bodies=[
                (
                    identity.conversation_id,
                    identity.message_id,
                    record_revision(body.record),
                    canonical_record_json(body.record),
                    json.dumps(
                        [
                            {
                                "provider_key": link.provider_key,
                                "provider_revision": link.provider_revision,
                                "input_sha256": link.input_sha256,
                            }
                            for link in sorted(body.artifacts, key=lambda link: link.provider_key)
                        ],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                for identity, body in sorted(bodies.items())
            ],
        )

    async def bodies(self, source_id: str, collection_attempt_id: str) -> dict[MessageIdentity, CollectedBody]:
        rows = await self._store.list_conversation_collection_bodies(
            workspace_id=self._workspace_id, source_id=source_id, collection_attempt_id=collection_attempt_id
        )
        return {
            MessageIdentity(conversation_id, message_id): CollectedBody(
                record=json.loads(record),
                artifacts=tuple(ArtifactLink(**link) for link in json.loads(artifacts)),
            )
            for conversation_id, message_id, _revision, record, artifacts in rows
        }

    async def uploaded_artifacts(
        self, source_id: str, bodies: Mapping[MessageIdentity, CollectedBody]
    ) -> dict[MessageIdentity, tuple[MemberArtifact, ...]]:
        """The stored artifacts each body's uploads hold.

        An upload belongs to one message: it was sent for the message's
        conversation, as an artifact of that message, under the artifact
        identity the body names.
        """

        links = {link.input_sha256 for body in bodies.values() for link in body.artifacts}
        if not links:
            return {}
        inputs = {
            source_input.raw_sha256: source_input
            for source_input in await self._store.get_source_sync_inputs_by_raw_hashes(
                source_id=source_id, workspace_id=self._workspace_id, raw_sha256s=sorted(links)
            )
        }
        resolved: dict[MessageIdentity, tuple[MemberArtifact, ...]] = {}
        for identity, body in bodies.items():
            artifacts = []
            for link in body.artifacts:
                source_input = inputs.get(link.input_sha256)
                upload = artifact_input(source_input) if source_input is not None else None
                if upload is None or (
                    upload.scope_key,
                    upload.parent_observation_type,
                    upload.parent_provider_key,
                    upload.provider_key,
                    upload.provider_revision,
                ) != (
                    identity.conversation_id,
                    MESSAGE_OBSERVATION_TYPE,
                    identity.message_id,
                    link.provider_key,
                    link.provider_revision,
                ):
                    raise ValueError(f"artifact upload {link.input_sha256} does not belong to message {identity}")
                artifacts.append(
                    MemberArtifact(
                        provider_key=upload.provider_key,
                        provider_revision=upload.provider_revision,
                        filename=upload.filename,
                        media_type=upload.media_type,
                        size_bytes=upload.size_bytes,
                        sha256=upload.content_sha256,
                        uri=upload.uri,
                    )
                )
            if artifacts:
                resolved[identity] = tuple(artifacts)
        return resolved


def _json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
