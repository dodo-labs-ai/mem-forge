"""Provider-neutral binary Source Artifact contracts.

Genes own provider enumeration and download. This module validates and stores
the resulting bytes before Source Projection turns them into revision-pinned
Observations.
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Protocol


SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
MAX_SOURCE_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_SOURCE_ARTIFACTS_PER_UNIT = 20
MAX_SOURCE_ARTIFACT_BYTES_PER_UNIT = 30 * 1024 * 1024


class SourceArtifactContractError(ValueError):
    """Provider Artifact materialization violated the shared contract."""


class SourceArtifactByteStore(Protocol):
    def store_source_artifact(
        self,
        *,
        source_id: str,
        artifact_id: str,
        filename: str,
        content: bytes,
        content_type: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class RawSourceArtifact:
    """Exact provider bytes plus stable attachment identity."""

    provider_key: str
    parent_observation_type: str
    parent_provider_key: str
    provider_revision: str
    filename: str
    media_type: str
    body: bytes
    declared_size_bytes: int | None = None
    locator: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.provider_key,
            self.parent_observation_type,
            self.parent_provider_key,
            self.provider_revision,
            self.filename,
            self.media_type,
        )
        if any(not str(value).strip() for value in required):
            raise SourceArtifactContractError("Source Artifact identity and media fields are required")


@dataclass(frozen=True, slots=True)
class StoredSourceArtifact:
    """Validated immutable bytes ready for Source Projection."""

    id: str
    provider_key: str
    parent_observation_type: str
    parent_provider_key: str
    provider_revision: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    uri: str
    locator: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceArtifactRevision:
    """One immutable Artifact revision reconstructed from Source Projection."""

    artifact_id: str
    observation_id: str
    observation_revision_id: str
    source_id: str
    source_unit_id: str
    parent_observation_id: str
    provider_revision: str
    filename: str
    media_type: str
    size_bytes: int
    sha256: str
    uri: str

    @property
    def resource_url(self) -> str:
        from urllib.parse import quote

        revision_id = quote(self.observation_revision_id, safe="")
        return f"/api/source-artifacts/{revision_id}"

    def metadata(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "observation_id": self.observation_id,
            "observation_revision_id": self.observation_revision_id,
            "parent_observation_id": self.parent_observation_id,
            "filename": self.filename,
            "content_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "url": self.resource_url,
        }


@dataclass(frozen=True, slots=True)
class SourceArtifactEvidence:
    """Active Support linkage for one exact Artifact revision."""

    memory_id: str
    evidence_reference_id: str
    evidence_unit_id: str
    role: str
    artifact: SourceArtifactRevision

    def metadata(self) -> dict[str, object]:
        return {
            **self.artifact.metadata(),
            "evidence_reference_id": self.evidence_reference_id,
            "evidence_unit_id": self.evidence_unit_id,
            "evidence_role": self.role,
        }


def source_artifact_revision_from_metadata(
    *,
    observation_id: str,
    observation_revision_id: str,
    source_id: str,
    source_unit_id: str,
    metadata: Mapping[str, object],
) -> SourceArtifactRevision | None:
    """Parse strict immutable Artifact metadata from one Observation revision."""

    raw = metadata.get("source_artifact")
    if not isinstance(raw, Mapping):
        return None
    try:
        artifact = SourceArtifactRevision(
            artifact_id=str(raw["artifact_id"]),
            observation_id=observation_id,
            observation_revision_id=observation_revision_id,
            source_id=source_id,
            source_unit_id=source_unit_id,
            parent_observation_id=str(raw["parent_observation_id"]),
            provider_revision=str(raw["provider_revision"]),
            filename=str(raw["filename"]),
            media_type=str(raw["media_type"]),
            size_bytes=int(raw["size_bytes"]),
            sha256=str(raw["sha256"]),
            uri=str(raw["uri"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not artifact.artifact_id
        or not artifact.parent_observation_id
        or not artifact.provider_revision
        or not artifact.filename
        or artifact.media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES
        or artifact.size_bytes < 0
        or len(artifact.sha256) != 64
        or not artifact.uri
    ):
        return None
    return artifact


def source_artifact_identity(
    *,
    source_id: str,
    source_unit_key: str,
    provider_key: str,
) -> str:
    """Return a stable opaque identity independent of filename and title."""

    digest = sha256(
        "\x1f".join((source_id, source_unit_key, provider_key)).encode("utf-8")
    ).hexdigest()[:24]
    return f"artifact-{digest}"


def materialize_source_artifact(
    *,
    source_id: str,
    source_unit_key: str,
    artifact: RawSourceArtifact,
    store: SourceArtifactByteStore,
) -> StoredSourceArtifact:
    """Validate one provider payload and store its exact bytes."""

    media_type = artifact.media_type.split(";", 1)[0].strip().lower()
    if media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
        raise SourceArtifactContractError(f"unsupported Source Artifact media type: {media_type}")
    size_bytes = len(artifact.body)
    if artifact.declared_size_bytes is not None and artifact.declared_size_bytes != size_bytes:
        raise SourceArtifactContractError(
            "Source Artifact declared size does not match downloaded bytes"
        )
    if size_bytes > MAX_SOURCE_ARTIFACT_BYTES:
        raise SourceArtifactContractError(
            f"Source Artifact exceeds {MAX_SOURCE_ARTIFACT_BYTES} byte limit"
        )
    artifact_id = source_artifact_identity(
        source_id=source_id,
        source_unit_key=source_unit_key,
        provider_key=artifact.provider_key,
    )
    uri = store.store_source_artifact(
        source_id=source_id,
        artifact_id=artifact_id,
        filename=artifact.filename,
        content=artifact.body,
        content_type=media_type,
    )
    return StoredSourceArtifact(
        id=artifact_id,
        provider_key=artifact.provider_key,
        parent_observation_type=artifact.parent_observation_type,
        parent_provider_key=artifact.parent_provider_key,
        provider_revision=artifact.provider_revision,
        filename=Path(artifact.filename).name,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha256(artifact.body).hexdigest(),
        uri=uri,
        locator=dict(artifact.locator),
    )


def materialize_source_artifacts(
    *,
    source_id: str,
    source_unit_key: str,
    artifacts: tuple[RawSourceArtifact, ...],
    store: SourceArtifactByteStore,
) -> tuple[StoredSourceArtifact, ...]:
    """Validate the bounded Artifact set and store each exact payload."""

    if len(artifacts) > MAX_SOURCE_ARTIFACTS_PER_UNIT:
        raise SourceArtifactContractError(
            f"Source Unit exceeds {MAX_SOURCE_ARTIFACTS_PER_UNIT} Artifact limit"
        )
    total_bytes = sum(len(artifact.body) for artifact in artifacts)
    if total_bytes > MAX_SOURCE_ARTIFACT_BYTES_PER_UNIT:
        raise SourceArtifactContractError(
            f"Source Unit Artifacts exceed {MAX_SOURCE_ARTIFACT_BYTES_PER_UNIT} byte aggregate limit"
        )
    provider_keys = [artifact.provider_key for artifact in artifacts]
    if len(set(provider_keys)) != len(provider_keys):
        raise SourceArtifactContractError("Source Unit contains duplicate Artifact provider identity")
    return tuple(
        materialize_source_artifact(
            source_id=source_id,
            source_unit_key=source_unit_key,
            artifact=artifact,
            store=store,
        )
        for artifact in artifacts
    )


def extension_for_media_type(media_type: str) -> str:
    """Return a safe filename extension for an Artifact media type."""

    normalized = media_type.split(";", 1)[0].strip().lower()
    if normalized == "image/jpeg":
        return ".jpg"
    return mimetypes.guess_extension(normalized) or ".bin"
