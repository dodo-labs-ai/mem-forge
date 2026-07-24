"""Provider-neutral binary Source Artifact contracts.

Genes own provider enumeration and download. This module validates and stores
the resulting bytes before Source Projection turns them into revision-pinned
Observations.
"""

from __future__ import annotations

import mimetypes
import tempfile
from collections.abc import AsyncIterable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Mapping, Protocol


SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
MAX_SOURCE_ARTIFACT_DESCRIPTORS_PER_UNIT = 200
MAX_SOURCE_ARTIFACTS_PER_UNIT = 100
MAX_SOURCE_ARTIFACT_STORAGE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT = 128 * 1024 * 1024
MAX_SOURCE_ARTIFACT_INFERENCE_BYTES = 10 * 1024 * 1024
MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH = 30 * 1024 * 1024
SOURCE_ARTIFACT_STREAM_CHUNK_BYTES = 256 * 1024
SOURCE_ARTIFACT_SPOOL_MEMORY_BYTES = 1024 * 1024


class SourceArtifactContractError(ValueError):
    """Provider Artifact materialization violated the shared contract."""


def normalize_source_artifact_media_type(value: object) -> str:
    """Return one canonical media type for provider descriptors and payloads."""

    return str(value or "").split(";", 1)[0].strip().lower()


def parse_source_artifact_content_length(value: object) -> int | None:
    """Parse an optional non-negative HTTP Content-Length."""

    text = str(value or "").strip()
    if not text:
        return None
    try:
        length = int(text)
    except ValueError as exc:
        raise SourceArtifactContractError("Source Artifact has an invalid transport length") from exc
    if length < 0:
        raise SourceArtifactContractError("Source Artifact has an invalid transport length")
    return length


class SourceArtifactByteStore(Protocol):
    def store_source_artifact(
        self,
        *,
        source_id: str,
        artifact_id: str,
        filename: str,
        content: BinaryIO,
        content_type: str,
        size_bytes: int,
        sha256: str,
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class RawSourceArtifact:
    """Stable provider descriptor whose body is opened only by the owning Gene."""

    provider_key: str
    parent_observation_type: str
    parent_provider_key: str
    provider_revision: str
    filename: str
    media_type: str
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
class SourceArtifactDownload:
    """One opened provider body plus transport metadata."""

    chunks: AsyncIterable[bytes]
    media_type: str | None = None
    content_length: int | None = None
    content_encoding: str | None = None


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
    inference_eligible: bool
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
    inference_eligible: bool

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
            "inference_eligible": self.inference_eligible,
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
        size_bytes = int(raw["size_bytes"])
        inference_eligible = raw.get("inference_eligible")
        if inference_eligible is None:
            # A missing decision is safe only within the inference budget.
            inference_eligible = size_bytes <= MAX_SOURCE_ARTIFACT_INFERENCE_BYTES
        elif not isinstance(inference_eligible, bool):
            return None
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
            size_bytes=size_bytes,
            sha256=str(raw["sha256"]),
            uri=str(raw["uri"]),
            inference_eligible=inference_eligible,
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

    digest = sha256("\x1f".join((source_id, source_unit_key, provider_key)).encode("utf-8")).hexdigest()[:24]
    return f"artifact-{digest}"


@dataclass(slots=True)
class _DownloadedSourceArtifact:
    descriptor: RawSourceArtifact
    content: BinaryIO
    media_type: str
    size_bytes: int
    sha256: str


async def _download_source_artifact(
    *,
    artifact: RawSourceArtifact,
    remaining_unit_bytes: int,
    open_artifact: Callable[
        [RawSourceArtifact],
        AbstractAsyncContextManager[SourceArtifactDownload],
    ],
) -> _DownloadedSourceArtifact:
    """Stream one provider body into a bounded spool and validate exact bytes."""

    media_type = normalize_source_artifact_media_type(artifact.media_type)
    if media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
        raise SourceArtifactContractError(f"unsupported Source Artifact media type: {media_type}")
    if artifact.declared_size_bytes is not None and artifact.declared_size_bytes > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
        raise SourceArtifactContractError(
            f"Source Artifact exceeds {MAX_SOURCE_ARTIFACT_STORAGE_BYTES} byte storage limit"
        )
    if artifact.declared_size_bytes is not None and artifact.declared_size_bytes > remaining_unit_bytes:
        raise SourceArtifactContractError(
            "Source Unit Artifacts exceed "
            f"{MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT} byte storage aggregate limit"
        )
    content = tempfile.SpooledTemporaryFile(
        max_size=SOURCE_ARTIFACT_SPOOL_MEMORY_BYTES,
        mode="w+b",
    )
    digest = sha256()
    size_bytes = 0
    try:
        async with open_artifact(artifact) as download:
            response_media_type = normalize_source_artifact_media_type(download.media_type)
            if response_media_type and response_media_type != media_type:
                raise SourceArtifactContractError("Source Artifact media type changed during download")
            if download.content_length is not None and download.content_length > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
                raise SourceArtifactContractError(
                    f"Source Artifact exceeds {MAX_SOURCE_ARTIFACT_STORAGE_BYTES} byte storage limit"
                )
            if download.content_length is not None and download.content_length > remaining_unit_bytes:
                raise SourceArtifactContractError(
                    "Source Unit Artifacts exceed "
                    f"{MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT} byte storage aggregate limit"
                )
            async for chunk in download.chunks:
                if not isinstance(chunk, bytes):
                    chunk = bytes(chunk)
                if not chunk:
                    continue
                size_bytes += len(chunk)
                if size_bytes > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
                    raise SourceArtifactContractError(
                        f"Source Artifact exceeds {MAX_SOURCE_ARTIFACT_STORAGE_BYTES} byte storage limit"
                    )
                if size_bytes > remaining_unit_bytes:
                    raise SourceArtifactContractError(
                        "Source Unit Artifacts exceed "
                        f"{MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT} byte storage aggregate limit"
                    )
                digest.update(chunk)
                content.write(chunk)
            if (
                download.content_length is not None
                and not str(download.content_encoding or "").strip()
                and download.content_length != size_bytes
            ):
                raise SourceArtifactContractError("Source Artifact transport length does not match downloaded bytes")
        if artifact.declared_size_bytes is not None and artifact.declared_size_bytes != size_bytes:
            raise SourceArtifactContractError("Source Artifact declared size does not match downloaded bytes")
        content.rollover()
        content.seek(0)
        return _DownloadedSourceArtifact(
            descriptor=artifact,
            content=content,
            media_type=media_type,
            size_bytes=size_bytes,
            sha256=digest.hexdigest(),
        )
    except Exception:
        content.close()
        raise


def _store_downloaded_source_artifact(
    *,
    source_id: str,
    source_unit_key: str,
    artifact: _DownloadedSourceArtifact,
    store: SourceArtifactByteStore,
) -> StoredSourceArtifact:
    """Store one fully validated spooled body without materializing it in memory."""

    descriptor = artifact.descriptor
    artifact_id = source_artifact_identity(
        source_id=source_id,
        source_unit_key=source_unit_key,
        provider_key=descriptor.provider_key,
    )
    uri = store.store_source_artifact(
        source_id=source_id,
        artifact_id=artifact_id,
        filename=descriptor.filename,
        content=artifact.content,
        content_type=artifact.media_type,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
    )
    return StoredSourceArtifact(
        id=artifact_id,
        provider_key=descriptor.provider_key,
        parent_observation_type=descriptor.parent_observation_type,
        parent_provider_key=descriptor.parent_provider_key,
        provider_revision=descriptor.provider_revision,
        filename=Path(descriptor.filename).name,
        media_type=artifact.media_type,
        size_bytes=artifact.size_bytes,
        sha256=artifact.sha256,
        uri=uri,
        inference_eligible=artifact.size_bytes <= MAX_SOURCE_ARTIFACT_INFERENCE_BYTES,
        locator=dict(descriptor.locator),
    )


async def materialize_source_artifacts(
    *,
    source_id: str,
    source_unit_key: str,
    artifacts: tuple[RawSourceArtifact, ...],
    store: SourceArtifactByteStore,
    open_artifact: Callable[
        [RawSourceArtifact],
        AbstractAsyncContextManager[SourceArtifactDownload],
    ],
) -> tuple[StoredSourceArtifact, ...]:
    """Validate one bounded descriptor set, spool it, then persist exact bodies."""

    if len(artifacts) > MAX_SOURCE_ARTIFACTS_PER_UNIT:
        raise SourceArtifactContractError(f"Source Unit exceeds {MAX_SOURCE_ARTIFACTS_PER_UNIT} Artifact limit")
    provider_keys = [artifact.provider_key for artifact in artifacts]
    if len(set(provider_keys)) != len(provider_keys):
        raise SourceArtifactContractError("Source Unit contains duplicate Artifact provider identity")
    declared_total = sum(
        artifact.declared_size_bytes
        for artifact in artifacts
        if artifact.declared_size_bytes is not None
    )
    if declared_total > MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT:
        raise SourceArtifactContractError(
            "Source Unit Artifacts exceed "
            f"{MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT} byte storage aggregate limit"
        )
    downloaded: list[_DownloadedSourceArtifact] = []
    try:
        total_bytes = 0
        for descriptor in artifacts:
            item = await _download_source_artifact(
                artifact=descriptor,
                remaining_unit_bytes=(
                    MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT - total_bytes
                ),
                open_artifact=open_artifact,
            )
            downloaded.append(item)
            total_bytes += item.size_bytes
        return tuple(
            _store_downloaded_source_artifact(
                source_id=source_id,
                source_unit_key=source_unit_key,
                artifact=artifact,
                store=store,
            )
            for artifact in downloaded
        )
    finally:
        for artifact in downloaded:
            artifact.content.close()


def extension_for_media_type(media_type: str) -> str:
    """Return a safe filename extension for an Artifact media type."""

    normalized = normalize_source_artifact_media_type(media_type)
    if normalized == "image/jpeg":
        return ".jpg"
    return mimetypes.guess_extension(normalized) or ".bin"
