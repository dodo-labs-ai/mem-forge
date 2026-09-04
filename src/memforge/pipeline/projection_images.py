"""Load current, revision-pinned image Artifacts for structured LLM calls."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection

from memforge.llm.structured_images import StructuredLlmImage
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH,
    source_artifact_revision_from_metadata,
)
from memforge.source_projection import SourceProjection
from memforge.storage.document_store import DocumentStore


class ProjectionImageLoadError(ValueError):
    """A current image Artifact cannot safely enter model inference."""

    def __init__(self, *, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


PROJECTION_IMAGE_INFERENCE_CAPABILITY_VERSION = 1


def projection_inference_capability_hash() -> str:
    """Identify the exact binary input contract used by projection extraction."""

    payload = {
        "version": PROJECTION_IMAGE_INFERENCE_CAPABILITY_VERSION,
        "media_types": ("image/gif", "image/jpeg", "image/png", "image/webp"),
        "max_batch_bytes": MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def projection_inference_image_observation_ids(
    projection: SourceProjection,
    *,
    observation_ids: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Return current image Artifact identities admitted by revision metadata."""

    if len(projection.source_units) != 1 or len(projection.source_unit_revisions) != 1:
        raise ProjectionImageLoadError(error_code="invalid_projection_scope")
    requested = set(observation_ids) if observation_ids is not None else None
    current_revision_ids = set(projection.source_unit_revisions[0].observation_revision_ids)
    source_unit_id = projection.source_units[0].id
    output: list[str] = []
    for revision in projection.observation_revisions:
        if revision.id not in current_revision_ids:
            continue
        if requested is not None and revision.observation_id not in requested:
            continue
        if "source_artifact" not in revision.metadata:
            continue
        artifact = source_artifact_revision_from_metadata(
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            source_id=projection.source_id,
            source_unit_id=source_unit_id,
            metadata=revision.metadata,
        )
        if artifact is None:
            raise ProjectionImageLoadError(error_code="invalid_artifact_metadata")
        if artifact.media_type.startswith("image/") and artifact.inference_eligible:
            output.append(revision.observation_id)
    return tuple(output)


def load_projection_images(
    *,
    projection: SourceProjection,
    observation_ids: Collection[str],
    document_store: DocumentStore,
) -> tuple[StructuredLlmImage, ...]:
    """Read exact current image bytes after metadata, size, hash, and budget checks."""

    admitted_ids = set(
        projection_inference_image_observation_ids(
            projection,
            observation_ids=observation_ids,
        )
    )
    current_revision_ids = set(projection.source_unit_revisions[0].observation_revision_ids)
    source_unit_id = projection.source_units[0].id
    images: list[StructuredLlmImage] = []
    total_bytes = 0
    for revision in projection.observation_revisions:
        if revision.id not in current_revision_ids or revision.observation_id not in admitted_ids:
            continue
        artifact = source_artifact_revision_from_metadata(
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            source_id=projection.source_id,
            source_unit_id=source_unit_id,
            metadata=revision.metadata,
        )
        if artifact is None:  # guarded by the metadata-only pass above
            raise ProjectionImageLoadError(error_code="invalid_artifact_metadata")
        total_bytes += artifact.size_bytes
        if total_bytes > MAX_SOURCE_ARTIFACT_INFERENCE_BYTES_PER_BATCH:
            raise ProjectionImageLoadError(error_code="image_batch_too_large")
        try:
            body = document_store.read_artifact(artifact.uri)
        except Exception as exc:
            raise ProjectionImageLoadError(error_code="artifact_unavailable") from exc
        if len(body) != artifact.size_bytes or hashlib.sha256(body).hexdigest() != artifact.sha256:
            raise ProjectionImageLoadError(error_code="artifact_integrity_failed")
        images.append(
            StructuredLlmImage(
                source_observation_id=revision.observation_id,
                media_type=artifact.media_type,
                body=body,
            )
        )
    return tuple(images)
