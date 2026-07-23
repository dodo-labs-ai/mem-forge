from __future__ import annotations

import hashlib

import pytest

from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_BYTES,
    MAX_SOURCE_ARTIFACTS_PER_UNIT,
    RawSourceArtifact,
    SourceArtifactContractError,
    materialize_source_artifact,
    materialize_source_artifacts,
)
from memforge.storage.document_store import LocalDocumentStore


def test_materialized_source_artifact_round_trips_exact_bytes_and_identity(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))
    payload = b"\x89PNG\r\n\x1a\nknown-image-bytes"
    raw = RawSourceArtifact(
        provider_key="attachment-42",
        parent_observation_type="comment",
        parent_provider_key="comment-7",
        provider_revision="3",
        filename="diagram.png",
        media_type="image/png",
        body=payload,
        declared_size_bytes=len(payload),
    )

    artifact = materialize_source_artifact(
        source_id="source-a",
        source_unit_key="issue-10",
        artifact=raw,
        store=store,
    )

    assert artifact.provider_key == "attachment-42"
    assert artifact.parent_observation_type == "comment"
    assert artifact.parent_provider_key == "comment-7"
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)
    assert artifact.media_type == "image/png"
    assert store.read_artifact(artifact.uri) == payload


def test_materialization_rejects_provider_size_drift_before_projection(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))
    raw = RawSourceArtifact(
        provider_key="attachment-42",
        parent_observation_type="page_body",
        parent_provider_key="page-1:body",
        provider_revision="1",
        filename="diagram.png",
        media_type="image/png",
        body=b"actual",
        declared_size_bytes=99,
    )

    with pytest.raises(SourceArtifactContractError, match="declared size"):
        materialize_source_artifact(
            source_id="source-a",
            source_unit_key="page-1",
            artifact=raw,
            store=store,
        )


def test_source_artifact_set_preserves_multiple_identities_and_rejects_duplicates(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))

    def artifact(provider_key: str) -> RawSourceArtifact:
        return RawSourceArtifact(
            provider_key=provider_key,
            parent_observation_type="page_body",
            parent_provider_key="page-1:body",
            provider_revision="1",
            filename=f"{provider_key}.png",
            media_type="image/png",
            body=provider_key.encode(),
        )

    stored = materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=(artifact("attachment-1"), artifact("attachment-2")),
        store=store,
    )

    assert len({item.id for item in stored}) == 2
    with pytest.raises(SourceArtifactContractError, match="duplicate"):
        materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(artifact("attachment-1"), artifact("attachment-1")),
            store=store,
        )


def test_materialization_accepts_customer_sized_bounded_artifact_set(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))
    artifact_count = 56

    stored = materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=tuple(
            RawSourceArtifact(
                provider_key=f"attachment-{index}",
                parent_observation_type="page_body",
                parent_provider_key="page-1:body",
                provider_revision="1",
                filename=f"attachment-{index}.png",
                media_type="image/png",
                body=f"image-{index}".encode(),
            )
            for index in range(artifact_count)
        ),
        store=store,
    )

    assert artifact_count <= MAX_SOURCE_ARTIFACTS_PER_UNIT
    assert len(stored) == artifact_count


def test_materialization_rejects_artifact_set_beyond_persistence_budget(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))

    with pytest.raises(SourceArtifactContractError, match="Artifact limit"):
        materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=tuple(
                RawSourceArtifact(
                    provider_key=f"attachment-{index}",
                    parent_observation_type="page_body",
                    parent_provider_key="page-1:body",
                    provider_revision="1",
                    filename=f"attachment-{index}.png",
                    media_type="image/png",
                    body=b"x",
                )
                for index in range(MAX_SOURCE_ARTIFACTS_PER_UNIT + 1)
            ),
            store=store,
        )


@pytest.mark.parametrize(
    ("media_type", "body", "error"),
    [
        ("application/octet-stream", b"binary", "unsupported"),
        ("image/png", b"x" * (MAX_SOURCE_ARTIFACT_BYTES + 1), "byte limit"),
    ],
)
def test_materialization_rejects_unsupported_or_oversized_artifact(
    tmp_path,
    media_type: str,
    body: bytes,
    error: str,
) -> None:
    store = LocalDocumentStore(str(tmp_path))
    raw = RawSourceArtifact(
        provider_key="attachment-42",
        parent_observation_type="page_body",
        parent_provider_key="page-1:body",
        provider_revision="1",
        filename="artifact.bin",
        media_type=media_type,
        body=body,
    )

    with pytest.raises(SourceArtifactContractError, match=error):
        materialize_source_artifact(
            source_id="source-a",
            source_unit_key="page-1",
            artifact=raw,
            store=store,
        )
