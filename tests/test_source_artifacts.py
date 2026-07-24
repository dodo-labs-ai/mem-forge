from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager

import pytest

from memforge import source_artifacts
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACTS_PER_UNIT,
    RawSourceArtifact,
    SourceArtifactContractError,
    SourceArtifactDownload,
    materialize_source_artifacts,
)
from memforge.storage.document_store import LocalDocumentStore


def _artifact(provider_key: str, payload: bytes, **overrides) -> RawSourceArtifact:
    values = {
        "provider_key": provider_key,
        "parent_observation_type": "page_body",
        "parent_provider_key": "page-1:body",
        "provider_revision": "1",
        "filename": f"{provider_key}.png",
        "media_type": "image/png",
        "declared_size_bytes": len(payload),
        "locator": {"payload_key": provider_key},
    }
    values.update(overrides)
    return RawSourceArtifact(**values)


def _opener(payloads: dict[str, bytes], *, transport_length_delta: int = 0):
    @asynccontextmanager
    async def open_artifact(artifact: RawSourceArtifact):
        payload = payloads[str(artifact.locator["payload_key"])]

        async def chunks():
            midpoint = max(1, len(payload) // 2)
            yield payload[:midpoint]
            yield payload[midpoint:]

        yield SourceArtifactDownload(
            chunks=chunks(),
            media_type=artifact.media_type,
            content_length=len(payload) + transport_length_delta,
        )

    return open_artifact


@pytest.mark.asyncio
async def test_materialization_streams_exact_bytes_and_identity(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))
    payload = b"\x89PNG\r\n\x1a\nknown-image-bytes"

    (artifact,) = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="issue-10",
        artifacts=(_artifact("attachment-42", payload),),
        store=store,
        open_artifact=_opener({"attachment-42": payload}),
    )

    assert artifact.provider_key == "attachment-42"
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.size_bytes == len(payload)
    assert artifact.inference_eligible is True
    assert store.read_artifact(artifact.uri) == payload

    replacement = b"\x89PNG\r\n\x1a\nrevised-image-bytes"
    (revised,) = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="issue-10",
        artifacts=(_artifact("attachment-42", replacement),),
        store=store,
        open_artifact=_opener({"attachment-42": replacement}),
    )

    assert revised.uri != artifact.uri
    assert store.read_artifact(artifact.uri) == payload
    assert store.read_artifact(revised.uri) == replacement


@pytest.mark.asyncio
async def test_storage_and_inference_budgets_are_independent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_artifacts, "MAX_SOURCE_ARTIFACT_INFERENCE_BYTES", 4)
    monkeypatch.setattr(source_artifacts, "MAX_SOURCE_ARTIFACT_STORAGE_BYTES", 32)
    payload = b"retrievable-original"

    (artifact,) = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=(_artifact("large", payload),),
        store=LocalDocumentStore(str(tmp_path)),
        open_artifact=_opener({"large": payload}),
    )

    assert artifact.size_bytes > source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES
    assert artifact.inference_eligible is False


@pytest.mark.asyncio
async def test_materialization_rejects_descriptor_or_transport_size_drift(tmp_path) -> None:
    payload = b"actual"
    descriptor = _artifact("attachment-42", payload, declared_size_bytes=99)

    with pytest.raises(SourceArtifactContractError, match="declared size"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(descriptor,),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({"attachment-42": payload}),
        )

    with pytest.raises(SourceArtifactContractError, match="transport length"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(_artifact("attachment-42", payload),),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({"attachment-42": payload}, transport_length_delta=1),
        )


@pytest.mark.asyncio
async def test_materialization_validates_the_set_before_persistence(tmp_path) -> None:
    payloads = {"one": b"one", "two": b"two"}
    stored = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=tuple(_artifact(key, body) for key, body in payloads.items()),
        store=LocalDocumentStore(str(tmp_path)),
        open_artifact=_opener(payloads),
    )
    assert len({item.id for item in stored}) == 2

    with pytest.raises(SourceArtifactContractError, match="duplicate"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(
                _artifact("one", payloads["one"]),
                _artifact("one", payloads["one"]),
            ),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener(payloads),
        )


@pytest.mark.asyncio
async def test_materialization_rejects_count_and_storage_limits(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"12345"
    with pytest.raises(SourceArtifactContractError, match="Artifact limit"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=tuple(
                _artifact(f"attachment-{index}", payload) for index in range(MAX_SOURCE_ARTIFACTS_PER_UNIT + 1)
            ),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({}),
        )

    monkeypatch.setattr(source_artifacts, "MAX_SOURCE_ARTIFACT_STORAGE_BYTES", 4)
    with pytest.raises(SourceArtifactContractError, match="storage limit"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(_artifact("one", payload),),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({"one": payload}),
        )

    monkeypatch.setattr(source_artifacts, "MAX_SOURCE_ARTIFACT_STORAGE_BYTES", 10)
    monkeypatch.setattr(
        source_artifacts,
        "MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT",
        8,
    )
    with pytest.raises(SourceArtifactContractError, match="storage aggregate"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(
                _artifact("one", payload),
                _artifact("two", payload),
            ),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({"one": payload, "two": payload}),
        )

    with pytest.raises(SourceArtifactContractError, match="storage aggregate"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(
                _artifact("one", payload, declared_size_bytes=None),
                _artifact("two", payload, declared_size_bytes=None),
            ),
            store=LocalDocumentStore(str(tmp_path)),
            open_artifact=_opener({"one": payload, "two": payload}),
        )
