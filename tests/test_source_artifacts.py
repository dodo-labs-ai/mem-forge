from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from io import BytesIO

import pytest
from PIL import Image

from memforge import source_artifacts
from memforge.source_artifacts import (
    RawSourceArtifact,
    SourceArtifactContractError,
    SourceArtifactDownload,
    SourceArtifactSummary,
    source_artifact_inference_eligibility,
    source_artifact_revision_from_metadata,
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


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (1, 1), color=(10, 20, 30)).save(
        output,
        format="PNG",
    )
    return output.getvalue()


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


def _artifact_metadata(
    *,
    size_bytes: object = 4,
    **inference_fields: object,
) -> dict[str, object]:
    return {
        "source_artifact": {
            "artifact_id": "artifact-1",
            "parent_observation_id": "obs-parent",
            "provider_revision": "1",
            "filename": "diagram.png",
            "media_type": "image/png",
            "size_bytes": size_bytes,
            "sha256": "a" * 64,
            "uri": "artifact://diagram.png",
            **inference_fields,
        }
    }


@pytest.mark.parametrize(
    ("size_bytes", "inference_fields", "expected"),
    (
        (4, {}, (True, None)),
        (source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES, {}, (True, None)),
        (
            source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES + 1,
            {},
            (False, "inference_byte_limit"),
        ),
        (4, {"inference_eligible": True}, (True, None)),
        (
            4,
            {
                "inference_eligible": False,
                "inference_ineligible_reason": "invalid_image_structure",
            },
            (False, "invalid_image_structure"),
        ),
        (
            4,
            {
                "inference_eligible": False,
                "inference_ineligible_reason": "inference_byte_limit",
            },
            None,
        ),
        (
            source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES + 1,
            {
                "inference_eligible": False,
                "inference_ineligible_reason": "invalid_image_structure",
            },
            None,
        ),
        (4, {"inference_eligible": False}, None),
        (
            source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES + 1,
            {"inference_eligible": False},
            (False, "inference_byte_limit"),
        ),
        (
            4,
            {
                "inference_eligible": True,
                "inference_ineligible_reason": "invalid_image_structure",
            },
            None,
        ),
        (4, {"inference_eligible": None}, None),
        (4, {"inference_ineligible_reason": None}, None),
        (
            4,
            {"inference_ineligible_reason": "invalid_image_structure"},
            None,
        ),
        (
            4,
            {
                "inference_eligible": False,
                "inference_ineligible_reason": "unknown_reason",
            },
            None,
        ),
        (-1, {}, None),
        ("not-a-size", {}, None),
    ),
)
def test_artifact_inference_metadata_has_one_legacy_current_decision_table(
    size_bytes: object,
    inference_fields: dict[str, object],
    expected: tuple[bool, str | None] | None,
) -> None:
    metadata = _artifact_metadata(
        size_bytes=size_bytes,
        **inference_fields,
    )

    revision = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata=metadata,
    )

    assert source_artifact_inference_eligibility(metadata) == (
        expected[0] if expected is not None else None
    )
    assert (
        (revision.inference_eligible, revision.inference_ineligible_reason)
        if revision is not None
        else None
    ) == expected


def test_artifact_revision_summary_is_revision_pinned_and_legacy_optional() -> None:
    metadata = {
        "source_artifact": {
            "artifact_id": "artifact-1",
            "parent_observation_id": "obs-parent",
            "provider_revision": "1",
            "filename": "diagram.png",
            "media_type": "image/png",
            "size_bytes": 4,
            "sha256": "a" * 64,
            "uri": "artifact://diagram.png",
            "inference_eligible": True,
            "summary": "Architecture diagram showing the request flow.",
        }
    }

    revision = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata=metadata,
    )
    legacy = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata={
            "source_artifact": {
                key: value
                for key, value in metadata["source_artifact"].items()
                if key != "summary"
            }
        },
    )

    assert revision is not None
    assert revision.summary == "Architecture diagram showing the request flow."
    assert revision.metadata()["summary"] == revision.summary
    assert legacy is not None
    assert legacy.summary is None
    assert SourceArtifactSummary(" obs-image ", "  Visible flow   overview. ") == (
        SourceArtifactSummary("obs-image", "Visible flow overview.")
    )


def test_artifact_revision_parses_legacy_inference_metadata_conservatively() -> None:
    revision = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata={
            "source_artifact": {
                "artifact_id": "artifact-1",
                "parent_observation_id": "obs-parent",
                "provider_revision": "1",
                "filename": "diagram.png",
                "media_type": "image/png",
                "size_bytes": 4,
                "sha256": "a" * 64,
                "uri": "artifact://diagram.png",
            }
        },
    )

    assert revision is not None
    assert revision.inference_eligible is True
    assert revision.inference_ineligible_reason is None

    unexplained_ineligible = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata={
            "source_artifact": {
                "artifact_id": "artifact-1",
                "parent_observation_id": "obs-parent",
                "provider_revision": "1",
                "filename": "diagram.png",
                "media_type": "image/png",
                "size_bytes": 4,
                "sha256": "a" * 64,
                "uri": "artifact://diagram.png",
                "inference_eligible": False,
            }
        },
    )

    assert unexplained_ineligible is None

    byte_limited = source_artifact_revision_from_metadata(
        observation_id="obs-image",
        observation_revision_id="obsrev-image",
        source_id="src-1",
        source_unit_id="unit-1",
        metadata={
            "source_artifact": {
                "artifact_id": "artifact-1",
                "parent_observation_id": "obs-parent",
                "provider_revision": "1",
                "filename": "diagram.png",
                "media_type": "image/png",
                "size_bytes": source_artifacts.MAX_SOURCE_ARTIFACT_INFERENCE_BYTES + 1,
                "sha256": "a" * 64,
                "uri": "artifact://diagram.png",
                "inference_eligible": False,
            }
        },
    )

    assert byte_limited is not None
    assert byte_limited.inference_eligible is False
    assert byte_limited.inference_ineligible_reason == "inference_byte_limit"


@pytest.mark.asyncio
async def test_materialization_streams_exact_bytes_and_identity(tmp_path) -> None:
    store = LocalDocumentStore(str(tmp_path))
    payload = _png_bytes()

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

    replacement_output = BytesIO()
    Image.new("RGB", (1, 1), color=(30, 20, 10)).save(
        replacement_output,
        format="PNG",
    )
    replacement = replacement_output.getvalue()
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
async def test_materialization_does_not_read_stream_after_store_takes_ownership() -> None:
    class ClosingStore:
        def __init__(self) -> None:
            self.body: bytes | None = None

        def store_source_artifact(self, **kwargs) -> str:
            content = kwargs["content"]
            self.body = content.read()
            content.close()
            return "artifact://stored"

    payload = _png_bytes()
    store = ClosingStore()

    (artifact,) = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=(_artifact("image", payload),),
        store=store,
        open_artifact=_opener({"image": payload}),
    )

    assert store.body == payload
    assert artifact.inference_eligible is True
    assert artifact.uri == "artifact://stored"


@pytest.mark.asyncio
async def test_invalid_image_is_stored_but_excluded_from_inference(tmp_path) -> None:
    payload = b"\x89PNG\r\n\x1a\nnot-a-decodable-image"

    (artifact,) = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=(_artifact("invalid-image", payload),),
        store=LocalDocumentStore(str(tmp_path)),
        open_artifact=_opener({"invalid-image": payload}),
    )

    assert artifact.inference_eligible is False
    assert artifact.inference_ineligible_reason == "invalid_image_structure"
    assert LocalDocumentStore(str(tmp_path)).read_artifact(artifact.uri) == payload


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
async def test_materialization_persists_each_exact_artifact_before_next_download(
    tmp_path,
) -> None:
    class RecordingStore(LocalDocumentStore):
        def __init__(self, root: str) -> None:
            super().__init__(root)
            self.uris: list[str] = []

        def store_source_artifact(self, **kwargs) -> str:
            uri = super().store_source_artifact(**kwargs)
            self.uris.append(uri)
            return uri

    payloads = {"one": _png_bytes(), "two": _png_bytes()}
    base_opener = _opener(payloads)

    @asynccontextmanager
    async def fail_second(artifact: RawSourceArtifact):
        if artifact.provider_key == "two":
            raise RuntimeError("provider interrupted")
        async with base_opener(artifact) as download:
            yield download

    store = RecordingStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="provider interrupted"):
        await materialize_source_artifacts(
            source_id="source-a",
            source_unit_key="page-1",
            artifacts=(
                _artifact("one", payloads["one"]),
                _artifact("two", payloads["two"]),
            ),
            store=store,
            open_artifact=fail_second,
        )

    assert len(store.uris) == 1
    assert store.read_artifact(store.uris[0]) == payloads["one"]


@pytest.mark.asyncio
async def test_materialization_bounds_bytes_without_rejecting_source_cardinality(
    tmp_path,
    monkeypatch,
) -> None:
    payload = b"12345"
    many_payloads = {
        f"attachment-{index}": payload
        for index in range(125)
    }
    many = await materialize_source_artifacts(
        source_id="source-a",
        source_unit_key="page-1",
        artifacts=tuple(
            _artifact(provider_key, body)
            for provider_key, body in many_payloads.items()
        ),
        store=LocalDocumentStore(str(tmp_path)),
        open_artifact=_opener(many_payloads),
    )
    assert len(many) == 125

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
