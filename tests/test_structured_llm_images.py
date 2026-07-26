from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from memforge.llm.structured_images import (
    MAX_STRUCTURED_LLM_IMAGE_DIMENSION,
    MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES,
    StructuredLlmImage,
    StructuredLlmImageError,
    prepare_structured_llm_images,
)


def _image_bytes(
    *,
    image_format: str,
    width: int,
    height: int,
    mode: str = "RGB",
) -> bytes:
    output = BytesIO()
    color = (20, 40, 60, 128) if mode == "RGBA" else (20, 40, 60)
    Image.new(mode, (width, height), color=color).save(output, format=image_format)
    return output.getvalue()


def test_prepare_structured_llm_images_preserves_valid_portable_image() -> None:
    body = _image_bytes(image_format="PNG", width=32, height=24)

    prepared = prepare_structured_llm_images(
        (
            StructuredLlmImage(
                source_observation_id="obs-image-1",
                media_type="image/png",
                body=body,
            ),
        )
    )

    assert prepared.images == (
        StructuredLlmImage(
            source_observation_id="obs-image-1",
            media_type="image/png",
            body=body,
        ),
    )
    assert prepared.original_bytes == len(body)
    assert prepared.transport_bytes == len(body)
    assert prepared.normalized_count == 0


def test_prepare_structured_llm_images_normalizes_dimensions_and_transport_bytes() -> None:
    body = _image_bytes(image_format="PNG", width=2400, height=1200, mode="RGBA")

    prepared = prepare_structured_llm_images(
        (
            StructuredLlmImage(
                source_observation_id="obs-image-1",
                media_type="image/png",
                body=body,
            ),
        )
    )

    assert prepared.normalized_count == 1
    assert prepared.images[0].source_observation_id == "obs-image-1"
    assert prepared.images[0].media_type == "image/jpeg"
    assert len(prepared.images[0].body) <= MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES
    with Image.open(BytesIO(prepared.images[0].body)) as image:
        assert image.mode == "RGB"
        assert max(image.size) <= MAX_STRUCTURED_LLM_IMAGE_DIMENSION


def test_prepare_structured_llm_images_rejects_declared_image_with_invalid_bytes() -> None:
    with pytest.raises(StructuredLlmImageError) as raised:
        prepare_structured_llm_images(
            (
                StructuredLlmImage(
                    source_observation_id="obs-image-1",
                    media_type="image/png",
                    body=b"not-an-image",
                ),
            )
        )

    assert raised.value.error_code == "invalid_image_evidence"
