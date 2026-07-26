"""Portable image preparation for structured multimodal LLM calls.

Stored Source Artifact bytes remain immutable evidence. This module derives a
bounded transport representation only when the original image is not portable
across supported multimodal endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


MAX_STRUCTURED_LLM_IMAGE_DIMENSION = 2000
MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES = 3_750_000
MAX_STRUCTURED_LLM_IMAGE_PIXELS = 64_000_000

_SUPPORTED_MEDIA_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)
_FORMAT_MEDIA_TYPES = {
    "GIF": "image/gif",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


@dataclass(frozen=True, slots=True)
class StructuredLlmImage:
    """One revision-pinned image supplied to a structured logical call."""

    source_observation_id: str
    media_type: str
    body: bytes

    def __post_init__(self) -> None:
        if not self.source_observation_id.strip():
            raise ValueError("image source_observation_id is required")
        if self.media_type not in _SUPPORTED_MEDIA_TYPES:
            raise ValueError(f"unsupported structured LLM image type: {self.media_type}")
        if not self.body:
            raise ValueError("structured LLM image body is required")


class StructuredLlmImageError(ValueError):
    """Safe failure raised before invalid image evidence reaches a provider."""

    def __init__(self, *, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class PreparedStructuredLlmImages:
    """One logical call's reusable image transport representation."""

    images: tuple[StructuredLlmImage, ...]
    original_bytes: int
    transport_bytes: int
    normalized_count: int


def prepare_structured_llm_images(
    images: tuple[StructuredLlmImage, ...],
) -> PreparedStructuredLlmImages:
    """Validate images and normalize only those outside the portable envelope."""

    prepared: list[StructuredLlmImage] = []
    normalized_count = 0
    for image in images:
        prepared_image, normalized = _prepare_image(image)
        prepared.append(prepared_image)
        normalized_count += int(normalized)
    return PreparedStructuredLlmImages(
        images=tuple(prepared),
        original_bytes=sum(len(image.body) for image in images),
        transport_bytes=sum(len(image.body) for image in prepared),
        normalized_count=normalized_count,
    )


def _prepare_image(image: StructuredLlmImage) -> tuple[StructuredLlmImage, bool]:
    try:
        with Image.open(BytesIO(image.body)) as verifier:
            verifier.verify()
        with Image.open(BytesIO(image.body)) as opened:
            width, height = opened.size
            actual_media_type = _FORMAT_MEDIA_TYPES.get(opened.format or "")
            orientation = opened.getexif().get(274)
            animated = bool(getattr(opened, "is_animated", False))
            if (
                actual_media_type is None
                or width <= 0
                or height <= 0
                or width * height > MAX_STRUCTURED_LLM_IMAGE_PIXELS
            ):
                raise StructuredLlmImageError(error_code="invalid_image_evidence")
    except StructuredLlmImageError:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        raise StructuredLlmImageError(error_code="invalid_image_evidence") from exc

    portable = (
        actual_media_type == image.media_type
        and not animated
        and orientation in (None, 1)
        and max(width, height) <= MAX_STRUCTURED_LLM_IMAGE_DIMENSION
        and len(image.body) <= MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES
    )
    if portable:
        return image, False

    return (
        StructuredLlmImage(
            source_observation_id=image.source_observation_id,
            media_type="image/jpeg",
            body=_normalize_to_jpeg(image.body),
        ),
        True,
    )


def _normalize_to_jpeg(body: bytes) -> bytes:
    try:
        with Image.open(BytesIO(body)) as opened:
            if opened.format == "JPEG":
                opened.draft(
                    "RGB",
                    (
                        MAX_STRUCTURED_LLM_IMAGE_DIMENSION,
                        MAX_STRUCTURED_LLM_IMAGE_DIMENSION,
                    ),
                )
            opened.seek(0)
            frame = ImageOps.exif_transpose(opened)
            try:
                frame.thumbnail(
                    (
                        MAX_STRUCTURED_LLM_IMAGE_DIMENSION,
                        MAX_STRUCTURED_LLM_IMAGE_DIMENSION,
                    ),
                    Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )
                rgb = _flatten_to_rgb(frame)
                try:
                    encoded = _encode_jpeg(rgb, quality=90)
                    if len(encoded) > MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES:
                        encoded = _encode_jpeg(rgb, quality=80)
                finally:
                    rgb.close()
            finally:
                if frame is not opened:
                    frame.close()
    except StructuredLlmImageError:
        raise
    except (Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError) as exc:
        raise StructuredLlmImageError(error_code="invalid_image_evidence") from exc

    if len(encoded) > MAX_STRUCTURED_LLM_IMAGE_TRANSPORT_BYTES:
        raise StructuredLlmImageError(error_code="image_evidence_too_large")
    return encoded


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        try:
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        finally:
            rgba.close()
    return image.convert("RGB")


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()
