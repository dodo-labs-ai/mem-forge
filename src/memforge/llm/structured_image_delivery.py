"""One verified Artifact with complete, ordered transport reading views.

The original remains Evidence. The fixed experimental capacity profile is
documented only; it does not attest managed-route resolution or capacity.
"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from typing import TYPE_CHECKING

from PIL import Image, __version__ as pillow_version

from memforge.raster_views import complete_rectangles, encode_region, open_upright_png, pixel_digest
from memforge.llm.structured_images import StructuredLlmImage, StructuredLlmImageError, prepare_structured_llm_images

if TYPE_CHECKING:
    from memforge.pipeline.projection_fragments import ProjectionFragmentCatalog

PROFILE = "whole-complete-overlap-png-documented-standard-v2"
DOCUMENTED_BOUNDS = dict(max_edge=1568, max_visual_tokens=1568, patch_edge=28)
_BINDING_FIELDS = {"source_id", "source_unit_id", "source_unit_revision_id", "observation_id",
                   "observation_revision_id", "artifact_ref", "access_context_hash", "authority_hash", "source_sha256",
                   "catalog_digest"}


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class DeliveryView:
    kind: str
    rectangle: tuple[int, int, int, int]
    source_pixel_sha256: str
    media_type: str
    body: bytes
    dimensions: tuple[int, int]

    def identity(self) -> dict:
        return {"kind": self.kind, "rectangle": self.rectangle, "source_pixel_sha256": self.source_pixel_sha256,
                "media_type": self.media_type, "wire_sha256": sha256(self.body).hexdigest(),
                "dimensions": self.dimensions}


@dataclass(frozen=True, slots=True)
class ArtifactImageDelivery:
    original: StructuredLlmImage
    source_binding: tuple[tuple[str, str], ...]
    model: str
    budget_identity: str
    views: tuple[DeliveryView, ...]
    manifest_sha256: str
    profile: str = PROFILE
    max_images: int = 20
    max_encoded_image_bytes: int = 5_000_000
    max_encoded_request_bytes: int = 32_000_000

    @classmethod
    def build(cls, original: StructuredLlmImage, *, catalog: "ProjectionFragmentCatalog", artifact_ref: str,
              authority_hash: str,
              model: str, budget_identity: str, max_images: int = 20,
              max_encoded_request_bytes: int = 32_000_000) -> "ArtifactImageDelivery":
        fragments = [f for f in catalog.fragments if f.reference == artifact_ref and f.kind.value == "artifact"]
        if len(fragments) != 1 or not catalog.usable:
            raise StructuredLlmImageError(error_code="image_delivery_catalog_binding_invalid")
        fragment = fragments[0]
        metadata = catalog.artifact_metadata_by_revision_id.get(fragment.anchor.observation_revision_id, {})
        if (metadata.get("inference_eligible") is not True or metadata.get("media_type") != original.media_type
                or metadata.get("size_bytes") != len(original.body)):
            raise StructuredLlmImageError(error_code="image_delivery_catalog_binding_invalid")
        binding = tuple(sorted({"source_id": catalog.source_id, "source_unit_id": catalog.source_unit_id,
                                "source_unit_revision_id": catalog.target_unit_revision_id,
                                "observation_id": fragment.anchor.observation_id,
                                "observation_revision_id": fragment.anchor.observation_revision_id,
                                "artifact_ref": artifact_ref, "access_context_hash": catalog.access_context_hash,
                                "authority_hash": authority_hash, "source_sha256": fragment.raw_content_sha256,
                                "catalog_digest": catalog.digest}.items()))
        provisional = cls(original, binding, model, budget_identity, (), "", max_images=max_images,
                          max_encoded_request_bytes=max_encoded_request_bytes)
        provisional._validate_binding()
        views = provisional._derive_views()
        result = cls(original, binding, model, budget_identity, views, "", max_images=max_images,
                     max_encoded_request_bytes=max_encoded_request_bytes)
        from dataclasses import replace
        result = replace(result, manifest_sha256=digest(result.identity()))
        result.validate()
        return result

    def _validate_binding(self) -> None:
        binding = dict(self.source_binding)
        if (len(binding) != len(self.source_binding) or set(binding) != _BINDING_FIELDS
                or not all(isinstance(v, str) and v.strip() for v in binding.values())
                or binding["observation_id"] != self.original.source_observation_id
                or binding["source_sha256"] != sha256(self.original.body).hexdigest()
                or self.original.media_type != "image/png" or not self.model or not self.budget_identity
                or self.profile != PROFILE or not 1 <= self.max_images <= 20
                or self.max_encoded_image_bytes != 5_000_000
                or not 1 <= self.max_encoded_request_bytes <= 32_000_000):
            raise StructuredLlmImageError(error_code="image_delivery_binding_invalid")

    def _derive_views(self) -> tuple[DeliveryView, ...]:
        try:
            with open_upright_png(self.original.body) as image:
                whole = prepare_structured_llm_images((self.original,)).images[0]
                with Image.open(BytesIO(whole.body)) as wire:
                    dimensions = wire.size
                views = [DeliveryView("whole", (0, 0, image.width, image.height), pixel_digest(image),
                                      whole.media_type, whole.body, dimensions)]
                rectangles = complete_rectangles(image.width, image.height, self.max_images - 1,
                                                 **DOCUMENTED_BOUNDS, overlap_fraction=0.2)
                if not rectangles and whole.body != self.original.body:
                    if self.max_images < 2:
                        raise ValueError("complete_raster_views_exceed_image_count")
                    rectangles = ((0, 0, image.width, image.height),)
                for rectangle in rectangles:
                    with image.crop(rectangle) as region:
                        views.append(DeliveryView("detail", rectangle, pixel_digest(region), "image/png",
                                                  encode_region(image, rectangle), region.size))
                return tuple(views)
        except StructuredLlmImageError:
            raise
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            raise StructuredLlmImageError(error_code="image_delivery_profile_unavailable") from exc

    def identity(self) -> dict:
        return {"original_sha256": sha256(self.original.body).hexdigest(), "source_binding": self.source_binding,
                "profile": self.profile, "pillow_version": pillow_version, "model": self.model,
                "budget_identity": self.budget_identity, "views": [v.identity() for v in self.views],
                "max_images": self.max_images, "max_encoded_image_bytes": self.max_encoded_image_bytes,
                "max_encoded_request_bytes": self.max_encoded_request_bytes,
                "documented_bounds": dict(DOCUMENTED_BOUNDS), "overlap_fraction": 0.2,
                "renderer_contract": "parent-artifact-rectangle-reading-view-v2",
                "instruction_order": "views_before_query",
                "capacity_evidence": "documented_only_route_unverified"}

    def validate(self, *, model: str | None = None, budget_identity: str | None = None) -> None:
        self._validate_binding()
        if ((model is not None and model != self.model)
                or (budget_identity is not None and budget_identity != self.budget_identity)
                or digest(self.identity()) != self.manifest_sha256 or self.views != self._derive_views()
                or any(4 * ((len(v.body) + 2) // 3) > self.max_encoded_image_bytes for v in self.views)):
            raise StructuredLlmImageError(error_code="image_delivery_integrity_failed")

    @property
    def visual_token_upper_bound(self) -> int:
        # Each declared standard-tier view is capped at 1568 visual tokens.
        # This conservative documented bound is not managed-route verification.
        return 1568 * len(self.views)

    @property
    def original_bytes(self) -> int:
        return len(self.original.body)

    @property
    def transport_bytes(self) -> int:
        return sum(len(v.body) for v in self.views)

    @property
    def normalized_count(self) -> int:
        return int(self.views[0].body != self.original.body)

    @property
    def images(self) -> "ArtifactImageDelivery":
        return self

    def __len__(self) -> int:
        return len(self.views)
