"""Complete revision context shared by extraction and fixed-claim Support assessment.

Indexes are operation-local. Full and delta inputs share the same current
fragment identities; old coordinates never select a new revision's Evidence.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from typing import Literal

from memforge.llm.batch_runner import LlmRequest, RequestTooLarge
from memforge.models import RawMemory
from memforge.pipeline.evidence_fragments import (
    EvidenceFragment,
    EvidenceFragmentKind,
    RevisionFragmentIndex,
    build_revision_fragment_index,
    canonical_record_field_ranges,
    canonical_record_is_tombstoned,
    revision_changed_structural_ranges,
    revision_structural_ranges,
    _revision_structural_identities,
)
from memforge.pipeline.projection_context import observation_is_inference_eligible
from memforge.pipeline.projection_images import ProjectionImageLoadError
from memforge.pipeline.projection_fragments import (
    ProjectionFragmentCatalog,
    SupportRevalidationLimitation,
    SupportRevalidationLimitationCode,
    _compose_projection_fragment_catalog,
)
from memforge.source_projection import SourceObservationRevision, SourceProjection

# Versions how a fixed Support is revalidated against a revision: it enters the
# reconciliation manifest and each revalidated Support's ``support_validation``.
REVISION_SUPPORT_CONTRACT = "revision-support-v4"
# Versions how revision Fragments are compiled into catalogs and how Claim
# Extraction chooses its reading scope. Every catalog this context composes,
# for extraction or for Support, carries it in its identity.
REVISION_INPUT_POLICY = "revision-input-v6"


def revision_inference_capability_hash(client, *, extraction_model=None, extraction_max_tokens=None) -> str:
    from memforge.pipeline.projection_images import projection_inference_capability_hash

    payload = {
        "images": projection_inference_capability_hash(),
        "revision_input_policy": REVISION_INPUT_POLICY,
        "revision_input": client.input_policy_identity_for(extraction_model) if client is not None else None,
        "extraction_model": extraction_model,
        "extraction_max_tokens": extraction_max_tokens,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SupportAssessment:
    # None preserves the existing claim without certifying current support.
    supported: bool | None
    reason: str
    memory: RawMemory | None
    # Why a kept claim could not be judged: an UNKNOWN Evidence part under partial
    # coverage, or one ReadingGroup that alone exceeds the model's capacity.
    unresolved: Literal["partial_coverage", "capacity"] | None = None


def _changed_ranges(base: SourceObservationRevision, target: SourceObservationRevision):
    if base.id == target.id:
        return ()
    profile = target.evidence_profile
    if base.evidence_profile == profile and profile is not None:
        if profile.name in {"markdown-structural", "plain-text"}:
            return revision_changed_structural_ranges(base, target)
        if profile.name == "canonical-record":
            previous = {
                field.descriptor.json_pointer: field.comparison_value for field in canonical_record_field_ranges(base)
            }
            return tuple(
                (field.start, field.end)
                for field in canonical_record_field_ranges(target)
                if field.descriptor.json_pointer not in previous
                or previous[field.descriptor.json_pointer] != field.comparison_value
            )
    return None


def _in_ranges(fragment: EvidenceFragment, ranges) -> bool:
    if ranges is None:
        return True
    start, end = fragment.anchor.range_start, fragment.anchor.range_end
    return start is not None and end is not None and any(start < right and left < end for left, right in ranges)


class RevisionAssessmentContext:
    """Own one Unit's indexes and net delta; consumers own semantic judgments."""

    def __init__(
        self,
        *,
        projection: SourceProjection,
        base: SourceProjection | None,
        access_context_hash: str,
        images: tuple = (),
        image_loader=None,
        indexes: dict | None = None,
        known_observations: tuple = (),
    ):
        """``known_observations`` describe carried Observations that neither the target
        nor the baseline returns, such as those of the committed Source Unit revision."""
        self.projection = projection
        self.base = base
        self.access_context_hash = access_context_hash
        self.images = images
        self.image_loader = image_loader
        self.indexes: dict[str, RevisionFragmentIndex] = indexes if indexes is not None else {}
        self.reading_indexes = {}
        self._images = {}
        self.current = {
            r.observation_id: r
            for r in projection.observation_revisions
            if r.id in projection.source_unit_revisions[0].observation_revision_ids
        }
        observations = {o.id: o for o in (*known_observations, *(base.observations if base else ()))}
        observations.update({o.id: o for o in projection.observations})
        if set(self.current) - set(observations):
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                "current observation eligibility metadata is incomplete",
            )
        self.members = dict(self.current)
        self.tombstoned = {
            key
            for key, revision in self.current.items()
            if revision.evidence_profile
            and revision.evidence_profile.name == "canonical-record"
            and canonical_record_is_tombstoned(revision)
        }
        self.current = {
            key: r
            for key, r in self.current.items()
            if key not in self.tombstoned
            and observation_is_inference_eligible(observations[key].observation_type, r.metadata)
        }
        self.previous = {r.observation_id: r for r in base.observation_revisions} if base else {}
        self.full_fragments = tuple(f for revision in self.current.values() for f in self.index(revision).fragments)
        self._delta = None
        self.structural_context = {}
        self.canonical_fields = {
            revision.id: canonical_record_field_ranges(revision)
            for revision in (*self.previous.values(), *self.current.values())
            if revision.evidence_profile and revision.evidence_profile.name == "canonical-record"
        }
        # Both sides, so removed old text keeps its heading path too.
        for revision in {r.id: r for r in (*self.previous.values(), *self.current.values())}.values():
            if revision.evidence_profile and revision.evidence_profile.name == "markdown-structural":
                units = revision_structural_ranges(revision)
                identities = _revision_structural_identities(revision, units)
                self.structural_context[revision.id] = tuple(
                    (unit.start, unit.end, identity[1]) for unit, identity in zip(units, identities, strict=True)
                )

    def canonical_context(self, fragment):
        """Keep field and event identity on both sides of a canonical delta."""
        anchor = fragment.anchor
        fields = self.canonical_fields.get(anchor.observation_revision_id, ())
        owner = next((item for item in fields
                      if item.start <= (anchor.range_start or 0) < item.end), None)
        if owner is None:
            return {}
        parent = owner.descriptor.json_pointer.rsplit("/", 1)[0]
        return {"field": owner.descriptor.json_pointer, "context": {
            item.descriptor.json_pointer: item.value
            for item in fields if item.descriptor.contextual
            and item.descriptor.json_pointer.rsplit("/", 1)[0] in {"", parent}
        }}

    def images_for(self, catalog):
        """The current bytes of the catalog's Artifacts, read once per set for this operation.

        Split, corrected and repeated requests render the same Artifacts again.
        """
        ids = frozenset(f.anchor.observation_id for f in catalog.fragments if f.kind is EvidenceFragmentKind.ARTIFACT)
        if not ids:
            return ()
        if ids not in self._images:
            images = self.image_loader(set(ids)) if self.image_loader is not None else self.images
            images = tuple(image for image in images if image.source_observation_id in ids)
            if ids != {image.source_observation_id for image in images}:
                raise SupportRevalidationLimitation(
                    SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                    "selected current Artifact bytes were not supplied",
                )
            self._images[ids] = images
        return self._images[ids]

    def attach_images(self, request: LlmRequest, catalog, *, fits) -> LlmRequest:
        """Attach the catalog's Artifact images to a request whose text already fits.

        Text alone can reject a request without reading attachments. An image
        batch over the loader's byte limit means this slice of work is too large.
        """
        if not fits(request):
            raise RequestTooLarge("request text exceeds input capacity")
        try:
            images = self.images_for(catalog)
        except ProjectionImageLoadError as error:
            if error.error_code != "image_batch_too_large":
                raise
            raise RequestTooLarge(error.error_code) from error
        return replace(request, images=images)

    def heading_context(self, fragment) -> tuple[str, ...]:
        """The Markdown heading path a current or baseline Fragment sits under; empty elsewhere."""
        anchor = fragment.anchor
        return tuple(next(
            (headings for start, end, headings in self.structural_context.get(
                anchor.observation_revision_id, ()
            ) if start <= (anchor.range_start or 0) < end), ()
        ))

    def _scope(self, fragment) -> tuple[tuple[str, ...], str | None]:
        """A Fragment's heading path and canonical record field, when its representation has them."""
        anchor = fragment.anchor
        field = next((field.descriptor.json_pointer for field in self.canonical_fields.get(anchor.observation_revision_id, ())
                      if field.start <= (anchor.range_start or 0) and (anchor.range_end or 0) <= field.end), None)
        return self.heading_context(fragment), field

    def fragment_scope(self, fragment) -> dict:
        """Where a Fragment sits, for a reader who sees its text without its neighbours."""
        headings, field = self._scope(fragment)
        return {
            **({"heading_context": list(headings)} if headings else {}),
            **({"field": field} if field is not None else {}),
        }

    def model_payload(self, catalog):
        payload = dict(catalog.model_payload())
        groups: dict[tuple[str, str, tuple[str, ...], str | None], list[str]] = {}
        for fragment in catalog.fragments:
            anchor = fragment.anchor
            headings, field = self._scope(fragment)
            key = (anchor.observation_id, anchor.observation_revision_id, headings, field)
            groups.setdefault(key, []).append(fragment.reference)
        # Ancestor text is supplementary structure. The original heading Fragment
        # remains selectable in its authorized role; groups never create Evidence.
        aliases = {key: f"o{index}" for index, key in enumerate(dict.fromkeys(
            (observation, revision) for observation, revision, _, _ in groups
        ))}
        payload["observations"] = {alias: {"observation_id": observation, "revision_id": revision}
                                   for (observation, revision), alias in aliases.items()}
        payload["structural_groups"] = tuple(
            {"source": aliases[observation, revision], "refs": refs,
             **({"heading_context": headings} if headings else {}),
             **({"field": field} if field is not None else {})}
            for (observation, revision, headings, field), refs in groups.items()
        )
        return payload

    def index(self, revision):
        if revision.id not in self.indexes:
            index = build_revision_fragment_index(revision)
            if any(error.fatal for error in index.errors):
                raise SupportRevalidationLimitation(
                    SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION
                    if any(error.code.value == "unsupported_profile" for error in index.errors)
                    else SupportRevalidationLimitationCode.COMPILER_FAILURE,
                    "revision cannot be compiled completely",
                )
            self.indexes[revision.id] = index
        return self.indexes[revision.id]

    def reading_index(self, revision):
        """Reuse one exact-revision structural reading index for this operation."""
        if revision.id not in self.reading_indexes:
            from memforge.pipeline.revision_reading import build_revision_reading_index

            self.reading_indexes[revision.id] = build_revision_reading_index(
                revision, self.index(revision).fragments
            )
        return self.reading_indexes[revision.id]

    def catalog(self, fragments) -> ProjectionFragmentCatalog:
        return _compose_projection_fragment_catalog(
            projection=self.projection,
            access_context_hash=self.access_context_hash,
            catalog_identity={"input_policy": REVISION_INPUT_POLICY, "purpose": "fixed_claim_support"},
            compiled_fragments=list(fragments),
            errors=[],
            component_digests=[],
            authority_payload=[],
            artifact_metadata={r.id: dict(r.metadata.get("source_artifact", {})) for r in self.current.values()},
            max_fragments=max(1, len(fragments)),
            max_presentation_chars=max(1, sum(len(f.presentation_text) for f in fragments)),
        )

    def delta_fragments(self) -> tuple[tuple[EvidenceFragment, ...], tuple[EvidenceFragment, ...]]:
        """The added or modified current Fragments, and the baseline Fragments this revision removed."""
        if self._delta is not None:
            return self._delta
        if self.base is None:
            raise SupportRevalidationLimitation(
                SupportRevalidationLimitationCode.UNSUPPORTED_REPRESENTATION,
                "delta assessment requires a complete applicable baseline",
            )
        changed, removed = [], []
        for key, current in self.current.items():
            old = self.previous.get(key)
            ranges = _changed_ranges(old, current) if old else None
            changed.extend(f for f in self.index(current).fragments if _in_ranges(f, ranges))
        for key, old in self.previous.items():
            current = self.current.get(key)
            if key in self.members and current is None and key not in self.tombstoned:
                continue
            ranges = _changed_ranges(current, old) if current else None
            removed.extend(f for f in self.index(old).fragments if _in_ranges(f, ranges))
        self._delta = tuple(changed), tuple(removed)
        return self._delta

    def removed_entry(self, fragment) -> dict:
        """One removed baseline Fragment's old text, with its canonical record field and event identity."""
        anchor = fragment.anchor
        return {
            "observation_id": anchor.observation_id,
            "revision_id": anchor.observation_revision_id,
            "text": fragment.presentation_text,
            **self.canonical_context(fragment),
        }

    def delta(self):
        """The changed current Fragments and the removed old text, as Claim Extraction reads them."""
        changed, removed = self.delta_fragments()
        return changed, [self.removed_entry(fragment) for fragment in removed]
