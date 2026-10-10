"""Compose and resolve one authorized projection-extraction Fragment catalog.

The representation compiler remains revision-local.  This module owns the
single-call catalog boundary: it combines current revisions from one Source
Unit, reassigns catalog-local references, and resolves model selections back to
exact application-owned Evidence parts.  It never writes Evidence or lifecycle
state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping

from memforge.derivation_work import payload_hash
from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
    ResolvedEvidencePart,
    ResolvedEvidenceSelection,
)
from memforge.pipeline.evidence_fragments import (
    COMPILER_CONTRACT_VERSION,
    DEFAULT_MAX_FRAGMENTS,
    DEFAULT_MAX_PRESENTATION_CHARS,
    EvidenceCandidateRange,
    EvidenceFragment,
    EvidenceFragmentKind,
    FragmentCompilationError,
    FragmentCompilationErrorCode,
    compile_fragments,
)
from memforge.pipeline.projection_context import ExtractionAuthority
from memforge.pipeline.projection_images import (
    projection_inference_capability_hash,
)
from memforge.source_time import parse_source_time
from memforge.source_projection import (
    AnchorKind,
    EvidenceCoordinateSpace,
    SourceAnchor,
    SourceObservationRevision,
    SourceProjection,
)


class FragmentSelectionErrorCode(str, Enum):
    CATALOG_UNUSABLE = "catalog_unusable"
    UNKNOWN_REF = "unknown_ref"
    DUPLICATE_REF = "duplicate_ref"
    INELIGIBLE_ROLE = "ineligible_role"
    INVALID_SELECTION = "invalid_selection"


class FragmentSelectionError(ValueError):
    """Typed fail-closed selector rejection safe for bounded telemetry."""

    def __init__(self, code: FragmentSelectionErrorCode, message: str, *,
                 location: str | None = None, received: str | None = None,
                 allowed_refs: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.location = location
        self.received = received
        self.allowed_refs = allowed_refs


class SupportRevalidationLimitationCode(str, Enum):
    UNSUPPORTED_REPRESENTATION = "unsupported_representation"
    COMPILER_FAILURE = "compiler_failure"
    EVIDENCE_INTEGRITY = "evidence_integrity"


class SupportRevalidationLimitation(RuntimeError):
    """A product-processing limitation requiring operational attention."""

    retryable = False

    def __init__(
        self,
        code: SupportRevalidationLimitationCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ProjectionFragmentCatalog:
    source_id: str
    source_unit_id: str
    target_unit_revision_id: str
    access_context_hash: str
    fragments: tuple[EvidenceFragment, ...]
    errors: tuple[FragmentCompilationError, ...]
    digest: str
    max_fragments: int
    max_presentation_chars: int
    artifact_metadata_by_revision_id: Mapping[str, Mapping[str, object]]

    @property
    def usable(self) -> bool:
        return bool(self.fragments) and not any(error.fatal for error in self.errors)

    def subset(self, references) -> ProjectionFragmentCatalog:
        """The Fragments with these references, keeping each reference and its authority."""

        selected = frozenset(references)
        return replace(
            self,
            fragments=tuple(fragment for fragment in self.fragments if fragment.reference in selected),
            digest=payload_hash([self.digest, sorted(selected)]),
        )

    def model_payload(self) -> Mapping[str, tuple[tuple[object, ...], ...]]:
        """Present exact text and selectable refs; provenance stays in this catalog."""

        primary: list[tuple[object, ...]] = []
        required: list[tuple[object, ...]] = []
        for fragment in self.fragments:
            row: tuple[object, ...] = (fragment.reference, fragment.presentation_text)
            if (
                (fragment.fragment_type.startswith("html-") and fragment.fragment_type != "html-p")
                or fragment.fragment_type.startswith("canonical-")
            ):
                row += ({"format": fragment.fragment_type},)
            if fragment.kind is EvidenceFragmentKind.ARTIFACT:
                metadata = self.artifact_metadata_by_revision_id.get(fragment.anchor.observation_revision_id, {})
                row += ({"image_source_observation_id": fragment.anchor.observation_id,
                         **{key: metadata[key] for key in ("filename", "parent_observation_id") if key in metadata}},)
            (primary if fragment.primary_eligible else required).append(row)
        return {
            "primary_candidates": tuple(primary),
            "required_only_candidates": tuple(required),
        }

    def selection_fingerprint(
        self,
        *,
        candidate_content_hash: str,
        primary_ref: str,
        required_refs: tuple[str, ...] | list[str] = (),
    ) -> str:
        """Return one content-free identity for a model candidate and selectors."""

        payload = {
            "catalog_digest": self.digest,
            "candidate_content_hash": candidate_content_hash,
            "primary_ref": primary_ref,
            "required_refs": sorted(required_refs),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

    def resolve_selection(
        self,
        *,
        primary_ref: str,
        required_refs: tuple[str, ...] | list[str] = (),
        display_text_by_ref: Mapping[str, str] | None = None,
    ) -> ResolvedEvidenceSelection:
        """Resolve one v9 model selection without guessing or widening."""

        if not self.usable:
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.CATALOG_UNUSABLE,
                "Evidence Fragment catalog is not usable",
            )
        if not isinstance(primary_ref, str) or not primary_ref.strip():
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.INVALID_SELECTION,
                "primary_ref is required",
            )
        normalized_required = tuple(required_refs)
        if any(not isinstance(value, str) or not value.strip() for value in normalized_required):
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.INVALID_SELECTION,
                "required_refs must contain non-empty references",
            )
        if primary_ref in normalized_required or len(set(normalized_required)) != len(normalized_required):
            raise FragmentSelectionError(
                FragmentSelectionErrorCode.DUPLICATE_REF,
                "Primary and Required references must be duplicate-free",
            )

        by_reference = {fragment.reference: fragment for fragment in self.fragments}
        selected: list[tuple[EvidenceRole, EvidenceFragment]] = []
        for index, (role, reference) in enumerate((
            (EvidenceRole.PRIMARY, primary_ref),
            *((EvidenceRole.REQUIRED, value) for value in normalized_required),
        )):
            location = "primary_ref" if index == 0 else f"required_refs[{index - 1}]"
            fragment = by_reference.get(reference)
            if fragment is None:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.UNKNOWN_REF,
                    f"unknown, stale, or cross-catalog Fragment reference: {reference}",
                    location=location, received=reference, allowed_refs=sorted(by_reference),
                )
            if role is EvidenceRole.PRIMARY and not fragment.primary_eligible:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.INELIGIBLE_ROLE,
                    f"Fragment reference is not eligible for {role.value}: {reference}",
                    location=location, received=reference,
                    allowed_refs=sorted(f.reference for f in self.fragments if f.primary_eligible),
                )
            selected.append((role, fragment))

        order = {fragment.reference: index for index, fragment in enumerate(self.fragments)}
        primary = selected[0]
        required = tuple(sorted(selected[1:], key=lambda item: order[item[1].reference]))
        parts = tuple(
            replace(self._resolved_part(role=role, fragment=fragment),
                    display_text=(display_text_by_ref or {}).get(fragment.reference))
            for role, fragment in (primary, *required)
        )
        return ResolvedEvidenceSelection(
            source_id=self.source_id,
            source_unit_id=self.source_unit_id,
            target_unit_revision_id=self.target_unit_revision_id,
            access_context_hash=self.access_context_hash,
            catalog_digest=self.digest,
            compiler_contract_version=COMPILER_CONTRACT_VERSION,
            parts=parts,
        )

    def _resolved_part(
        self,
        *,
        role: EvidenceRole,
        fragment: EvidenceFragment,
    ) -> ResolvedEvidencePart:
        artifact_metadata = (
            self.artifact_metadata_by_revision_id.get(
                fragment.anchor.observation_revision_id,
                {},
            )
            if fragment.kind is EvidenceFragmentKind.ARTIFACT
            else {}
        )
        return ResolvedEvidencePart(
            role=role,
            kind=(
                EvidencePartKind.ARTIFACT
                if fragment.kind is EvidenceFragmentKind.ARTIFACT
                else EvidencePartKind.TEXT
            ),
            anchor=fragment.anchor,
            raw_content_sha256=fragment.raw_content_sha256,
            presentation_sha256=fragment.presentation_sha256,
            excerpt=(
                fragment.presentation_text
                if fragment.kind is EvidenceFragmentKind.TEXT
                else None
            ),
            artifact_metadata=dict(artifact_metadata),
            text_view=fragment.text_view,
        )


@dataclass(frozen=True, slots=True)
class AgentEventSourceRange:
    event_id: str
    role: EvidenceRole
    anchor: SourceAnchor


@dataclass(frozen=True, slots=True)
class AgentEventSourceRangeReceipt:
    """Audit-only mapping from authorized prompt events to projected Markdown."""

    target_unit_revision_id: str
    catalog_digest: str
    claim_anchor: SourceAnchor
    event_ranges: tuple[AgentEventSourceRange, ...]

    def to_payload(self) -> Mapping[str, object]:
        return {
            "target_unit_revision_id": self.target_unit_revision_id,
            "catalog_digest": self.catalog_digest,
            "claim_anchor": _anchor_payload(self.claim_anchor),
            "event_ranges": [
                {
                    "event_id": item.event_id,
                    "role": item.role.value,
                    "anchor": _anchor_payload(item.anchor),
                }
                for item in self.event_ranges
            ],
        }


def compile_projection_fragment_catalog(
    projection: SourceProjection,
    authority: ExtractionAuthority,
    *,
    catalog_id: str,
    access_context_hash: str,
    max_fragments: int = DEFAULT_MAX_FRAGMENTS,
    max_presentation_chars: int = DEFAULT_MAX_PRESENTATION_CHARS,
) -> ProjectionFragmentCatalog:
    """Compile one immutable selection catalog of a Source Unit revision's authorized Observations.

    Each authorized Observation is compiled against its exact Primary ranges.
    """

    if len(projection.source_units) != 1 or len(projection.source_unit_revisions) != 1:
        raise ValueError("projection Fragment catalog requires exactly one Source Unit revision")
    if not access_context_hash:
        raise ValueError("projection Fragment catalog requires an access context hash")
    if max_fragments <= 0 or max_presentation_chars <= 0:
        raise ValueError("projection Fragment catalog limits must be positive")

    unit_revision = projection.source_unit_revisions[0]
    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
        if revision.id in set(unit_revision.observation_revision_ids)
    }
    observation_ids = {observation.id for observation in projection.observations}
    primary_ids = set(authority.ranges_by_observation_id)
    if not primary_ids.issubset(revisions) or not primary_ids.issubset(observation_ids):
        raise ValueError("projection Fragment catalog contains stale Observation identity")

    compiled_fragments: list[EvidenceFragment] = []
    errors: list[FragmentCompilationError] = []
    component_digests: list[str] = []
    authority_payload: list[Mapping[str, object]] = []
    artifact_metadata: dict[str, Mapping[str, object]] = {}

    for observation_id in sorted(primary_ids, key=lambda value: revisions[value].id):
        revision = revisions[observation_id]
        spans = authority.ranges_by_observation_id[observation_id]
        whole_artifact = (
            revision.evidence_profile is not None
            and revision.evidence_profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT
        )
        if whole_artifact:
            raw_artifact = revision.metadata.get("source_artifact")
            artifact_metadata[revision.id] = (
                dict(raw_artifact) if isinstance(raw_artifact, Mapping) else {}
            )
        ranges = (
            (_whole_range(revision, primary_eligible=True),)
            if spans is None or whole_artifact or revision.evidence_profile is None
            else tuple(
                _text_range(revision, start, end, primary_eligible=True)
                for start, end in _merged_spans(spans)
            )
        )

        authority_payload.extend(_authority_payload(revision, ranges))
        compiled = compile_fragments(
            revision,
            ranges,
            max_fragments=max_fragments,
            max_presentation_chars=max_presentation_chars,
        )
        component_digests.append(compiled.digest)
        compiled_fragments.extend(compiled.fragments)
        errors.extend(compiled.errors)

    return _compose_projection_fragment_catalog(
        projection=projection,
        access_context_hash=access_context_hash,
        catalog_identity={
            "catalog_id": catalog_id,
            "inference_capability_hash": projection_inference_capability_hash(),
        },
        compiled_fragments=compiled_fragments,
        errors=errors,
        component_digests=component_digests,
        authority_payload=authority_payload,
        artifact_metadata=artifact_metadata,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )


def _compose_projection_fragment_catalog(
    *,
    projection: SourceProjection,
    access_context_hash: str,
    catalog_identity: Mapping[str, object],
    compiled_fragments: list[EvidenceFragment],
    errors: list[FragmentCompilationError],
    component_digests: list[str],
    authority_payload: list[Mapping[str, object]],
    artifact_metadata: Mapping[str, Mapping[str, object]],
    max_fragments: int,
    max_presentation_chars: int,
) -> ProjectionFragmentCatalog:
    """Compose revision-local compiler outputs behind one catalog interface."""

    source_times = {revision.id: _source_time(revision) for revision in projection.observation_revisions}
    ordered = tuple(
        sorted(
            compiled_fragments,
            key=lambda fragment: _fragment_sort_key(
                fragment, source_times.get(fragment.anchor.observation_revision_id)
            ),
        )
    )
    presentation_chars = sum(
        len(fragment.presentation_text) for fragment in ordered
    )
    if len(ordered) > max_fragments or presentation_chars > max_presentation_chars:
        representative = next(iter(projection.observation_revisions))
        errors.append(
            _fatal_error(
                representative,
                FragmentCompilationErrorCode.CATALOG_TOO_LARGE,
                "composed Fragment catalog exceeds its explicit limits",
            )
        )
    exposed = () if any(error.fatal for error in errors) else ordered
    fragments = tuple(
        replace(
            fragment,
            reference=(
                f"p{index:06d}"
                if fragment.primary_eligible
                else f"r{index:06d}"
            ),
        )
        for index, fragment in enumerate(exposed, start=1)
    )
    digest = _catalog_digest(
        projection=projection,
        catalog_identity=catalog_identity,
        access_context_hash=access_context_hash,
        authority_payload=authority_payload,
        component_digests=component_digests,
        ordered_fragments=ordered,
        errors=errors,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )
    source_unit = projection.source_units[0]
    unit_revision = projection.source_unit_revisions[0]
    return ProjectionFragmentCatalog(
        source_id=projection.source_id,
        source_unit_id=source_unit.id,
        target_unit_revision_id=unit_revision.id,
        access_context_hash=access_context_hash,
        fragments=fragments,
        errors=tuple(errors),
        digest=digest,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
        artifact_metadata_by_revision_id=dict(artifact_metadata),
    )


def resolve_projected_agent_claim_fragment(
    projection: SourceProjection,
    *,
    claim_text: str,
    access_context_hash: str,
    primary_event_id: str | None = None,
    required_event_ids: tuple[str, ...] = (),
) -> tuple[ResolvedEvidenceSelection, AgentEventSourceRangeReceipt]:
    """Resolve one deterministic managed-agent claim from its owned Markdown.

    Event ids authorize the upstream command only.  The returned Evidence is
    the exact projected Markdown Fragment; the receipt is audit provenance and
    cannot be supplied to lifecycle mutations as an Evidence identity.
    """

    claim = claim_text.strip()
    if not claim:
        raise ValueError("projected agent claim must not be blank")
    normalized_required = tuple(value.strip() for value in required_event_ids)
    if any(not value for value in normalized_required):
        raise ValueError("Required agent event ids must not be blank")
    if len(set(normalized_required)) != len(normalized_required):
        raise ValueError("Required agent event ids must be duplicate-free")
    if primary_event_id is not None:
        primary_event_id = primary_event_id.strip() or None
    if primary_event_id in normalized_required:
        raise ValueError("Primary agent event cannot also be Required")

    matches: list[tuple[SourceObservationRevision, int, int]] = []
    for revision in projection.observation_revisions:
        cursor = 0
        while True:
            start = revision.content.find(claim, cursor)
            if start < 0:
                break
            matches.append((revision, start, start + len(claim)))
            cursor = start + 1
    if len(matches) != 1:
        raise ValueError("projected agent claim must map to one unique current Markdown range")
    revision, claim_start, claim_end = matches[0]
    catalog = compile_projection_fragment_catalog(
        projection,
        ExtractionAuthority({revision.observation_id: None}),
        catalog_id=(
            "agent-fragment-"
            + hashlib.sha256(
                "\x1f".join(
                    (
                        projection.source_id,
                        projection.source_unit_revisions[0].id,
                        revision.id,
                        str(claim_start),
                        str(claim_end),
                    )
                ).encode("utf-8")
            ).hexdigest()[:20]
        ),
        access_context_hash=access_context_hash,
    )
    if not catalog.usable:
        raise ValueError("projected agent claim Fragment catalog is unusable")
    candidates = [
        fragment
        for fragment in catalog.fragments
        if fragment.kind is EvidenceFragmentKind.TEXT
        and fragment.primary_eligible
        and fragment.anchor.observation_revision_id == revision.id
        and fragment.anchor.range_start is not None
        and fragment.anchor.range_end is not None
        and fragment.anchor.range_start <= claim_start
        and claim_end <= fragment.anchor.range_end
    ]
    if len(candidates) == 1:
        selected_fragments = (candidates[0],)
        claim_anchor = candidates[0].anchor
    elif candidates:
        raise ValueError("projected agent claim maps to ambiguous enclosing Fragments")
    else:
        selected_fragments = _claim_covering_fragments(
            catalog.fragments,
            revision=revision,
            claim_start=claim_start,
            claim_end=claim_end,
        )
        claim_anchor = SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=claim_start,
            range_end=claim_end,
        )
    selection = catalog.resolve_selection(
        primary_ref=selected_fragments[0].reference,
        required_refs=tuple(
            fragment.reference for fragment in selected_fragments[1:]
        ),
    )
    selected_events: list[tuple[EvidenceRole, str]] = []
    if primary_event_id is not None:
        selected_events.append((EvidenceRole.PRIMARY, primary_event_id))
    selected_events.extend(
        (EvidenceRole.REQUIRED, value) for value in normalized_required
    )
    event_ranges = tuple(
        AgentEventSourceRange(
            event_id=event_id,
            role=role,
            anchor=claim_anchor,
        )
        for role, event_id in selected_events
    )
    return selection, AgentEventSourceRangeReceipt(
        target_unit_revision_id=catalog.target_unit_revision_id,
        catalog_digest=catalog.digest,
        claim_anchor=claim_anchor,
        event_ranges=event_ranges,
    )


def _claim_covering_fragments(
    fragments: tuple[EvidenceFragment, ...],
    *,
    revision: SourceObservationRevision,
    claim_start: int,
    claim_end: int,
) -> tuple[EvidenceFragment, ...]:
    """Return the minimal ordered structural set covering one exact claim."""

    selected = tuple(
        sorted(
            (
                fragment
                for fragment in fragments
                if fragment.kind is EvidenceFragmentKind.TEXT
                and fragment.anchor.observation_revision_id == revision.id
                and fragment.anchor.range_start is not None
                and fragment.anchor.range_end is not None
                and fragment.anchor.range_start < claim_end
                and claim_start < fragment.anchor.range_end
            ),
            key=lambda fragment: (
                fragment.anchor.range_start,
                fragment.anchor.range_end,
                fragment.reference,
            ),
        )
    )
    if not selected or not selected[0].primary_eligible:
        raise ValueError("projected agent claim has no Primary Fragment coverage")

    cursor = claim_start
    previous_fragment_end: int | None = None
    for fragment in selected:
        fragment_start = fragment.anchor.range_start
        fragment_end = fragment.anchor.range_end
        assert fragment_start is not None and fragment_end is not None
        if (
            previous_fragment_end is not None
            and fragment_start < previous_fragment_end
        ):
            raise ValueError("projected agent claim Fragment coverage is ambiguous")
        previous_fragment_end = fragment_end
        start = max(claim_start, fragment_start)
        end = min(claim_end, fragment_end)
        if start > cursor and revision.content[cursor:start].strip():
            raise ValueError("projected agent claim Fragment coverage has a content gap")
        cursor = max(cursor, end)
    if cursor < claim_end and revision.content[cursor:claim_end].strip():
        raise ValueError("projected agent claim Fragment coverage is incomplete")
    return selected


def _whole_range(
    revision: SourceObservationRevision,
    *,
    primary_eligible: bool,
) -> EvidenceCandidateRange:
    return EvidenceCandidateRange(
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
        ),
        primary_eligible=primary_eligible,
    )


def _text_range(
    revision: SourceObservationRevision,
    start: int,
    end: int,
    *,
    primary_eligible: bool,
) -> EvidenceCandidateRange:
    if start == 0 and end == len(revision.content):
        return _whole_range(revision, primary_eligible=primary_eligible)
    return EvidenceCandidateRange(
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=start,
            range_end=end,
        ),
        primary_eligible=primary_eligible,
    )


def _merged_spans(spans: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _source_time(revision: SourceObservationRevision) -> datetime | None:
    """The revision's source time on the timeline; ``None`` when its source gave no usable one."""
    try:
        return parse_source_time(revision.observed_at)
    except ValueError:
        return None


def _fragment_sort_key(fragment: EvidenceFragment, source_time: datetime | None) -> tuple[object, ...]:
    """Reading order: Observations by source time, then each one's Fragments by position.

    A conversation therefore reads in the order it was written. An Observation
    whose source reports no time follows those that have one.
    """
    anchor = fragment.anchor
    return (
        source_time is None,
        source_time or datetime.min.replace(tzinfo=timezone.utc),
        anchor.observation_revision_id,
        -1 if anchor.range_start is None else anchor.range_start,
        -1 if anchor.range_end is None else anchor.range_end,
        fragment.kind.value,
        fragment.raw_content_sha256,
        fragment.presentation_sha256,
    )


def _authority_payload(
    revision: SourceObservationRevision,
    ranges: tuple[EvidenceCandidateRange, ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "revision_id": revision.id,
            "profile": _profile_payload(revision),
            "anchor_kind": item.anchor.kind.value,
            "range_start": item.anchor.range_start,
            "range_end": item.anchor.range_end,
            "primary_eligible": item.primary_eligible,
        }
        for item in ranges
    )


def _profile_payload(revision: SourceObservationRevision) -> Mapping[str, object] | None:
    profile = revision.evidence_profile
    if profile is None:
        return None
    return {
        "name": profile.name,
        "version": profile.version,
        "coordinate_space": profile.coordinate_space.value,
        "schema_name": profile.schema_name,
        "schema_version": profile.schema_version,
    }


def _anchor_payload(anchor: SourceAnchor) -> Mapping[str, object]:
    return {
        "kind": anchor.kind.value,
        "observation_id": anchor.observation_id,
        "observation_revision_id": anchor.observation_revision_id,
        "fragment_id": anchor.fragment_id,
        "range_start": anchor.range_start,
        "range_end": anchor.range_end,
    }


def _catalog_digest(
    *,
    projection: SourceProjection,
    catalog_identity: Mapping[str, object],
    access_context_hash: str,
    authority_payload: list[Mapping[str, object]],
    component_digests: list[str],
    ordered_fragments: tuple[EvidenceFragment, ...],
    errors: list[FragmentCompilationError],
    max_fragments: int,
    max_presentation_chars: int,
) -> str:
    payload = {
        "compiler_contract_version": COMPILER_CONTRACT_VERSION,
        "source_id": projection.source_id,
        "source_unit_id": projection.source_units[0].id,
        "target_unit_revision_id": projection.source_unit_revisions[0].id,
        **dict(catalog_identity),
        "access_context_hash": access_context_hash,
        "authority_ranges": authority_payload,
        "component_digests": component_digests,
        "limits": {
            "max_fragments": max_fragments,
            "max_presentation_chars": max_presentation_chars,
        },
        "fragments": [
            {
                "kind": fragment.kind.value,
                "type": fragment.fragment_type,
                "anchor_kind": fragment.anchor.kind.value,
                "observation_id": fragment.anchor.observation_id,
                "observation_revision_id": fragment.anchor.observation_revision_id,
                "range_start": fragment.anchor.range_start,
                "range_end": fragment.anchor.range_end,
                "primary_eligible": fragment.primary_eligible,
                "raw_sha256": fragment.raw_content_sha256,
                "presentation_sha256": fragment.presentation_sha256,
            }
            for fragment in ordered_fragments
        ],
        "errors": [
            {
                "code": error.code.value,
                "revision_id": error.observation_revision_id,
                "start": error.range_start,
                "end": error.range_end,
                "fatal": error.fatal,
            }
            for error in errors
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _fatal_error(
    revision: SourceObservationRevision,
    code: FragmentCompilationErrorCode,
    message: str,
) -> FragmentCompilationError:
    return FragmentCompilationError(
        code=code,
        observation_revision_id=revision.id,
        message=message,
        fatal=True,
    )
