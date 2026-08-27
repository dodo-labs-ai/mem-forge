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
from enum import Enum
from typing import Mapping

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
    EvidenceAuthorityRange,
    EvidenceFragment,
    EvidenceFragmentKind,
    FragmentCompilationError,
    FragmentCompilationErrorCode,
    compile_fragments,
)
from memforge.pipeline.projection_context import ProjectionExtractionBatch
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

    def __init__(self, code: FragmentSelectionErrorCode, message: str) -> None:
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

    def model_payload(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "ref": fragment.reference,
                "kind": fragment.kind.value,
                "type": fragment.fragment_type,
                "text": fragment.presentation_text,
                "eligible_roles": sorted(role.value for role in fragment.eligible_roles),
            }
            for fragment in self.fragments
        )

    def resolve_selection(
        self,
        *,
        primary_ref: str,
        required_refs: tuple[str, ...] | list[str] = (),
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
        for role, reference in (
            (EvidenceRole.PRIMARY, primary_ref),
            *((EvidenceRole.REQUIRED, value) for value in normalized_required),
        ):
            fragment = by_reference.get(reference)
            if fragment is None:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.UNKNOWN_REF,
                    f"unknown, stale, or cross-catalog Fragment reference: {reference}",
                )
            if role not in fragment.eligible_roles:
                raise FragmentSelectionError(
                    FragmentSelectionErrorCode.INELIGIBLE_ROLE,
                    f"Fragment reference is not eligible for {role.value}: {reference}",
                )
            selected.append((role, fragment))

        order = {fragment.reference: index for index, fragment in enumerate(self.fragments)}
        primary = selected[0]
        required = tuple(sorted(selected[1:], key=lambda item: order[item[1].reference]))
        parts = tuple(
            self._resolved_part(role=role, fragment=fragment)
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
    batch: ProjectionExtractionBatch,
    *,
    access_context_hash: str,
    required_authority_observation_ids: tuple[str, ...] | None = None,
    supplied_artifact_observation_ids: tuple[str, ...] = (),
    max_fragments: int = DEFAULT_MAX_FRAGMENTS,
    max_presentation_chars: int = DEFAULT_MAX_PRESENTATION_CHARS,
) -> ProjectionFragmentCatalog:
    """Compile one immutable, source/scope-bound v9 selection catalog."""

    if len(projection.source_units) != 1 or len(projection.source_unit_revisions) != 1:
        raise ValueError("projection Fragment catalog requires exactly one Source Unit revision")
    if not access_context_hash:
        raise ValueError("projection Fragment catalog requires an access context hash")
    if max_fragments <= 0 or max_presentation_chars <= 0:
        raise ValueError("projection Fragment catalog limits must be positive")

    source_unit = projection.source_units[0]
    unit_revision = projection.source_unit_revisions[0]
    if batch.source_unit_id != source_unit.id or unit_revision.source_unit_id != source_unit.id:
        raise ValueError("projection Fragment batch belongs to another Source Unit")

    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
        if revision.id in set(unit_revision.observation_revision_ids)
    }
    observation_ids = {observation.id for observation in projection.observations}
    required_ids = set(
        batch.required_authority_observation_ids
        if required_authority_observation_ids is None
        else required_authority_observation_ids
    )
    if not required_ids.issubset(batch.context_observation_ids):
        raise ValueError("Required authority must come from the bounded batch dependency input")
    selectable_ids = set(batch.primary_observation_ids) | required_ids
    if not selectable_ids.issubset(revisions) or not selectable_ids.issubset(observation_ids):
        raise ValueError("projection Fragment batch contains stale Observation identity")
    supplied_artifacts = set(supplied_artifact_observation_ids)
    if not supplied_artifacts.issubset(selectable_ids):
        raise ValueError("supplied Artifact belongs to another catalog")

    primary_spans = _primary_spans_by_observation(batch)
    compiled_fragments: list[EvidenceFragment] = []
    errors: list[FragmentCompilationError] = []
    component_digests: list[str] = []
    authority_payload: list[Mapping[str, object]] = []
    artifact_metadata: dict[str, Mapping[str, object]] = {}

    for observation_id in sorted(selectable_ids, key=lambda value: revisions[value].id):
        revision = revisions[observation_id]
        is_primary = observation_id in set(batch.primary_observation_ids)
        if revision.evidence_profile is None:
            ranges = (_whole_range(revision, _roles(is_primary)),)
        elif revision.evidence_profile.coordinate_space is EvidenceCoordinateSpace.WHOLE_ARTIFACT:
            if observation_id not in supplied_artifacts:
                if is_primary:
                    errors.append(
                        _fatal_error(
                            revision,
                            FragmentCompilationErrorCode.ARTIFACT_INELIGIBLE,
                            "Primary Artifact was not supplied to the extraction model",
                        )
                    )
                continue
            ranges = (_whole_range(revision, _roles(is_primary)),)
            raw_artifact = revision.metadata.get("source_artifact")
            artifact_metadata[revision.id] = (
                dict(raw_artifact) if isinstance(raw_artifact, Mapping) else {}
            )
        elif is_primary:
            spans = primary_spans.get(observation_id, ())
            if not spans:
                errors.append(
                    _fatal_error(
                        revision,
                        FragmentCompilationErrorCode.INVALID_AUTHORITY_RANGE,
                        "Primary Observation has no exact authority span",
                    )
                )
                continue
            ranges = tuple(
                _text_range(revision, start, end, _roles(True))
                for start, end in spans
            )
        else:
            ranges = (_whole_range(revision, _roles(False)),)

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

    ordered = tuple(sorted(compiled_fragments, key=_fragment_sort_key))
    presentation_chars = sum(len(fragment.presentation_text) for fragment in ordered)
    if len(ordered) > max_fragments or presentation_chars > max_presentation_chars:
        representative = next(iter(revisions.values()))
        errors.append(
            _fatal_error(
                representative,
                FragmentCompilationErrorCode.CATALOG_TOO_LARGE,
                "composed Fragment catalog exceeds its explicit limits",
            )
        )
    exposed = () if any(error.fatal for error in errors) else ordered
    fragments = tuple(
        replace(fragment, reference=f"f{index:06d}")
        for index, fragment in enumerate(exposed, start=1)
    )
    digest = _catalog_digest(
        projection=projection,
        batch=batch,
        access_context_hash=access_context_hash,
        authority_payload=authority_payload,
        component_digests=component_digests,
        ordered_fragments=ordered,
        errors=errors,
        max_fragments=max_fragments,
        max_presentation_chars=max_presentation_chars,
    )
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
        artifact_metadata_by_revision_id=artifact_metadata,
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
    batch = ProjectionExtractionBatch(
        id=(
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
        source_unit_id=projection.source_units[0].id,
        primary_image_bytes=0,
        primary_observation_ids=(revision.observation_id,),
        primary_content_by_observation_id=((revision.observation_id, revision.content),),
        context_observation_ids=(),
        context_observation_ids_by_primary=((revision.observation_id, ()),),
        primary_markdown=revision.content,
        context_markdown="",
        primary_authority_spans=((revision.observation_id, 0, revision.content),),
    )
    catalog = compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash=access_context_hash,
    )
    if not catalog.usable:
        raise ValueError("projected agent claim Fragment catalog is unusable")
    candidates = [
        fragment
        for fragment in catalog.fragments
        if fragment.kind is EvidenceFragmentKind.TEXT
        and EvidenceRole.PRIMARY in fragment.eligible_roles
        and fragment.anchor.observation_revision_id == revision.id
        and fragment.anchor.range_start is not None
        and fragment.anchor.range_end is not None
        and fragment.anchor.range_start <= claim_start
        and claim_end <= fragment.anchor.range_end
    ]
    if len(candidates) != 1:
        raise ValueError("projected agent claim must map to one claim-coherent Fragment")
    selection = catalog.resolve_selection(primary_ref=candidates[0].reference)
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
            anchor=candidates[0].anchor,
        )
        for role, event_id in selected_events
    )
    return selection, AgentEventSourceRangeReceipt(
        target_unit_revision_id=catalog.target_unit_revision_id,
        catalog_digest=catalog.digest,
        claim_anchor=candidates[0].anchor,
        event_ranges=event_ranges,
    )


def _roles(primary: bool) -> frozenset[EvidenceRole]:
    return (
        frozenset({EvidenceRole.PRIMARY, EvidenceRole.REQUIRED})
        if primary
        else frozenset({EvidenceRole.REQUIRED})
    )


def _whole_range(
    revision: SourceObservationRevision,
    roles: frozenset[EvidenceRole],
) -> EvidenceAuthorityRange:
    return EvidenceAuthorityRange(
        anchor=SourceAnchor(
            kind=AnchorKind.WHOLE_OBSERVATION,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
        ),
        eligible_roles=roles,
    )


def _text_range(
    revision: SourceObservationRevision,
    start: int,
    end: int,
    roles: frozenset[EvidenceRole],
) -> EvidenceAuthorityRange:
    if start == 0 and end == len(revision.content):
        return _whole_range(revision, roles)
    return EvidenceAuthorityRange(
        anchor=SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=start,
            range_end=end,
        ),
        eligible_roles=roles,
    )


def _primary_spans_by_observation(
    batch: ProjectionExtractionBatch,
) -> dict[str, tuple[tuple[int, int], ...]]:
    raw_spans = batch.primary_authority_spans or tuple(
        (observation_id, 0, content)
        for observation_id, content in batch.primary_content_by_observation_id
    )
    grouped: dict[str, list[tuple[int, int]]] = {}
    for observation_id, start, content in raw_spans:
        if not content:
            continue
        grouped.setdefault(observation_id, []).append((start, start + len(content)))
    merged: dict[str, tuple[tuple[int, int], ...]] = {}
    for observation_id, spans in grouped.items():
        output: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(output[-1][1], end))
            else:
                output.append((start, end))
        merged[observation_id] = tuple(output)
    return merged


def _fragment_sort_key(fragment: EvidenceFragment) -> tuple[object, ...]:
    anchor = fragment.anchor
    return (
        anchor.observation_revision_id,
        -1 if anchor.range_start is None else anchor.range_start,
        -1 if anchor.range_end is None else anchor.range_end,
        fragment.kind.value,
        fragment.raw_content_sha256,
        fragment.presentation_sha256,
    )


def _authority_payload(
    revision: SourceObservationRevision,
    ranges: tuple[EvidenceAuthorityRange, ...],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "revision_id": revision.id,
            "profile": _profile_payload(revision),
            "anchor_kind": item.anchor.kind.value,
            "range_start": item.anchor.range_start,
            "range_end": item.anchor.range_end,
            "eligible_roles": sorted(role.value for role in item.eligible_roles),
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
    batch: ProjectionExtractionBatch,
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
        "batch_id": batch.id,
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
                "roles": sorted(role.value for role in fragment.eligible_roles),
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
