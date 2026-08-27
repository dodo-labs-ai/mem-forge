"""Bind extracted claims to revision-pinned Source Projection evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    SupportScopeVersion,
    evidence_part_set_digest,
    evidence_reference_id_for,
    evidence_unit_id_v2,
)
from memforge.models import RawMemory, content_hash
from memforge.pipeline.claim_evidence import (
    ClaimEvidenceWorkKind,
    localize_claim_evidence,
)
from memforge.pipeline.projection_context import (
    context_observation_ids_for,
    observation_is_inference_eligible,
)
from memforge.source_projection import (
    AnchorKind,
    SourceAnchor,
    SourceObservationRevision,
    SourceProjection,
)


@dataclass(frozen=True, slots=True)
class ProjectedClaimEvidence:
    units: tuple[EvidenceUnit, ...]
    references: tuple[EvidenceReference, ...]
    reference_ids_by_claim_hash: Mapping[str, tuple[str, ...]]
    canonical_memories_by_claim_hash: Mapping[str, RawMemory]
    evidence_unit_ids_by_claim_hash: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )


def build_projected_claim_evidence(
    *,
    projection: SourceProjection,
    raw_memories: Sequence[RawMemory],
    doc_id: str,
    source_type: str,
    project_key: str | None,
    visibility: str,
    owner_user_id: str | None,
    repo_identifier: str | None,
    access_context_hash: str,
    extractor_run_id: str | None,
    observed_at: str | None = None,
    support_scope_version: SupportScopeVersion = SupportScopeVersion.REFERENCE_SET_V1,
) -> ProjectedClaimEvidence:
    """Build deterministic evidence staged for the atomic Lifecycle Plan.

    Candidate localization is proof-oriented: an exact quote match selects one
    Observation; otherwise a single current changed-or-added Observation is an
    acceptable Whole Observation fallback. Multiple possible Observations are
    rejected instead of assigning invented lineage.
    """

    if len(projection.source_units) != 1 or len(projection.source_unit_revisions) != 1:
        raise ValueError("claim evidence materialization requires one Source Unit projection")
    source_unit = projection.source_units[0]
    unit_revision = projection.source_unit_revisions[0]
    observations_by_id = {item.id: item for item in projection.observations}
    revisions_by_observation = {item.observation_id: item for item in projection.observation_revisions}
    inference_eligible_ids = {
        observation_id
        for observation_id, revision in revisions_by_observation.items()
        if observation_id in observations_by_id
        and observation_is_inference_eligible(
            observations_by_id[observation_id].observation_type,
            revision.metadata,
        )
    }
    ordered_observation_ids = [item.id for item in projection.observations if item.id in inference_eligible_ids]
    current_evidence_ids = {
        anchor.observation_id
        for delta in projection.deltas
        for anchor in delta.changed_anchors
        if anchor.observation_id in inference_eligible_ids
    } | {
        observation_id
        for delta in projection.deltas
        for observation_id in delta.added_observation_ids
        if observation_id in inference_eligible_ids
    }
    candidate_ids = current_evidence_ids or set(ordered_observation_ids)
    artifact_observation_ids = {
        observation_id
        for observation_id, observation in observations_by_id.items()
        if observation.observation_type == "binary_artifact" and observation_id in inference_eligible_ids
    }

    units_by_id: dict[str, EvidenceUnit] = {}
    references_by_id: dict[str, EvidenceReference] = {}
    reference_ids_by_claim_hash: dict[str, tuple[str, ...]] = {}
    canonical_memories_by_claim_hash: dict[str, RawMemory] = {}
    evidence_unit_ids_by_claim_hash: dict[str, tuple[str, ...]] = {}
    for raw in raw_memories:
        if (
            support_scope_version is SupportScopeVersion.EVIDENCE_UNIT_SET_V2
            and raw.resolved_evidence_selection is not None
        ):
            unit, supporting_references, context_references, canonical_memory = (
                _materialize_v2_claim_evidence(
                    projection=projection,
                    raw=raw,
                    doc_id=doc_id,
                    source_type=source_type,
                    project_key=project_key,
                    visibility=visibility,
                    owner_user_id=owner_user_id,
                    repo_identifier=repo_identifier,
                    access_context_hash=access_context_hash,
                    extractor_run_id=extractor_run_id,
                    observed_at=observed_at,
                )
            )
            units_by_id.setdefault(unit.id, unit)
            persisted = tuple(
                EvidenceReference(
                    id=reference.id
                    or evidence_reference_id_for(unit.id, reference),
                    evidence_unit_id=unit.id,
                    role=reference.role,
                    anchor=reference.anchor,
                    kind=reference.kind,
                    raw_content_sha256=reference.raw_content_sha256,
                    presentation_sha256=reference.presentation_sha256,
                    excerpt=reference.excerpt,
                    artifact_metadata=dict(reference.artifact_metadata),
                )
                for reference in (*supporting_references, *context_references)
            )
            for reference in persisted:
                references_by_id.setdefault(str(reference.id), reference)
            claim_hash = content_hash(raw.content.strip())
            reference_ids_by_claim_hash[claim_hash] = tuple(
                str(reference.id)
                for reference in persisted
                if reference.role in {EvidenceRole.PRIMARY, EvidenceRole.REQUIRED}
            )
            evidence_unit_ids_by_claim_hash[claim_hash] = (unit.id,)
            canonical_memories_by_claim_hash[claim_hash] = canonical_memory
            continue
        quote = raw.evidence_quote or ""
        primary_id = _primary_observation_id(
            candidate_ids=candidate_ids,
            revisions_by_observation=revisions_by_observation,
            quote=quote,
            observation_hint=raw.source_observation_id,
            revalidated_noop=raw.evidence_anchor == "revalidated_noop",
            empty_quote_observation_ids=(
                artifact_observation_ids if raw.evidence_anchor == "source_artifact" else set()
            ),
        )

        primary_revision = revisions_by_observation[primary_id]
        artifact_evidence = (
            raw.evidence_anchor in {"source_artifact", "revalidated_noop"}
            and primary_id in artifact_observation_ids
        )
        canonical_memory = replace(
            raw,
            evidence_quote=None,
            extraction_context=None,
        )
        if quote and not artifact_evidence:
            localized = localize_claim_evidence(
                raw,
                authority_text=primary_revision.content,
                work_kind=ClaimEvidenceWorkKind.OBSERVATION,
                allow_short_whole_authority=(
                    primary_revision.metadata.get("claim_evidence_scope")
                    == "atomic"
                ),
            )
            if not localized.accepted:
                raise ValueError(
                    "evidence quote is not canonical for the selected Source Observation"
                )
            canonical_memory = localized.memory
            quote = canonical_memory.evidence_quote or ""
        evidence_content = "" if artifact_evidence else quote
        evidence_unit_id = _stable_id(
            "eu-projected",
            projection.run_id,
            unit_revision.id,
            content_hash(raw.content.strip()),
            primary_revision.id,
        )
        unit = EvidenceUnit(
            id=evidence_unit_id,
            source_id=projection.source_id,
            doc_id=doc_id,
            doc_revision_id=unit_revision.id,
            source_type=source_type,
            source_anchor=primary_id,
            source_lineage_id=source_unit.id,
            project_key=project_key,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            content=evidence_content,
            excerpt=quote or None,
            evidence_provenance=(
                EvidenceContentProvenance.SOURCE_ARTIFACT
                if artifact_evidence
                else (EvidenceContentProvenance.SOURCE_EXCERPT if quote else EvidenceContentProvenance.NO_EXCERPT)
            ),
            source_metadata={
                "projection_run_id": projection.run_id,
                "source_unit_revision_id": unit_revision.id,
                "observation_type": observations_by_id[primary_id].observation_type,
                **_resolved_fragment_audit_metadata(raw),
            },
            observed_at=observed_at or primary_revision.observed_at,
            extractor_run_id=extractor_run_id,
            access_context_hash=access_context_hash,
        )
        units_by_id.setdefault(unit.id, unit)

        context_ids = context_observation_ids_for(projection, primary_id)
        required_ids = tuple(dict.fromkeys(raw.required_source_observation_ids))
        if primary_id in required_ids:
            raise ValueError("PRIMARY observation cannot also be REQUIRED")
        if any(observation_id not in context_ids for observation_id in required_ids):
            raise ValueError("required source observation is outside the extraction context")
        required_set = set(required_ids)
        claim_references = [
            EvidenceReference(
                role=EvidenceRole.PRIMARY,
                anchor=_primary_evidence_anchor(
                    revision=primary_revision,
                    quote=quote,
                    resolved_range_start=canonical_memory.evidence_range_start,
                    resolved_range_end=canonical_memory.evidence_range_end,
                ),
                evidence_unit_id=unit.id,
            )
        ]
        claim_references.extend(
            EvidenceReference(
                role=EvidenceRole.REQUIRED,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=observation_id,
                    observation_revision_id=revisions_by_observation[observation_id].id,
                ),
                evidence_unit_id=unit.id,
            )
            for observation_id in required_ids
        )
        claim_references.extend(
            EvidenceReference(
                role=EvidenceRole.CONTEXT,
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=observation_id,
                    observation_revision_id=revisions_by_observation[observation_id].id,
                ),
                evidence_unit_id=unit.id,
            )
            for observation_id in context_ids
            if observation_id not in required_set
        )
        persisted = tuple(
            EvidenceReference(
                id=item.id or evidence_reference_id_for(unit.id, item),
                evidence_unit_id=unit.id,
                role=item.role,
                anchor=item.anchor,
            )
            for item in claim_references
        )
        support_ids = tuple(item.id or "" for item in persisted if item.grants_support)
        if not support_ids:
            raise ValueError("projected claim has no support-granting evidence")
        for item in persisted:
            assert item.id is not None
            references_by_id.setdefault(item.id, item)
        claim_hash = content_hash(raw.content.strip())
        reference_ids_by_claim_hash[claim_hash] = support_ids
        canonical_memories_by_claim_hash[claim_hash] = canonical_memory
    return ProjectedClaimEvidence(
        units=tuple(units_by_id.values()),
        references=tuple(references_by_id.values()),
        reference_ids_by_claim_hash=reference_ids_by_claim_hash,
        canonical_memories_by_claim_hash=canonical_memories_by_claim_hash,
        evidence_unit_ids_by_claim_hash=evidence_unit_ids_by_claim_hash,
    )


def _resolved_fragment_audit_metadata(raw: RawMemory) -> dict[str, object]:
    selection = raw.resolved_evidence_selection
    if selection is None:
        return {}
    metadata: dict[str, object] = {
        "projection_extraction_v9_shadow_validated": True,
        "fragment_catalog_digest": selection.catalog_digest,
        "fragment_compiler_contract_version": selection.compiler_contract_version,
        "resolved_fragment_part_count": len(selection.parts),
    }
    receipt = raw.support_validation.get("agent_event_source_range_receipt")
    if isinstance(receipt, Mapping):
        metadata["agent_event_source_range_receipt"] = dict(receipt)
    return metadata


def _materialize_v2_claim_evidence(
    *,
    projection: SourceProjection,
    raw: RawMemory,
    doc_id: str,
    source_type: str,
    project_key: str | None,
    visibility: str,
    owner_user_id: str | None,
    repo_identifier: str | None,
    access_context_hash: str,
    extractor_run_id: str | None,
    observed_at: str | None,
) -> tuple[
    EvidenceUnit,
    tuple[EvidenceReference, ...],
    tuple[EvidenceReference, ...],
    RawMemory,
]:
    selection = raw.resolved_evidence_selection
    assert selection is not None
    source_unit = projection.source_units[0]
    unit_revision = projection.source_unit_revisions[0]
    if (
        selection.source_id != projection.source_id
        or selection.source_unit_id != source_unit.id
        or selection.target_unit_revision_id != unit_revision.id
        or selection.access_context_hash != access_context_hash
    ):
        raise ValueError("resolved Evidence selection belongs to another projection scope")
    member_revision_ids = set(unit_revision.observation_revision_ids)
    revisions = {
        revision.observation_id: revision
        for revision in projection.observation_revisions
    }
    observations = {observation.id: observation for observation in projection.observations}
    supporting: list[EvidenceReference] = []
    for part in selection.parts:
        if part.anchor.observation_revision_id not in member_revision_ids:
            raise ValueError("resolved Evidence part belongs to another Source Unit revision")
        revision = revisions.get(part.anchor.observation_id)
        if revision is None or revision.id != part.anchor.observation_revision_id:
            raise ValueError("resolved Evidence part is stale or unavailable")
        supporting.append(
            EvidenceReference(
                role=part.role,
                anchor=part.anchor,
                kind=part.kind,
                raw_content_sha256=part.raw_content_sha256,
                presentation_sha256=part.presentation_sha256,
                excerpt=part.excerpt,
                artifact_metadata=dict(part.artifact_metadata),
            )
        )
    supporting_tuple = tuple(supporting)
    part_digest = evidence_part_set_digest(supporting_tuple)
    unit_id = evidence_unit_id_v2(
        source_unit_id=source_unit.id,
        claim_content=raw.content,
        part_set_digest=part_digest,
        access_context_hash=access_context_hash,
    )
    primary = next(reference for reference in supporting_tuple if reference.role is EvidenceRole.PRIMARY)
    required_observation_ids = {
        reference.anchor.observation_id
        for reference in supporting_tuple
        if reference.role is EvidenceRole.REQUIRED
    }
    context_ids = tuple(
        observation_id
        for observation_id in context_observation_ids_for(
            projection,
            primary.anchor.observation_id,
        )
        if observation_id not in required_observation_ids
    )
    contexts = tuple(
        EvidenceReference(
            role=EvidenceRole.CONTEXT,
            anchor=SourceAnchor(
                kind=AnchorKind.WHOLE_OBSERVATION,
                observation_id=observation_id,
                observation_revision_id=revisions[observation_id].id,
            ),
            evidence_unit_id=unit_id,
        )
        for observation_id in context_ids
    )
    unit = EvidenceUnit(
        id=unit_id,
        source_id=projection.source_id,
        doc_id=doc_id,
        doc_revision_id=unit_revision.id,
        source_type=source_type,
        source_anchor=primary.anchor.observation_id,
        source_lineage_id=source_unit.id,
        project_key=project_key,
        visibility=visibility,
        owner_user_id=owner_user_id,
        repo_identifier=repo_identifier,
        content=primary.excerpt or "",
        excerpt=primary.excerpt,
        evidence_provenance=(
            EvidenceContentProvenance.SOURCE_ARTIFACT
            if primary.kind is EvidencePartKind.ARTIFACT
            else EvidenceContentProvenance.SOURCE_EXCERPT
        ),
        source_metadata={
            "projection_run_id": projection.run_id,
            "source_unit_revision_id": unit_revision.id,
            "observation_type": observations[primary.anchor.observation_id].observation_type,
            "fragment_catalog_digest": selection.catalog_digest,
            "fragment_compiler_contract_version": selection.compiler_contract_version,
            **_resolved_fragment_audit_metadata(raw),
        },
        observed_at=(
            observed_at
            or revisions[primary.anchor.observation_id].observed_at
        ),
        extractor_run_id=extractor_run_id,
        access_context_hash=access_context_hash,
        part_set_digest=part_digest,
    )
    canonical = replace(
        raw,
        evidence_quote=primary.excerpt,
        extraction_context=primary.excerpt,
        evidence_range_start=primary.anchor.range_start,
        evidence_range_end=primary.anchor.range_end,
        evidence_anchor="projection_fragment_catalog",
        source_observation_id=primary.anchor.observation_id,
        required_source_observation_ids=sorted(required_observation_ids),
    )
    return unit, supporting_tuple, contexts, canonical


def _primary_evidence_anchor(
    *,
    revision: SourceObservationRevision,
    quote: str,
    resolved_range_start: int | None = None,
    resolved_range_end: int | None = None,
) -> SourceAnchor:
    """Use an exact unique quote range; otherwise retain conservative authority."""

    if (
        resolved_range_start is not None
        and resolved_range_end is not None
        and 0 <= resolved_range_start < resolved_range_end <= len(revision.content)
        and revision.content[resolved_range_start:resolved_range_end] == quote
    ):
        return SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=resolved_range_start,
            range_end=resolved_range_end,
        )

    start = revision.content.find(quote) if quote else -1
    if start >= 0 and revision.content.find(quote, start + 1) < 0:
        return SourceAnchor(
            kind=AnchorKind.REVISION_RANGE,
            observation_id=revision.observation_id,
            observation_revision_id=revision.id,
            range_start=start,
            range_end=start + len(quote),
        )
    return SourceAnchor(
        kind=AnchorKind.WHOLE_OBSERVATION,
        observation_id=revision.observation_id,
        observation_revision_id=revision.id,
    )


def _primary_observation_id(
    *,
    candidate_ids: set[str],
    revisions_by_observation: Mapping[str, SourceObservationRevision],
    quote: str,
    observation_hint: str | None,
    revalidated_noop: bool,
    empty_quote_observation_ids: set[str],
) -> str:
    exact_quote_matches = [
        observation_id
        for observation_id in candidate_ids
        if quote and quote in revisions_by_observation[observation_id].content
    ]
    if observation_hint is None:
        if len(exact_quote_matches) == 1:
            return exact_quote_matches[0]
        if len(candidate_ids) == 1:
            return next(iter(candidate_ids))
        raise ValueError("extracted Memory cannot be localized to exactly one changed Source Observation")

    if observation_hint in candidate_ids or revalidated_noop:
        if observation_hint not in revisions_by_observation:
            raise ValueError("explicit source observation is unavailable in the current revision")
        # A revalidated NOOP hint is copied from active PRIMARY Support, not
        # supplied by extraction. Its unchanged whole-Observation authority
        # therefore remains valid without inventing a claim-local excerpt.
        if not quote and (
            revalidated_noop
            or observation_hint in empty_quote_observation_ids
        ):
            return observation_hint
        if not quote or quote not in revisions_by_observation[observation_hint].content:
            raise ValueError("explicit source observation does not contain the evidence quote")
        return observation_hint

    # Extractor-provided identities are localization hints, not evidence. The
    # current projection can safely repair a hint only with one exact match.
    if len(exact_quote_matches) == 1:
        return exact_quote_matches[0]
    raise ValueError("explicit source observation is outside the current evidence scope")


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(value) for value in values).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"
