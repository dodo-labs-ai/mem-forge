"""Bind extracted claims to revision-pinned Source Projection evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    ResolvedEvidenceSelection,
    evidence_part_set_digest,
    evidence_reference_id_for,
    evidence_unit_id_v2,
)
from memforge.models import RawMemory, content_hash
from memforge.pipeline.projection_context import (
    context_observation_ids_for,
)
from memforge.source_projection import (
    AnchorKind,
    SourceAnchor,
    SourceProjection,
)


@dataclass(frozen=True, slots=True)
class ProjectedClaimEvidence:
    units: tuple[EvidenceUnit, ...]
    references: tuple[EvidenceReference, ...]
    canonical_memories_by_claim_hash: Mapping[str, RawMemory]
    evidence_unit_ids_by_claim_hash: Mapping[str, tuple[str, ...]]


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
) -> ProjectedClaimEvidence:
    """Build deterministic Evidence Units staged for the atomic Lifecycle Plan.

    Every claim must carry the application-resolved Fragment selection that
    names its exact supporting parts; each claim becomes one Evidence Unit
    whose identity hashes those parts.
    """

    if len(projection.source_units) != 1 or len(projection.source_unit_revisions) != 1:
        raise ValueError("claim evidence materialization requires one Source Unit projection")
    units_by_id: dict[str, EvidenceUnit] = {}
    references_by_id: dict[str, EvidenceReference] = {}
    canonical_memories_by_claim_hash: dict[str, RawMemory] = {}
    evidence_unit_ids_by_claim_hash: dict[str, tuple[str, ...]] = {}
    for raw in raw_memories:
        if raw.resolved_evidence_selection is None:
            raise ValueError("projected claim lacks a resolved Evidence selection")
        unit, supporting_references, context_references, canonical_memory = (
            _materialize_claim_evidence(
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
        for reference in (*supporting_references, *context_references):
            persisted = EvidenceReference(
                id=reference.id or evidence_reference_id_for(unit.id, reference),
                evidence_unit_id=unit.id,
                role=reference.role,
                anchor=reference.anchor,
                kind=reference.kind,
                raw_content_sha256=reference.raw_content_sha256,
                presentation_sha256=reference.presentation_sha256,
                excerpt=reference.excerpt,
                artifact_metadata=dict(reference.artifact_metadata),
            )
            references_by_id.setdefault(str(persisted.id), persisted)
        claim_hash = content_hash(raw.content.strip())
        evidence_unit_ids_by_claim_hash[claim_hash] = (unit.id,)
        canonical_memories_by_claim_hash[claim_hash] = canonical_memory
    return ProjectedClaimEvidence(
        units=tuple(units_by_id.values()),
        references=tuple(references_by_id.values()),
        canonical_memories_by_claim_hash=canonical_memories_by_claim_hash,
        evidence_unit_ids_by_claim_hash=evidence_unit_ids_by_claim_hash,
    )


def _resolved_fragment_audit_metadata(
    raw: RawMemory,
    selection: ResolvedEvidenceSelection,
) -> dict[str, object]:
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


def _materialize_claim_evidence(
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
            "observation_type": observations[primary.anchor.observation_id].observation_type,
            "fragment_catalog_digest": selection.catalog_digest,
            "fragment_compiler_contract_version": selection.compiler_contract_version,
            **_resolved_fragment_audit_metadata(raw, selection),
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
