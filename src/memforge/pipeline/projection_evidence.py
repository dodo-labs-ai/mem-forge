"""Bind extracted claims to revision-pinned Source Projection evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    evidence_reference_id_for,
)
from memforge.models import RawMemory, content_hash
from memforge.pipeline.projection_context import context_observation_ids_for
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
    ordered_observation_ids = [
        item.id for item in projection.observations if item.id in revisions_by_observation
    ]
    current_evidence_ids = {
        anchor.observation_id
        for delta in projection.deltas
        for anchor in delta.changed_anchors
        if anchor.observation_id in revisions_by_observation
    } | {
        observation_id
        for delta in projection.deltas
        for observation_id in delta.added_observation_ids
        if observation_id in revisions_by_observation
    }
    candidate_ids = current_evidence_ids or set(ordered_observation_ids)

    units_by_id: dict[str, EvidenceUnit] = {}
    references_by_id: dict[str, EvidenceReference] = {}
    reference_ids_by_claim_hash: dict[str, tuple[str, ...]] = {}
    for raw in raw_memories:
        quote = (raw.evidence_quote or raw.extraction_context or "").strip()
        primary_id = _primary_observation_id(
            candidate_ids=candidate_ids,
            revisions_by_observation=revisions_by_observation,
            quote=quote,
            observation_hint=raw.source_observation_id,
            revalidated_noop=raw.evidence_anchor == "revalidated_noop",
        )

        primary_revision = revisions_by_observation[primary_id]
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
            content=primary_revision.content,
            excerpt=quote or None,
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
            source_metadata={
                "projection_run_id": projection.run_id,
                "source_unit_revision_id": unit_revision.id,
                "observation_type": observations_by_id[primary_id].observation_type,
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
                anchor=SourceAnchor(
                    kind=AnchorKind.WHOLE_OBSERVATION,
                    observation_id=primary_id,
                    observation_revision_id=primary_revision.id,
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
        reference_ids_by_claim_hash[content_hash(raw.content.strip())] = support_ids
    return ProjectedClaimEvidence(
        units=tuple(units_by_id.values()),
        references=tuple(references_by_id.values()),
        reference_ids_by_claim_hash=reference_ids_by_claim_hash,
    )


def _primary_observation_id(
    *,
    candidate_ids: set[str],
    revisions_by_observation: Mapping[str, SourceObservationRevision],
    quote: str,
    observation_hint: str | None,
    revalidated_noop: bool,
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
        raise ValueError(
            "extracted Memory cannot be localized to exactly one changed Source Observation"
        )

    if observation_hint in candidate_ids or revalidated_noop:
        if observation_hint not in revisions_by_observation:
            raise ValueError("explicit source observation is unavailable in the current revision")
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
