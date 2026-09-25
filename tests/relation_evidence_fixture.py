"""A current Evidence Unit with one Primary text item and its Observation Revision, for relation tests."""

from __future__ import annotations

from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
    MemoryEvidenceItemProjection,
    MemoryEvidenceUnitProjection,
)
from memforge.source_projection import AnchorKind, SourceAnchor, SourceObservationRevision
from memforge.source_representation import MARKDOWN_STRUCTURAL_PROFILE

_UNUSED_SHA256 = "0" * 64
PRIMARY_OBSERVED_AT = "2026-03-25T10:00:00.000+0000"


def primary_observation_revision_fixture(memory_id: str) -> SourceObservationRevision:
    return SourceObservationRevision(
        id=f"rev-{memory_id}",
        observation_id=f"obs-{memory_id}",
        semantic_hash=_UNUSED_SHA256,
        content=f"Evidence for {memory_id}.",
        observed_at=PRIMARY_OBSERVED_AT,
        evidence_profile=MARKDOWN_STRUCTURAL_PROFILE,
    )


def primary_evidence_unit_fixture(memory_id: str) -> MemoryEvidenceUnitProjection:
    anchor = SourceAnchor(
        kind=AnchorKind.WHOLE_OBSERVATION,
        observation_id=f"obs-{memory_id}",
        observation_revision_id=f"rev-{memory_id}",
    )
    return MemoryEvidenceUnitProjection(
        evidence_unit_id=f"eu-{memory_id}",
        support_ids=(f"support-{memory_id}",),
        source_id=f"src-{memory_id}",
        source_type="jira",
        source_unit_id=f"unit-{memory_id}",
        source_unit_revision_id=f"rev-{memory_id}",
        doc_id=f"doc-{memory_id}",
        current=True,
        legacy_limited=False,
        items=(
            MemoryEvidenceItemProjection(
                reference_id=f"ref-{memory_id}",
                role=EvidenceRole.PRIMARY,
                kind=EvidencePartKind.TEXT,
                anchor=anchor,
                excerpt=f"Raw observation for {memory_id}",
                raw_content_sha256=_UNUSED_SHA256,
                presentation_sha256=_UNUSED_SHA256,
                current=True,
            ),
        ),
    )
