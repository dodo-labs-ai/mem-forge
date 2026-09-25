"""A current Evidence Unit projection with one Primary text item, for relation tests."""

from __future__ import annotations

from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceRole,
    MemoryEvidenceItemProjection,
    MemoryEvidenceUnitProjection,
    SupportScopeVersion,
)
from memforge.source_projection import AnchorKind, SourceAnchor

_UNUSED_SHA256 = "0" * 64


def primary_evidence_unit_fixture(memory_id: str) -> MemoryEvidenceUnitProjection:
    anchor = SourceAnchor(
        kind=AnchorKind.WHOLE_OBSERVATION,
        observation_id=f"obs-{memory_id}",
        observation_revision_id=f"rev-{memory_id}",
    )
    return MemoryEvidenceUnitProjection(
        evidence_unit_id=f"eu-{memory_id}",
        support_ids=(f"support-{memory_id}",),
        support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
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
                excerpt=f"Excerpt for {memory_id}",
                raw_content_sha256=_UNUSED_SHA256,
                presentation_sha256=_UNUSED_SHA256,
                current=True,
            ),
        ),
    )
