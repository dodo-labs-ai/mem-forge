"""Seed Evidence Unit Support the way a committed Lifecycle Plan leaves it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from memforge.memory.evidence import (
    ActiveSupportEvidence,
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemoryUnitSupportAssertion,
    evidence_part_set_digest,
    evidence_reference_id_for,
    memory_unit_support_assertion_id,
)
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.storage.database import Database

_BINARY_ARTIFACT_PROFILE = "binary-artifact"
_EMPTY_PRESENTATION_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _complete_part(db: Database, reference: EvidenceReference) -> EvidenceReference:
    """Fill the exact part fields from the stored Observation Revision."""

    if reference.kind is not None and reference.raw_content_sha256 is not None:
        return reference
    anchor = reference.anchor
    [row] = await db.db.execute_fetchall(
        "SELECT content, metadata_json, profile_name FROM source_observation_revisions WHERE id = ?",
        (anchor.observation_revision_id,),
    )
    if row["profile_name"] == _BINARY_ARTIFACT_PROFILE:
        artifact = dict(json.loads(row["metadata_json"] or "{}")["source_artifact"])
        return replace(
            reference,
            kind=EvidencePartKind.ARTIFACT,
            raw_content_sha256=str(artifact["sha256"]).lower(),
            presentation_sha256=_EMPTY_PRESENTATION_SHA256,
            excerpt=None,
            artifact_metadata=artifact,
        )
    content = str(row["content"])
    if anchor.kind is AnchorKind.REVISION_RANGE:
        assert anchor.range_start is not None and anchor.range_end is not None
        raw = content[anchor.range_start : anchor.range_end]
    else:
        assert anchor.kind is AnchorKind.WHOLE_OBSERVATION
        raw = content
    return replace(
        reference,
        kind=EvidencePartKind.TEXT,
        raw_content_sha256=_sha256(raw),
        presentation_sha256=_sha256(raw),
        excerpt=raw,
    )


async def complete_unit_parts(
    db: Database,
    unit: EvidenceUnit,
    references: Sequence[EvidenceReference],
) -> tuple[EvidenceUnit, tuple[EvidenceReference, ...]]:
    """Return ``unit`` and its parts as a Lifecycle Plan stages them.

    References without part fields are completed from the stored Observation
    Revision: a Revision Range or Whole Observation becomes a text part, and a
    binary artifact Observation becomes an artifact part.
    """

    completed = tuple([await _complete_part(db, reference) for reference in references])
    staged_unit = replace(unit, part_set_digest=evidence_part_set_digest(completed))
    staged_references = tuple(
        replace(
            reference,
            id=reference.id or evidence_reference_id_for(staged_unit.id, reference),
            evidence_unit_id=staged_unit.id,
        )
        for reference in completed
    )
    return staged_unit, staged_references


async def record_unit_support(
    db: Database,
    *,
    memory_id: str,
    unit: EvidenceUnit,
    references: Sequence[EvidenceReference],
) -> tuple[EvidenceReference, ...]:
    """Persist ``unit`` with its parts and make it active Support for ``memory_id``."""

    persisted_unit, parts = await complete_unit_parts(db, unit, references)
    await db.upsert_evidence_unit(persisted_unit)
    recorded = await db.record_evidence_references(persisted_unit.id, parts)
    access_context_hash = persisted_unit.access_context_hash or ""
    await db.upsert_memory_unit_support_assertion(
        MemoryUnitSupportAssertion(
            id=memory_unit_support_assertion_id(
                memory_id=memory_id,
                evidence_unit_id=persisted_unit.id,
                source_id=persisted_unit.source_id,
                access_context_hash=access_context_hash,
            ),
            memory_id=memory_id,
            evidence_unit_id=persisted_unit.id,
            source_id=persisted_unit.source_id,
            access_context_hash=access_context_hash,
        )
    )
    return recorded


def primary_reference(anchor: SourceAnchor) -> EvidenceReference:
    return EvidenceReference(role=EvidenceRole.PRIMARY, anchor=anchor)


async def active_support_evidence(
    db: Database,
    memory_id: str,
    *,
    source_id: str | None = None,
) -> tuple[ActiveSupportEvidence, ...]:
    """Return one Memory's active Support Evidence through the batched reader."""

    evidence = await db.get_active_memory_support_evidence_many(
        (memory_id,),
        source_id=source_id,
    )
    return evidence[memory_id]


async def withdraw_lifecycle_gate(db: Database, source_id: str) -> None:
    """Return a Source to the gated default that a Source without an enabled gate has."""

    await db.db.execute(
        "DELETE FROM source_lifecycle_gates WHERE source_id = ?",
        (source_id,),
    )
    await db.db.commit()
