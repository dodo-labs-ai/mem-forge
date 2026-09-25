"""Seed Evidence Unit Support the way a committed Lifecycle Plan leaves it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemoryUnitSupportAssertion,
    evidence_part_set_digest,
    evidence_reference_id_for,
    memory_unit_support_assertion_id,
)
from memforge.models import RawMemory
from memforge.pipeline.evidence_fragments import EvidenceFragment
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.source_projection import AnchorKind, SourceAnchor, SourceProjection
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


def select_quoted_fragments(
    projection: SourceProjection,
    raw_memories: Sequence[RawMemory],
    *,
    access_context_hash: str,
    base: SourceProjection | None = None,
) -> list[RawMemory]:
    """Resolve each candidate's quote to the current Fragments an extractor would select.

    The quote is the candidate's evidence quote, or its content when it has
    none. A quote equal to a Fragment's text selects the first such Fragment
    as the Primary, and a quote inside one Fragment selects that one. A
    quote spanning several Fragments selects the first as the Primary and the
    rest as Required parts. The Fragments of each Observation named in
    ``required_source_observation_ids``, such as an image, are Required too.
    """

    context = RevisionAssessmentContext(
        projection=projection,
        base=base,
        access_context_hash=access_context_hash,
    )
    catalog = context.catalog(context.full_fragments)

    def fragments_for(quote: str) -> list[EvidenceFragment]:
        exact = [fragment for fragment in catalog.fragments if fragment.presentation_text == quote]
        if exact:
            return exact[:1]
        containing = [fragment for fragment in catalog.fragments if quote in fragment.presentation_text]
        if containing:
            [fragment] = containing
            return [fragment]
        spanned = [
            fragment
            for fragment in catalog.fragments
            if fragment.presentation_text and fragment.presentation_text in quote
        ]
        assert spanned, f"no current Fragment matches quote: {quote!r}"
        return spanned

    selected: list[RawMemory] = []
    for raw in raw_memories:
        primary, *required = fragments_for(raw.evidence_quote or raw.content)
        required += [
            fragment
            for fragment in catalog.fragments
            if fragment.anchor.observation_id in raw.required_source_observation_ids
            and fragment not in (primary, *required)
        ]
        selected.append(
            replace(
                raw,
                source_observation_id=primary.anchor.observation_id,
                required_source_observation_ids=[fragment.anchor.observation_id for fragment in required],
                resolved_evidence_selection=catalog.resolve_selection(
                    primary_ref=primary.reference,
                    required_refs=tuple(fragment.reference for fragment in required),
                ),
            )
        )
    return selected
