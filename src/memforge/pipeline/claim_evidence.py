"""Canonicalize untrusted extractor Evidence before durable use."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from memforge.models import RawMemory
from memforge.pipeline.document_update import quote_overlaps_current_changes
from memforge.pipeline.evidence_catalog import localize_quote


MAX_INLINE_WHOLE_AUTHORITY_BYTES = 4_096
MAX_INLINE_CLAIM_EVIDENCE_BYTES = 32_768


class ClaimEvidenceWorkKind(str, Enum):
    """Provider-neutral authority shape for one extracted claim."""

    CHANGED_RANGE = "changed_range"
    STRUCTURAL_UNIT = "structural_unit"
    OBSERVATION = "observation"
    REVALIDATED = "revalidated"
    SOURCE_ARTIFACT = "source_artifact"


@dataclass(frozen=True, slots=True)
class ClaimEvidenceLocalization:
    """One canonical compatibility-safe view of an extractor candidate."""

    memory: RawMemory
    accepted: bool
    omission_reason: str | None = None


def localize_claim_evidence(
    memory: RawMemory,
    *,
    authority_text: str,
    work_kind: ClaimEvidenceWorkKind,
    current_changed_ranges: tuple[tuple[int, int], ...] = (),
    allow_short_whole_authority: bool = False,
) -> ClaimEvidenceLocalization:
    """Return one canonical excerpt derived only from the proposed quote."""

    proposed_quote = memory.evidence_quote or ""
    quote = proposed_quote
    localized_quote = localize_quote(authority_text, proposed_quote)
    if localized_quote is not None:
        quote = localized_quote[0]
    exact_required = work_kind in {
        ClaimEvidenceWorkKind.CHANGED_RANGE,
        ClaimEvidenceWorkKind.STRUCTURAL_UNIT,
        ClaimEvidenceWorkKind.OBSERVATION,
        ClaimEvidenceWorkKind.REVALIDATED,
    }
    if exact_required and localized_quote is None:
        return ClaimEvidenceLocalization(
            memory=replace(
                memory,
                evidence_quote=None,
                extraction_context=None,
            ),
            accepted=False,
            omission_reason="quote_not_in_authority",
        )

    quote_bytes = len(quote.encode("utf-8"))
    omission_reason = None
    resolved_start = memory.evidence_range_start
    resolved_end = memory.evidence_range_end
    range_authority_text = authority_text[resolved_start:resolved_end] if (
        resolved_start is not None
        and resolved_end is not None
        and 0 <= resolved_start < resolved_end <= len(authority_text)
    ) else None
    has_resolved_range = (
        range_authority_text == quote
    )
    proposed_is_whole_authority = proposed_quote == authority_text
    if proposed_is_whole_authority and (
        not memory.evidence_resolved_from_block
        and (
            work_kind is ClaimEvidenceWorkKind.CHANGED_RANGE
            or not allow_short_whole_authority
            or quote_bytes > MAX_INLINE_WHOLE_AUTHORITY_BYTES
        )
    ):
        omission_reason = "whole_authority_not_claim_local"
    elif quote_bytes > MAX_INLINE_CLAIM_EVIDENCE_BYTES:
        omission_reason = "inline_evidence_safety_envelope_exceeded"
    if omission_reason is not None:
        return ClaimEvidenceLocalization(
            memory=replace(
                memory,
                evidence_quote=None,
                extraction_context=None,
            ),
            accepted=(work_kind is not ClaimEvidenceWorkKind.CHANGED_RANGE),
            omission_reason=omission_reason,
        )

    overlaps_change = (
        any(
            resolved_start < range_end and resolved_end > range_start
            for range_start, range_end in current_changed_ranges
        )
        if has_resolved_range
        else quote_overlaps_current_changes(
            authority_text,
            quote,
            current_changed_ranges,
        )
    )
    if work_kind is ClaimEvidenceWorkKind.CHANGED_RANGE and not overlaps_change:
        return ClaimEvidenceLocalization(
            memory=replace(
                memory,
                evidence_quote=None,
                extraction_context=None,
            ),
            accepted=False,
            omission_reason="quote_outside_changed_ranges",
        )

    canonical_excerpt = quote if quote.strip() else None
    return ClaimEvidenceLocalization(
        memory=replace(
            memory,
            evidence_quote=canonical_excerpt,
            extraction_context=canonical_excerpt,
        ),
        accepted=True,
    )
