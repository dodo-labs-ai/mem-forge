"""Canonicalize untrusted extractor Evidence before durable use."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from memforge.models import RawMemory
from memforge.pipeline.document_update import quote_overlaps_current_changes


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

    quote = memory.evidence_quote or ""
    exact_match = bool(quote.strip() and quote in authority_text)
    exact_required = work_kind in {
        ClaimEvidenceWorkKind.CHANGED_RANGE,
        ClaimEvidenceWorkKind.STRUCTURAL_UNIT,
        ClaimEvidenceWorkKind.OBSERVATION,
        ClaimEvidenceWorkKind.REVALIDATED,
    }
    if exact_required and not exact_match:
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
    if quote == authority_text and (
        work_kind is ClaimEvidenceWorkKind.CHANGED_RANGE
        or not allow_short_whole_authority
        or quote_bytes > MAX_INLINE_WHOLE_AUTHORITY_BYTES
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

    if work_kind is ClaimEvidenceWorkKind.CHANGED_RANGE and not quote_overlaps_current_changes(
        authority_text,
        quote,
        current_changed_ranges,
    ):
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
