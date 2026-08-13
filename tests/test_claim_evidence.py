from __future__ import annotations

from memforge.models import RawMemory
from memforge.pipeline.claim_evidence import (
    ClaimEvidenceWorkKind,
    localize_claim_evidence,
)


def test_long_claim_local_quote_is_preserved_verbatim_as_one_canonical_excerpt() -> None:
    quote = (
        "The payroll validation workflow keeps the incumbent result active while "
        "a conflicting proposal is reviewed, because review creation alone does "
        "not authorize support removal. "
    ) * 4
    authority_text = f"# Decision\n\nBefore.\n\n{quote}\n\nAfter."
    raw = RawMemory(
        content="Conflicting payroll validation results remain active during review.",
        memory_type="decision",
        evidence_quote=quote,
        extraction_context="provider supplied a different compatibility value",
    )

    localized = localize_claim_evidence(
        raw,
        authority_text=authority_text,
        work_kind=ClaimEvidenceWorkKind.OBSERVATION,
    )

    assert len(quote) > 200
    assert localized.accepted is True
    assert localized.omission_reason is None
    assert localized.memory.evidence_quote == quote
    assert localized.memory.extraction_context == quote


def test_large_whole_observation_is_omitted_instead_of_truncated() -> None:
    whole_observation = "Large source paragraph.\n" * 300
    raw = RawMemory(
        content="The source establishes one durable operating rule.",
        memory_type="fact",
        evidence_quote=whole_observation,
        extraction_context=whole_observation,
    )

    localized = localize_claim_evidence(
        raw,
        authority_text=whole_observation,
        work_kind=ClaimEvidenceWorkKind.OBSERVATION,
    )

    assert len(whole_observation.encode("utf-8")) > 4_096
    assert localized.accepted is True
    assert localized.omission_reason == "whole_authority_not_claim_local"
    assert localized.memory.evidence_quote is None
    assert localized.memory.extraction_context is None


def test_short_atomic_observation_can_use_its_whole_text() -> None:
    whole_message = "**Alice** (10:05): Keep the source adapter provider-specific."
    raw = RawMemory(
        content="Source adapters own provider-specific behavior.",
        memory_type="decision",
        evidence_quote=whole_message,
    )

    localized = localize_claim_evidence(
        raw,
        authority_text=whole_message,
        work_kind=ClaimEvidenceWorkKind.OBSERVATION,
        allow_short_whole_authority=True,
    )

    assert localized.accepted is True
    assert localized.memory.evidence_quote == whole_message
    assert localized.memory.extraction_context == whole_message


def test_changed_range_rejects_whole_document_as_false_localization() -> None:
    whole_document = "Unchanged context.\n" * 300 + "New durable decision.\n"
    changed_start = whole_document.index("New durable decision.")
    raw = RawMemory(
        content="A new durable decision was adopted.",
        memory_type="decision",
        evidence_quote=whole_document,
        extraction_context=whole_document,
    )

    localized = localize_claim_evidence(
        raw,
        authority_text=whole_document,
        work_kind=ClaimEvidenceWorkKind.CHANGED_RANGE,
        current_changed_ranges=((changed_start, len(whole_document)),),
    )

    assert localized.accepted is False
    assert localized.omission_reason == "whole_authority_not_claim_local"
    assert localized.memory.evidence_quote is None
    assert localized.memory.extraction_context is None


def test_changed_range_accepts_whole_short_document_resolved_from_block() -> None:
    whole_document = "New durable decision."
    raw = RawMemory(
        content="A new durable decision was adopted.",
        memory_type="decision",
        evidence_quote=whole_document,
        extraction_context=whole_document,
        evidence_resolved_from_block=True,
        evidence_range_start=0,
        evidence_range_end=len(whole_document),
    )

    localized = localize_claim_evidence(
        raw,
        authority_text=whole_document,
        work_kind=ClaimEvidenceWorkKind.CHANGED_RANGE,
        current_changed_ranges=((0, len(whole_document)),),
    )

    assert localized.accepted is True
    assert localized.omission_reason is None
    assert localized.memory.evidence_quote == whole_document
