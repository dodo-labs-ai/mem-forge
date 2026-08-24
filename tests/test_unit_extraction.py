from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.llm.structured import MemoryCandidate, MemoryExtractionResponse
from memforge.models import ContentItem, NormalizedContent, RawContent, content_hash
from memforge.pipeline.document_units import ExtractionContext, ExtractionUnit
from memforge.pipeline.extraction_contract import DURABLE_MEMORY_QUALITY_RULES
from memforge.pipeline.memory_extractor import (
    MEMORY_CHANGE_EXTRACTION_PROMPT,
    MEMORY_EXTRACTION_PROMPT,
    PROJECTION_BATCH_EXTRACTION_PROMPT,
    UNIT_MEMORY_EXTRACTION_PROMPT,
    MemoryExtractor,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.source_projection_adapters import project_source_item


class RecordingStructuredMemoryClient:
    def __init__(self, response: MemoryExtractionResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ) -> MemoryExtractionResponse:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens, "model": model})
        return self.response


def _context() -> ExtractionContext:
    unit = ExtractionUnit(
        doc_id="doc-1",
        unit_id="doc-1::tracking",
        path_id="tracking",
        content_fingerprint="abc123",
        segmentation_version="v1",
        unit_kind="content",
        heading_path=("Guide", "Tracking"),
        start_line=5,
        end_line=12,
        split_depth=2,
        split_reason="chosen_depth",
        unit_markdown="## Tracking\n\nTracking uses [UnifiedContextApi](../uca) for explicit API calls.",
    )
    return ExtractionContext(
        document_title="Guide",
        document_url="https://example.test/guide",
        source_type="github_pages",
        unit=unit,
        document_outline="# Guide\n  ## Tracking\n  ## Terminology",
        glossary_appendix="UnifiedContextApi (UCA) is a process tracking API.",
        entities=["UnifiedContextApi"],
    )


@pytest.mark.asyncio
async def test_extract_unit_memories_requires_block_and_exact_unit_quote():
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Tracking uses UnifiedContextApi for explicit API calls.",
                    memory_type="fact",
                    confidence=0.9,
                    entity_refs=["UnifiedContextApi"],
                    extraction_context=(
                        "Tracking uses [UnifiedContextApi](../uca) for explicit API calls."
                    ),
                    evidence_quote=(
                        "Tracking uses [UnifiedContextApi](../uca) for explicit API calls."
                    ),
                    evidence_block_id="EB-002",
                    evidence_anchor="unit",
                ),
                MemoryCandidate(
                    content="An invented unit claim.",
                    memory_type="fact",
                    evidence_quote="This quote is absent from the owned unit.",
                    evidence_block_id="EB-999",
                    evidence_anchor="unit",
                ),
            ]
        )
    )
    extractor = MemoryExtractor(structured_llm_client=client)

    result = await extractor.extract_unit_memories(_context(), doc_type="reference")

    assert [memory.content for memory in result.memories] == ["Tracking uses UnifiedContextApi for explicit API calls."]
    assert result.memories[0].evidence_anchor == "unit"
    assert result.memories[0].extraction_context == (
        "Tracking uses [UnifiedContextApi](../uca) for explicit API calls."
    )
    assert result.metadata["structured_llm_calls"] == 1
    assert result.metadata["prompt_chars"] == len(client.calls[0]["prompt"])
    assert "glossary_appendix" in client.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_structural_block_fallback_keeps_nonempty_projected_excerpt():
    body = "A compact durable Confluence rule."
    context = replace(
        _context(),
        source_type="confluence",
        unit=replace(_context().unit, unit_markdown=body),
    )
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Confluence defines a compact durable rule.",
                    memory_type="decision",
                    evidence_block_id="EB-001",
                    evidence_quote="not copied from the source",
                )
            ]
        )
    )
    result = await MemoryExtractor(
        structured_llm_client=client
    ).extract_unit_memories(context, doc_type="reference")
    item = ContentItem(
        item_id="confluence-1",
        title="Rule",
        source_url="https://confluence.example.test/pages/1",
        last_modified=datetime(2026, 8, 12, tzinfo=timezone.utc),
        version="1",
        extra={"page_id": "1", "space_key": "ENG"},
    )
    projection = project_source_item(
        source_id="src-confluence",
        source_type="confluence",
        run_id="run-confluence",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=body),
    )

    staged = build_projected_claim_evidence(
        projection=projection,
        raw_memories=result.memories,
        doc_id=item.item_id,
        source_type="confluence",
        project_key=None,
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        access_context_hash="workspace",
        extractor_run_id="run-confluence",
    )

    assert result.memories[0].evidence_resolved_from_block is True
    assert result.memories[0].evidence_range_start is None
    assert staged.units[0].excerpt == body
    canonical = staged.canonical_memories_by_claim_hash[
        content_hash(result.memories[0].content)
    ]
    assert canonical.evidence_quote == body


@pytest.mark.asyncio
async def test_extract_unit_memories_preserves_long_claim_local_quote_without_truncation():
    quote = "The review gate preserves incumbent support until approval. " * 8
    context = _context()
    context = replace(
        context,
        unit=replace(
            context.unit,
            unit_markdown=f"## Review contract\n\nBefore.\n\n{quote}\n\nAfter.",
        ),
    )
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Review creation does not remove incumbent support.",
                    memory_type="decision",
                    extraction_context=context.unit.unit_markdown,
                    evidence_quote=quote,
                    evidence_block_id="EB-003",
                    evidence_anchor="unit",
                )
            ]
        )
    )

    result = await MemoryExtractor(
        structured_llm_client=client
    ).extract_unit_memories(context, doc_type="reference")

    assert len(quote) > 200
    assert result.memories[0].evidence_quote == quote
    assert result.memories[0].extraction_context == quote


def test_full_scope_extraction_prompts_delegate_history_to_lifecycle():
    for prompt in (
        MEMORY_EXTRACTION_PROMPT,
        UNIT_MEMORY_EXTRACTION_PROMPT,
        PROJECTION_BATCH_EXTRACTION_PROMPT,
    ):
        assert "reconciliation" in prompt.lower()
        assert "existing_memories" not in prompt


def test_all_extraction_prompts_share_the_durable_memory_quality_contract():
    required_rules = (
        "PREFER EMPTY",
        "CODE-RECOVERABLE FACTS ARE NOT MEMORIES",
        "ONE CLAIM, ONE MEMORY",
        "FOLD REJECTED ALTERNATIVES INTO THE CHOSEN DECISION",
        "FUTURE USEFULNESS CHECK",
        "NO META-MEMORIES",
        "OPERATIONAL DOES NOT MEAN TRANSIENT",
    )

    for prompt in (
        MEMORY_EXTRACTION_PROMPT,
        MEMORY_CHANGE_EXTRACTION_PROMPT,
        UNIT_MEMORY_EXTRACTION_PROMPT,
        PROJECTION_BATCH_EXTRACTION_PROMPT,
    ):
        assert DURABLE_MEMORY_QUALITY_RULES in prompt
        for rule in required_rules:
            assert rule in prompt


def test_all_extraction_prompts_share_the_owned_evidence_language_contract():
    required_rules = (
        "For each candidate, preserve the language of its owned source evidence.",
        "When that evidence is primarily Chinese, write memory.content in Chinese.",
        "unless the evidence itself is English or mixed-language phrasing is necessary",
        "Read-only context may resolve meaning but must not change the candidate's language.",
    )

    for rule in required_rules:
        assert rule in DURABLE_MEMORY_QUALITY_RULES

    for prompt in (
        MEMORY_EXTRACTION_PROMPT,
        MEMORY_CHANGE_EXTRACTION_PROMPT,
        UNIT_MEMORY_EXTRACTION_PROMPT,
        PROJECTION_BATCH_EXTRACTION_PROMPT,
    ):
        assert DURABLE_MEMORY_QUALITY_RULES in prompt
