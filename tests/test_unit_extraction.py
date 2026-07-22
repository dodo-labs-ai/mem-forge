from __future__ import annotations

import pytest

from memforge.llm.structured import MemoryCandidate, MemoryExtractionResponse
from memforge.pipeline.document_units import ExtractionContext, ExtractionUnit
from memforge.pipeline.memory_extractor import (
    MEMORY_EXTRACTION_PROMPT,
    PROJECTION_BATCH_EXTRACTION_PROMPT,
    UNIT_MEMORY_EXTRACTION_PROMPT,
    MemoryExtractor,
)


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
async def test_extract_unit_memories_trusts_unit_anchor_as_boundary_contract():
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Tracking uses UnifiedContextApi for explicit API calls.",
                    memory_type="fact",
                    confidence=0.9,
                    entity_refs=["UnifiedContextApi"],
                    extraction_context="Tracking uses UnifiedContextApi for explicit API calls.",
                    evidence_quote="Tracking uses UnifiedContextApi for explicit API calls.",
                    evidence_anchor="unit",
                ),
                MemoryCandidate(
                    content="Tracking uses UnifiedContextApi for explicit API calls.",
                    memory_type="fact",
                    confidence=0.9,
                    entity_refs=["UnifiedContextApi"],
                    extraction_context="Tracking uses UnifiedContextApi for explicit API calls.",
                    evidence_quote="Tracking uses UnifiedContextApi for explicit API calls.",
                    evidence_anchor="glossary",
                ),
            ]
        )
    )
    extractor = MemoryExtractor(structured_llm_client=client)

    result = await extractor.extract_unit_memories(_context(), doc_type="reference")

    assert [memory.content for memory in result.memories] == ["Tracking uses UnifiedContextApi for explicit API calls."]
    assert result.memories[0].evidence_anchor == "unit"
    assert result.memories[0].extraction_context == "Tracking uses UnifiedContextApi for explicit API calls."
    assert result.metadata["structured_llm_calls"] == 1
    assert result.metadata["prompt_chars"] == len(client.calls[0]["prompt"])
    assert "glossary_appendix" in client.calls[0]["prompt"]


def test_full_scope_extraction_prompts_delegate_history_to_lifecycle():
    for prompt in (
        MEMORY_EXTRACTION_PROMPT,
        UNIT_MEMORY_EXTRACTION_PROMPT,
        PROJECTION_BATCH_EXTRACTION_PROMPT,
    ):
        assert "reconciliation" in prompt.lower()
        assert "existing_memories" not in prompt
