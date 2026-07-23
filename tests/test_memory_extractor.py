from __future__ import annotations

import pytest

from memforge.llm.structured import MemoryCandidate, MemoryExtractionResponse, StructuredLlmError
from memforge.pipeline.memory_extractor import MemoryExtractor


class RecordingStructuredMemoryClient:
    def __init__(self, response: MemoryExtractionResponse | None = None, error: Exception | None = None) -> None:
        self.response = response or MemoryExtractionResponse(memories=[])
        self.error = error
        self.calls: list[dict] = []

    async def extract_memories(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
        images=(),
    ) -> MemoryExtractionResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "model": model,
                "images": images,
            }
        )
        if self.error:
            raise self.error
        return self.response


@pytest.mark.asyncio
async def test_memory_extractor_uses_structured_schema_client():
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Service A uses PostgreSQL 16.",
                    memory_type="fact",
                    confidence=0.9,
                    entity_refs=["Service A"],
                    extraction_context="Service A uses PostgreSQL 16",
                )
            ]
        )
    )
    extractor = MemoryExtractor(structured_llm_client=client, max_tokens=1234)

    result = await extractor.extract_memories(
        content="# Service A\n\nService A uses PostgreSQL 16.",
        source_type="github_pages",
        doc_type="reference",
    )

    assert result.error_type is None
    assert len(result.memories) == 1
    assert result.memories[0].content == "Service A uses PostgreSQL 16."
    assert result.memories[0].entity_refs == ["Service A"]
    assert client.calls[0]["max_tokens"] == 1234
    assert client.calls[0]["model"] == "claude-sonnet-4-20250514"
    assert "github_pages" in client.calls[0]["prompt"]
    assert "existing_memories" not in client.calls[0]["prompt"]
    assert '"tags"' not in client.calls[0]["prompt"]
    assert result.metadata["structured_llm_calls"] == 1
    assert result.metadata["prompt_chars"] == len(client.calls[0]["prompt"])
    assert result.metadata["structured_llm_elapsed_ms"] >= 0


@pytest.mark.asyncio
async def test_memory_extractor_reports_structured_output_failure():
    client = RecordingStructuredMemoryClient(error=StructuredLlmError("response_format unsupported"))
    extractor = MemoryExtractor(structured_llm_client=client)

    result = await extractor.extract_memories(content="Durable content")

    assert result.memories == []
    assert result.error_type == "structured_llm_error"
    assert result.error == "response_format unsupported"
    assert result.metadata["structured_llm_calls"] == 1
