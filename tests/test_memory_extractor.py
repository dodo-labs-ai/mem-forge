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
                    evidence_quote="Service A uses PostgreSQL 16",
                    evidence_block_id="EB-002",
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
    assert result.memories[0].evidence_quote == "Service A uses PostgreSQL 16"
    assert result.memories[0].evidence_block_id is None
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
    client = RecordingStructuredMemoryClient(
        error=StructuredLlmError(
            "response_format unsupported",
            error_code="ValidationError",
            validation_fields=(
                ("memories.0.memory_type", "literal_error"),
            ),
        )
    )
    extractor = MemoryExtractor(structured_llm_client=client)

    result = await extractor.extract_memories(content="Durable content")

    assert result.memories == []
    assert result.error_type == "structured_llm_error"
    assert result.error == "response_format unsupported"
    assert result.metadata["structured_llm_calls"] == 1
    assert result.metadata["safe_error_code"] == "ValidationError"
    assert result.metadata["safe_validation_fields"] == [
        {
            "location": "memories.0.memory_type",
            "type": "literal_error",
        }
    ]


@pytest.mark.asyncio
async def test_memory_extractor_uses_block_as_nonempty_fallback_for_changed_quote():
    source = "validation\\_rule\\_agent records traceId for later CLS lookup."
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="Use the recorded traceId for later CLS lookup.",
                    memory_type="procedure",
                    evidence_block_id="EB-001",
                    evidence_quote="validation_rule_agent records a trace ID.",
                )
            ]
        )
    )

    result = await MemoryExtractor(
        structured_llm_client=client
    ).extract_memories(source, source_type="teams")

    assert [memory.evidence_quote for memory in result.memories] == [source]
    assert result.metadata["evidence_refinement_counts"] == {"block_fallback": 1}
    assert 'id="EB-001"' in client.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_memory_change_extractor_exposes_only_current_changed_ranges():
    document = "Unchanged fact.\nChanged durable rule.\nUnchanged context."
    changed_start = document.index("Changed durable rule.")
    changed_end = changed_start + len("Changed durable rule.")
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="The durable rule changed.",
                    memory_type="decision",
                    evidence_block_id="EB-001",
                )
            ]
        )
    )

    result = await MemoryExtractor(
        structured_llm_client=client
    ).extract_memory_changes(
        changed_hunks="-Old durable rule.\n+Changed durable rule.",
        updated_document=document,
        current_changed_ranges=((changed_start, changed_end),),
    )

    prompt = client.calls[0]["prompt"]
    changed_catalog = prompt.split("<changed_evidence_catalog>", 1)[1].split(
        "</changed_evidence_catalog>", 1
    )[0]
    assert "Changed durable rule." in changed_catalog
    assert "Unchanged fact." not in changed_catalog
    assert "Unchanged context." not in changed_catalog
    assert result.memories[0].evidence_quote == "Changed durable rule."


@pytest.mark.asyncio
async def test_memory_change_extractor_catalogs_changes_beyond_context_prefix():
    changed = "Durable rule after the bounded context prefix."
    document = ("x" * 100_001) + changed
    changed_start = document.index(changed)
    client = RecordingStructuredMemoryClient(
        MemoryExtractionResponse(
            memories=[
                MemoryCandidate(
                    content="The durable rule remains authoritative.",
                    memory_type="decision",
                    evidence_block_id="EB-001",
                )
            ]
        )
    )

    result = await MemoryExtractor(
        structured_llm_client=client
    ).extract_memory_changes(
        changed_hunks=f"+{changed}",
        updated_document=document,
        current_changed_ranges=((changed_start, len(document)),),
    )

    prompt = client.calls[0]["prompt"]
    changed_catalog = prompt.split("<changed_evidence_catalog>", 1)[1].split(
        "</changed_evidence_catalog>", 1
    )[0]
    context_document = prompt.split("<updated_document>", 1)[1].split(
        "</updated_document>", 1
    )[0]
    assert changed in changed_catalog
    assert changed not in context_document
    assert result.memories[0].evidence_quote == changed
