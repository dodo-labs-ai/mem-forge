from __future__ import annotations

import pytest
from pydantic import ValidationError

from memforge.llm.structured import (
    ContradictionResponse,
    EntityValidationResponse,
    EnrichmentResponse,
    LiteLlmStructuredClient,
    MemoryCandidate,
    MemoryExtractionResponse,
    ReconciliationResponse,
    RerankResponse,
    SourceSupportDecision,
    SourceSupportResponse,
    StructuredLlmConfig,
    StructuredLlmError,
    litellm_model_name,
)


class ChoiceMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class Choice:
    def __init__(self, content: str | None) -> None:
        self.message = ChoiceMessage(content)


class CompletionResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [Choice(content)]


class ToolFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class ToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.function = ToolFunction(name, arguments)


class ToolChoiceMessage:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.content = None
        self.tool_calls = tool_calls


class ToolChoice:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.message = ToolChoiceMessage(tool_calls)


class ToolCompletionResponse:
    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.choices = [ToolChoice(tool_calls)]


def test_source_support_response_accepts_decision_list():
    response = SourceSupportResponse.model_validate({
        "decisions": [
            {
                "memory_id": "mem-1",
                "supported": True,
                "excerpt": "The document states the rule.",
                "reason": "direct statement",
            }
        ]
    })

    assert response.decisions == [
        SourceSupportDecision(
            memory_id="mem-1",
            supported=True,
            excerpt="The document states the rule.",
            reason="direct statement",
        )
    ]


def test_source_support_response_rejects_top_level_array():
    with pytest.raises(ValidationError):
        SourceSupportResponse.model_validate([
            {"memory_id": "mem-1", "supported": True}
        ])


def test_memory_extraction_response_accepts_memory_list():
    response = MemoryExtractionResponse.model_validate({
        "memories": [
            {
                "content": "Service A uses PostgreSQL 16 for transactional storage.",
                "memory_type": "fact",
                "confidence": 0.9,
                "entity_refs": ["Service A"],
                "tags": ["database", "storage"],
                "valid_from": None,
                "valid_until": None,
                "extraction_context": "Service A uses PostgreSQL 16",
            }
        ]
    })

    assert response.memories == [
        MemoryCandidate(
            content="Service A uses PostgreSQL 16 for transactional storage.",
            memory_type="fact",
            confidence=0.9,
            entity_refs=["Service A"],
            tags=["database", "storage"],
            valid_from=None,
            valid_until=None,
            extraction_context="Service A uses PostgreSQL 16",
        )
    ]


def test_memory_extraction_response_rejects_top_level_array():
    with pytest.raises(ValidationError):
        MemoryExtractionResponse.model_validate([
            {"content": "Fact", "memory_type": "fact"}
        ])


def test_litellm_model_name_preserves_explicit_provider_prefix():
    assert litellm_model_name("anthropic/claude-sonnet") == "anthropic/claude-sonnet"
    assert litellm_model_name("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_litellm_model_name_defaults_to_anthropic_provider():
    assert litellm_model_name("anthropic--claude-sonnet-latest") == (
        "anthropic/anthropic--claude-sonnet-latest"
    )


@pytest.mark.asyncio
async def test_litellm_structured_client_requires_response_schema(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"decisions":[{"memory_id":"mem-1","supported":true,"excerpt":"Exact text","reason":"match"}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    response = await client.verify_source_support("prompt")

    assert response.decisions[0].memory_id == "mem-1"
    assert calls[0]["model"] == "anthropic/anthropic--claude-sonnet-latest"
    assert calls[0]["api_base"] == "http://localhost:6655/anthropic"
    assert calls[0]["api_key"] == "local-key"
    assert calls[0]["timeout"] == 120.0
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0]["response_format"] is SourceSupportResponse
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_litellm_structured_client_requires_memory_extraction_schema(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"memories":[{"content":"Service A uses PostgreSQL 16.","memory_type":"fact","confidence":0.9}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    response = await client.extract_memories("prompt", max_tokens=8192)

    assert response.memories[0].content == "Service A uses PostgreSQL 16."
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0]["response_format"] is MemoryExtractionResponse
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_litellm_structured_client_supports_all_pipeline_schemas(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        schema = kwargs["response_format"]
        if schema is EnrichmentResponse:
            return CompletionResponse('{"summary":"Summary","tags":["tag"],"entities":[],"relationships":[],"doc_type":"reference","complexity":"low"}')
        if schema is ReconciliationResponse:
            return CompletionResponse('{"decisions":[{"action":"ADD","index":0,"reason":"new"}]}')
        if schema is ContradictionResponse:
            return CompletionResponse('{"decisions":[{"pair_index":0,"classification":"unrelated","reason":"different topic"}]}')
        if schema is EntityValidationResponse:
            return CompletionResponse('{"same_entity":true,"matched_id":7,"confidence":0.95}')
        if schema is RerankResponse:
            return CompletionResponse('{"ranking":[2,0,1]}')
        raise AssertionError(f"unexpected schema {schema}")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    assert (await client.enrich_document("prompt", max_tokens=64000)).summary == "Summary"
    assert (await client.reconcile_memories("prompt")).decisions[0].action == "ADD"
    assert (await client.detect_contradictions("prompt")).decisions[0].classification == "unrelated"
    assert (await client.validate_entity_match("prompt")).matched_id == 7
    assert (await client.rerank_memories("prompt")).ranking == [2, 0, 1]

    assert [call["response_format"] for call in calls] == [
        EnrichmentResponse,
        ReconciliationResponse,
        ContradictionResponse,
        EntityValidationResponse,
        RerankResponse,
    ]


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_on_invalid_response_schema(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse("{}")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError):
        await client.extract_memories("prompt", max_tokens=8192)
    assert len(calls) == 1
    assert calls[0]["response_format"] is MemoryExtractionResponse


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_when_litellm_rejects_schema(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise Exception("response_format unsupported")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError, match="response_format unsupported"):
        await client.verify_source_support("prompt")


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_on_missing_content(monkeypatch):
    async def fake_acompletion(**kwargs):
        return CompletionResponse(None)

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError, match="missing structured response content"):
        await client.verify_source_support("prompt")


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_on_invalid_schema(monkeypatch):
    async def fake_acompletion(**kwargs):
        return CompletionResponse('[{"memory_id":"mem-1","supported":true}]')

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError):
        await client.verify_source_support("prompt")


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_when_decisions_missing(monkeypatch):
    async def fake_acompletion(**kwargs):
        return CompletionResponse("{}")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError):
        await client.verify_source_support("prompt")


@pytest.mark.asyncio
async def test_litellm_structured_client_passes_num_retries(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse('{"memories":[]}')

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)

    # Default: transient gateway/connection blips are retried by litellm.
    default_client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )
    await default_client.extract_memories("prompt", max_tokens=8192)
    assert calls[0]["num_retries"] == 2

    # An explicit retry budget flows through unchanged.
    calls.clear()
    tuned_client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key="local-key",
            timeout_s=120.0,
            num_retries=5,
        )
    )
    await tuned_client.extract_memories("prompt", max_tokens=8192)
    assert calls[0]["num_retries"] == 5
