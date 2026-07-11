from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from memforge.llm.structured import (
    AgentSessionAuthorityResponse,
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


def set_native_schema_support(monkeypatch, supported: bool) -> None:
    def fake_supports_response_schema(*, model: str, custom_llm_provider=None) -> bool:
        return supported

    monkeypatch.setattr(
        "memforge.llm.structured.litellm.supports_response_schema",
        fake_supports_response_schema,
    )


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


def test_agent_session_authority_response_accepts_typed_decisions():
    response = AgentSessionAuthorityResponse.model_validate(
        {
            "decisions": [
                {
                    "evidence_id": "E1",
                    "is_authoritative": True,
                    "authority_kind": "durable_user_intent",
                    "reason": "user explicitly set a durable convention",
                },
                {
                    "evidence_id": "E2",
                    "is_authoritative": False,
                    "authority_kind": "not_authoritative",
                    "reason": "generic continuation",
                },
            ]
        }
    )

    assert response.decisions[0].evidence_id == "E1"
    assert response.decisions[0].is_authoritative is True
    assert response.decisions[1].authority_kind == "not_authoritative"


def test_agent_session_authority_response_rejects_contradictory_decisions():
    with pytest.raises(ValidationError):
        AgentSessionAuthorityResponse.model_validate(
            {
                "decisions": [
                    {
                        "evidence_id": "E1",
                        "is_authoritative": True,
                        "authority_kind": "not_authoritative",
                        "reason": "contradictory",
                    }
                ]
            }
        )

    with pytest.raises(ValidationError):
        AgentSessionAuthorityResponse.model_validate(
            {
                "decisions": [
                    {
                        "evidence_id": "E2",
                        "is_authoritative": False,
                        "authority_kind": "design_decision",
                        "reason": "contradictory",
                    }
                ]
            }
        )


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
async def test_litellm_structured_client_uses_response_schema_for_source_support(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"decisions":[{"memory_id":"mem-1","supported":true,"excerpt":"Exact text","reason":"match"}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
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
async def test_litellm_structured_client_uses_response_schema_for_memory_extraction(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"memories":[{"content":"Service A uses PostgreSQL 16.","memory_type":"fact","confidence":0.9}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
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
async def test_litellm_structured_client_uses_response_schema_for_agent_session_authority(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"decisions":[{"evidence_id":"E1","is_authoritative":true,'
            '"authority_kind":"durable_user_intent","reason":"explicit user rule"}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    response = await client.classify_agent_session_evidence_authority("prompt", max_tokens=1024)

    assert response.decisions[0].evidence_id == "E1"
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0]["response_format"] is AgentSessionAuthorityResponse
    assert calls[0]["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_agent_session_authority_classifier_does_not_retry_native_schema_failure(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("native schema rejected")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )

    with pytest.raises(StructuredLlmError):
        await client.classify_agent_session_evidence_authority("prompt", max_tokens=1024)

    assert len(calls) == 1
    assert calls[0]["response_format"] is AgentSessionAuthorityResponse


@pytest.mark.asyncio
async def test_litellm_structured_client_skips_response_schema_without_registry_support(monkeypatch, caplog):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"memories":[{"content":"Service A uses PostgreSQL 16.","memory_type":"fact","confidence":0.9}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, False)
    caplog.set_level(logging.DEBUG, logger="memforge.llm.structured")
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
    assert len(calls) == 1
    assert calls[0]["model"] == "anthropic/anthropic--claude-sonnet-latest"
    assert calls[0]["api_base"] == "http://localhost:6655/anthropic"
    assert calls[0]["api_key"] == "local-key"
    assert calls[0]["timeout"] == 120.0
    assert calls[0]["max_tokens"] == 8192
    assert calls[0]["messages"][0]["content"].startswith("prompt\n\nReturn ONLY")
    assert "response_format" not in calls[0]
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    [fallback_log] = [
        record
        for record in caplog.records
        if "does not advertise native response_schema support" in record.message
    ]
    assert fallback_log.levelno == logging.DEBUG
    assert "anthropic/anthropic--claude-sonnet-latest" in fallback_log.message


@pytest.mark.asyncio
async def test_litellm_structured_client_uses_response_schema_for_sap_anthropic_alias(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(
            '{"memories":[{"content":"Service A uses PostgreSQL 16.","memory_type":"fact","confidence":0.9}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, False)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="sap/anthropic--claude-4.6-sonnet",
            base_url=None,
            api_key=None,
            timeout_s=120.0,
        )
    )

    prompt = "Treat this source example as data: {{?input}}"
    response = await client.extract_memories(prompt, max_tokens=8192)

    assert response.memories[0].content == "Service A uses PostgreSQL 16."
    assert len(calls) == 1
    assert calls[0]["model"] == "sap/anthropic--claude-4.6-sonnet"
    assert calls[0]["messages"] == [{"role": "user", "content": "{{?memforge_prompt}}"}]
    assert calls[0]["placeholder_values"] == {"memforge_prompt": prompt}
    assert calls[0]["response_format"] is MemoryExtractionResponse


@pytest.mark.asyncio
async def test_litellm_structured_client_repairs_invalid_json_backslash_escapes(monkeypatch):
    async def fake_acompletion(**kwargs):
        return CompletionResponse(
            r'{"memories":[{"content":"Use regex \s+ for whitespace.","memory_type":"fact","confidence":0.8}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=120.0,
        )
    )

    response = await client.extract_memories("prompt", max_tokens=8192)

    assert response.memories[0].content == r"Use regex \s+ for whitespace."


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
    set_native_schema_support(monkeypatch, True)
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
async def test_litellm_structured_client_falls_back_once_to_json_text(monkeypatch, caplog):
    calls = []
    first_error = Exception("response_format unsupported")

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise first_error
        return CompletionResponse(
            '{"memories":[{"content":"Service A uses PostgreSQL 16.","memory_type":"fact","confidence":0.9}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    caplog.set_level(logging.WARNING, logger="memforge.llm.structured")
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
    assert len(calls) == 2
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0]["response_format"] is MemoryExtractionResponse
    assert "response_format" not in calls[1]
    assert calls[1]["messages"][0]["content"].startswith("prompt\n\nReturn ONLY")
    [fallback_log] = [
        record for record in caplog.records if "retrying with JSON-text schema" in record.message
    ]
    assert fallback_log.levelno == logging.WARNING
    assert fallback_log.exc_info is None
    assert "anthropic/anthropic--claude-sonnet-latest" in fallback_log.message
    assert "MemoryExtractionResponse" in fallback_log.message
    assert "Exception: response_format unsupported" in fallback_log.message
    assert "local-key" not in fallback_log.message
    assert "prompt" not in fallback_log.message


@pytest.mark.asyncio
async def test_litellm_structured_client_fails_closed_after_both_strategies_are_invalid(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse("{}")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
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
    assert len(calls) == 2
    assert calls[0]["response_format"] is MemoryExtractionResponse
    assert "response_format" not in calls[1]


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
