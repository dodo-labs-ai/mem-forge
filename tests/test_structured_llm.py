from __future__ import annotations

import asyncio
import logging
from time import perf_counter

import pytest
from pydantic import ValidationError

from memforge.llm.structured import (
    AgentSessionAuthorityResponse,
    CandidateLedgerResponse,
    EntityValidationResponse,
    LiteLlmStructuredClient,
    MemoryCandidate,
    MemoryExtractionResponse,
    MemoryRelationResponse,
    MemorySupportValidationResponse,
    ReconciliationResponse,
    RerankResponse,
    SourceSupportDecision,
    SourceSupportResponse,
    StructuredLlmCallTelemetry,
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
    response = SourceSupportResponse.model_validate(
        {
            "decisions": [
                {
                    "memory_id": "mem-1",
                    "supported": True,
                    "excerpt": "The document states the rule.",
                    "reason": "direct statement",
                }
            ]
        }
    )

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
        SourceSupportResponse.model_validate([{"memory_id": "mem-1", "supported": True}])


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
    response = MemoryExtractionResponse.model_validate(
        {
            "memories": [
                {
                    "content": "Service A uses PostgreSQL 16 for transactional storage.",
                    "memory_type": "fact",
                    "confidence": 0.9,
                    "entity_refs": ["Service A"],
                    "valid_from": None,
                    "valid_until": None,
                    "extraction_context": "Service A uses PostgreSQL 16",
                }
            ]
        }
    )

    assert response.memories == [
        MemoryCandidate(
            content="Service A uses PostgreSQL 16 for transactional storage.",
            memory_type="fact",
            confidence=0.9,
            entity_refs=["Service A"],
            valid_from=None,
            valid_until=None,
            extraction_context="Service A uses PostgreSQL 16",
        )
    ]


def test_memory_extraction_response_rejects_top_level_array():
    with pytest.raises(ValidationError):
        MemoryExtractionResponse.model_validate([{"content": "Fact", "memory_type": "fact"}])


def test_litellm_model_name_preserves_explicit_provider_prefix():
    assert litellm_model_name("anthropic/claude-sonnet") == "anthropic/claude-sonnet"
    assert litellm_model_name("openai/gpt-4o-mini") == "openai/gpt-4o-mini"


def test_litellm_model_name_defaults_to_anthropic_provider():
    assert litellm_model_name("anthropic--claude-sonnet-latest") == ("anthropic/anthropic--claude-sonnet-latest")


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
    assert calls[0]["timeout"] == pytest.approx(120.0, abs=0.01)
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
async def test_agent_session_authority_classifier_retries_invalid_native_schema_as_json_text(
    monkeypatch,
):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return CompletionResponse("not valid json")
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
    assert response.decisions[0].is_authoritative is True
    assert len(calls) == 2
    assert calls[0]["response_format"] is AgentSessionAuthorityResponse
    assert "response_format" not in calls[1]
    assert calls[1]["messages"][0]["content"].startswith("prompt\n\nReturn ONLY")


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
    assert calls[0]["timeout"] == pytest.approx(120.0, abs=0.01)
    assert calls[0]["max_tokens"] == 8192
    assert calls[0]["messages"][0]["content"].startswith("prompt\n\nReturn ONLY")
    assert "response_format" not in calls[0]
    assert "tools" not in calls[0]
    assert "tool_choice" not in calls[0]
    [fallback_log] = [
        record for record in caplog.records if "does not advertise native response_schema support" in record.message
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
@pytest.mark.parametrize(
    "content",
    [
        (
            "Looking at the updated document, this memory should be removed.\n"
            '{"decisions":[{"action":"DELETE","memory_id":"mem-1",'
            '"reason":"unsupported","flag_for_review":false}]}'
        ),
        (
            "Here is the corrected ledger:\n```json\n"
            '{"decisions":[{"action":"DELETE","memory_id":"mem-1",'
            '"reason":"unsupported","flag_for_review":false}]}\n```'
        ),
    ],
)
async def test_litellm_structured_client_accepts_one_schema_valid_json_object_with_commentary(
    monkeypatch,
    content,
    caplog,
):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(content)

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

    response = await client.reconcile_memories("prompt")

    assert response.decisions[0].action == "DELETE"
    assert response.decisions[0].memory_id == "mem-1"
    assert len(calls) == 1
    assert any("recovered exactly one schema-valid JSON object" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_litellm_structured_client_rejects_ambiguous_schema_valid_json_objects(monkeypatch):
    calls = []
    decision = '{"decisions":[{"action":"DELETE","memory_id":"mem-1","reason":"unsupported","flag_for_review":false}]}'

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse(f"{decision}\n{decision}")

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

    with pytest.raises(StructuredLlmError, match="ambiguous structured JSON objects"):
        await client.reconcile_memories("prompt")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_litellm_structured_client_supports_all_pipeline_schemas(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        schema = kwargs["response_format"]
        if schema is ReconciliationResponse:
            return CompletionResponse('{"decisions":[{"action":"ADD","index":0,"reason":"new"}]}')
        if schema is CandidateLedgerResponse:
            return CompletionResponse('{"decisions":[{"index":0,"action":"KEEP"}]}')
        if schema is MemoryRelationResponse:
            return CompletionResponse(
                '{"decisions":[{"pair_index":0,"classification":"refines",'
                '"direction":"challenger_to_candidate","reason":"adds a condition"}]}'
            )
        if schema is MemorySupportValidationResponse:
            return CompletionResponse('{"supported":true,"reason":"still entailed"}')
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

    assert (await client.select_memory_candidates("prompt")).decisions[0].action == "KEEP"
    assert (await client.reconcile_memories("prompt")).decisions[0].action == "ADD"
    assert (await client.classify_memory_relations("prompt")).decisions[0].direction == "challenger_to_candidate"
    assert (await client.validate_memory_support("prompt")).supported is True
    assert (await client.validate_entity_match("prompt")).matched_id == 7
    assert (await client.rerank_memories("prompt")).ranking == [2, 0, 1]

    assert [call["response_format"] for call in calls] == [
        CandidateLedgerResponse,
        ReconciliationResponse,
        MemoryRelationResponse,
        MemorySupportValidationResponse,
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
    [fallback_log] = [record for record in caplog.records if "retrying with JSON-text schema" in record.message]
    assert fallback_log.levelno == logging.WARNING
    assert fallback_log.exc_info is None
    assert "anthropic/anthropic--claude-sonnet-latest" in fallback_log.message
    assert "MemoryExtractionResponse" in fallback_log.message
    assert "error_type=Exception" in fallback_log.message
    assert "response_format unsupported" not in fallback_log.message
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
async def test_litellm_structured_client_bounds_native_and_json_fallback_by_one_deadline(
    monkeypatch,
):
    calls = []
    telemetry: list[StructuredLlmCallTelemetry] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            await asyncio.sleep(0.02)
            return CompletionResponse("{}")
        await asyncio.sleep(1)
        raise AssertionError("the logical deadline should cancel the fallback")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=0.05,
        ),
        telemetry_sink=telemetry.append,
    )

    started = perf_counter()
    with pytest.raises(StructuredLlmError, match="logical deadline exceeded"):
        await client.extract_memories("prompt", max_tokens=1024)
    elapsed = perf_counter() - started

    assert elapsed < 0.25
    assert len(calls) == 2
    assert calls[0]["timeout"] <= 0.05
    assert calls[1]["timeout"] < calls[0]["timeout"]
    assert calls[0]["num_retries"] == 0
    assert calls[1]["num_retries"] == 0
    assert telemetry == [
        StructuredLlmCallTelemetry(
            operation="memory_extraction",
            attempt_count=2,
            retry_count=0,
            fallback_count=1,
            final_mode="json_text",
            elapsed_ms=pytest.approx(50, abs=40),
            terminal_category="deadline_exceeded",
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        )
    ]


@pytest.mark.asyncio
async def test_litellm_structured_client_shares_one_transport_retry_budget_across_modes(
    monkeypatch,
):
    calls = []
    telemetry: list[StructuredLlmCallTelemetry] = []

    class RetryableFailure(Exception):
        status_code = 503

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RetryableFailure("temporary")
        if len(calls) == 2:
            return CompletionResponse("{}")
        return CompletionResponse(
            '{"memories":[{"content":"A durable fact.","memory_type":"fact"}]}'
        )

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=1.0,
            num_retries=1,
        ),
        telemetry_sink=telemetry.append,
    )

    response = await client.extract_memories("prompt", max_tokens=1024)

    assert response.memories[0].content == "A durable fact."
    assert len(calls) == 3
    assert telemetry[0].attempt_count == 3
    assert telemetry[0].retry_count == 1
    assert telemetry[0].fallback_count == 1
    assert telemetry[0].final_mode == "json_text"
    assert telemetry[0].terminal_category == "success"


@pytest.mark.asyncio
async def test_litellm_structured_client_aggregates_available_usage_without_estimating_missing_tokens(
    monkeypatch,
):
    telemetry: list[StructuredLlmCallTelemetry] = []
    response = CompletionResponse(
        '{"memories":[{"content":"A durable fact.","memory_type":"fact"}]}'
    )
    response.usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }

    async def fake_acompletion(**kwargs):
        return response

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=1.0,
        ),
        telemetry_sink=telemetry.append,
    )

    await client.extract_memories("prompt", max_tokens=1024)

    assert telemetry[0].prompt_tokens == 11
    assert telemetry[0].completion_tokens == 7
    assert telemetry[0].total_tokens == 18


@pytest.mark.asyncio
async def test_litellm_structured_client_does_not_use_schema_fallback_for_provider_outage(
    monkeypatch,
):
    calls = []
    telemetry: list[StructuredLlmCallTelemetry] = []

    class ProviderUnavailable(Exception):
        status_code = 503

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        raise ProviderUnavailable("temporary")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=1.0,
            num_retries=1,
        ),
        telemetry_sink=telemetry.append,
    )

    with pytest.raises(StructuredLlmError) as raised:
        await client.extract_memories("prompt", max_tokens=1024)

    assert raised.value.terminal_category == "provider_error"
    assert len(calls) == 2
    assert telemetry[0].attempt_count == 2
    assert telemetry[0].retry_count == 1
    assert telemetry[0].fallback_count == 0
    assert telemetry[0].terminal_category == "provider_error"


@pytest.mark.asyncio
async def test_litellm_structured_client_distinguishes_provider_timeout_from_logical_deadline(
    monkeypatch,
):
    telemetry: list[StructuredLlmCallTelemetry] = []

    async def fake_acompletion(**kwargs):
        raise TimeoutError("provider request timed out")

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)
    set_native_schema_support(monkeypatch, True)
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url=None,
            api_key=None,
            timeout_s=1.0,
            num_retries=0,
        ),
        telemetry_sink=telemetry.append,
    )

    with pytest.raises(StructuredLlmError) as raised:
        await client.extract_memories("prompt", max_tokens=1024)

    assert raised.value.terminal_category == "provider_error"
    assert telemetry[0].terminal_category == "provider_error"
    assert telemetry[0].fallback_count == 0


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
async def test_litellm_structured_client_disables_nested_litellm_retries(monkeypatch):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return CompletionResponse('{"memories":[]}')

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", fake_acompletion)

    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="anthropic--claude-sonnet-latest",
            base_url="http://localhost:6655/anthropic",
            api_key="local-key",
            timeout_s=120.0,
        )
    )
    await client.extract_memories("prompt", max_tokens=8192)

    # The adapter owns one exact logical retry budget; allowing LiteLLM to
    # retry again would multiply both the deadline and attempt telemetry.
    assert calls[0]["num_retries"] == 0
