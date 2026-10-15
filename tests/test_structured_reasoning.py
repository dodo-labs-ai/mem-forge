"""Closed-choice calls ask for a direct answer, and get their output back when reasoning spends it."""

from __future__ import annotations

import json

import litellm
import pytest

from memforge.evals.agent_evaluation import QualitySignalCollector, quality_signal_scope
from memforge.llm.capability import CapabilityCheckResponse
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.structured import (
    ChangeImpactWireResponse,
    LiteLlmStructuredClient,
    StructuredLlmCallTelemetry,
    StructuredLlmConfig,
    StructuredLlmError,
    StructuredLlmMetricsCollector,
    SupportAssessmentWireResponse,
)

OPENROUTER_ROUTE = "openrouter/anthropic/claude-haiku-5.5"
OPENROUTER_REASONING_OFF = {"extra_body": {"reasoning": {"enabled": False}}}
# The cross-document relation budget for one pair: a 256-token envelope and 32 per pair.
ONE_PAIR_OUTPUT_TOKENS = 288
ONE_PAIR_ANSWER = '{"decisions":[{"pair_index":0,"label":"none"}]}'


@pytest.fixture(autouse=True)
def model_capacity(monkeypatch):
    """Every route has these limits; LiteLLM's model map still tells what else it knows."""
    known = litellm.get_model_info

    def model_info(*args, **kwargs):
        try:
            info = dict(known(*args, **kwargs))
        except Exception:
            info = {}
        return {**info, "max_input_tokens": 200_000, "context_window": 200_000, "max_output_tokens": 64_000}

    monkeypatch.setattr("memforge.llm.structured.litellm.get_model_info", model_info)


@pytest.fixture(autouse=True)
def json_text_schema(monkeypatch):
    """The route answers with the schema written in its prompt, as an OpenRouter alias does."""
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: False)


def _response(
    content: str,
    *,
    finish_reason: str = "stop",
    prompt_tokens: int = 1576,
    completion_tokens: int = 20,
    reasoning_tokens: int | None = None,
) -> litellm.ModelResponse:
    """A provider reply in the shape LiteLLM returns for an OpenRouter chat completion."""
    usage: dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"audio_tokens": 0, "reasoning_tokens": reasoning_tokens, "image_tokens": 0}
    return litellm.ModelResponse(
        id="gen-1",
        model="anthropic/claude-haiku-5.5",
        created=1,
        object="chat.completion",
        choices=[{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
        usage=usage,
    )


def _reasoned_away(max_tokens: int, *, reasoning_tokens: int | None = None, content: str = "") -> litellm.ModelResponse:
    """A reply whose reasoning used the output budget before the answer was written."""
    return _response(
        content,
        finish_reason="length",
        completion_tokens=max_tokens,
        reasoning_tokens=max_tokens if reasoning_tokens is None else reasoning_tokens,
    )


def _client(transport, *, provider: str | None = "openrouter", model: str = OPENROUTER_ROUTE, **settings):
    telemetry: list[StructuredLlmCallTelemetry] = []
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model=model, base_url=None, api_key=None, timeout_s=5.0, provider=provider, **settings,
        ),
        telemetry_sink=telemetry.append,
        completion_transport=transport,
    )
    return client, telemetry


def _recording(*responses):
    calls: list[dict] = []
    queue = list(responses)

    async def transport(**kwargs):
        calls.append(kwargs)
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply(kwargs) if callable(reply) else reply

    return transport, calls


async def _call(client: LiteLlmStructuredClient, operation: str) -> None:
    calls = {
        "classify_memory_relations": lambda: client.classify_memory_relations("prompt", max_tokens=544),
        "classify_cross_document_relations": lambda: client.classify_cross_document_relations(
            "prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS),
        "change_impact": lambda: client.evaluate_revision_work(
            "prompt", response_format=ChangeImpactWireResponse, max_tokens=1024),
        "admit_candidates": lambda: client.admit_candidates("prompt", max_tokens=1024),
        "classify_agent_session_evidence_authority": lambda: client.classify_agent_session_evidence_authority("prompt"),
        "validate_entity_batch": lambda: client.validate_entity_batch("prompt"),
        "rerank_memories": lambda: client.rerank_memories("prompt"),
        "judge_offline_semantics": lambda: client.judge_offline_semantics("prompt"),
        "generation_check": lambda: client.answer_schema(
            "prompt", response_format=CapabilityCheckResponse, max_tokens=256),
        "extract_projection_fragment_memories": lambda: client.extract_projection_fragment_memories(
            "prompt", max_tokens=8192),
        "correct_projection_fragment_selectors": lambda: client.correct_projection_fragment_selectors(
            "prompt", max_tokens=2048),
        "assess_claim_revisions": lambda: client.assess_claim_revisions("prompt"),
        "support_assessment": lambda: client.evaluate_revision_work(
            "prompt", response_format=SupportAssessmentWireResponse, max_tokens=4096),
        "generate_agent_knowledge_patch": lambda: client.generate_agent_knowledge_patch("prompt"),
    }
    try:
        await calls[operation]()
    except StructuredLlmError:
        pass


DIRECT_ANSWER_OPERATIONS = (
    "classify_memory_relations",
    "classify_cross_document_relations",
    "change_impact",
    "admit_candidates",
    "classify_agent_session_evidence_authority",
    "validate_entity_batch",
    "rerank_memories",
    "judge_offline_semantics",
    "generation_check",
)
REASONED_OPERATIONS = (
    "extract_projection_fragment_memories",
    "correct_projection_fragment_selectors",
    "assess_claim_revisions",
    "support_assessment",
    "generate_agent_knowledge_patch",
)


@pytest.mark.parametrize("operation", DIRECT_ANSWER_OPERATIONS)
async def test_a_closed_choice_call_asks_the_provider_for_a_direct_answer(operation) -> None:
    transport, calls = _recording(*[_response("{}")] * 3)
    client, _telemetry = _client(transport)

    await _call(client, operation)

    assert calls
    assert all(call["extra_body"] == OPENROUTER_REASONING_OFF["extra_body"] for call in calls)


@pytest.mark.parametrize("operation", REASONED_OPERATIONS)
async def test_an_open_generation_call_keeps_the_provider_default(operation) -> None:
    transport, calls = _recording(*[_response("{}")] * 3)
    client, _telemetry = _client(transport)

    await _call(client, operation)

    assert calls
    assert all("extra_body" not in call for call in calls)


@pytest.mark.parametrize(
    "invocation",
    [
        ResolvedInvocation(route="litellm_proxy/claude-sonnet", provider="litellm"),
        ResolvedInvocation(route="openrouter/anthropic/claude-haiku-5.5", provider="litellm"),
        ResolvedInvocation(route="litellm_proxy/claude-sonnet"),
        ResolvedInvocation(route="openai/gpt-5-mini", api_base="https://models.example.test/v1",
                           provider="openai_compatible"),
    ],
)
async def test_a_route_without_a_known_safe_parameter_is_sent_the_call_as_before(invocation) -> None:
    transport, calls = _recording(_response(ONE_PAIR_ANSWER))
    client = LiteLlmStructuredClient(
        StructuredLlmConfig.for_invocation(invocation, timeout_s=5.0), completion_transport=transport,
    )

    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    [call] = calls
    assert set(call) == {"model", "messages", "timeout", "max_tokens", "num_retries", *invocation.request_kwargs()}


async def test_the_decision_model_is_asked_for_a_direct_answer_by_its_own_route() -> None:
    transport, calls = _recording(_response(ONE_PAIR_ANSWER))
    client, _telemetry = _client(
        transport, provider="openai", model="openai/gpt-5.1", decision_model="openai/gpt-5-mini",
    )

    await client.classify_cross_document_relations(
        "prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS, model=client.decision_model,
    )

    [call] = calls
    assert (call["model"], call["reasoning_effort"]) == ("openai/gpt-5-mini", "minimal")


async def test_a_route_that_reasons_by_default_answers_a_one_pair_relation_directly() -> None:
    """Asked with the provider's default, this route spends all 288 output tokens
    reasoning and returns no content in JSON-text mode; asked to turn reasoning
    off, it returns the answer in 24 tokens with no reasoning."""

    def openrouter(request: dict) -> litellm.ModelResponse:
        if request.get("extra_body", {}).get("reasoning") == {"enabled": False}:
            return _response(ONE_PAIR_ANSWER, completion_tokens=24, reasoning_tokens=0)
        return _reasoned_away(request["max_tokens"])

    transport, calls = _recording(openrouter)
    client, telemetry = _client(transport)

    result = await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert [(decision.pair_index, decision.label) for decision in result.decisions] == [(0, "none")]
    [call] = calls
    assert (call["model"], call["max_tokens"]) == (OPENROUTER_ROUTE, ONE_PAIR_OUTPUT_TOKENS)
    assert telemetry[-1].terminal_category == "success"
    assert telemetry[-1].reasoning_budget_retries == 0


async def test_reasoning_that_spends_the_budget_anyway_gets_it_back_once() -> None:
    collector = QualitySignalCollector()
    metrics = StructuredLlmMetricsCollector()
    transport, calls = _recording(
        _reasoned_away(ONE_PAIR_OUTPUT_TOKENS),
        _response(ONE_PAIR_ANSWER, completion_tokens=310, reasoning_tokens=286),
    )
    client, telemetry = _client(transport)

    with quality_signal_scope(collector), client.metrics_scope(metrics):
        result = await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert result.decisions[0].label == "none"
    assert [call["max_tokens"] for call in calls] == [ONE_PAIR_OUTPUT_TOKENS, 2 * ONE_PAIR_OUTPUT_TOKENS]
    assert all(call["extra_body"] == OPENROUTER_REASONING_OFF["extra_body"] for call in calls)
    [call_telemetry] = telemetry
    assert call_telemetry.terminal_category == "success"
    assert call_telemetry.reasoning_budget_retries == 1
    assert (call_telemetry.attempt_count, call_telemetry.retry_count, call_telemetry.fallback_count) == (2, 0, 0)
    [attempt] = call_telemetry.diagnostic_attempts
    assert (attempt.error_code, attempt.requested_max_tokens, attempt.finish_reason) == (
        "output_truncated", ONE_PAIR_OUTPUT_TOKENS, "length")
    outcome = next(signal for signal in collector.snapshot() if signal.event_name == "structured_output_outcome")
    assert (outcome.outcome, outcome.reason_code) == ("degraded", "reasoning_budget_recovered")
    summary = metrics.summary(source_unit_elapsed_ms=1)
    assert summary.reasoning_budget_retries == 1
    assert summary.to_payload()["reasoning_budget_retries"] == 1


async def test_the_raised_budget_stays_within_the_models_output_limit() -> None:
    transport, calls = _recording(
        _reasoned_away(ONE_PAIR_OUTPUT_TOKENS),
        _response(ONE_PAIR_ANSWER),
    )
    client, _telemetry = _client(transport, max_output_tokens=400)

    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert [call["max_tokens"] for call in calls] == [ONE_PAIR_OUTPUT_TOKENS, 400]


async def test_the_raised_budget_stays_within_the_context_window_left_by_the_prompt() -> None:
    transport, calls = _recording(
        _response("", finish_reason="length", prompt_tokens=199_500,
                  completion_tokens=ONE_PAIR_OUTPUT_TOKENS, reasoning_tokens=ONE_PAIR_OUTPUT_TOKENS),
        _response(ONE_PAIR_ANSWER),
    )
    client, _telemetry = _client(transport)

    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert [call["max_tokens"] for call in calls] == [ONE_PAIR_OUTPUT_TOKENS, 500]
    # A context window with no room left past the first budget gets no second request.
    transport, calls = _recording(
        _response("", finish_reason="length", prompt_tokens=199_712,
                  completion_tokens=ONE_PAIR_OUTPUT_TOKENS, reasoning_tokens=ONE_PAIR_OUTPUT_TOKENS),
    )
    client, _telemetry = _client(transport)
    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)
    assert failed.value.error_code == "output_truncated"
    assert len(calls) == 1


async def test_a_second_truncation_fails_the_call_without_another_request() -> None:
    transport, calls = _recording(
        _reasoned_away(ONE_PAIR_OUTPUT_TOKENS),
        _reasoned_away(2 * ONE_PAIR_OUTPUT_TOKENS, content='{"decisions":[{"pair_in'),
    )
    client, telemetry = _client(transport)

    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert failed.value.error_code == "output_truncated"
    assert len(calls) == 2
    assert telemetry[-1].reasoning_budget_retries == 1
    assert [attempt.requested_max_tokens for attempt in telemetry[-1].diagnostic_attempts] == [
        ONE_PAIR_OUTPUT_TOKENS, 2 * ONE_PAIR_OUTPUT_TOKENS]


@pytest.mark.parametrize(
    "reply",
    [
        # The answer itself outgrew its budget.
        _reasoned_away(ONE_PAIR_OUTPUT_TOKENS, reasoning_tokens=40, content='{"decisions":[{"pair_in'),
        # The provider does not report reasoning.
        _response("", finish_reason="length", completion_tokens=ONE_PAIR_OUTPUT_TOKENS),
    ],
)
async def test_a_truncation_reasoning_did_not_cause_is_not_sent_again(reply) -> None:
    transport, calls = _recording(reply)
    client, telemetry = _client(transport)

    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert failed.value.error_code == "output_truncated"
    assert len(calls) == 1
    assert telemetry[-1].reasoning_budget_retries == 0


async def test_an_open_call_whose_reasoning_spent_its_budget_also_gets_it_back_once() -> None:
    transport, calls = _recording(
        _reasoned_away(2048),
        _response(json.dumps({"memories": []})),
    )
    client, telemetry = _client(transport)

    await client.extract_projection_fragment_memories("prompt", max_tokens=2048)

    assert [call["max_tokens"] for call in calls] == [2048, 4096]
    assert all("extra_body" not in call for call in calls)
    assert telemetry[-1].reasoning_budget_retries == 1


async def test_a_provider_that_refuses_to_turn_reasoning_off_is_asked_without_it_from_then_on() -> None:
    refusal = litellm.BadRequestError(
        message="Reasoning is mandatory for this endpoint and cannot be disabled.",
        model=OPENROUTER_ROUTE,
        llm_provider="openrouter",
    )
    transport, calls = _recording(refusal, _response(ONE_PAIR_ANSWER), _response(ONE_PAIR_ANSWER))
    client, telemetry = _client(transport)

    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)
    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert ["extra_body" in call for call in calls] == [True, False, False]
    assert [entry.terminal_category for entry in telemetry] == ["success", "success"]
    assert telemetry[0].attempt_count == 2


async def test_a_rejection_that_is_not_about_reasoning_is_not_sent_again_without_it() -> None:
    rejected = litellm.BadRequestError(message="invalid message", model=OPENROUTER_ROUTE, llm_provider="openrouter")
    transport, calls = _recording(rejected, _response(ONE_PAIR_ANSWER))
    client, _telemetry = _client(transport)

    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)
    await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert failed.value.terminal_category == "request_error"
    assert ["extra_body" in call for call in calls] == [True, True]


async def test_a_route_that_rejects_native_schema_falls_back_with_reasoning_still_off(monkeypatch) -> None:
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: True)

    async def transport(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise litellm.BadRequestError(
                message="response_format json_schema is not supported", model=OPENROUTER_ROUTE, llm_provider="openrouter")
        return _response(ONE_PAIR_ANSWER)

    calls: list[dict] = []
    client, telemetry = _client(transport)

    for _ in range(2):
        await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    # Each call is one rejected native-schema request and one JSON-text request.
    assert [("response_format" in call, "extra_body" in call) for call in calls] == [
        (True, True), (False, True), (True, True), (False, True)]
    assert [entry.attempt_count for entry in telemetry] == [2, 2]


@pytest.mark.parametrize("native_schema", [False, True], ids=["json_text_repair", "json_text_fallback"])
async def test_the_attempt_after_an_invalid_raised_budget_reply_keeps_the_raised_budget(
    monkeypatch, native_schema,
) -> None:
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: native_schema)
    invalid = '{"decisions":[{"pair_index":0,"label":"unrelated"}]}'
    transport, calls = _recording(
        _reasoned_away(ONE_PAIR_OUTPUT_TOKENS),
        _response(invalid, completion_tokens=310, reasoning_tokens=286),
        _response(ONE_PAIR_ANSWER, completion_tokens=310, reasoning_tokens=286),
    )
    # The route is sent no reasoning-off parameter, so it keeps reasoning on every attempt.
    client, telemetry = _client(transport, provider="litellm")

    result = await client.classify_cross_document_relations("prompt", max_tokens=ONE_PAIR_OUTPUT_TOKENS)

    assert result.decisions[0].label == "none"
    assert [call["max_tokens"] for call in calls] == [ONE_PAIR_OUTPUT_TOKENS] + [2 * ONE_PAIR_OUTPUT_TOKENS] * 2
    assert telemetry[-1].reasoning_budget_retries == 1
