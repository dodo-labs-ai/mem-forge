"""Closed-choice calls ask for a direct answer; open generation calls keep the provider's reasoning default."""

from __future__ import annotations

import litellm
import pytest

from memforge.llm.capability import CapabilityCheckResponse
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.structured import (
    ChangeImpactWireResponse,
    LiteLlmStructuredClient,
    StructuredLlmCallTelemetry,
    StructuredLlmConfig,
    StructuredLlmError,
    SupportAssessmentWireResponse,
)

OPENROUTER_ROUTE = "openrouter/anthropic/claude-haiku-5.5"
OPENROUTER_REASONING_OFF = {"extra_body": {"reasoning": {"enabled": False}}}
# The route's output capacity under these limits: a quarter of the context window.
ROUTE_OUTPUT_TOKENS = 50_000
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
    """A reply whose reasoning used the requested output before the answer was written."""
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
        "classify_memory_relations": lambda: client.classify_memory_relations("prompt"),
        "classify_cross_document_relations": lambda: client.classify_cross_document_relations("prompt"),
        "change_impact": lambda: client.evaluate_revision_work(
            "prompt", response_format=ChangeImpactWireResponse),
        "admit_candidates": lambda: client.admit_candidates("prompt"),
        "validate_entity_batch": lambda: client.validate_entity_batch("prompt"),
        "rerank_memories": lambda: client.rerank_memories("prompt"),
        "judge_offline_semantics": lambda: client.judge_offline_semantics("prompt"),
        "generation_check": lambda: client.answer_schema("prompt", response_format=CapabilityCheckResponse),
        "extract_projection_fragment_memories": lambda: client.extract_projection_fragment_memories("prompt"),
        "assess_claim_revisions": lambda: client.assess_claim_revisions("prompt"),
        "support_assessment": lambda: client.evaluate_revision_work(
            "prompt", response_format=SupportAssessmentWireResponse),
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
    "validate_entity_batch",
    "rerank_memories",
    "judge_offline_semantics",
    "generation_check",
)
REASONED_OPERATIONS = (
    "extract_projection_fragment_memories",
    "assess_claim_revisions",
    "support_assessment",
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
        ResolvedInvocation(route="litellm_proxy/team-claude-sonnet", provider="litellm"),
        ResolvedInvocation(route="openrouter/anthropic/claude-haiku-5.5", provider="litellm"),
        ResolvedInvocation(route="litellm_proxy/team-claude-sonnet"),
        ResolvedInvocation(route="openai/gpt-5-mini", api_base="https://models.example.test/v1",
                           provider="openai_compatible"),
    ],
)
async def test_a_route_without_a_known_safe_parameter_is_sent_the_call_as_before(invocation) -> None:
    transport, calls = _recording(_response(ONE_PAIR_ANSWER))
    client = LiteLlmStructuredClient(
        StructuredLlmConfig.for_invocation(invocation, timeout_s=5.0), completion_transport=transport,
    )

    await client.classify_cross_document_relations("prompt")

    [call] = calls
    assert set(call) == {"model", "messages", "timeout", "max_tokens", "num_retries", *invocation.request_kwargs()}


async def test_the_decision_model_is_asked_for_a_direct_answer_by_its_own_route() -> None:
    transport, calls = _recording(_response(ONE_PAIR_ANSWER))
    client, _telemetry = _client(
        transport, provider="openai", model="openai/gpt-5.1", decision_model="openai/gpt-5-mini",
    )

    await client.classify_cross_document_relations("prompt", model=client.decision_model)

    [call] = calls
    assert (call["model"], call["reasoning_effort"]) == ("openai/gpt-5-mini", "minimal")


async def test_a_route_that_reasons_by_default_answers_a_one_pair_relation_directly() -> None:
    """Asked with the provider's default, this route spends all its output
    reasoning and returns no content in JSON-text mode; asked to turn reasoning
    off, it returns the answer in 24 tokens with no reasoning."""

    def openrouter(request: dict) -> litellm.ModelResponse:
        if request.get("extra_body", {}).get("reasoning") == {"enabled": False}:
            return _response(ONE_PAIR_ANSWER, completion_tokens=24, reasoning_tokens=0)
        return _reasoned_away(request["max_tokens"])

    transport, calls = _recording(openrouter)
    client, telemetry = _client(transport)

    result = await client.classify_cross_document_relations("prompt")

    assert [(decision.pair_index, decision.label) for decision in result.decisions] == [(0, "none")]
    [call] = calls
    assert (call["model"], call["max_tokens"]) == (OPENROUTER_ROUTE, ROUTE_OUTPUT_TOKENS)
    assert telemetry[-1].terminal_category == "success"


@pytest.mark.parametrize(
    "reply",
    [
        # Reasoning used the route's output capacity.
        _reasoned_away(ROUTE_OUTPUT_TOKENS),
        # The answer itself outgrew the route's output capacity.
        _reasoned_away(ROUTE_OUTPUT_TOKENS, reasoning_tokens=40, content='{"decisions":[{"pair_in'),
        # The provider does not report reasoning.
        _response("", finish_reason="length", completion_tokens=ROUTE_OUTPUT_TOKENS),
    ],
    ids=["reasoning", "answer", "unreported"],
)
async def test_a_truncated_reply_fails_the_call_without_another_request(reply) -> None:
    transport, calls = _recording(reply)
    client, telemetry = _client(transport)

    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt")

    assert failed.value.error_code == "output_truncated"
    assert [call["max_tokens"] for call in calls] == [ROUTE_OUTPUT_TOKENS]
    assert telemetry[-1].terminal_category == "invalid_response"


async def test_a_provider_that_refuses_to_turn_reasoning_off_is_asked_without_it_from_then_on() -> None:
    refusal = litellm.BadRequestError(
        message="Reasoning is mandatory for this endpoint and cannot be disabled.",
        model=OPENROUTER_ROUTE,
        llm_provider="openrouter",
    )
    transport, calls = _recording(refusal, _response(ONE_PAIR_ANSWER), _response(ONE_PAIR_ANSWER))
    client, telemetry = _client(transport)

    await client.classify_cross_document_relations("prompt")
    await client.classify_cross_document_relations("prompt")

    assert ["extra_body" in call for call in calls] == [True, False, False]
    assert [entry.terminal_category for entry in telemetry] == ["success", "success"]
    assert telemetry[0].attempt_count == 2


async def test_a_rejection_that_is_not_about_reasoning_is_not_sent_again_without_it() -> None:
    rejected = litellm.BadRequestError(message="invalid message", model=OPENROUTER_ROUTE, llm_provider="openrouter")
    transport, calls = _recording(rejected, _response(ONE_PAIR_ANSWER))
    client, _telemetry = _client(transport)

    with pytest.raises(StructuredLlmError) as failed:
        await client.classify_cross_document_relations("prompt")
    await client.classify_cross_document_relations("prompt")

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
        await client.classify_cross_document_relations("prompt")

    # Each call is one rejected native-schema request and one JSON-text request.
    assert [("response_format" in call, "extra_body" in call) for call in calls] == [
        (True, True), (False, True), (True, True), (False, True)]
    assert [entry.attempt_count for entry in telemetry] == [2, 2]
