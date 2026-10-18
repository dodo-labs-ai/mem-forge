"""Capability checks: one real request through the runtime's own clients."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
import pytest

from memforge.llm.capability import (
    CapabilityCheckResponse,
    check_embedding,
    check_generation,
)
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig

GENERATION = ResolvedInvocation(
    route="openai/check-model",
    api_base="https://generate.example.test/v1",
    api_key="generation-secret-key",
)
EMBEDDING = ResolvedInvocation(
    route="openai/check-embedding",
    api_base="https://embed.example.test/v1",
    api_key="embedding-secret-key",
)
EXPECTED_REPLY = {"probe": {"label": "memforge", "confirmed": True, "items": ["alpha", "beta", "gamma"]}}


def _completion(content: Any, *, model: str | None = "check-model-2026-10") -> SimpleNamespace:
    text = content if isinstance(content, str) else json.dumps(content)
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
    )


class Provider:
    """Stand in for the generation provider behind LiteLLM and record every request."""

    def __init__(self, monkeypatch, *steps: Any, native_schema: bool = True) -> None:
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []
        monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", self)
        monkeypatch.setattr(
            "memforge.llm.structured.litellm.supports_response_schema",
            lambda **_kwargs: native_schema,
        )

    async def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        step = self._steps.pop(0) if self._steps else _completion(EXPECTED_REPLY)
        if isinstance(step, BaseException):
            raise step
        return step


def _refused_connection() -> Exception:
    """A refused connection as LiteLLM reports it on an OpenAI-compatible route."""
    failure = litellm.InternalServerError("Connection error.", "openai", "check-model")
    failure.__cause__ = httpx.ConnectError("connection refused")
    return failure


TOKEN_LIMITS = {"max_input_tokens": 32_000, "context_window_tokens": 32_000, "max_output_tokens": 4_000}


def _client(*, token_limits: dict[str, int] | None = None, **settings: Any) -> LiteLlmStructuredClient:
    """The runtime's client for a model LiteLLM has no limits for, with the limits given."""
    settings.setdefault("timeout_s", 5.0)
    limits = TOKEN_LIMITS if token_limits is None else token_limits
    return LiteLlmStructuredClient(StructuredLlmConfig.for_invocation(GENERATION, **limits, **settings))


def _schema_property(name: str) -> dict[str, Any]:
    schema = CapabilityCheckResponse.model_json_schema()
    return schema["$defs"]["CapabilityProbe"]["properties"][name]


def test_the_generation_check_asks_for_one_strict_nested_object() -> None:
    schema = CapabilityCheckResponse.model_json_schema()
    probe = schema["$defs"]["CapabilityProbe"]

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["probe"]
    assert probe["additionalProperties"] is False
    assert probe["required"] == ["label", "confirmed", "items"]
    assert _schema_property("label")["type"] == "string"
    assert _schema_property("confirmed")["type"] == "boolean"
    assert _schema_property("items") == {"items": {"type": "string"}, "title": "Items", "type": "array"}


async def test_a_generation_model_that_returns_the_object_passes(monkeypatch) -> None:
    provider = Provider(monkeypatch)

    check = await check_generation(_client())

    assert check.passed is True
    assert check.error_code is None
    assert check.facts == {
        "latency_ms": check.facts["latency_ms"],
        "token_limits_known": True,
        "structured_output_mode": "native_schema",
        "schema_fallback": False,
        "provider_model": "check-model-2026-10",
    }
    request = provider.requests[0]
    assert request["model"] == "openai/check-model"
    assert request["api_base"] == "https://generate.example.test/v1"
    assert request["api_key"] == "generation-secret-key"
    assert request["num_retries"] == 0
    assert request["response_format"] is CapabilityCheckResponse


async def test_the_check_follows_the_runtimes_fallback_from_native_schema_to_json_text(monkeypatch) -> None:
    provider = Provider(
        monkeypatch,
        litellm.BadRequestError("response_format is not supported", "check-model", "openai"),
    )

    check = await check_generation(_client())

    assert check.passed is True
    assert check.facts["structured_output_mode"] == "json_text"
    assert check.facts["schema_fallback"] is True
    assert "response_format" in provider.requests[0]
    assert "response_format" not in provider.requests[1]


async def test_a_model_without_native_schema_support_is_checked_with_the_schema_as_text(monkeypatch) -> None:
    provider = Provider(monkeypatch, native_schema=False)

    check = await check_generation(_client())

    assert check.passed is True
    assert check.facts["structured_output_mode"] == "json_text"
    assert check.facts["schema_fallback"] is False
    assert len(provider.requests) == 1
    assert "response_format" not in provider.requests[0]


@pytest.mark.parametrize(
    "reply",
    [
        {"probe": {"label": "memforge", "confirmed": False, "items": ["alpha", "beta", "gamma"]}},
        {"probe": {"label": "something else", "confirmed": True, "items": ["alpha", "beta", "gamma"]}},
        {"probe": {"label": "memforge", "confirmed": True, "items": ["gamma", "beta", "alpha"]}},
        {"probe": {"label": "memforge", "confirmed": True, "items": []}},
    ],
    ids=["boolean", "string", "list_order", "list_empty"],
)
async def test_a_schema_valid_reply_with_other_values_fails(monkeypatch, reply) -> None:
    Provider(monkeypatch, _completion(reply))

    check = await check_generation(_client())

    assert check.error_code == "structured_output_invalid"
    assert check.passed is False


async def test_a_reply_that_never_matches_the_schema_fails(monkeypatch) -> None:
    provider = Provider(
        monkeypatch,
        _completion({"probe": {"label": "memforge", "confirmed": "yes"}}),
        _completion("I cannot produce JSON."),
        _completion({"probe": {"label": 7, "confirmed": True, "items": "alpha"}}),
    )

    check = await check_generation(_client())

    assert check.error_code == "structured_output_invalid"
    assert len(provider.requests) == 3


@pytest.mark.parametrize(
    ("failure", "error_code", "status_code"),
    [
        (litellm.AuthenticationError("invalid x-api-key", "openai", "check-model"), "authentication_failed", 401),
        (litellm.NotFoundError("model not found", "check-model", "openai"), "model_not_found", 404),
        (
            litellm.APIConnectionError("nodename nor servname provided", "openai", "check-model"),
            "provider_unreachable",
            None,
        ),
        (_refused_connection(), "provider_unreachable", None),
        (litellm.InternalServerError("upstream failed", "openai", "check-model"), "provider_error", 500),
    ],
    ids=["rejected_key", "unknown_model", "unresolved_host", "refused_connection", "server_fault"],
)
async def test_a_provider_failure_is_reported_by_code_and_the_status_it_was_answered_with(
    monkeypatch, failure, error_code, status_code
) -> None:
    Provider(monkeypatch, failure, failure, failure, failure)

    check = await check_generation(_client(num_retries=0))

    assert check.error_code == error_code
    assert check.facts.get("status_code") == status_code
    assert check.facts["latency_ms"] >= 0


async def test_a_model_whose_token_limits_nobody_knows_fails_after_it_has_answered(monkeypatch) -> None:
    provider = Provider(monkeypatch)

    unknown = await check_generation(_client(token_limits={}))
    configured = await check_generation(_client())

    assert unknown.passed is False
    assert unknown.error_code == "token_limits_required"
    assert unknown.facts["token_limits_known"] is False
    assert unknown.facts["structured_output_mode"] == "native_schema"
    assert configured.passed is True
    assert configured.facts["token_limits_known"] is True
    assert len(provider.requests) == 2


async def test_a_failed_request_is_reported_before_missing_token_limits(monkeypatch) -> None:
    Provider(monkeypatch, litellm.AuthenticationError("invalid api key", "openai", "check-model"))

    check = await check_generation(_client(token_limits={}))

    assert check.error_code == "authentication_failed"


async def test_a_generation_check_that_outlasts_its_deadline_times_out(monkeypatch) -> None:
    import asyncio

    async def never_answers(**_request: Any) -> Any:
        await asyncio.sleep(60)

    monkeypatch.setattr("memforge.llm.structured.litellm.acompletion", never_answers)
    monkeypatch.setattr("memforge.llm.structured.litellm.supports_response_schema", lambda **_kwargs: True)

    check = await check_generation(_client(timeout_s=0.05))

    assert check.error_code == "timeout"


async def test_a_generation_check_keeps_neither_the_prompt_nor_the_reply(monkeypatch, caplog) -> None:
    Provider(monkeypatch)

    with caplog.at_level("DEBUG"):
        check = await check_generation(_client())

    assert set(check.facts) == {
        "latency_ms",
        "token_limits_known",
        "structured_output_mode",
        "schema_fallback",
        "provider_model",
    }
    assert "generation-secret-key" not in caplog.text
    assert "alpha" not in repr(check)


# -- Embedding ----------------------------------------------------------------------


def _vectors(dimension: int, *, model: str | None = "check-embedding-v2"):
    """A provider answer of one vector of ``dimension`` numbers per input."""

    def respond(texts: list[str]) -> dict[str, Any]:
        return {
            "model": model,
            "data": [
                {"index": index, "embedding": [float(len(text))] + [0.5] * (dimension - 1)}
                for index, text in enumerate(texts)
            ],
        }

    return respond


class EmbeddingProvider:
    """Answer each embedding request from a script and record what was asked."""

    def __init__(self, *steps: Any) -> None:
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        step = self._steps.pop(0) if self._steps else _vectors(3)
        if isinstance(step, BaseException):
            raise step
        return step(list(request["input"])) if callable(step) else step


class Rejection(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider answered {status_code}")
        self.status_code = status_code


async def test_an_embedding_model_passes_after_two_requests_of_two_distinct_inputs() -> None:
    provider = EmbeddingProvider()

    check = await check_embedding(EMBEDDING, transport=provider)

    assert check.passed is True
    assert check.facts == {
        "dimension": 3,
        "provider_model": "check-embedding-v2",
        "latency_ms": check.facts["latency_ms"],
    }
    assert len(provider.requests) == 2
    for request in provider.requests:
        assert len(request["input"]) == 2
        assert request["input"][0] != request["input"][1]
        assert request["model"] == "openai/check-embedding"
        assert request["api_base"] == "https://embed.example.test/v1"
        assert request["api_key"] == "embedding-secret-key"
        assert request["max_retries"] == 0


@pytest.mark.parametrize(
    "response",
    [
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 0, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 2, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": [0.1, math.nan]}, {"index": 1, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": [0.1, math.inf]}, {"index": 1, "embedding": [0.3, 0.4]}]},
        {"data": [{"index": 0, "embedding": [0.1, 0.2]}, {"index": 1, "embedding": [0.3, 0.4, 0.5]}]},
    ],
    ids=["count", "duplicate_index", "index_out_of_range", "not_a_number", "infinite", "two_dimensions"],
)
async def test_a_response_without_one_valid_vector_per_input_fails(response) -> None:
    provider = EmbeddingProvider(response)

    check = await check_embedding(EMBEDDING, transport=provider)

    assert check.error_code == "embedding_response_invalid"
    assert len(provider.requests) == 1


async def test_out_of_order_items_are_matched_to_their_inputs() -> None:
    def reordered(texts: list[str]) -> dict[str, Any]:
        return {"data": list(reversed(_vectors(3)(texts)["data"]))}

    provider = EmbeddingProvider(reordered, reordered)

    check = await check_embedding(EMBEDDING, transport=provider)

    assert check.passed is True


async def test_a_dimension_that_changes_on_the_repeat_fails() -> None:
    provider = EmbeddingProvider(_vectors(3), _vectors(4))

    check = await check_embedding(EMBEDDING, transport=provider)

    assert check.error_code == "embedding_dimension_unstable"
    assert check.facts["dimension"] == 3


async def test_a_requested_dimension_is_sent_and_has_to_be_the_one_observed() -> None:
    requested = ResolvedInvocation(
        route=EMBEDDING.route,
        api_base=EMBEDDING.api_base,
        api_key=EMBEDDING.api_key,
        extra_params={"dimensions": 4},
    )
    matching = EmbeddingProvider(_vectors(4), _vectors(4))
    ignoring = EmbeddingProvider(_vectors(3))

    passed = await check_embedding(requested, transport=matching)
    failed = await check_embedding(requested, transport=ignoring)

    assert passed.passed is True
    assert passed.facts["dimension"] == 4
    assert matching.requests[0]["dimensions"] == 4
    assert failed.error_code == "embedding_dimension_mismatch"
    assert failed.facts["dimension"] == 3
    assert len(ignoring.requests) == 1


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (400, "request_rejected"),
        (401, "authentication_failed"),
        (403, "authentication_failed"),
        (404, "model_not_found"),
        (418, "provider_error"),
    ],
)
async def test_an_embedding_provider_rejection_is_reported_by_code_and_status(status_code, error_code) -> None:
    check = await check_embedding(EMBEDDING, transport=EmbeddingProvider(Rejection(status_code)))

    assert check.error_code == error_code
    assert check.facts["status_code"] == status_code
    assert "dimension" not in check.facts


async def test_a_dimension_the_route_does_not_take_is_reported_as_such_and_not_as_a_provider_refusal() -> None:
    """LiteLLM refuses the parameter itself, so no request is made."""
    check = await check_embedding(
        ResolvedInvocation(
            route="mistral/mistral-embed",
            api_base=EMBEDDING.api_base,
            api_key=EMBEDDING.api_key,
            extra_params={"dimensions": 4},
        )
    )

    assert check.error_code == "parameter_not_supported_by_route"
    assert "dimension" not in check.facts


async def test_an_unreachable_embedding_endpoint_is_reported_as_unreachable(monkeypatch) -> None:
    async def no_pause(_seconds: float) -> None:
        return None

    monkeypatch.setattr("memforge.llm.embedding.asyncio.sleep", no_pause)
    provider = EmbeddingProvider(ConnectionError("refused"), ConnectionError("refused"))

    check = await check_embedding(EMBEDDING, transport=provider)

    assert check.error_code == "provider_unreachable"
    assert "status_code" not in check.facts
    assert len(provider.requests) == 2


async def test_an_embedding_check_reaches_litellm_through_the_embedding_runtime(monkeypatch) -> None:
    provider = EmbeddingProvider(_vectors(3, model=None), _vectors(3, model=None))
    monkeypatch.setattr("litellm.aembedding", provider)

    check = await check_embedding(EMBEDDING)

    assert check.passed is True
    assert "provider_model" not in check.facts
    assert len(provider.requests) == 2
