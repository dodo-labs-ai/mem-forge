"""The Embedding Runtime: invocation, attempts, deadline, slicing, validation, cache."""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import pytest

from memforge.llm.embedding import (
    EmbeddingDeadlineExceeded,
    EmbeddingNotConfigured,
    EmbeddingProviderError,
    EmbeddingRuntime,
    EmbeddingSettings,
    InvalidEmbeddingResponse,
)
from memforge.llm.providers import ResolvedInvocation
from tests.embedding_fixture import embedding_response

_OLLAMA = ResolvedInvocation(route="ollama/embedding-model")


class ProviderFailure(Exception):
    def __init__(self, status_code: int, *, retry_after: str | None = None) -> None:
        super().__init__(f"provider answered {status_code}")
        self.status_code = status_code
        if retry_after is not None:
            self.litellm_response_headers = {"retry-after": retry_after}


class ScriptedProvider:
    """Answer each request from a script and record what was asked."""

    def __init__(self, *steps: Any) -> None:
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    @property
    def inputs(self) -> list[list[str]]:
        return [list(request["input"]) for request in self.requests]

    async def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        step = self._steps.pop(0) if self._steps else _text_length_vectors
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(list(request["input"]))
        return step


def _text_length_vectors(texts: list[str]) -> dict[str, Any]:
    return embedding_response([[float(len(text)), 1.0] for text in texts])


@pytest.fixture
def recorded_pauses(monkeypatch) -> list[float]:
    """Replace the pause between attempts with a record of how long it would be."""
    pauses: list[float] = []
    real_sleep = asyncio.sleep

    async def record_pause(seconds: float) -> None:
        pauses.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr("memforge.llm.embedding.asyncio.sleep", record_pause)
    return pauses


def _runtime(provider: ScriptedProvider, **settings: Any) -> EmbeddingRuntime:
    return EmbeddingRuntime(
        EmbeddingSettings(invocation=_OLLAMA, **settings),
        transport=provider,
    )


def test_a_runtime_is_configured_when_it_has_an_invocation() -> None:
    assert EmbeddingRuntime(EmbeddingSettings(invocation=_OLLAMA)).configured is True
    assert EmbeddingRuntime(EmbeddingSettings(invocation=None)).configured is False


def test_a_keyless_invocation_is_configured() -> None:
    keyless = ResolvedInvocation(
        route="openai/local-embedding",
        api_base="http://localhost:11434/v1",
        api_key="no-api-key",
    )

    assert EmbeddingRuntime(EmbeddingSettings(invocation=keyless)).configured is True


def test_settings_repr_does_not_show_the_api_key() -> None:
    settings = EmbeddingSettings(
        invocation=ResolvedInvocation(route="openai/bare-model", api_key="secret-key"),
    )

    assert "secret-key" not in repr(settings)


async def test_a_runtime_without_a_model_fails_before_any_request() -> None:
    provider = ScriptedProvider()
    runtime = EmbeddingRuntime(EmbeddingSettings(invocation=None), transport=provider)

    with pytest.raises(EmbeddingNotConfigured):
        await runtime.embed_query("alpha")

    assert provider.requests == []


async def test_a_request_carries_the_route_endpoint_key_and_extra_parameters_of_the_invocation() -> None:
    provider = ScriptedProvider()
    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=ResolvedInvocation(
                route="openai/BAAI/bge-m3",
                api_base="https://embed.test/v1",
                api_key="key",
                extra_params={"dimensions": 256},
            ),
        ),
        transport=provider,
    )

    await runtime.embed_query("alpha")

    request = provider.requests[0]
    assert request == {
        "model": "openai/BAAI/bge-m3",
        "input": ["alpha"],
        "timeout": request["timeout"],
        "max_retries": 0,
        "api_base": "https://embed.test/v1",
        "api_key": "key",
        "dimensions": 256,
        "allowed_openai_params": ["dimensions"],
    }


@pytest.mark.parametrize(
    ("route", "extra_params"),
    [
        ("openai/BAAI/bge-m3", {}),
        ("openai/BAAI/bge-m3", {"organization": "org-1"}),
        ("voyage/voyage-3", {"dimensions": 256}),
        ("ollama/nomic-embed-text", {"dimensions": 256}),
    ],
    ids=["no_dimension", "another_parameter", "voyage_route", "ollama_route"],
)
async def test_only_a_set_dimension_on_the_openai_route_is_named_for_the_sdk_to_send(route, extra_params) -> None:
    provider = ScriptedProvider()
    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=ResolvedInvocation(
                route=route,
                api_base="https://embed.test/v1",
                api_key="key",
                extra_params=extra_params,
            ),
        ),
        transport=provider,
    )

    await runtime.embed_query("alpha")

    assert "allowed_openai_params" not in provider.requests[0]


async def test_a_parameter_the_route_does_not_take_fails_without_a_request_and_says_so() -> None:
    """LiteLLM itself refuses the parameter before it sends anything."""
    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=ResolvedInvocation(
                route="mistral/mistral-embed",
                api_base="https://embed.test/v1",
                api_key="key",
                extra_params={"dimensions": 256},
            ),
        ),
    )

    with pytest.raises(EmbeddingProviderError) as raised:
        await runtime.embed_query("alpha")

    assert raised.value.parameter_refused is True
    assert raised.value.status_code is None
    assert raised.value.unreachable is False


async def test_an_invocation_without_endpoint_or_key_leaves_both_to_the_provider() -> None:
    provider = ScriptedProvider()

    await _runtime(provider).embed_query("alpha")

    request = provider.requests[0]
    assert request["model"] == "ollama/embedding-model"
    assert "api_base" not in request
    assert "api_key" not in request


async def test_the_runtime_reports_the_model_id_the_provider_answered_with() -> None:
    provider = ScriptedProvider({**embedding_response([[1.0, 0.0]]), "model": "embedding-model-2026"})
    runtime = _runtime(provider)

    assert runtime.provider_model is None
    await runtime.embed_query("alpha")

    assert runtime.provider_model == "embedding-model-2026"


async def test_out_of_order_response_items_return_in_input_order() -> None:
    provider = ScriptedProvider(
        {
            "data": [
                {"index": 2, "embedding": [3.0]},
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [2.0]},
            ]
        }
    )

    vectors = await _runtime(provider).embed_documents(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        pytest.param({}, "missing data", id="no_data"),
        pytest.param(embedding_response([[0.1]]), "expected 2, got 1", id="missing_item"),
        pytest.param(
            embedding_response([[0.1], [0.2], [0.3]]),
            "expected 2, got 3",
            id="extra_item",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"embedding": [0.2]}]},
            "indices are incomplete",
            id="missing_index",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}]},
            "indices do not match inputs",
            id="duplicate_index",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1}]},
            "missing embedding",
            id="missing_vector",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": []}]},
            "missing embedding",
            id="empty_vector",
        ),
        pytest.param(
            embedding_response([[0.1, math.nan], [0.2, 0.3]]),
            "non-finite",
            id="nan",
        ),
        pytest.param(
            embedding_response([[0.1, 0.2], [math.inf, 0.3]]),
            "non-finite",
            id="infinity",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": ["x"]}]},
            "non-numeric",
            id="non_numeric",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": ["0.2"]}]},
            "non-numeric",
            id="numeric_string",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [True]}]},
            "non-numeric",
            id="boolean",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [None]}]},
            "non-numeric",
            id="null",
        ),
        pytest.param(
            {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [10**400]}]},
            "non-finite",
            id="integer_beyond_float_range",
        ),
        pytest.param(
            embedding_response([[0.1, 0.2], [0.3]]),
            "do not share one dimension",
            id="mixed_dimension",
        ),
    ],
)
async def test_an_invalid_response_is_rejected_without_another_attempt(response, message) -> None:
    provider = ScriptedProvider(response)
    runtime = _runtime(provider)

    with pytest.raises(InvalidEmbeddingResponse, match=message):
        await runtime.embed_documents(["alpha", "beta"])

    assert len(provider.requests) == 1


async def test_whole_number_values_are_returned_as_floats() -> None:
    provider = ScriptedProvider({"data": [{"index": 0, "embedding": [1, 0, -2]}]})

    vector = await _runtime(provider).embed_query("alpha")

    assert vector == [1.0, 0.0, -2.0]
    assert all(type(value) is float for value in vector)


async def test_a_later_response_with_another_dimension_is_rejected() -> None:
    provider = ScriptedProvider(
        embedding_response([[0.1, 0.2]]),
        embedding_response([[0.1, 0.2, 0.3]]),
    )
    runtime = _runtime(provider)

    assert await runtime.embed_query("alpha") == [0.1, 0.2]
    assert runtime.dimension == 2
    with pytest.raises(InvalidEmbeddingResponse, match="dimension 3 does not match the dimension 2"):
        await runtime.embed_query("beta")

    assert runtime.dimension == 2


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(ProviderFailure(408), id="request_timeout"),
        pytest.param(ProviderFailure(409), id="conflict"),
        pytest.param(ProviderFailure(429), id="rate_limited"),
        pytest.param(ProviderFailure(503), id="server_error"),
        pytest.param(TimeoutError("timed out"), id="timeout"),
        pytest.param(ConnectionError("connection reset"), id="connection"),
    ],
)
async def test_a_retryable_failure_is_attempted_again(recorded_pauses, failure) -> None:
    provider = ScriptedProvider(failure)

    vectors = await _runtime(provider).embed_documents(["alpha"])

    assert vectors == [[5.0, 1.0]]
    assert provider.inputs == [["alpha"], ["alpha"]]
    assert [request["max_retries"] for request in provider.requests] == [0, 0]
    assert recorded_pauses == [2.0]


async def test_retries_pause_long_enough_to_outlast_a_short_outage(recorded_pauses) -> None:
    provider = ScriptedProvider(*[ProviderFailure(503) for _ in range(3)])

    vectors = await _runtime(provider).embed_documents(["alpha"])

    assert vectors == [[5.0, 1.0]]
    assert len(provider.requests) == 4
    assert recorded_pauses == [2.0, 4.0, 8.0]


async def test_the_last_pause_serves_every_later_retry(recorded_pauses) -> None:
    provider = ScriptedProvider(*[ProviderFailure(503) for _ in range(4)])

    await _runtime(provider, num_retries=4, retry_pauses_s=(0.5, 1.5)).embed_documents(["alpha"])

    assert recorded_pauses == [0.5, 1.5, 1.5, 1.5]


async def test_a_retry_waits_as_long_as_the_provider_asks(recorded_pauses) -> None:
    provider = ScriptedProvider(
        ProviderFailure(429, retry_after="3"),
        ProviderFailure(429, retry_after="1"),
        ProviderFailure(429, retry_after="soon"),
    )

    vectors = await _runtime(provider).embed_documents(["alpha"])

    assert vectors == [[5.0, 1.0]]
    assert recorded_pauses == [3.0, 4.0, 8.0]


async def test_a_pause_that_outlasts_the_deadline_fails_at_once_with_the_provider_error(
    recorded_pauses,
) -> None:
    provider = ScriptedProvider(ProviderFailure(429, retry_after="120"))

    with pytest.raises(EmbeddingProviderError) as raised:
        await _runtime(provider, timeout_s=60.0).embed_documents(["alpha"])

    assert raised.value.status_code == 429
    assert len(provider.requests) == 1
    assert recorded_pauses == []


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
async def test_a_non_retryable_failure_is_not_attempted_again(status_code) -> None:
    provider = ScriptedProvider(ProviderFailure(status_code))

    with pytest.raises(EmbeddingProviderError) as raised:
        await _runtime(provider).embed_documents(["alpha"])

    assert len(provider.requests) == 1
    assert raised.value.status_code == status_code
    assert isinstance(raised.value.__cause__, ProviderFailure)


async def test_retries_stop_when_the_budget_is_spent(recorded_pauses) -> None:
    provider = ScriptedProvider(*[ProviderFailure(503) for _ in range(5)])

    with pytest.raises(EmbeddingProviderError, match="Embedding provider request failed"):
        await _runtime(provider, num_retries=2).embed_documents(["alpha"])

    assert len(provider.requests) == 3


async def test_one_retry_budget_covers_every_slice(recorded_pauses) -> None:
    provider = ScriptedProvider(
        ProviderFailure(503),
        _text_length_vectors,
        ProviderFailure(503),
    )

    with pytest.raises(EmbeddingProviderError):
        await _runtime(provider, num_retries=1, max_inputs_per_request=1).embed_documents(["a", "bb"])

    assert provider.inputs == [["a"], ["a"], ["bb"]]


async def test_a_connection_failure_reads_as_an_unreachable_provider() -> None:
    provider = ScriptedProvider(ConnectionError("[Errno 111] Connection refused"))

    with pytest.raises(EmbeddingProviderError, match="Embedding provider unreachable: .*Connection refused"):
        await _runtime(provider, num_retries=0).embed_query("alpha")


async def test_each_attempt_gets_only_what_is_left_of_the_deadline() -> None:
    provider = ScriptedProvider(*[TimeoutError("timed out") for _ in range(50)])

    with pytest.raises(EmbeddingProviderError, match="TimeoutError"):
        await _runtime(
            provider,
            timeout_s=0.4,
            num_retries=50,
            retry_pauses_s=(0.05,),
        ).embed_documents(["alpha"])

    timeouts = [request["timeout"] for request in provider.requests]
    assert 2 <= len(timeouts) <= 9
    assert all(0 < timeout <= 0.4 for timeout in timeouts)
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[1] <= timeouts[0] - 0.05


async def test_a_deadline_that_passes_during_an_attempt_ends_the_call() -> None:
    async def slow_then_fails(**_request: Any) -> Any:
        await asyncio.sleep(0.03)
        raise TimeoutError("timed out")

    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=_OLLAMA,
            timeout_s=0.1,
            num_retries=50,
            retry_pauses_s=(0.0,),
        ),
        transport=slow_then_fails,
    )

    with pytest.raises(EmbeddingDeadlineExceeded, match="after 0.1s"):
        await runtime.embed_documents(["alpha"])


async def test_a_retryable_failure_that_ends_after_the_deadline_ends_the_call_at_the_deadline() -> None:
    async def fails_after_the_deadline(**_request: Any) -> Any:
        # Holding the event loop keeps the deadline from interrupting the
        # attempt, so the attempt itself ends after the deadline.
        time.sleep(0.1)
        raise TimeoutError("timed out")

    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=_OLLAMA,
            timeout_s=0.05,
            num_retries=3,
            retry_pauses_s=(0.0,),
        ),
        transport=fails_after_the_deadline,
    )

    with pytest.raises(EmbeddingDeadlineExceeded, match="after 0.05s"):
        await runtime.embed_documents(["alpha"])


async def test_a_request_that_never_answers_ends_at_the_deadline() -> None:
    async def never_answers(**_request: Any) -> Any:
        await asyncio.sleep(60)

    runtime = EmbeddingRuntime(
        EmbeddingSettings(invocation=_OLLAMA, timeout_s=0.05),
        transport=never_answers,
    )
    started = asyncio.get_running_loop().time()

    with pytest.raises(EmbeddingDeadlineExceeded):
        await runtime.embed_query("alpha")

    assert asyncio.get_running_loop().time() - started < 5


async def test_one_request_carries_at_most_32_inputs_by_default() -> None:
    provider = ScriptedProvider()

    await _runtime(provider).embed_documents([f"text-{index}" for index in range(70)])

    assert [len(inputs) for inputs in provider.inputs] == [32, 32, 6]


async def test_a_batch_is_sliced_by_the_request_item_limit() -> None:
    provider = ScriptedProvider()

    vectors = await _runtime(provider, max_inputs_per_request=2).embed_documents(
        ["a", "bb", "ccc", "dddd", "eeeee"]
    )

    assert provider.inputs == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]]


async def test_a_failed_slice_fails_the_whole_batch_and_caches_nothing() -> None:
    provider = ScriptedProvider(_text_length_vectors, ProviderFailure(400))
    runtime = _runtime(provider, max_inputs_per_request=2)

    with pytest.raises(EmbeddingProviderError):
        await runtime.embed_documents(["a", "bb", "ccc"])

    assert provider.inputs == [["a", "bb"], ["ccc"]]

    vectors = await runtime.embed_documents(["a", "bb", "ccc"])

    assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    assert provider.inputs[2:] == [["a", "bb"], ["ccc"]]


async def test_an_invalid_slice_fails_the_whole_batch_and_caches_nothing() -> None:
    provider = ScriptedProvider(_text_length_vectors, embedding_response([[math.nan, 1.0]]))
    runtime = _runtime(provider, max_inputs_per_request=2)

    with pytest.raises(InvalidEmbeddingResponse):
        await runtime.embed_documents(["a", "bb", "ccc"])

    await runtime.embed_query("a")

    assert provider.inputs[2:] == [["a"]]


async def test_a_text_is_embedded_once_for_queries_and_documents() -> None:
    provider = ScriptedProvider()
    runtime = _runtime(provider)

    first = await runtime.embed_documents(["alpha", "beta"])
    second = await runtime.embed_query("alpha")
    third = await runtime.embed_documents(["beta", "gamma"])

    assert second == first[0]
    assert third[0] == first[1]
    assert provider.inputs == [["alpha", "beta"], ["gamma"]]


async def test_a_repeated_text_in_one_batch_is_sent_once() -> None:
    provider = ScriptedProvider()

    vectors = await _runtime(provider).embed_documents(["alpha", "be", "alpha"])

    assert provider.inputs == [["alpha", "be"]]
    assert vectors == [[5.0, 1.0], [2.0, 1.0], [5.0, 1.0]]


async def test_a_caller_cannot_change_a_cached_vector() -> None:
    runtime = _runtime(ScriptedProvider())

    first = await runtime.embed_query("alpha")
    first[0] = 99.0

    assert await runtime.embed_query("alpha") == [5.0, 1.0]


async def test_the_cache_keeps_the_most_recently_used_vectors() -> None:
    provider = ScriptedProvider()
    runtime = _runtime(provider, cache_size=2)

    await runtime.embed_query("a")
    await runtime.embed_query("bb")
    await runtime.embed_query("a")
    await runtime.embed_query("ccc")
    await runtime.embed_query("a")
    await runtime.embed_query("bb")

    assert provider.inputs == [["a"], ["bb"], ["ccc"], ["bb"]]


async def test_a_runtime_never_serves_another_runtimes_vectors() -> None:
    small = ScriptedProvider(embedding_response([[1.0, 0.0]]))
    large = ScriptedProvider(embedding_response([[0.0, 1.0, 0.0]]))
    small_runtime = EmbeddingRuntime(
        EmbeddingSettings(invocation=ResolvedInvocation(route="ollama/small")),
        transport=small,
    )
    large_runtime = EmbeddingRuntime(
        EmbeddingSettings(invocation=ResolvedInvocation(route="ollama/large")),
        transport=large,
    )

    assert await small_runtime.embed_query("alpha") == [1.0, 0.0]
    assert await large_runtime.embed_query("alpha") == [0.0, 1.0, 0.0]
    assert await small_runtime.embed_query("alpha") == [1.0, 0.0]

    assert small.inputs == [["alpha"]]
    assert large.inputs == [["alpha"]]


async def test_an_empty_batch_needs_no_request() -> None:
    provider = ScriptedProvider()

    assert await _runtime(provider).embed_documents([]) == []
    assert provider.requests == []
