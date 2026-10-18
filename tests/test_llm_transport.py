"""Transport policy shared by generation and embedding."""

from __future__ import annotations

import asyncio
import json

import httpx
import litellm
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from memforge.llm.bindings import build_binding, compile_binding
from memforge.llm.embedding import EmbeddingRuntime, EmbeddingSettings
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
from memforge.llm.transport import (
    end_provider_client_exchanges_at_their_timeout,
    install_no_redirect_http_client,
    is_retryable_provider_error,
    keep_ollama_model_lookups_off_request_endpoints,
    provider_retry_after_seconds,
    provider_status_code,
    refuse_redirects_in_provider_clients,
    send_within_timeout,
)
from tests.model_endpoint_fixture import InMemorySharedClient, OllamaLookups

# The time a call that is not about its deadline is given. The first request a
# process sends through a provider's SDK also imports that SDK's modules, which
# on a busy machine takes seconds of the call's own time.
_UNHURRIED_CALL_S = 60.0


class _StatusError(Exception):
    def __init__(self, status_code: object) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_importing_a_model_client_installs_a_client_that_refuses_redirects() -> None:
    import memforge.llm.structured  # noqa: F401

    session = litellm.aclient_session

    assert isinstance(session, httpx.AsyncClient)
    assert session.follow_redirects is False


def _redirect_elsewhere(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(307, headers={"location": "https://elsewhere.example.test/v1/embeddings"})


async def test_the_shared_client_returns_a_redirect_even_when_asked_to_follow_it() -> None:
    client = InMemorySharedClient(_redirect_elsewhere)

    response = await client.post(
        "https://models.example.test/v1/embeddings",
        json={"input": ["alpha"]},
        follow_redirects=True,
    )
    await client.aclose()

    assert response.status_code == 307
    assert [str(request.url) for request in client.requests] == ["https://models.example.test/v1/embeddings"]


def test_the_shared_client_keeps_one_connection_pool_per_event_loop() -> None:
    client = InMemorySharedClient(lambda _request: httpx.Response(200, json={"ok": True}))

    async def twice() -> list[int]:
        first = await client.get("https://models.example.test/v1/models")
        second = await client.get("https://models.example.test/v1/models")
        return [first.status_code, second.status_code]

    assert asyncio.run(twice()) == [200, 200]
    assert client.pools_opened == 1
    assert asyncio.run(twice()) == [200, 200]
    assert client.pools_opened == 2
    assert len(client._pools) == 1


def test_install_replaces_a_client_that_follows_redirects(monkeypatch) -> None:
    following = httpx.AsyncClient(follow_redirects=True)
    monkeypatch.setattr(litellm, "aclient_session", following)

    installed = install_no_redirect_http_client()

    assert installed is litellm.aclient_session
    assert installed is not following
    assert installed.follow_redirects is False
    asyncio.run(following.aclose())
    asyncio.run(installed.aclose())


def test_install_keeps_a_client_that_already_refuses_redirects(monkeypatch) -> None:
    refusing = httpx.AsyncClient(follow_redirects=False)
    monkeypatch.setattr(litellm, "aclient_session", refusing)

    assert install_no_redirect_http_client() is refusing
    assert litellm.aclient_session is refusing
    asyncio.run(refusing.aclose())


def test_install_sets_a_client_when_none_is_set(monkeypatch) -> None:
    monkeypatch.setattr(litellm, "aclient_session", None)

    installed = install_no_redirect_http_client()

    assert litellm.aclient_session is installed
    assert installed.follow_redirects is False
    asyncio.run(installed.aclose())


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (TimeoutError(), True),
        (ConnectionError(), True),
        (_StatusError(408), True),
        (_StatusError(409), True),
        (_StatusError(429), True),
        (_StatusError(500), True),
        (_StatusError("503"), True),
        (_StatusError(400), False),
        (_StatusError(401), False),
        (_StatusError(404), False),
        (_StatusError(None), False),
        (_StatusError("unknown"), False),
        (ValueError("bad request"), False),
    ],
)
def test_only_transient_provider_failures_are_retryable(error, retryable) -> None:
    assert is_retryable_provider_error(error) is retryable


def _answered(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://models.example.test/api/embed")
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=request,
        response=httpx.Response(status_code, request=request),
    )


def _connection_error(cause: BaseException | None) -> litellm.APIConnectionError:
    error = litellm.APIConnectionError(message="request failed", llm_provider="ollama", model="m")
    error.__cause__ = cause
    return error


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (401, False), (404, False), (307, False), (429, True), (503, True)],
)
def test_an_answer_reported_as_a_connection_error_is_judged_by_its_own_status(
    status_code, retryable
) -> None:
    error = _connection_error(_answered(status_code))

    assert provider_status_code(error) == status_code
    assert is_retryable_provider_error(error) is retryable


def test_an_answer_is_found_under_the_provider_error_that_wraps_it() -> None:
    provider_error = RuntimeError("provider rejected the request")
    provider_error.__context__ = _answered(401)

    assert provider_status_code(_connection_error(provider_error)) == 401


def test_a_connection_error_without_an_answer_stays_retryable() -> None:
    error = _connection_error(httpx.ConnectError("connection refused"))

    assert provider_status_code(error) == 500
    assert is_retryable_provider_error(error) is True


@pytest.mark.parametrize(
    ("headers", "seconds"),
    [
        ({"retry-after": "3"}, 3.0),
        (httpx.Headers({"Retry-After": "1.5"}), 1.5),
        ({"retry-after": "0"}, 0.0),
        ({"retry-after": "-1"}, None),
        ({"retry-after": "nan"}, None),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_retry_after_is_read_in_seconds_from_the_failed_response(headers, seconds) -> None:
    error = _StatusError(429)
    error.litellm_response_headers = headers

    assert provider_retry_after_seconds(error) == seconds


def test_an_error_without_response_headers_names_no_retry_after() -> None:
    assert provider_retry_after_seconds(_StatusError(429)) is None


def test_the_sdk_default_retry_count_is_zero() -> None:
    assert litellm.DEFAULT_MAX_RETRIES == 0


def test_the_sdk_help_notice_is_silenced() -> None:
    assert litellm.suppress_debug_info is True


def _endpoint_runtime(base_url: str, *, api_key: str | None = "endpoint-key") -> EmbeddingRuntime:
    """A runtime for an OpenAI-compatible endpoint, with a key or with the none auth strategy."""
    binding = build_binding(
        kind="embedding",
        provider="openai_compatible",
        model_id="text-embedding-test",
        endpoint=base_url,
        auth_strategy="api_key" if api_key else "none",
        secret_ref="secret:test" if api_key else None,
        config_source="settings",
    )
    return EmbeddingRuntime(
        EmbeddingSettings(invocation=compile_binding(binding, api_key), num_retries=0)
    )


def _embeddings(request: httpx.Request) -> httpx.Response:
    inputs = json.loads(request.content)["input"]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": "text-embedding-test",
            "data": [
                {"object": "embedding", "index": index, "embedding": [float(len(text)), 0.5]}
                for index, text in enumerate(inputs)
            ],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        },
    )


@pytest.fixture
async def endpoint(monkeypatch):
    """An in-memory OpenAI-compatible embeddings endpoint behind LiteLLM's HTTP client."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    client = InMemorySharedClient(_embeddings)
    monkeypatch.setattr(litellm, "aclient_session", client)
    yield client
    # LiteLLM records each call on a queue bound to this test's event loop.
    await GLOBAL_LOGGING_WORKER.flush()
    await client.aclose()


async def test_a_keyless_endpoint_is_never_sent_a_key_from_the_environment(monkeypatch, endpoint) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    await _endpoint_runtime("https://embed-keyless.example.test/v1", api_key=None).embed_query("alpha")

    assert endpoint.requests[0].headers["authorization"] == "Bearer no-api-key"


def test_a_client_litellm_builds_for_a_provider_refuses_redirects() -> None:
    handler = AsyncHTTPHandler()

    assert handler.client.follow_redirects is False
    asyncio.run(handler.close())


def test_refusing_redirects_in_provider_clients_wraps_the_builder_once() -> None:
    builder = AsyncHTTPHandler.create_client

    refuse_redirects_in_provider_clients()

    assert AsyncHTTPHandler.create_client is builder


@pytest.fixture
async def ollama_endpoint(monkeypatch):
    """An in-memory Ollama server behind LiteLLM, and the model lookups LiteLLM makes."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    requests: list[httpx.Request] = []

    def chat(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "private-chat-model",
                "message": {"role": "assistant", "content": json.dumps({"ranked_ids": []})},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )

    lookups = OllamaLookups()
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_options: httpx.MockTransport(chat)),
    )
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(litellm, "module_level_client", lookups)
    yield requests, lookups
    await GLOBAL_LOGGING_WORKER.flush()


async def test_a_keyless_ollama_endpoint_is_never_sent_the_ollama_key_of_the_environment(
    monkeypatch, ollama_endpoint
) -> None:
    """LiteLLM looks the model up once more after the request, at the address the request went to."""
    requests, lookups = ollama_endpoint
    monkeypatch.setenv("OLLAMA_API_KEY", "environment-ollama-key")
    monkeypatch.setenv("OLLAMA_API_BASE", "https://ollama-environment.example.test")
    getattr(litellm.get_model_info, "cache_clear", lambda: None)()
    loop = asyncio.get_running_loop()
    looked_up_after_the_request = asyncio.Event()
    look_up = litellm.OllamaConfig.get_model_info

    def watched_look_up(self, model, api_base=None):
        if api_base is not None:
            loop.call_soon_threadsafe(looked_up_after_the_request.set)
        return look_up(self, model, api_base)

    monkeypatch.setattr(litellm.OllamaConfig, "get_model_info", watched_look_up)
    binding = build_binding(
        kind="generation",
        provider="ollama",
        model_id="private-chat-model",
        endpoint="https://ollama-keyless.example.test",
        auth_strategy="none",
        config_source="settings",
    )
    client = LiteLlmStructuredClient(
        StructuredLlmConfig.for_invocation(compile_binding(binding, None), timeout_s=_UNHURRIED_CALL_S)
    )

    await client.rerank_memories("Rank the matching memories")
    await asyncio.wait_for(looked_up_after_the_request.wait(), timeout=10)

    assert {request.url.host for request in requests} == {"ollama-keyless.example.test"}
    assert {request.headers.get("authorization") for request in requests} == {"Bearer no-api-key"}
    assert lookups.requests
    assert {httpx.URL(url).host for url, _headers in lookups.requests} == {"ollama-environment.example.test"}


def test_an_ollama_lookup_at_a_request_endpoint_reports_a_model_with_no_known_limits(monkeypatch) -> None:
    lookups = OllamaLookups()
    monkeypatch.setattr(litellm, "module_level_client", lookups)
    monkeypatch.setenv("OLLAMA_API_KEY", "environment-ollama-key")

    info = litellm.OllamaConfig().get_model_info(
        "ollama_chat/private-chat-model", api_base="https://ollama-keyless.example.test/api/chat"
    )

    assert lookups.requests == []
    assert (info["key"], info["max_input_tokens"], info["input_cost_per_token"]) == (
        "ollama_chat/private-chat-model",
        None,
        0.0,
    )


def test_keeping_ollama_lookups_off_request_endpoints_wraps_the_lookup_once() -> None:
    lookup = litellm.OllamaConfig.get_model_info

    keep_ollama_model_lookups_off_request_endpoints()

    assert litellm.OllamaConfig.get_model_info is lookup


# -- Request timeouts ---------------------------------------------------------------


async def test_an_exchange_that_outlasts_its_timeout_fails_as_a_read_timeout() -> None:
    request = httpx.Request("POST", "https://models.example.test/v1/chat/completions")
    request.extensions["timeout"] = httpx.Timeout(0.05).as_dict()

    async def never_answers(_request: httpx.Request, **_options) -> httpx.Response:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    with pytest.raises(httpx.ReadTimeout) as raised:
        await send_within_timeout(never_answers, request)

    assert raised.value.request is request


async def test_a_request_without_a_timeout_is_sent_unbounded() -> None:
    request = httpx.Request("GET", "https://models.example.test/v1/models")
    request.extensions["timeout"] = httpx.Timeout(None).as_dict()

    async def answers(_request: httpx.Request, **_options) -> httpx.Response:
        return httpx.Response(200)

    assert (await send_within_timeout(answers, request)).status_code == 200


async def test_a_timeout_the_send_raises_itself_is_left_as_it_is() -> None:
    request = httpx.Request("GET", "https://models.example.test/v1/models")
    request.extensions["timeout"] = httpx.Timeout(5.0).as_dict()
    own = TimeoutError("the transport's own")

    async def raises(_request: httpx.Request, **_options) -> httpx.Response:
        raise own

    with pytest.raises(TimeoutError) as raised:
        await send_within_timeout(raises, request)

    assert raised.value is own


def test_ending_provider_client_exchanges_at_their_timeout_wraps_the_builder_once() -> None:
    builder = AsyncHTTPHandler.create_client

    end_provider_client_exchanges_at_their_timeout()

    assert AsyncHTTPHandler.create_client is builder
