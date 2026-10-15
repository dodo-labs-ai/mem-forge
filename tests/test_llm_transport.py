"""Transport policy shared by generation and embedding."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import litellm
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from pydantic import ConfigDict

from memforge.llm.bindings import build_binding, compile_binding
from memforge.llm.embedding import (
    EmbeddingDeadlineExceeded,
    EmbeddingProviderError,
    EmbeddingRuntime,
    EmbeddingSettings,
)
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
    StructuredResponseModel,
    failure_retryable,
)
from memforge.llm.transport import (
    NoRedirectAsyncClient,
    end_provider_client_exchanges_at_their_timeout,
    install_no_redirect_http_client,
    is_retryable_provider_error,
    keep_ollama_model_lookups_off_request_endpoints,
    provider_retry_after_seconds,
    provider_status_code,
    refuse_redirects_in_provider_clients,
    send_within_timeout,
)
from tests.stalling_endpoint_fixture import GIVES_UP_AFTER_S, STALL_MODES, stalling_endpoint


class _StatusError(Exception):
    def __init__(self, status_code: object) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_importing_a_model_client_installs_a_client_that_refuses_redirects() -> None:
    import memforge.llm.structured  # noqa: F401

    session = litellm.aclient_session

    assert isinstance(session, httpx.AsyncClient)
    assert session.follow_redirects is False


class _InMemoryClient(NoRedirectAsyncClient):
    """The shared client with each connection pool replaced by an in-memory server."""

    def __init__(self, respond) -> None:
        super().__init__()
        self.respond = respond
        self.requests: list[httpx.Request] = []
        self.pools_opened = 0

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def _new_pool(self) -> httpx.AsyncClient:
        self.pools_opened += 1
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handle), follow_redirects=False)


def _redirect_elsewhere(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(307, headers={"location": "https://elsewhere.example.test/v1/embeddings"})


async def test_the_shared_client_returns_a_redirect_even_when_asked_to_follow_it() -> None:
    client = _InMemoryClient(_redirect_elsewhere)

    response = await client.post(
        "https://models.example.test/v1/embeddings",
        json={"input": ["alpha"]},
        follow_redirects=True,
    )
    await client.aclose()

    assert response.status_code == 307
    assert [str(request.url) for request in client.requests] == ["https://models.example.test/v1/embeddings"]


def test_the_shared_client_keeps_one_connection_pool_per_event_loop() -> None:
    client = _InMemoryClient(lambda _request: httpx.Response(200, json={"ok": True}))

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

    client = _InMemoryClient(_embeddings)
    monkeypatch.setattr(litellm, "aclient_session", client)
    yield client
    # LiteLLM records each call on a queue bound to this test's event loop.
    await GLOBAL_LOGGING_WORKER.flush()
    await client.aclose()


async def test_an_openai_compatible_model_reaches_the_configured_endpoint_through_litellm(endpoint) -> None:
    vectors = await _endpoint_runtime("https://embed-ok.example.test/v1").embed_documents(["alpha", "be"])

    assert vectors == [[5.0, 0.5], [2.0, 0.5]]
    assert [str(request.url) for request in endpoint.requests] == [
        "https://embed-ok.example.test/v1/embeddings"
    ]
    assert endpoint.requests[0].headers["authorization"] == "Bearer endpoint-key"


async def test_a_keyless_endpoint_is_never_sent_a_key_from_the_environment(monkeypatch, endpoint) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")

    await _endpoint_runtime("https://embed-keyless.example.test/v1", api_key=None).embed_query("alpha")

    assert endpoint.requests[0].headers["authorization"] == "Bearer no-api-key"


async def test_a_set_dimension_reaches_an_openai_compatible_endpoint_whatever_the_model_is_called(
    endpoint,
) -> None:
    binding = build_binding(
        kind="embedding",
        provider="openai_compatible",
        model_id="BAAI/bge-m3",
        endpoint="https://embed-sized.example.test/v1",
        parameters={"dimensions": 2},
        secret_ref="secret:test",
        config_source="settings",
    )
    runtime = EmbeddingRuntime(
        EmbeddingSettings(invocation=compile_binding(binding, "endpoint-key"), num_retries=0)
    )

    await runtime.embed_query("alpha")

    sent = json.loads(endpoint.requests[0].content)
    assert (sent["model"], sent["dimensions"]) == ("BAAI/bge-m3", 2)
    assert "allowed_openai_params" not in sent


async def test_one_attempt_is_one_request_because_the_sdk_does_not_retry(endpoint) -> None:
    endpoint.respond = lambda _request: httpx.Response(503, json={"error": {"message": "busy"}})

    with pytest.raises(EmbeddingProviderError) as raised:
        await _endpoint_runtime("https://embed-busy.example.test/v1").embed_query("alpha")

    assert raised.value.status_code == 503
    assert len(endpoint.requests) == 1


async def test_a_redirecting_endpoint_fails_and_the_request_is_not_sent_on(endpoint) -> None:
    endpoint.respond = _redirect_elsewhere

    with pytest.raises(EmbeddingProviderError):
        await _endpoint_runtime("https://embed-moved.example.test/v1").embed_query("alpha")

    assert [request.url.host for request in endpoint.requests] == ["embed-moved.example.test"]


@pytest.fixture
async def provider_endpoint(monkeypatch):
    """An in-memory endpoint behind the HTTP clients LiteLLM builds for a provider."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    requests: list[httpx.Request] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": f"https://elsewhere.example.test{request.url.path}"},
        )

    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_options: httpx.MockTransport(redirect)),
    )
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    yield requests
    await GLOBAL_LOGGING_WORKER.flush()


def test_a_client_litellm_builds_for_a_provider_refuses_redirects() -> None:
    handler = AsyncHTTPHandler()

    assert handler.client.follow_redirects is False
    asyncio.run(handler.close())


def test_refusing_redirects_in_provider_clients_wraps_the_builder_once() -> None:
    builder = AsyncHTTPHandler.create_client

    refuse_redirects_in_provider_clients()

    assert AsyncHTTPHandler.create_client is builder


async def test_a_bare_generation_model_does_not_send_its_key_and_prompt_on_to_a_redirect_target(
    provider_endpoint,
) -> None:
    client = LiteLlmStructuredClient(
        StructuredLlmConfig(
            model="claude-sonnet-4-20250514",
            base_url="https://generate-moved.example.test",
            api_key="endpoint-key",
            timeout_s=5.0,
        )
    )

    with pytest.raises(StructuredLlmError):
        await client.rerank_memories("Rank the matching memories")

    assert provider_endpoint
    assert {request.url.host for request in provider_endpoint} == {"generate-moved.example.test"}
    assert provider_endpoint[0].headers["x-api-key"] == "endpoint-key"


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("ollama/nomic-embed-text", "https://embed-moved.example.test"),
        ("voyage/voyage-3", "https://embed-moved.example.test/v1"),
        ("openrouter/openai/text-embedding-3-small", "https://embed-moved.example.test/v1"),
    ],
)
async def test_a_provider_route_embedding_is_not_sent_on_to_a_redirect_target(
    provider_endpoint, model, base_url
) -> None:
    runtime = EmbeddingRuntime(
        EmbeddingSettings(
            invocation=ResolvedInvocation(route=model, api_base=base_url, api_key="endpoint-key"),
            num_retries=3,
        )
    )

    with pytest.raises(EmbeddingProviderError) as raised:
        await runtime.embed_query("alpha")

    assert raised.value.status_code == 307
    assert [request.url.host for request in provider_endpoint] == ["embed-moved.example.test"]


class _OllamaLookups:
    """Stand in for the client LiteLLM asks an Ollama server about a model with.

    It records each question and answers none, as when no server is listening.
    """

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, str]]] = []

    def post(self, url: str, json=None, headers=None, **_options) -> httpx.Response:
        self.requests.append((url, dict(headers or {})))
        raise httpx.ConnectError("connection refused")


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

    lookups = _OllamaLookups()
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
        StructuredLlmConfig.for_invocation(compile_binding(binding, None), timeout_s=5.0)
    )

    await client.rerank_memories("Rank the matching memories")
    await asyncio.wait_for(looked_up_after_the_request.wait(), timeout=10)

    assert {request.url.host for request in requests} == {"ollama-keyless.example.test"}
    assert {request.headers.get("authorization") for request in requests} == {"Bearer no-api-key"}
    assert lookups.requests
    assert {httpx.URL(url).host for url, _headers in lookups.requests} == {"ollama-environment.example.test"}


def test_an_ollama_lookup_at_a_request_endpoint_reports_a_model_with_no_known_limits(monkeypatch) -> None:
    lookups = _OllamaLookups()
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


class _RankingAnswer(StructuredResponseModel):
    """An answer the model may reason toward: a schema that is not a direct answer."""

    model_config = ConfigDict(extra="ignore")

    ranking: list[int] = []


def _chat_reply(request: httpx.Request) -> httpx.Response:
    """Answer a chat request in the wire shape of the provider it was sent to."""
    answer = json.dumps({"ranking": []})
    if request.url.path.endswith("/v1/messages"):
        return httpx.Response(200, json={
            "id": "msg-1", "type": "message", "role": "assistant", "model": "chat-model",
            "content": [{"type": "text", "text": answer}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
    if request.url.path.endswith("/api/chat"):
        return httpx.Response(200, json={
            "model": "chat-model", "message": {"role": "assistant", "content": answer},
            "done": True, "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1,
        })
    return httpx.Response(200, json={
        "id": "chat-1", "object": "chat.completion", "created": 1, "model": "chat-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


@pytest.fixture
async def chat_endpoint(monkeypatch):
    """An in-memory chat endpoint behind every HTTP client LiteLLM sends generation requests through."""
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _chat_reply(request)

    shared = _InMemoryClient(respond)
    monkeypatch.setattr(litellm, "aclient_session", shared)
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_options: httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(litellm, "module_level_client", _OllamaLookups())
    yield requests
    await GLOBAL_LOGGING_WORKER.flush()
    await shared.aclose()


@pytest.mark.parametrize(
    ("provider", "model_id", "endpoint_url", "reasoning_off"),
    [
        ("openrouter", "anthropic/claude-haiku-5.5", "https://router.example.test/api/v1",
         {"reasoning": {"enabled": False}}),
        ("ollama", "qwen3", "https://ollama-chat.example.test", {"think": False}),
        ("openai", "gpt-5-mini", "https://openai-chat.example.test/v1", {"reasoning_effort": "minimal"}),
        ("anthropic", "claude-haiku-4-5", "https://anthropic-chat.example.test", {}),
    ],
)
async def test_only_a_direct_answer_asks_the_provider_to_answer_without_reasoning(
    chat_endpoint, provider, model_id, endpoint_url, reasoning_off
) -> None:
    """LiteLLM writes the descriptor's reasoning-off parameters into the provider's request body."""
    keyless = provider == "ollama"
    binding = build_binding(
        kind="generation",
        provider=provider,
        model_id=model_id,
        endpoint=endpoint_url,
        auth_strategy="none" if keyless else "api_key",
        secret_ref=None if keyless else "secret:test",
        config_source="settings",
    )
    client = LiteLlmStructuredClient(
        StructuredLlmConfig.for_invocation(compile_binding(binding, None if keyless else "endpoint-key"), timeout_s=5.0)
    )

    await client.rerank_memories("Rank the matching memories")
    await client.answer_schema("Rank the matching memories", response_format=_RankingAnswer, max_tokens=256)

    direct, reasoned = (json.loads(request.content) for request in chat_endpoint)
    assert {name: direct[name] for name in set(direct) - set(reasoned)} == reasoning_off
    assert set(reasoned) <= set(direct)


# -- Request timeouts ---------------------------------------------------------------

_REQUEST_TIMEOUT_S = 0.5
# A request ended at its timeout ends well before the stalling endpoint gives up.
_ENDED_AT_ITS_TIMEOUT_S = GIVES_UP_AFTER_S / 2
_MESSAGES = [{"role": "user", "content": "Rank the matching memories"}]


@pytest.fixture
async def fresh_provider_clients(monkeypatch):
    """Provider clients built for this test, under the policy the runtime installs.

    LiteLLM keeps one module-level client for Ollama requests. This test gets
    its own, so it never sends through one built under another test's stand-in.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

    assert isinstance(litellm.aclient_session, NoRedirectAsyncClient)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    module_client = AsyncHTTPHandler(timeout=litellm.request_timeout)
    monkeypatch.setattr(litellm, "module_level_aclient", module_client)
    yield
    await GLOBAL_LOGGING_WORKER.flush()
    await close_litellm_async_clients()
    await module_client.close()


def _completion(route: str, *, path: str = "/v1", **request):
    async def call(base_url: str):
        return await litellm.acompletion(
            model=route,
            messages=_MESSAGES,
            api_base=f"{base_url}{path}",
            api_key="endpoint-key",
            timeout=_REQUEST_TIMEOUT_S,
            num_retries=0,
            **request,
        )

    return call


def _embedding(route: str, *, path: str = "/v1"):
    async def call(base_url: str):
        return await litellm.aembedding(
            model=route,
            input=["alpha"],
            api_base=f"{base_url}{path}",
            api_key="endpoint-key",
            timeout=_REQUEST_TIMEOUT_S,
            max_retries=0,
        )

    return call


_SDK_REQUESTS = {
    "openrouter completion": _completion(
        "openrouter/anthropic/claude-haiku-test", extra_body={"reasoning": {"effort": "none"}}
    ),
    "openai completion": _completion("openai/chat-test"),
    "anthropic completion": _completion("anthropic/claude-test", path=""),
    "openai embedding": _embedding("openai/text-embedding-test"),
    "openrouter embedding": _embedding("openrouter/openai/text-embedding-test"),
}


@pytest.mark.parametrize("mode", STALL_MODES)
@pytest.mark.parametrize("request_name", list(_SDK_REQUESTS))
async def test_a_model_request_ends_at_its_timeout_however_the_endpoint_stalls(
    fresh_provider_clients, mode, request_name
) -> None:
    """A provider that keeps the connection busy cannot hold a request past its timeout."""
    with stalling_endpoint(mode) as base_url:
        started = time.monotonic()
        with pytest.raises(litellm.Timeout) as raised:
            await _SDK_REQUESTS[request_name](base_url)
        elapsed_s = time.monotonic() - started

    assert elapsed_s < _ENDED_AT_ITS_TIMEOUT_S
    assert is_retryable_provider_error(raised.value) is True


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


@pytest.mark.parametrize(
    "route", ["openrouter/anthropic/claude-haiku-test", "openai/chat-test", "anthropic/claude-test"]
)
async def test_a_structured_call_to_a_trickling_endpoint_ends_at_its_deadline_as_a_transient_failure(
    fresh_provider_clients, route
) -> None:
    with stalling_endpoint("trickle") as base_url:
        client = LiteLlmStructuredClient(
            StructuredLlmConfig(
                model=route,
                base_url=base_url if route.startswith("anthropic/") else f"{base_url}/v1",
                api_key="endpoint-key",
                timeout_s=_REQUEST_TIMEOUT_S,
            )
        )
        started = time.monotonic()
        with pytest.raises(StructuredLlmError) as raised:
            await client.rerank_memories("Rank the matching memories")
        elapsed_s = time.monotonic() - started

    assert elapsed_s < _ENDED_AT_ITS_TIMEOUT_S
    assert raised.value.terminal_category == "deadline_exceeded"
    assert failure_retryable(raised.value) is True


@pytest.mark.parametrize(
    ("route", "path"),
    [
        ("openai/text-embedding-test", "/v1"),
        ("openrouter/openai/text-embedding-test", "/v1"),
        # LiteLLM sends an Ollama embedding without the caller's timeout; the
        # runtime's own deadline ends it.
        ("ollama/nomic-embed-text", ""),
    ],
)
async def test_an_embedding_from_a_trickling_endpoint_ends_at_its_deadline_as_a_transient_failure(
    fresh_provider_clients, route, path
) -> None:
    with stalling_endpoint("trickle") as base_url:
        runtime = EmbeddingRuntime(
            EmbeddingSettings(
                invocation=ResolvedInvocation(
                    route=route, api_base=f"{base_url}{path}", api_key="endpoint-key"
                ),
                timeout_s=_REQUEST_TIMEOUT_S,
                cache_size=0,
            )
        )
        started = time.monotonic()
        with pytest.raises(EmbeddingDeadlineExceeded) as raised:
            await runtime.embed_query("alpha")
        elapsed_s = time.monotonic() - started

    assert elapsed_s < _ENDED_AT_ITS_TIMEOUT_S
    assert failure_retryable(raised.value) is True
