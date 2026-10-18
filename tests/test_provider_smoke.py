"""The provider smoke script: which providers it tries, how it reports them and its exit status.

The script's checks are the product's own, so the end-to-end tests here send
them through LiteLLM and the model listing to an in-memory provider, as the
provider request contract matrix does. No request leaves the process.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import litellm
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from memforge.llm.bindings import ModelBinding
from memforge.llm.capability import CapabilityCheck
from memforge.llm.providers import ResolvedInvocation
from memforge.server import model_settings_api
from memforge.server.model_settings_api import ModelDiscoveryResponse, ModelOption
from tests.model_endpoint_fixture import InMemorySharedClient, OllamaLookups

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "provider_smoke.py"
SECRET = "sk-or-smoke-0123456789abcdef"
OLLAMA = "http://127.0.0.1:11434"
_CAPABILITY_ANSWER = {"probe": {"label": "memforge", "confirmed": True, "items": ["alpha", "beta", "gamma"]}}


def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("provider_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def _no_server(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _ollama_version(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/version"
    return httpx.Response(200, json={"version": "0.12.0"})


# -- Which providers are tried ------------------------------------------------------


def test_a_provider_whose_key_is_set_is_tried_and_every_other_one_is_reported_skipped() -> None:
    options = smoke.parse_args([])

    targets, skipped = smoke.plan_targets(
        options,
        {"OPENROUTER_API_KEY": SECRET, "OPENAI_API_KEY": "  "},
        ollama_answers=False,
    )

    assert [(target.descriptor, target.auth_strategy, target.key_variable) for target in targets] == [
        ("openrouter", "api_key", "OPENROUTER_API_KEY")
    ]
    assert targets[0].models == {
        "generation": "anthropic/claude-haiku-4.5",
        "embedding": "openai/text-embedding-3-small",
    }
    assert [(row.provider, row.result, row.detail) for row in skipped] == [
        ("openai", "skipped", "no key (OPENAI_API_KEY not set)"),
        ("anthropic", "skipped", "no key (ANTHROPIC_API_KEY not set)"),
        ("ollama", "skipped", f"not reachable ({OLLAMA})"),
        ("litellm", "skipped", "no --litellm-route given"),
    ]


def test_anthropic_is_tried_for_generation_only() -> None:
    targets, _skipped = smoke.plan_targets(smoke.parse_args([]), {"ANTHROPIC_API_KEY": SECRET}, ollama_answers=False)

    assert [(target.descriptor, dict(target.models)) for target in targets] == [
        ("anthropic", {"generation": "claude-haiku-4-5"})
    ]


def test_ollama_is_tried_without_a_key_at_the_endpoint_given() -> None:
    options = smoke.parse_args(
        ["--ollama-endpoint", "http://gpu.example.test:11434", "--ollama-embedding-model", "bge-m3"]
    )

    targets, _skipped = smoke.plan_targets(options, {}, ollama_answers=True)

    [ollama] = targets
    assert (ollama.endpoint, ollama.auth_strategy, ollama.key_variable) == (
        "http://gpu.example.test:11434",
        "none",
        None,
    )
    assert ollama.models == {"generation": "qwen3:0.6b", "embedding": "bge-m3"}


def test_a_litellm_route_is_tried_with_the_providers_own_environment() -> None:
    options = smoke.parse_args(
        ["--litellm-route", "voyage/voyage-3", "--kind", "embedding", "--endpoint", "https://models.example.test"]
    )

    targets, skipped = smoke.plan_targets(options, {}, ollama_answers=False)

    [route] = targets
    assert (route.descriptor, route.endpoint, route.auth_strategy, route.key_variable) == (
        "litellm",
        "https://models.example.test",
        "environment",
        None,
    )
    assert route.models == {"embedding": "voyage/voyage-3"}
    assert "litellm" not in {row.provider for row in skipped}


@pytest.mark.parametrize(
    ("respond", "answers"),
    [
        (_ollama_version, True),
        (lambda _request: httpx.Response(404), False),
        (_no_server, False),
    ],
    ids=["server", "something_else", "nothing"],
)
async def test_ollama_counts_as_available_only_when_its_server_answers(respond, answers) -> None:
    assert await smoke.ollama_answers(OLLAMA, transport=httpx.MockTransport(respond)) is answers


# -- The report ---------------------------------------------------------------------


def _report(*rows: Any) -> Any:
    return smoke.Report(date="2026-10-18", memforge_version="1.2.3", litellm_version="1.80.0", rows=rows)


def test_the_report_is_a_markdown_table_of_one_row_per_check_under_the_versions_it_ran_with() -> None:
    report = _report(
        smoke.Row("ollama", "pass", "generation", "structured answer", "qwen3:0.6b", 850, "json_text"),
        smoke.Row("ollama", "pass", "embedding", "embedding", "nomic-embed-text", 1234, "dimension 768"),
        smoke.Row("openrouter", "fail", "generation", "model listing", "a|b", 12, "authentication_failed (HTTP 401)"),
        smoke.Row("anthropic", "skipped", detail="no key (ANTHROPIC_API_KEY not set)"),
    )

    assert smoke.render_report(report).splitlines() == [
        "**Provider smoke** 2026-10-18 · MemForge 1.2.3 · LiteLLM 1.80.0",
        "",
        "| Provider | Kind | Check | Model | Result | Latency | Detail |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| ollama | generation | structured answer | `qwen3:0.6b` | pass | 850 ms | json_text |",
        "| ollama | embedding | embedding | `nomic-embed-text` | pass | 1.2 s | dimension 768 |",
        "| openrouter | generation | model listing | `a\\|b` | fail | 12 ms | authentication_failed (HTTP 401) |",
        "| anthropic |  |  |  | skipped |  | no key (ANTHROPIC_API_KEY not set) |",
    ]


@pytest.mark.parametrize(
    ("results", "status"),
    [
        (("pass", "skipped"), 0),
        (("skipped", "skipped"), 0),
        (("pass", "fail", "skipped"), 1),
    ],
    ids=["passed", "nothing_ran", "one_failed"],
)
def test_the_exit_status_is_zero_only_when_every_check_that_ran_passed(results, status) -> None:
    assert smoke.exit_code(_report(*(smoke.Row("p", result) for result in results))) == status


# -- Deadlines ----------------------------------------------------------------------


class _StallingChecks:
    """Lists the model at once and never finishes a capability check."""

    async def listing(self, binding: ModelBinding, api_key: str | None) -> ModelDiscoveryResponse:
        return ModelDiscoveryResponse(
            ok=True, models_supported=True, models=[ModelOption(id=binding.model_id)], message=""
        )

    async def capability(self, binding: ModelBinding, invocation: ResolvedInvocation) -> CapabilityCheck:
        await asyncio.sleep(60)
        raise AssertionError("the check outlived its deadline")


async def test_a_check_that_outlasts_its_deadline_fails_and_the_next_one_still_runs() -> None:
    options = smoke.parse_args(["--timeout", "0.05"])

    report = await smoke.run(
        options,
        {},
        checks=_StallingChecks(),
        probe_transport=httpx.MockTransport(_ollama_version),
    )

    ollama = [(row.kind, row.check, row.result, row.detail) for row in report.rows if row.provider == "ollama"]
    assert ollama == [
        ("generation", "model listing", "pass", ""),
        ("generation", "structured answer", "fail", "deadline_exceeded"),
        ("embedding", "model listing", "pass", ""),
        ("embedding", "embedding", "fail", "deadline_exceeded"),
    ]
    assert smoke.exit_code(report) == 1


# -- Through the product path -------------------------------------------------------


class _Provider:
    """An in-memory provider behind LiteLLM's HTTP clients and the model listing's."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.respond: Callable[[httpx.Request], httpx.Response] = lambda _request: httpx.Response(404)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def sent(self) -> list[tuple[str, str]]:
        return [(request.method, str(request.url)) for request in self.requests]


@pytest.fixture
async def provider(monkeypatch):
    """Every model request the smoke sends, answered in memory.

    LiteLLM keeps module-level clients; this test gets its own, and a stand-in
    for the one it asks an Ollama server about a model with.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

    provider = _Provider()
    shared = InMemorySharedClient(provider.handle)
    monkeypatch.setattr(litellm, "aclient_session", shared)
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_options: httpx.MockTransport(provider.handle)),
    )
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(litellm, "module_level_client", OllamaLookups())
    module_client = AsyncHTTPHandler(timeout=litellm.request_timeout)
    monkeypatch.setattr(litellm, "module_level_aclient", module_client)

    def listing_client(**options: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(provider.handle), **options)

    monkeypatch.setattr(
        model_settings_api,
        "httpx",
        types.SimpleNamespace(**{**vars(httpx), "AsyncClient": listing_client}),
    )
    yield provider
    await GLOBAL_LOGGING_WORKER.flush()
    await close_litellm_async_clients()
    await module_client.close()
    await shared.aclose()


def _embeddings(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": "list",
        "model": body["model"],
        "data": [
            {"object": "embedding", "index": index, "embedding": [float(len(text)), 0.5]}
            for index, text in enumerate(body["input"])
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _ollama(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/tags":
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3:0.6b", "details": {"family": "qwen3"}},
                    {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
                ]
            },
        )
    body = json.loads(request.content)
    if path == "/api/chat":
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "created_at": "2026-10-18T00:00:00Z",
                "message": {"role": "assistant", "content": json.dumps(_CAPABILITY_ANSWER)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )
    if path == "/v1/embeddings":
        return httpx.Response(200, json=_embeddings(body))
    return httpx.Response(404, json={"error": "not found"})


async def test_a_running_ollama_passes_every_check_through_the_product_path(provider) -> None:
    provider.respond = _ollama

    report = await smoke.run(smoke.parse_args([]), {}, probe_transport=httpx.MockTransport(_ollama_version))

    ollama = [
        (row.kind, row.check, row.model, row.result, row.detail) for row in report.rows if row.provider == "ollama"
    ]
    assert ollama == [
        ("generation", "model listing", "qwen3:0.6b", "pass", ""),
        # LiteLLM asks the Ollama server for the model's limits, and none answers here.
        ("generation", "structured answer", "qwen3:0.6b", "pass", "json_text; token limits unknown to LiteLLM"),
        ("embedding", "model listing", "nomic-embed-text", "pass", ""),
        ("embedding", "embedding", "nomic-embed-text", "pass", "dimension 2"),
    ]
    assert smoke.exit_code(report) == 0
    assert set(provider.sent()) == {
        ("GET", f"{OLLAMA}/api/tags"),
        ("POST", f"{OLLAMA}/api/chat"),
        ("POST", f"{OLLAMA}/v1/embeddings"),
    }
    chat = next(json.loads(request.content) for request in provider.requests if request.url.path == "/api/chat")
    assert chat["think"] is False


def _openrouter(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/v1/models":
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "anthropic/claude-haiku-4.5",
                        "context_length": 200000,
                        "top_provider": {"max_completion_tokens": 64000},
                    }
                ]
            },
        )
    if path == "/api/v1/embeddings/models":
        return httpx.Response(200, json={"data": [{"id": "openai/text-embedding-3-large"}]})
    body = json.loads(request.content)
    if path == "/api/v1/chat/completions":
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(_CAPABILITY_ANSWER)},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    return httpx.Response(200, json=_embeddings(body))


async def test_a_model_the_listing_gives_limits_for_passes_and_one_it_does_not_list_is_noted(provider) -> None:
    """Settings takes the token limits a listing gives, so LiteLLM need not know the model."""
    provider.respond = _openrouter

    report = await smoke.run(
        smoke.parse_args([]),
        {"OPENROUTER_API_KEY": SECRET},
        probe_transport=httpx.MockTransport(_no_server),
    )

    openrouter = [(row.kind, row.check, row.result, row.detail) for row in report.rows if row.provider == "openrouter"]
    assert openrouter == [
        ("generation", "model listing", "pass", ""),
        ("generation", "structured answer", "pass", "json_text"),
        ("embedding", "model listing", "pass", "not among the 1 listed models"),
        ("embedding", "embedding", "pass", "dimension 2"),
    ]
    assert smoke.exit_code(report) == 0
    chat = next(json.loads(r.content) for r in provider.requests if r.url.path == "/api/v1/chat/completions")
    assert chat["reasoning"] == {"enabled": False}
    assert {request.headers.get("authorization") for request in provider.requests} == {f"Bearer {SECRET}"}


async def test_a_refused_key_fails_each_check_by_code_and_status_and_the_key_appears_nowhere(
    provider, capsys, caplog
) -> None:
    """The provider repeats the key in its refusal; the report keeps no response body."""
    provider.respond = lambda _request: httpx.Response(
        401, json={"error": {"message": f"Invalid key {SECRET}", "code": 401}}
    )

    with caplog.at_level("DEBUG"):
        report = await smoke.run(
            smoke.parse_args([]),
            {"OPENROUTER_API_KEY": SECRET},
            probe_transport=httpx.MockTransport(_no_server),
        )
        output = smoke.render_report(report)

    failed = [(row.kind, row.check, row.result, row.detail) for row in report.rows if row.provider == "openrouter"]
    assert [(kind, check, result) for kind, check, result, _detail in failed] == [
        ("generation", "model listing", "fail"),
        ("generation", "structured answer", "fail"),
        ("embedding", "model listing", "fail"),
        ("embedding", "embedding", "fail"),
    ]
    assert all(detail.endswith("(HTTP 401)") for *_rest, detail in failed)
    assert failed[1][3] == "authentication_failed (HTTP 401)"
    assert smoke.exit_code(report) == 1
    assert {request.headers.get("authorization") for request in provider.requests} == {f"Bearer {SECRET}"}
    captured = capsys.readouterr()
    assert SECRET not in output + captured.out + captured.err
    assert SECRET not in repr(report)
    assert SECRET not in caplog.text
