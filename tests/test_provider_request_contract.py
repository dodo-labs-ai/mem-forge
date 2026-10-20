"""The provider request contract: what MemForge sends each model provider, and how it reads the reply.

``MATRIX`` holds one row for every provider descriptor, and one for every
LiteLLM route the generic route descriptor is claimed to reach. A row states,
for each kind of model it serves, the request the product path writes on the
wire and the reply the provider answers with in its own format. The tests
send every request through the product's own clients and LiteLLM, and answer
it below LiteLLM at the HTTP transport, so LiteLLM's translation of the
request and its reading of the reply are both under test, and no request
leaves the process.

The features each kind of row is held to are in ``FEATURES``. A row covers
every feature of every kind its descriptor serves, or records why it does not
in ``excluded``. A request a row states that the product does not send yet is
recorded in ``known_gaps`` and expected to fail until it is sent.

Adding a provider is adding its row.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

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
from memforge.llm.providers import (
    PROVIDER_DESCRIPTORS,
    AuthStrategy,
    ModelKind,
    ResolvedInvocation,
    provider_descriptor,
)
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
    StructuredResponseModel,
    failure_retryable,
)
from memforge.llm.transport import NoRedirectAsyncClient, is_retryable_provider_error
from tests.model_endpoint_fixture import InMemorySharedClient, OllamaLookups
from tests.stalling_endpoint_fixture import GIVES_UP_AFTER_S, STALL_MODES, stalling_endpoint

Feature = Literal[
    "structured_generation",
    "credential",
    "reasoning_off",
    "embedding",
    "dimensions",
    "timeout",
    "redirect",
    "error_classification",
]

FEATURES: Mapping[ModelKind, tuple[Feature, ...]] = {
    "generation": (
        "structured_generation",
        "credential",
        "reasoning_off",
        "timeout",
        "redirect",
        "error_classification",
    ),
    "embedding": ("embedding", "credential", "dimensions", "timeout", "redirect", "error_classification"),
}

# How a structured request asks for its schema. ``json_schema``: an OpenAI
# ``response_format`` of type ``json_schema``. ``json_tool_call``: one tool the
# model is made to call, whose input is the answer. ``format``: the schema as
# Ollama's ``format``. ``json_text``: the schema written into the prompt, for a
# model nothing declares to take one.
SchemaShape = Literal["json_schema", "json_tool_call", "format", "json_text"]

# The credential an API-key row is configured with.
_KEY = "row-key"
_SECRET_REF = "secret:row"

# The time a call that is not about its deadline is given. The first request a
# process sends through a provider's SDK also imports that SDK's modules, which
# on a busy machine takes seconds of the call's own time.
_UNHURRIED_CALL_S = 60.0
_REQUEST_TIMEOUT_S = 0.5
# A request ended at its timeout ends well before the stalling endpoint gives up.
_ENDED_AT_ITS_TIMEOUT_S = GIVES_UP_AFTER_S / 2
# The limits every structured client's route states. A quarter of the context
# window is far above the output limit, so the output limit is the route's
# output capacity.
_CONTEXT_WINDOW = 4096
_OUTPUT_LIMIT = 64
_PROMPT = "Rank the matching memories"
_ANSWER = {"ranking": [3, 1]}
_JSON_TEXT_INSTRUCTION = "Return ONLY a single JSON object"
_ELSEWHERE = "elsewhere.example.test"



@dataclass(frozen=True)
class StatusClassification:
    """What a failed answer with one HTTP status leads to."""

    # The call sends the same request again within its retry budget.
    sent_again: bool
    # The run that made the call may succeed when it is retried
    # (``failure_retryable``).
    run_retryable: bool


_TRANSIENT = StatusClassification(sent_again=True, run_retryable=True)
_REFUSED = StatusClassification(sent_again=False, run_retryable=False)

# The HTTP answers a request can fail with, and what each leads to. Every row
# classifies them the same way, for both kinds of model.
STATUS_CLASSIFICATION: Mapping[int, StatusClassification] = {
    400: _REFUSED,
    401: _REFUSED,
    403: _REFUSED,
    404: _REFUSED,
    408: _TRANSIENT,
    409: _TRANSIENT,
    422: _REFUSED,
    429: _TRANSIENT,
    500: _TRANSIENT,
    503: _TRANSIENT,
}

# Keyword arguments MemForge gives the SDK that are the SDK's to read, never a
# provider's.
_SDK_ONLY_FIELDS = ("allowed_openai_params", "extra_body", "api_key", "api_base")


# -- Provider replies ---------------------------------------------------------------


def _openai_chat(body: dict[str, Any], answer: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": body["model"],
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_messages(body: dict[str, Any], answer: str) -> dict[str, Any]:
    """A Messages reply: the call of the tool the request made the model call, or text."""
    tool = (body.get("tool_choice") or {}).get("name")
    content = (
        [{"type": "tool_use", "id": "toolu_1", "name": tool, "input": json.loads(answer)}]
        if tool
        else [{"type": "text", "text": answer}]
    )
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": body["model"],
        "content": content,
        "stop_reason": "tool_use" if tool else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _ollama_chat(body: dict[str, Any], answer: str) -> dict[str, Any]:
    return {
        "model": body["model"],
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": answer},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 1,
        "eval_count": 1,
    }


def _vector(text: str) -> list[float]:
    return [float(len(text)), 0.5]


def _openai_embeddings(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": "list",
        "model": body["model"],
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(body["input"])
        ],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _voyage_embeddings(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "object": "list",
        "model": body["model"],
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text)}
            for index, text in enumerate(body["input"])
        ],
        "usage": {"total_tokens": 1},
    }


def _ollama_embed(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": body["model"],
        "embeddings": [_vector(text) for text in body["input"]],
        "total_duration": 1,
        "load_duration": 1,
        "prompt_eval_count": 1,
    }


def _openai_error(_status: int) -> dict[str, Any]:
    return {"error": {"message": "The request was refused", "type": "invalid_request_error", "param": None, "code": None}}


def _anthropic_error(_status: int) -> dict[str, Any]:
    return {"type": "error", "error": {"type": "api_error", "message": "The request was refused"}}


def _ollama_error(_status: int) -> dict[str, Any]:
    return {"error": "The request was refused"}


# -- The matrix ---------------------------------------------------------------------


@dataclass(frozen=True)
class ReasoningOff:
    """What a direct answer from one generation model asks its provider for.

    ``parameters`` are the SDK keyword arguments the descriptor gives, and
    ``wire`` the request fields they become; ``None`` when they are the same.
    """

    model_id: str
    parameters: Mapping[str, Any]
    wire: Mapping[str, Any] | None = None

    @property
    def sent(self) -> Mapping[str, Any]:
        return self.parameters if self.wire is None else self.wire


@dataclass(frozen=True)
class GenerationContract:
    model_id: str
    url: str
    # The model the request names.
    model: str
    # Request headers by name; ``None`` when the request carries no such header.
    headers: Mapping[str, str | None]
    schema: SchemaShape
    # Where in the request body the output limit is written.
    output_limit: tuple[str, ...]
    reply: Callable[[dict[str, Any], str], dict[str, Any]]
    error: Callable[[int], dict[str, Any]]
    reasoning_off: tuple[ReasoningOff, ...]
    # How LiteLLM reports an exchange it ended at its timeout.
    timeout_error: type[Exception]
    # Other fields the request body carries.
    fields: Mapping[str, Any] = field(default_factory=dict)
    # Whether the model listing declares that the model answers in a response
    # schema, as Settings binds a model chosen from the listing; ``None`` for a
    # model the listing declares nothing for.
    native_schema: bool | None = None


@dataclass(frozen=True)
class EmbeddingContract:
    model_id: str
    url: str
    model: str
    headers: Mapping[str, str | None]
    reply: Callable[[dict[str, Any]], dict[str, Any]]
    error: Callable[[int], dict[str, Any]]
    # The request field a set dimension is sent in.
    dimensions_field: str
    # How LiteLLM reports an exchange it ended at its timeout; ``None`` when
    # LiteLLM sends the request without its timeout, and only the runtime's
    # own deadline ends it.
    timeout_error: type[Exception] | None
    fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Row:
    id: str
    descriptor: str
    auth_strategy: AuthStrategy
    # The endpoint Settings carries; ``None`` takes the descriptor's default,
    # or for a LiteLLM route the provider's own address.
    endpoint: str | None
    # The path after the server address where the provider's API is, for a
    # request sent to a local server instead.
    api_path: str
    generation: GenerationContract | None = None
    embedding: EmbeddingContract | None = None
    # Features the row is not held to, with the reason.
    excluded: Mapping[tuple[ModelKind, Feature], str] = field(default_factory=dict)
    # Features whose stated request the product does not send yet, with the reason.
    known_gaps: Mapping[tuple[ModelKind, Feature], str] = field(default_factory=dict)

    def contract(self, kind: ModelKind) -> GenerationContract | EmbeddingContract | None:
        return self.generation if kind == "generation" else self.embedding


_OLLAMA_EMBEDDING_KEY_GAP = (
    "LiteLLM's ollama/ embedding route posts with no headers, so the API key of an Ollama "
    "server behind authentication never reaches it: https://github.com/BerriAI/litellm/issues/47261"
)
_SAME_REQUESTS_AS_KEYLESS_OLLAMA = "The keyless ollama row sends the same requests; only the credential differs."
_SAME_REQUESTS_AS_OPENROUTER = (
    "The openrouter row sends the same requests; only the structured request differs, for a "
    "model the listing declares no response schema support for."
)

MATRIX: tuple[Row, ...] = (
    Row(
        id="openai",
        descriptor="openai",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/v1",
        generation=GenerationContract(
            model_id="gpt-5-mini",
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-5-mini",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_schema",
            output_limit=("max_completion_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(
                ReasoningOff("gpt-5.1", {"reasoning_effort": "none"}),
                ReasoningOff("gpt-5-mini", {"reasoning_effort": "minimal"}),
                ReasoningOff("o4-mini", {"reasoning_effort": "low"}),
                ReasoningOff("gpt-4o-mini", {}),
                ReasoningOff("gpt-5-chat-latest", {}),
                ReasoningOff("a-model-litellm-does-not-know", {}),
            ),
            timeout_error=litellm.Timeout,
        ),
        embedding=EmbeddingContract(
            model_id="text-embedding-3-small",
            url="https://api.openai.com/v1/embeddings",
            model="text-embedding-3-small",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
            fields={"encoding_format": "float"},
        ),
    ),
    Row(
        id="anthropic",
        descriptor="anthropic",
        auth_strategy="api_key",
        endpoint=None,
        api_path="",
        generation=GenerationContract(
            model_id="claude-haiku-4-5",
            url="https://api.anthropic.com/v1/messages",
            model="claude-haiku-4-5",
            headers={"x-api-key": _KEY, "anthropic-version": "2023-06-01", "authorization": None},
            schema="json_tool_call",
            output_limit=("max_tokens",),
            reply=_anthropic_messages,
            error=_anthropic_error,
            reasoning_off=(ReasoningOff("claude-haiku-4-5", {}),),
            timeout_error=litellm.Timeout,
        ),
    ),
    Row(
        id="openrouter",
        descriptor="openrouter",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/api/v1",
        generation=GenerationContract(
            model_id="anthropic/claude-haiku-5.5",
            url="https://openrouter.ai/api/v1/chat/completions",
            model="anthropic/claude-haiku-5.5",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_schema",
            output_limit=("max_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(
                ReasoningOff(
                    "anthropic/claude-haiku-5.5",
                    {"extra_body": {"reasoning": {"enabled": False}}},
                    wire={"reasoning": {"enabled": False}},
                ),
                ReasoningOff(
                    "openai/gpt-5-mini",
                    {"extra_body": {"reasoning": {"enabled": False}}},
                    wire={"reasoning": {"enabled": False}},
                ),
            ),
            timeout_error=litellm.Timeout,
            fields={"usage": {"include": True}},
            native_schema=True,
        ),
        embedding=EmbeddingContract(
            model_id="openai/text-embedding-3-small",
            url="https://openrouter.ai/api/v1/embeddings",
            model="openai/text-embedding-3-small",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
        ),
    ),
    Row(
        id="openrouter-without-response-schema",
        descriptor="openrouter",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/api/v1",
        generation=GenerationContract(
            model_id="meituan/longcat-2.0",
            url="https://openrouter.ai/api/v1/chat/completions",
            model="meituan/longcat-2.0",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_text",
            output_limit=("max_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(),
            timeout_error=litellm.Timeout,
            fields={"usage": {"include": True}},
            native_schema=False,
        ),
        excluded={
            **{
                ("generation", feature): _SAME_REQUESTS_AS_OPENROUTER
                for feature in ("credential", "reasoning_off", "timeout", "redirect", "error_classification")
            },
            **{("embedding", feature): _SAME_REQUESTS_AS_OPENROUTER for feature in FEATURES["embedding"]},
        },
    ),
    Row(
        id="ollama",
        descriptor="ollama",
        auth_strategy="none",
        endpoint=None,
        api_path="",
        generation=GenerationContract(
            model_id="qwen3",
            url="http://localhost:11434/api/chat",
            model="qwen3",
            headers={"authorization": "Bearer no-api-key"},
            schema="format",
            output_limit=("options", "num_predict"),
            reply=_ollama_chat,
            error=_ollama_error,
            reasoning_off=(
                ReasoningOff("qwen3", {"think": False}),
                ReasoningOff("gpt-oss:20b", {"think": "low"}),
            ),
            timeout_error=litellm.APIConnectionError,
            fields={"stream": False},
        ),
        embedding=EmbeddingContract(
            model_id="nomic-embed-text",
            url="http://localhost:11434/v1/embeddings",
            model="nomic-embed-text",
            headers={"authorization": "Bearer no-api-key"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
            fields={"encoding_format": "float"},
        ),
    ),
    Row(
        id="ollama-with-key",
        descriptor="ollama",
        auth_strategy="api_key",
        endpoint="https://ollama.example.test",
        api_path="",
        generation=GenerationContract(
            model_id="qwen3",
            url="https://ollama.example.test/api/chat",
            model="qwen3",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="format",
            output_limit=("options", "num_predict"),
            reply=_ollama_chat,
            error=_ollama_error,
            reasoning_off=(),
            timeout_error=litellm.APIConnectionError,
            fields={"stream": False},
        ),
        embedding=EmbeddingContract(
            model_id="nomic-embed-text",
            url="https://ollama.example.test/v1/embeddings",
            model="nomic-embed-text",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
            fields={"encoding_format": "float"},
        ),
        excluded={
            (kind, feature): _SAME_REQUESTS_AS_KEYLESS_OLLAMA
            for kind, features in (
                ("generation", ("reasoning_off", "timeout", "redirect", "error_classification")),
                ("embedding", ("dimensions", "timeout", "redirect", "error_classification")),
            )
            for feature in features
        },
    ),
    Row(
        id="openai_compatible",
        descriptor="openai_compatible",
        auth_strategy="api_key",
        endpoint="https://models.example.test/v1",
        api_path="/v1",
        generation=GenerationContract(
            model_id="qwen3",
            url="https://models.example.test/v1/chat/completions",
            model="qwen3",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_text",
            output_limit=("max_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(ReasoningOff("gpt-5-mini", {}), ReasoningOff("qwen3", {})),
            timeout_error=litellm.Timeout,
        ),
        embedding=EmbeddingContract(
            model_id="BAAI/bge-m3",
            url="https://models.example.test/v1/embeddings",
            model="BAAI/bge-m3",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
            fields={"encoding_format": "float"},
        ),
    ),
    Row(
        id="litellm-openai",
        descriptor="litellm",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/v1",
        generation=GenerationContract(
            model_id="openai/gpt-5-mini",
            url="https://api.openai.com/v1/chat/completions",
            model="gpt-5-mini",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_schema",
            output_limit=("max_completion_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(ReasoningOff("openai/gpt-5-mini", {}),),
            timeout_error=litellm.Timeout,
        ),
        embedding=EmbeddingContract(
            model_id="openai/text-embedding-3-small",
            url="https://api.openai.com/v1/embeddings",
            model="text-embedding-3-small",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
            fields={"encoding_format": "float"},
        ),
    ),
    Row(
        id="litellm-anthropic",
        descriptor="litellm",
        auth_strategy="api_key",
        endpoint=None,
        api_path="",
        generation=GenerationContract(
            model_id="anthropic/claude-haiku-4-5",
            url="https://api.anthropic.com/v1/messages",
            model="claude-haiku-4-5",
            headers={"x-api-key": _KEY, "anthropic-version": "2023-06-01", "authorization": None},
            schema="json_tool_call",
            output_limit=("max_tokens",),
            reply=_anthropic_messages,
            error=_anthropic_error,
            reasoning_off=(ReasoningOff("anthropic/claude-haiku-4-5", {}),),
            timeout_error=litellm.Timeout,
        ),
        excluded={
            ("embedding", feature): "Anthropic serves no embedding models."
            for feature in FEATURES["embedding"]
        },
    ),
    Row(
        id="litellm-openrouter",
        descriptor="litellm",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/api/v1",
        generation=GenerationContract(
            model_id="openrouter/anthropic/claude-haiku-5.5",
            url="https://openrouter.ai/api/v1/chat/completions",
            model="anthropic/claude-haiku-5.5",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="json_text",
            output_limit=("max_tokens",),
            reply=_openai_chat,
            error=_openai_error,
            reasoning_off=(ReasoningOff("openrouter/anthropic/claude-haiku-5.5", {}),),
            timeout_error=litellm.Timeout,
            fields={"usage": {"include": True}},
        ),
        embedding=EmbeddingContract(
            model_id="openrouter/openai/text-embedding-3-small",
            url="https://openrouter.ai/api/v1/embeddings",
            model="openai/text-embedding-3-small",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_openai_embeddings,
            error=_openai_error,
            dimensions_field="dimensions",
            timeout_error=litellm.Timeout,
        ),
    ),
    Row(
        id="litellm-ollama",
        descriptor="litellm",
        auth_strategy="api_key",
        endpoint=None,
        api_path="",
        generation=GenerationContract(
            model_id="ollama_chat/qwen3",
            url="http://localhost:11434/api/chat",
            model="qwen3",
            headers={"authorization": f"Bearer {_KEY}"},
            schema="format",
            output_limit=("options", "num_predict"),
            reply=_ollama_chat,
            error=_ollama_error,
            reasoning_off=(ReasoningOff("ollama_chat/qwen3", {}),),
            timeout_error=litellm.APIConnectionError,
            fields={"stream": False},
        ),
        embedding=EmbeddingContract(
            model_id="ollama/nomic-embed-text",
            url="http://localhost:11434/api/embed",
            model="nomic-embed-text",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_ollama_embed,
            error=_ollama_error,
            dimensions_field="dimensions",
            timeout_error=None,
        ),
        known_gaps={("embedding", "credential"): _OLLAMA_EMBEDDING_KEY_GAP},
    ),
    Row(
        id="litellm-voyage",
        descriptor="litellm",
        auth_strategy="api_key",
        endpoint=None,
        api_path="/v1",
        embedding=EmbeddingContract(
            model_id="voyage/voyage-3",
            url="https://api.voyageai.com/v1/embeddings",
            model="voyage-3",
            headers={"authorization": f"Bearer {_KEY}"},
            reply=_voyage_embeddings,
            error=_openai_error,
            dimensions_field="output_dimension",
            timeout_error=litellm.APIConnectionError,
        ),
        excluded={
            ("generation", feature): "Voyage serves embedding models only."
            for feature in FEATURES["generation"]
        },
    ),
)

# Route providers the quickstart names for the LiteLLM-route descriptor, with
# the kind of model it names them for.
DOCUMENTED_ROUTES: tuple[tuple[str, ModelKind], ...] = (
    ("voyage", "embedding"),
    ("ollama", "embedding"),
)


def _route_provider(model_id: str) -> str:
    return model_id.partition("/")[0]


def _rows_for(kind: ModelKind, feature: Feature) -> list[Any]:
    """The rows held to ``feature`` of ``kind``, a known gap marked to fail until it is closed."""
    params = []
    for row in MATRIX:
        if row.contract(kind) is None or (kind, feature) in row.excluded:
            continue
        gap = row.known_gaps.get((kind, feature))
        marks = [pytest.mark.xfail(strict=True, reason=gap)] if gap else []
        params.append(pytest.param(row, id=row.id, marks=marks))
    return params


def _row_kinds(feature: Feature) -> list[Any]:
    return [
        pytest.param(param.values[0], kind, id=f"{param.id}-{kind}", marks=param.marks)
        for kind in ("generation", "embedding")
        for param in _rows_for(kind, feature)
    ]


def _reasoning_off_cases() -> list[Any]:
    return [
        pytest.param(param.values[0], case, id=f"{param.id}-{case.model_id}", marks=param.marks)
        for param in _rows_for("generation", "reasoning_off")
        for case in param.values[0].generation.reasoning_off
    ]


# -- Completeness -------------------------------------------------------------------


def test_every_provider_descriptor_has_a_row() -> None:
    assert {descriptor.id for descriptor in PROVIDER_DESCRIPTORS} == {row.descriptor for row in MATRIX}


def test_every_route_a_descriptor_compiles_or_the_docs_name_has_a_litellm_row() -> None:
    """A model named ``provider/model`` in the environment reaches its provider through the route descriptor."""
    claimed = {
        (prefix, kind)
        for descriptor in PROVIDER_DESCRIPTORS
        for kind, prefix in descriptor.route_prefixes.items()
    } | set(DOCUMENTED_ROUTES)
    covered = {
        (_route_provider(contract.model_id), kind)
        for row in MATRIX
        if row.descriptor == "litellm"
        for kind in ("generation", "embedding")
        if (contract := row.contract(kind)) is not None
    }

    assert claimed - covered == set()


def test_every_row_covers_each_feature_of_the_kinds_its_descriptor_serves() -> None:
    for row in MATRIX:
        kinds = provider_descriptor(row.descriptor).kinds
        for kind in ("generation", "embedding"):
            contract = row.contract(kind)
            if kind not in kinds:
                assert contract is None, f"{row.id} states {kind} requests its descriptor does not serve"
                continue
            for feature in FEATURES[kind]:
                reason = row.excluded.get((kind, feature))
                if reason is not None:
                    assert reason.strip(), f"{row.id} excludes {kind} {feature} without a reason"
                    continue
                assert contract is not None, f"{row.id} has no {kind} contract for {feature}"
                if feature == "reasoning_off":
                    assert contract.reasoning_off, f"{row.id} states no reasoning-off request"
        for kind, feature in [*row.excluded, *row.known_gaps]:
            assert feature in FEATURES[kind], f"{row.id} names {feature}, not a {kind} feature"


# -- The in-memory provider ---------------------------------------------------------


class _Wire:
    """Every request LiteLLM sends, answered by ``respond``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.respond: Callable[[httpx.Request], httpx.Response] = lambda _request: httpx.Response(
            404, json={"error": "no reply"}
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


@contextlib.asynccontextmanager
async def _fresh_provider_clients(monkeypatch) -> AsyncIterator[None]:
    """Provider clients built for this test, under the policy the runtime installs.

    LiteLLM keeps module-level clients: one Ollama embeddings are sent with,
    and one it asks an Ollama server about a model with. This test gets its
    own of the first and a stand-in for the second, so it never sends through
    one built under another test's stand-in, nor asks a real server.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients

    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(litellm, "module_level_client", OllamaLookups())
    module_client = AsyncHTTPHandler(timeout=litellm.request_timeout)
    monkeypatch.setattr(litellm, "module_level_aclient", module_client)
    yield
    # LiteLLM records each call on a queue bound to this test's event loop.
    await GLOBAL_LOGGING_WORKER.flush()
    await close_litellm_async_clients()
    await module_client.close()


@pytest.fixture
async def wire(monkeypatch):
    """An in-memory provider behind every HTTP client LiteLLM sends a request through."""
    wire = _Wire()
    shared = InMemorySharedClient(wire.handle)
    monkeypatch.setattr(litellm, "aclient_session", shared)
    monkeypatch.setattr(
        AsyncHTTPHandler,
        "_create_async_transport",
        staticmethod(lambda **_options: httpx.MockTransport(wire.handle)),
    )
    async with _fresh_provider_clients(monkeypatch):
        yield wire
    await shared.aclose()


@pytest.fixture
async def provider_clients(monkeypatch):
    """The HTTP clients the runtime installs, sending to real local servers."""
    assert isinstance(litellm.aclient_session, NoRedirectAsyncClient)
    async with _fresh_provider_clients(monkeypatch):
        yield


# -- The product path ---------------------------------------------------------------


class _Ranking(StructuredResponseModel):
    """An answer the model may reason toward: a schema that is not a direct answer."""

    model_config = ConfigDict(extra="ignore")

    ranking: list[int] = []


class _DirectRanking(_Ranking):
    """The same answer, asked for directly."""

    direct_answer = True


_DEFAULT_ENDPOINT = object()


def _invocation(
    row: Row,
    kind: ModelKind,
    *,
    model_id: str | None = None,
    endpoint: object = _DEFAULT_ENDPOINT,
    parameters: Mapping[str, Any] | None = None,
) -> ResolvedInvocation:
    """Compile the row's settings for ``kind`` as Settings would save them."""
    contract = row.contract(kind)
    binding = build_binding(
        kind=kind,
        provider=row.descriptor,
        model_id=model_id or contract.model_id,
        endpoint=row.endpoint if endpoint is _DEFAULT_ENDPOINT else endpoint,
        auth_strategy=row.auth_strategy,
        parameters=parameters,
        native_schema=contract.native_schema if kind == "generation" else None,
        secret_ref=_SECRET_REF if row.auth_strategy == "api_key" else None,
        config_source="settings",
    )
    return compile_binding(binding, _KEY)


def _structured_client(invocation: ResolvedInvocation, **settings: Any) -> LiteLlmStructuredClient:
    settings.setdefault("timeout_s", _UNHURRIED_CALL_S)
    settings.setdefault("max_input_tokens", _CONTEXT_WINDOW)
    settings.setdefault("context_window_tokens", _CONTEXT_WINDOW)
    settings.setdefault("max_output_tokens", _OUTPUT_LIMIT)
    return LiteLlmStructuredClient(StructuredLlmConfig.for_invocation(invocation, **settings))


def _embedding_runtime(invocation: ResolvedInvocation, **settings: Any) -> EmbeddingRuntime:
    return EmbeddingRuntime(EmbeddingSettings(invocation=invocation, cache_size=0, **settings))


def _answer(contract: GenerationContract) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=contract.reply(json.loads(request.content), json.dumps(_ANSWER)))

    return respond


def _embed(contract: EmbeddingContract) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=contract.reply(json.loads(request.content)))

    return respond


def _at(body: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = body
    for name in path:
        value = value[name]
    return value


def _prompt_text(body: Mapping[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


def _assert_nothing_but_the_request(request: httpx.Request) -> None:
    """The body carries no SDK-only argument and no credential."""
    body = json.loads(request.content)
    assert [name for name in _SDK_ONLY_FIELDS if name in body] == []
    assert _KEY not in request.content.decode()


# -- Credentials --------------------------------------------------------------------


@pytest.mark.parametrize(("row", "kind"), _row_kinds("credential"))
async def test_a_request_carries_the_configured_credential_in_the_header_its_provider_reads(
    wire, row, kind
) -> None:
    contract = row.contract(kind)
    if kind == "generation":
        wire.respond = _answer(contract)
        await _structured_client(_invocation(row, kind)).answer_schema(
            _PROMPT, response_format=_Ranking
        )
    else:
        wire.respond = _embed(contract)
        await _embedding_runtime(_invocation(row, kind), num_retries=0).embed_query("alpha")

    [request] = wire.requests
    assert {name: request.headers.get(name) for name in contract.headers} == dict(contract.headers)


# -- Structured generation ----------------------------------------------------------


@pytest.mark.parametrize("row", _rows_for("generation", "structured_generation"))
async def test_a_structured_request_reaches_the_provider_as_its_api_takes_it_and_its_answer_comes_back(
    wire, row
) -> None:
    contract = row.generation
    wire.respond = _answer(contract)
    client = _structured_client(_invocation(row, "generation"))

    answer = await client.answer_schema(_PROMPT, response_format=_Ranking)

    assert answer == _Ranking(**_ANSWER)
    [request] = wire.requests
    body = json.loads(request.content)
    assert (request.method, str(request.url)) == ("POST", contract.url)
    _assert_nothing_but_the_request(request)
    assert body["model"] == contract.model
    assert _at(body, contract.output_limit) == client.request_budget().output_tokens == _OUTPUT_LIMIT
    assert {name: body.get(name) for name in contract.fields} == dict(contract.fields)
    assert _PROMPT in _prompt_text(body)
    if contract.schema == "json_schema":
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert set(response_format["json_schema"]["schema"]["properties"]) == {"ranking"}
        assert "tools" not in body
    elif contract.schema == "json_tool_call":
        assert body["tool_choice"] == {"type": "tool", "name": "json_tool_call"}
        assert [tool["name"] for tool in body["tools"]] == ["json_tool_call"]
        assert set(body["tools"][0]["input_schema"]["properties"]) == {"ranking"}
        assert "response_format" not in body
    elif contract.schema == "format":
        assert set(body["format"]["properties"]) == {"ranking"}
        assert [name for name in ("response_format", "tools") if name in body] == []
        assert _JSON_TEXT_INSTRUCTION not in _prompt_text(body)
    else:
        assert [name for name in ("response_format", "tools", "format") if name in body] == []
        assert _JSON_TEXT_INSTRUCTION in _prompt_text(body)


@pytest.mark.parametrize(("row", "case"), _reasoning_off_cases())
async def test_only_a_direct_answer_asks_the_provider_to_answer_without_reasoning(wire, row, case) -> None:
    """The descriptor's reasoning-off parameters reach the provider's request body, and nothing else changes."""
    descriptor = provider_descriptor(row.descriptor)
    wire.respond = _answer(row.generation)
    invocation = _invocation(row, "generation", model_id=case.model_id)
    client = _structured_client(invocation)

    await client.answer_schema(_PROMPT, response_format=_DirectRanking)
    await client.answer_schema(_PROMPT, response_format=_Ranking)

    assert descriptor.reasoning_off(invocation.route) == case.parameters
    direct, reasoned = wire.bodies()
    assert {name: direct[name] for name in set(direct) - set(reasoned)} == dict(case.sent)
    assert set(reasoned) <= set(direct)


@pytest.mark.parametrize("descriptor", PROVIDER_DESCRIPTORS, ids=lambda descriptor: descriptor.id)
def test_only_ollama_is_held_to_the_context_window_a_request_is_sized_by(descriptor) -> None:
    """Ollama sizes its window per request; every other provider fixes it and states it."""
    expected = {"num_ctx": 8192} if descriptor.id == "ollama" else {}

    assert descriptor.context_window(8192) == expected


@pytest.mark.parametrize("row", _rows_for("generation", "structured_generation"))
async def test_an_ollama_request_states_the_context_window_of_its_route(wire, row) -> None:
    wire.respond = _answer(row.generation)
    client = _structured_client(_invocation(row, "generation"))

    await client.answer_schema(_PROMPT, response_format=_Ranking)

    [body] = wire.bodies()
    if row.descriptor == "ollama":
        assert body["options"]["num_ctx"] == client.request_budget().context_limit == _CONTEXT_WINDOW
    else:
        assert "num_ctx" not in json.dumps(body)


# -- Embedding ----------------------------------------------------------------------


@pytest.mark.parametrize("row", _rows_for("embedding", "embedding"))
async def test_an_embedding_request_reaches_the_provider_in_batches_and_its_vectors_come_back_in_order(
    wire, row
) -> None:
    contract = row.embedding
    wire.respond = _embed(contract)
    runtime = _embedding_runtime(_invocation(row, "embedding"), num_retries=0, max_inputs_per_request=2)

    vectors = await runtime.embed_documents(["alpha", "be", "cue"])

    assert vectors == [[5.0, 0.5], [2.0, 0.5], [3.0, 0.5]]
    assert [body["input"] for body in wire.bodies()] == [["alpha", "be"], ["cue"]]
    for request in wire.requests:
        body = json.loads(request.content)
        assert (request.method, str(request.url)) == ("POST", contract.url)
        _assert_nothing_but_the_request(request)
        assert body["model"] == contract.model
        assert {name: body.get(name) for name in contract.fields} == dict(contract.fields)
        assert contract.dimensions_field not in body


@pytest.mark.parametrize("row", _rows_for("embedding", "dimensions"))
async def test_a_set_dimension_reaches_the_provider_whatever_the_model_is_called(wire, row) -> None:
    contract = row.embedding
    wire.respond = _embed(contract)
    runtime = _embedding_runtime(_invocation(row, "embedding", parameters={"dimensions": 2}), num_retries=0)

    assert await runtime.embed_query("alpha") == [5.0, 0.5]

    [request] = wire.requests
    body = json.loads(request.content)
    assert (body["model"], body[contract.dimensions_field]) == (contract.model, 2)
    _assert_nothing_but_the_request(request)


# -- Timeouts -----------------------------------------------------------------------

_MESSAGES = [{"role": "user", "content": _PROMPT}]


async def _ended(call: Awaitable[Any]) -> tuple[BaseException | None, float]:
    """Await ``call``; return what it raised and how long it took."""
    started = time.monotonic()
    try:
        await call
    except Exception as exc:
        return exc, time.monotonic() - started
    return None, time.monotonic() - started


def _sdk_request(kind: ModelKind, invocation: ResolvedInvocation) -> Awaitable[Any]:
    """The request alone, as the SDK sends it with a timeout and no retries."""
    if kind == "generation":
        return litellm.acompletion(
            model=invocation.route,
            messages=_MESSAGES,
            timeout=_REQUEST_TIMEOUT_S,
            num_retries=0,
            **invocation.request_kwargs(),
        )
    return litellm.aembedding(
        model=invocation.route,
        input=["alpha"],
        timeout=_REQUEST_TIMEOUT_S,
        max_retries=0,
        **invocation.request_kwargs(),
    )


def _product_call(kind: ModelKind, invocation: ResolvedInvocation) -> Awaitable[Any]:
    """The call the product makes, bounded by its own deadline."""
    if kind == "generation":
        client = _structured_client(invocation, timeout_s=_REQUEST_TIMEOUT_S)
        return client.answer_schema(_PROMPT, response_format=_Ranking)
    return _embedding_runtime(invocation, timeout_s=_REQUEST_TIMEOUT_S).embed_query("alpha")


@pytest.mark.parametrize("mode", STALL_MODES)
@pytest.mark.parametrize(("row", "kind"), _row_kinds("timeout"))
async def test_a_request_ends_at_its_timeout_however_the_endpoint_stalls(
    provider_clients, row, kind, mode
) -> None:
    """A provider that keeps the connection busy holds neither the request nor the call past its time."""
    contract = row.contract(kind)
    with stalling_endpoint(mode) as address:
        invocation = _invocation(row, kind, endpoint=f"{address}{row.api_path}")
        calls = [_product_call(kind, invocation)]
        if contract.timeout_error is not None:
            calls.append(_sdk_request(kind, invocation))
        ended = await asyncio.gather(*(_ended(call) for call in calls))

    (product_error, product_s), *sdk = ended
    assert product_s < _ENDED_AT_ITS_TIMEOUT_S
    assert failure_retryable(product_error) is True
    if kind == "generation":
        assert isinstance(product_error, StructuredLlmError)
        assert product_error.terminal_category == "deadline_exceeded"
    else:
        assert isinstance(product_error, EmbeddingDeadlineExceeded)
    for sdk_error, sdk_s in sdk:
        assert sdk_s < _ENDED_AT_ITS_TIMEOUT_S
        assert isinstance(sdk_error, contract.timeout_error)
        assert is_retryable_provider_error(sdk_error) is True


# -- Redirects ----------------------------------------------------------------------


def _redirect_elsewhere(request: httpx.Request) -> httpx.Response:
    return httpx.Response(307, headers={"location": f"https://{_ELSEWHERE}{request.url.path}"})


@pytest.mark.parametrize(("row", "kind"), _row_kinds("redirect"))
async def test_a_redirect_fails_the_request_and_its_key_and_payload_are_not_sent_on(wire, row, kind) -> None:
    contract = row.contract(kind)
    wire.respond = _redirect_elsewhere

    if kind == "generation":
        client = _structured_client(_invocation(row, kind))
        with pytest.raises(StructuredLlmError) as raised:
            await client.answer_schema(_PROMPT, response_format=_Ranking)
    else:
        runtime = _embedding_runtime(_invocation(row, kind), num_retries=3)
        with pytest.raises(EmbeddingProviderError) as raised:
            await runtime.embed_query("alpha")
        assert len(wire.requests) == 1

    assert raised.value.status_code == 307
    assert {request.url.host for request in wire.requests} == {httpx.URL(contract.url).host}


# -- Error classification -----------------------------------------------------------


@pytest.mark.parametrize("status", list(STATUS_CLASSIFICATION))
@pytest.mark.parametrize(("row", "kind"), _row_kinds("error_classification"))
async def test_a_failed_request_is_sent_again_and_its_run_retried_only_when_its_status_is_transient(
    wire, row, kind, status
) -> None:
    """One retry is allowed; the provider's own error format reports the status the same way for every row."""
    contract = row.contract(kind)
    expected = STATUS_CLASSIFICATION[status]
    wire.respond = lambda _request: httpx.Response(status, json=contract.error(status))

    if kind == "generation":
        client = _structured_client(_invocation(row, kind), num_retries=1)
        with pytest.raises(StructuredLlmError) as raised:
            await client.answer_schema(_PROMPT, response_format=_Ranking)
        first = wire.requests[0].content
        # A rejected schema request may be followed by its JSON-text form,
        # which is another request, not the same one again.
        sent = sum(request.content == first for request in wire.requests)
    else:
        runtime = _embedding_runtime(_invocation(row, kind), num_retries=1, retry_pauses_s=(0.0,))
        with pytest.raises(EmbeddingProviderError) as raised:
            await runtime.embed_query("alpha")
        sent = len(wire.requests)

    assert sent == (2 if expected.sent_again else 1)
    assert failure_retryable(raised.value) is expected.run_retryable
    assert raised.value.status_code == status
    assert {request.url.host for request in wire.requests} == {httpx.URL(contract.url).host}
