"""Embedding Runtime: text to vectors through the LiteLLM SDK.

One runtime stands for one embedding configuration. It owns everything between
"embed these texts" and a list of usable vectors: the provider route, the
attempt loop, the logical deadline, the per-request item limit, validation of
what the provider returned, and the cache of vectors it has produced.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import litellm

from memforge.llm.providers import litellm_optional_kwargs
from memforge.llm.transport import (
    is_retryable_provider_error,
    provider_retry_after_seconds,
    provider_status_code,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingDeadlineExceeded",
    "EmbeddingError",
    "EmbeddingNotConfigured",
    "EmbeddingProviderError",
    "EmbeddingRuntime",
    "EmbeddingSettings",
    "InvalidEmbeddingResponse",
    "litellm_embedding_route",
]

EmbeddingTransport = Callable[..., Awaitable[Any]]

# An OpenAI-compatible endpoint that needs no key ignores the Authorization
# header, but the SDK builds no request without a key. A model served by such
# an endpoint sends this value when no key is configured, so a key from the
# process environment is never sent to an endpoint the operator did not pair
# it with.
_KEYLESS_API_KEY = "no-api-key"


class EmbeddingError(RuntimeError):
    """An embedding operation produced no usable vectors."""


class EmbeddingNotConfigured(EmbeddingError):
    """The settings name no model, or no endpoint for a model that needs one."""


class EmbeddingProviderError(EmbeddingError):
    """The provider request failed and the attempt budget did not recover it.

    This includes an answer the SDK cannot read as an embedding payload at
    all, such as an HTML page from a proxy: the SDK reports it as a server
    fault, so it is attempted again like one.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class EmbeddingDeadlineExceeded(EmbeddingError):
    """The logical deadline passed before every vector was produced."""


class InvalidEmbeddingResponse(EmbeddingError, ValueError):
    """The provider answered with an embedding payload that does not hold one valid vector per input.

    The same request would get the same answer, so it is not attempted again.
    """


@dataclass(frozen=True)
class EmbeddingSettings:
    """What defines one Embedding Runtime."""

    # A LiteLLM ``provider/model`` route, or the model id of an
    # OpenAI-compatible endpoint at ``base_url``.
    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    # One budget of time for a whole embed call: every request, every retry and
    # every pause between them.
    timeout_s: float = 60.0
    # One budget of retries for a whole embed call, spent on transient failures.
    num_retries: int = 3
    # The pause before each retry in turn; the last one serves every later
    # retry. Long enough for a short rate limit or a restarting model server
    # to clear within the retry budget.
    retry_pauses_s: tuple[float, ...] = (2.0, 4.0, 8.0)
    # The most inputs one provider request may carry. The provider client
    # decodes a response on the event loop, one number at a time, so this
    # bounds how long a single response keeps the loop from other work.
    max_inputs_per_request: int = 32
    cache_size: int = 256


def litellm_embedding_route(model: str) -> str:
    """Compile a configured embedding model into a LiteLLM route.

    A name whose first segment is a provider LiteLLM knows, as in
    ``ollama/nomic-embed-text``, names its own route. Any other name is the
    model id of an OpenAI-compatible embeddings endpoint and takes the
    ``openai/`` route; such an id may contain a slash itself, as
    ``BAAI/bge-m3`` does.
    """
    name = model.strip()
    if _names_provider_route(name):
        return name
    return f"openai/{name}"


def _names_provider_route(model: str) -> bool:
    provider, separator, provider_model = model.partition("/")
    if not separator or not provider_model:
        return False
    return provider in litellm.provider_list or any(
        registered.get("provider") == provider for registered in litellm.custom_provider_map
    )


class EmbeddingRuntime:
    """Embeds queries and documents for one embedding configuration.

    Queries and documents share one vector space and one cache. The cache
    belongs to this instance, so a runtime built from other settings can never
    serve vectors this one produced.
    """

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: EmbeddingTransport | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._cache = _VectorCache(settings.cache_size)
        self._dimension: int | None = None

    @property
    def configured(self) -> bool:
        """Whether the settings name a model and a way to reach it.

        A ``provider/model`` route may resolve its endpoint and credentials
        from the provider's own environment. The model id of an
        OpenAI-compatible endpoint needs both a base URL and an API key.
        """
        model = self.settings.model.strip()
        if not model:
            return False
        if _names_provider_route(model):
            return True
        return bool(self.settings.base_url and self.settings.api_key)

    @property
    def dimension(self) -> int | None:
        """The vector length this runtime has observed, once it has embedded."""
        return self._dimension

    async def embed_query(self, text: str) -> list[float]:
        """Return the vector for one search query."""
        return (await self._embed((text,)))[0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per document text, in the order given.

        The call succeeds for every text or fails as a whole: when any provider
        request fails, nothing is returned and nothing is cached.
        """
        return await self._embed(tuple(texts))

    async def _embed(self, texts: tuple[str, ...]) -> list[list[float]]:
        if not texts:
            return []
        vectors: dict[str, tuple[float, ...]] = {}
        uncached: list[str] = []
        for text in dict.fromkeys(texts):
            cached = self._cache.get(text)
            if cached is None:
                uncached.append(text)
            else:
                vectors[text] = cached
        if uncached:
            produced = await self._request_vectors(uncached)
            for text, vector in zip(uncached, produced, strict=True):
                self._cache.put(text, vector)
                vectors[text] = vector
        return [list(vectors[text]) for text in texts]

    async def _request_vectors(self, texts: list[str]) -> list[tuple[float, ...]]:
        route = self._route()
        limit = max(1, self.settings.max_inputs_per_request)
        timeout_s = max(0.0, self.settings.timeout_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        attempts = _AttemptBudget(retries_left=max(0, self.settings.num_retries))
        produced: list[tuple[float, ...]] = []
        try:
            async with asyncio.timeout_at(deadline):
                for start in range(0, len(texts), limit):
                    produced.extend(
                        await self._request_slice(
                            route,
                            texts[start : start + limit],
                            deadline=deadline,
                            attempts=attempts,
                        )
                    )
        except TimeoutError as exc:
            raise EmbeddingDeadlineExceeded(
                f"Embedding deadline exceeded after {timeout_s:g}s"
            ) from exc
        dimension = _single_dimension(produced, expected=self._dimension)
        self._dimension = dimension
        return produced

    async def _request_slice(
        self,
        route: str,
        texts: list[str],
        *,
        deadline: float,
        attempts: _AttemptBudget,
    ) -> list[tuple[float, ...]]:
        loop = asyncio.get_running_loop()
        transport = litellm.aembedding if self._transport is None else self._transport
        while True:
            remaining_s = deadline - loop.time()
            if remaining_s <= 0:
                raise TimeoutError
            try:
                response = await transport(
                    model=route,
                    input=texts,
                    timeout=remaining_s,
                    # This loop owns the retry budget, so that every attempt
                    # counts against the one deadline.
                    max_retries=0,
                    **self._endpoint_kwargs(),
                )
            except Exception as exc:
                if not is_retryable_provider_error(exc) or attempts.retries_left <= 0:
                    raise _provider_error(exc) from exc
                pause_s = self._retry_pause_s(exc, retries_used=attempts.retries_used)
                # A pause that outlasts the deadline leaves no attempt to wait for.
                if pause_s >= deadline - loop.time():
                    raise _provider_error(exc) from exc
                attempts.retries_left -= 1
                attempts.retries_used += 1
                logger.warning(
                    "Embedding request failed (%s, status=%s); retry %d in %.2fs",
                    type(exc).__name__,
                    provider_status_code(exc),
                    attempts.retries_used,
                    pause_s,
                )
                await asyncio.sleep(pause_s)
                continue
            return _response_vectors(response, expected_count=len(texts))

    def _retry_pause_s(self, exc: Exception, *, retries_used: int) -> float:
        """Return the pause before the next retry: the scheduled one, or longer if the provider asks."""
        pauses = self.settings.retry_pauses_s
        scheduled_s = pauses[min(retries_used, len(pauses) - 1)] if pauses else 0.0
        return max(scheduled_s, provider_retry_after_seconds(exc) or 0.0)

    def _endpoint_kwargs(self) -> dict[str, str]:
        kwargs = litellm_optional_kwargs(
            api_base=self.settings.base_url,
            api_key=self.settings.api_key,
        )
        if not _names_provider_route(self.settings.model.strip()):
            kwargs.setdefault("api_key", _KEYLESS_API_KEY)
        return kwargs

    def _route(self) -> str:
        model = self.settings.model.strip()
        if not model:
            raise EmbeddingNotConfigured("Embedding model is not configured")
        # Without a base URL the SDK would pick an endpoint from the process
        # environment and send it the configured key and the texts.
        if not _names_provider_route(model) and not self.settings.base_url:
            raise EmbeddingNotConfigured(
                f"Embedding base URL is not configured for model {model!r}"
            )
        return litellm_embedding_route(model)


@dataclass
class _AttemptBudget:
    retries_left: int
    retries_used: int = 0


class _VectorCache:
    """Least-recently-used vectors keyed by the exact input text."""

    def __init__(self, max_size: int) -> None:
        self._vectors: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._max_size = max(0, max_size)

    def get(self, text: str) -> tuple[float, ...] | None:
        vector = self._vectors.get(text)
        if vector is not None:
            self._vectors.move_to_end(text)
        return vector

    def put(self, text: str, vector: tuple[float, ...]) -> None:
        if self._max_size == 0:
            return
        self._vectors[text] = vector
        self._vectors.move_to_end(text)
        while len(self._vectors) > self._max_size:
            self._vectors.popitem(last=False)

    def __len__(self) -> int:
        return len(self._vectors)


def _provider_error(exc: Exception) -> EmbeddingProviderError:
    prefix = (
        "Embedding provider unreachable"
        if _is_connection_failure(exc)
        else "Embedding provider request failed"
    )
    return EmbeddingProviderError(
        f"{prefix}: {type(exc).__name__}: {exc}",
        status_code=provider_status_code(exc),
    )


def _is_connection_failure(exc: BaseException) -> bool:
    """Whether the request never reached the provider, anywhere in the error's chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, (ConnectionError, httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _response_vectors(response: object, *, expected_count: int) -> list[tuple[float, ...]]:
    """Return the response's vectors in input order, or reject the response."""
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        raise InvalidEmbeddingResponse("Embedding response is missing data")
    if len(data) != expected_count:
        raise InvalidEmbeddingResponse(
            "Embedding response count does not match input count: "
            f"expected {expected_count}, got {len(data)}"
        )
    by_index: dict[int, tuple[float, ...]] = {}
    for item in data:
        by_index[_item_index(item)] = _item_vector(item)
    if sorted(by_index) != list(range(expected_count)):
        raise InvalidEmbeddingResponse(
            "Embedding response indices do not match inputs: "
            f"expected 0..{expected_count - 1}"
        )
    return [by_index[index] for index in range(expected_count)]


def _item_field(item: object, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _item_index(item: object) -> int:
    raw = _item_field(item, "index")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise InvalidEmbeddingResponse("Embedding response indices are incomplete")
    return raw


def _item_vector(item: object) -> tuple[float, ...]:
    raw = _item_field(item, "embedding")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise InvalidEmbeddingResponse("Embedding response item is missing embedding")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in raw):
        raise InvalidEmbeddingResponse("Embedding response contains a non-numeric value")
    try:
        vector = tuple(float(value) for value in raw)
    except OverflowError as exc:
        raise InvalidEmbeddingResponse("Embedding response contains a non-finite value") from exc
    if not all(map(math.isfinite, vector)):
        raise InvalidEmbeddingResponse("Embedding response contains a non-finite value")
    return vector


def _single_dimension(vectors: list[tuple[float, ...]], *, expected: int | None) -> int:
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise InvalidEmbeddingResponse(
            "Embedding response vectors do not share one dimension: "
            f"got {sorted(dimensions)}"
        )
    (dimension,) = dimensions
    if expected is not None and dimension != expected:
        raise InvalidEmbeddingResponse(
            f"Embedding response dimension {dimension} does not match "
            f"the dimension {expected} this runtime already produced"
        )
    return dimension
