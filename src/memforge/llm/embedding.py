"""Embedding Runtime: text to vectors through the LiteLLM SDK.

One runtime stands for one embedding model, named by the invocation it is
built from. It owns everything between "embed these texts" and a list of
usable vectors: the attempt loop, the logical deadline, the per-request item
limit, validation of what the provider returned, and the cache of vectors it
has produced.

A runtime built with the identity of its model also knows the vector space it
embeds in: once the model has answered, the runtime has the fingerprint of its
Embedding Profile, and ``embed_documents_for_index`` and
``embed_query_for_index`` hand its vectors only to an index built with that
fingerprint.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import litellm

from memforge.llm.embedding_profile import EmbeddingIdentity, EmbeddingProfile
from memforge.llm.providers import ResolvedInvocation
from memforge.llm.transport import (
    is_connection_failure,
    is_retryable_provider_error,
    provider_retry_after_seconds,
    provider_status_code,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingDeadlineExceeded",
    "EmbeddingError",
    "EmbeddingIndexMismatch",
    "EmbeddingNotConfigured",
    "EmbeddingProviderError",
    "EmbeddingRuntime",
    "EmbeddingSettings",
    "InvalidEmbeddingResponse",
    "ProfileObserver",
    "embed_documents_for_index",
    "embed_query_for_index",
]

EmbeddingTransport = Callable[..., Awaitable[Any]]
ProfileObserver = Callable[[EmbeddingProfile], Awaitable[None]]


class EmbeddingError(RuntimeError):
    """An embedding operation produced no usable vectors."""


class EmbeddingNotConfigured(EmbeddingError):
    """No embedding model is configured."""


class EmbeddingProviderError(EmbeddingError):
    """The provider request failed and the attempt budget did not recover it.

    This includes an answer the SDK cannot read as an embedding payload at
    all, such as an HTML page from a proxy: the SDK reports it as a server
    fault, so it is attempted again like one.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        unreachable: bool = False,
        parameter_refused: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        # The request never reached the provider.
        self.unreachable = unreachable
        # The SDK sent nothing, because the route does not take a parameter the
        # request sets.
        self.parameter_refused = parameter_refused


class EmbeddingDeadlineExceeded(EmbeddingError):
    """The logical deadline passed before every vector was produced."""


class EmbeddingIndexMismatch(EmbeddingError):
    """The embedding model in use is not the one the vector index was built with.

    Its vectors belong to another vector space, so they are neither compared
    with the index's vectors nor written among them. The index has to be built
    again with this model before the two are used together.
    """

    code = "reindex_required"


class InvalidEmbeddingResponse(EmbeddingError, ValueError):
    """The provider answered with an embedding payload that does not hold one valid vector per input.

    The same request would get the same answer, so it is not attempted again.
    """


@dataclass(frozen=True)
class EmbeddingSettings:
    """What defines one Embedding Runtime."""

    # The route, endpoint and credential every request is made with. ``None``
    # is an installation with no embedding model.
    invocation: ResolvedInvocation | None
    # The settings that decide the model's vector space. A runtime without
    # them has no fingerprint, and only an index that records none takes its
    # vectors.
    identity: EmbeddingIdentity | None = None
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


class EmbeddingRuntime:
    """Embeds queries and documents with one embedding model.

    Queries and documents share one vector space and one cache. The cache
    belongs to this instance, so a runtime built from other settings can never
    serve vectors this one produced.

    ``profile_observer`` is told the runtime's Embedding Profile once, when the
    model's first answer makes it known and before that answer is returned.
    """

    def __init__(
        self,
        settings: EmbeddingSettings,
        *,
        transport: EmbeddingTransport | None = None,
        profile_observer: ProfileObserver | None = None,
    ) -> None:
        self.settings = settings
        self._transport = transport
        self._profile_observer = profile_observer
        self._cache = _VectorCache(settings.cache_size)
        self._dimension: int | None = None
        self._provider_model: str | None = None

    @property
    def configured(self) -> bool:
        """Whether an embedding model is configured."""
        return self.settings.invocation is not None

    @property
    def dimension(self) -> int | None:
        """The vector length this runtime has observed, once it has embedded."""
        return self._dimension

    @property
    def fingerprint(self) -> str | None:
        """The fingerprint of the Embedding Profile this runtime embeds in.

        It is known once the model has answered, because the vector length it
        answers with is part of the profile.
        """
        identity = self.settings.identity
        if identity is None or self._dimension is None:
            return None
        return identity.profile(self._dimension).fingerprint

    def embeds_for(self, index_fingerprint: str | None) -> bool | None:
        """Whether this runtime's vectors belong in an index with ``index_fingerprint``.

        An index that records no fingerprint is bound to no model and takes
        any vector. ``None`` means it cannot be said yet: the runtime knows its
        model and has not seen a vector of it.
        """
        if index_fingerprint is None:
            return True
        if self.settings.identity is not None and self._dimension is None:
            return None
        return self.fingerprint == index_fingerprint

    @property
    def provider_model(self) -> str | None:
        """The model id the provider reported in its latest answer, if it reported one."""
        return self._provider_model

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
        invocation = self.settings.invocation
        if invocation is None:
            raise EmbeddingNotConfigured("Embedding model is not configured")
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
                            invocation,
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
        first_answer = self._dimension is None
        self._dimension = dimension
        if first_answer:
            await self._announce_profile(dimension)
        return produced

    async def _announce_profile(self, dimension: int) -> None:
        identity = self.settings.identity
        if identity is None or self._profile_observer is None:
            return
        try:
            await self._profile_observer(identity.profile(dimension))
        except Exception:
            logger.exception("The observer of the Embedding Profile failed")

    async def _request_slice(
        self,
        invocation: ResolvedInvocation,
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
                    model=invocation.route,
                    input=texts,
                    timeout=remaining_s,
                    # This loop owns the retry budget, so that every attempt
                    # counts against the one deadline.
                    max_retries=0,
                    **invocation.request_kwargs(),
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
            vectors = _response_vectors(response, expected_count=len(texts))
            self._provider_model = _response_model(response)
            return vectors

    def _retry_pause_s(self, exc: Exception, *, retries_used: int) -> float:
        """Return the pause before the next retry: the scheduled one, or longer if the provider asks."""
        pauses = self.settings.retry_pauses_s
        scheduled_s = pauses[min(retries_used, len(pauses) - 1)] if pauses else 0.0
        return max(scheduled_s, provider_retry_after_seconds(exc) or 0.0)


class _FingerprintedIndex(Protocol):
    async def index_fingerprint(self) -> str | None: ...


async def embed_documents_for_index(
    runtime: EmbeddingRuntime,
    index: _FingerprintedIndex,
    texts: Sequence[str],
) -> list[list[float]]:
    """Embed document texts as vectors of ``index``.

    Raises ``EmbeddingIndexMismatch`` when the index was built with another
    embedding model. When that is already known no request is made. An index
    that records no fingerprint is asked again after the request: the
    runtime's first answer can be what gives an installation its first index.
    """
    index_fingerprint = await index.index_fingerprint()
    _require_index(runtime, index_fingerprint, unknown_passes=True)
    vectors = await runtime.embed_documents(texts)
    if index_fingerprint is None:
        index_fingerprint = await index.index_fingerprint()
    _require_index(runtime, index_fingerprint, unknown_passes=not texts)
    return vectors


async def embed_query_for_index(
    runtime: EmbeddingRuntime,
    index: _FingerprintedIndex,
    text: str,
) -> list[float]:
    """Embed one search query as a vector of ``index``, under the rule of ``embed_documents_for_index``."""
    index_fingerprint = await index.index_fingerprint()
    _require_index(runtime, index_fingerprint, unknown_passes=True)
    vector = await runtime.embed_query(text)
    if index_fingerprint is None:
        index_fingerprint = await index.index_fingerprint()
    _require_index(runtime, index_fingerprint, unknown_passes=False)
    return vector


def _require_index(
    runtime: EmbeddingRuntime,
    index_fingerprint: str | None,
    *,
    unknown_passes: bool,
) -> None:
    embeds = runtime.embeds_for(index_fingerprint)
    if embeds or (embeds is None and unknown_passes):
        return
    raise EmbeddingIndexMismatch(
        "The embedding model in use is not the one the vector index was built with; "
        "the index has to be rebuilt"
    )


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
    if isinstance(exc, litellm.UnsupportedParamsError):
        return EmbeddingProviderError(
            f"Embedding route does not take a parameter the request sets: {exc}",
            parameter_refused=True,
        )
    unreachable = is_connection_failure(exc)
    prefix = "Embedding provider unreachable" if unreachable else "Embedding provider request failed"
    return EmbeddingProviderError(
        f"{prefix}: {type(exc).__name__}: {exc}",
        status_code=provider_status_code(exc),
        unreachable=unreachable,
    )


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


def _response_model(response: object) -> str | None:
    model = _item_field(response, "model")
    return model.strip() or None if isinstance(model, str) else None


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
