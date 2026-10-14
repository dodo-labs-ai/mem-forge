"""Embedding Runtimes and provider stand-ins for tests that reach no provider."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from memforge.llm.embedding import EmbeddingRuntime, EmbeddingSettings

EmbeddingTransport = Callable[..., Awaitable[dict[str, Any]]]


def embedding_response(vectors: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Shape vectors as a provider response, one item per input in input order."""
    return {
        "data": [
            {"index": index, "embedding": list(vector)}
            for index, vector in enumerate(vectors)
        ]
    }


def texts_transport(embed: Callable[[list[str]], Sequence[Sequence[float]]]) -> EmbeddingTransport:
    """A provider that answers each request with ``embed(texts)``."""

    async def transport(*, input: Sequence[str], **_request: Any) -> dict[str, Any]:
        return embedding_response(embed(list(input)))

    return transport


def fixed_vector_transport(vector: Sequence[float]) -> EmbeddingTransport:
    """A provider that answers every text with the same vector."""
    return texts_transport(lambda texts: [vector for _ in texts])


def unconfigured_embedding() -> EmbeddingRuntime:
    """A runtime with no model: every embed call fails, as on a fresh install."""
    return EmbeddingRuntime(EmbeddingSettings(model=""))


def embedding_from_texts(
    embed: Callable[[list[str]], Sequence[Sequence[float]]],
) -> EmbeddingRuntime:
    """A configured runtime whose provider answers with ``embed(texts)``."""
    return EmbeddingRuntime(
        EmbeddingSettings(model="test-embedding", base_url="http://embed.test", api_key="key"),
        transport=texts_transport(embed),
    )


def fixed_vector_embedding(vector: Sequence[float]) -> EmbeddingRuntime:
    """A configured runtime that embeds every text as the same vector."""
    return embedding_from_texts(lambda texts: [vector for _ in texts])
