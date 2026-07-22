"""Embedding utilities and ChromaDB collection management."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any

import chromadb
import httpx
import litellm

from memforge.llm.providers import is_litellm_provider_model, litellm_optional_kwargs

logger = logging.getLogger(__name__)

__all__ = ["embed_texts", "get_chroma_collection"]

_EMBEDDING_INPUT_BATCH_LIMIT = 2048


# ---------------------------------------------------------------------------
# Embedding API (OpenAI-compatible)
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    base_url: str,
    api_key: str | None,
    model: str,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> list[list[float]]:
    """Call an OpenAI-compatible /v1/embeddings endpoint with retry.

    Returns a list of embedding vectors in the same order as input texts.
    Retries on 5xx errors, connection errors, and timeouts.
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBEDDING_INPUT_BATCH_LIMIT):
        batch = texts[start : start + _EMBEDDING_INPUT_BATCH_LIMIT]
        batch_vectors = _embed_text_batch(
            batch,
            base_url,
            api_key,
            model,
            timeout=timeout,
            max_retries=max_retries,
        )
        if len(batch_vectors) != len(batch):
            raise ValueError(
                "Embedding response count does not match input count: "
                f"expected {len(batch)}, got {len(batch_vectors)}"
            )
        vectors.extend(batch_vectors)
    return vectors


def _embed_text_batch(
    texts: list[str],
    base_url: str,
    api_key: str | None,
    model: str,
    *,
    timeout: float,
    max_retries: int,
) -> list[list[float]]:
    if is_litellm_provider_model(model):
        response = litellm.embedding(
            model=model,
            input=texts,
            timeout=timeout,
            num_retries=max_retries,
            **litellm_optional_kwargs(api_base=base_url or None, api_key=api_key),
        )
        return _embedding_vectors(response, expected_count=len(texts))

    url = f"{base_url.rstrip('/')}/embeddings"
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            resp = httpx.post(
                url,
                json={"model": model, "input": texts},
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                timeout=timeout,
            )
            resp.raise_for_status()
            return _embedding_vectors(resp.json(), expected_count=len(texts))
        except (httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                import time

                delay = 2.0 * (2 ** attempt)
                logger.warning(
                    "Embedding API call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error("Embedding API call failed after %d attempts: %s", max_retries + 1, e)

    raise last_error  # type: ignore[misc]


def _embedding_vectors(
    response: object,
    *,
    expected_count: int,
) -> list[list[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if not isinstance(data, list):
        raise ValueError("Embedding response is missing data")
    if len(data) != expected_count:
        raise ValueError(
            "Embedding response count does not match input count: "
            f"expected {expected_count}, got {len(data)}"
        )

    def item_index(item: object) -> int:
        if isinstance(item, dict):
            if "index" not in item:
                raise ValueError("Embedding response indices are incomplete")
            return int(item["index"])
        if not hasattr(item, "index"):
            raise ValueError("Embedding response indices are incomplete")
        return int(getattr(item, "index"))

    def item_embedding(item: object) -> list[float]:
        if isinstance(item, dict):
            embedding = item.get("embedding")
        else:
            embedding = getattr(item, "embedding", None)
        if not isinstance(embedding, list):
            raise ValueError("Embedding response item is missing embedding")
        return embedding

    indexed_items = [(item_index(item), item) for item in data]
    actual_indices = sorted(index for index, _item in indexed_items)
    expected_indices = list(range(expected_count))
    if actual_indices != expected_indices:
        raise ValueError(
            "Embedding response indices do not match inputs: "
            f"expected {expected_indices[0] if expected_indices else 0}.."
            f"{expected_indices[-1] if expected_indices else -1}"
        )
    return [
        item_embedding(item)
        for _index, item in sorted(indexed_items, key=lambda pair: pair[0])
    ]


# ---------------------------------------------------------------------------
# ChromaDB collection management
# ---------------------------------------------------------------------------

# Singleton client cache — avoids creating multiple PersistentClient instances
# to the same path, which causes SQLite lock contention inside ChromaDB.
_chroma_clients: dict[str, chromadb.ClientAPI] = {}
_chroma_lock = threading.Lock()


def get_chroma_collection(
    chroma_path: str,
    name: str = "memories",
) -> Any:
    """Open the Memory vector collection with cosine similarity.

    ``name`` remains explicit for tests and future contracted collections; the
    production default is Memory-only and never recreates the retired document
    vector projection.
    """
    with _chroma_lock:
        if chroma_path not in _chroma_clients:
            _chroma_clients[chroma_path] = chromadb.PersistentClient(path=chroma_path)
        client = _chroma_clients[chroma_path]
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------------------
# Embedding cache (LRU, for query embeddings)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """Simple LRU cache for query embeddings.

    Saves the ~50-200ms OpenAI API round-trip for repeated/similar queries.
    """

    def __init__(self, max_size: int = 256):
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def stats(self) -> dict:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}
