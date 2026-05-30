"""Embedding utilities and ChromaDB collection management."""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Any

import chromadb
import httpx

logger = logging.getLogger(__name__)

__all__ = ["embed_texts", "get_chroma_collection"]


# ---------------------------------------------------------------------------
# Embedding API (OpenAI-compatible)
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    base_url: str,
    api_key: str,
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
            data = resp.json()
            # Sort by index to guarantee order matches input
            sorted_items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_items]
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


# ---------------------------------------------------------------------------
# ChromaDB collection management
# ---------------------------------------------------------------------------

# Singleton client cache — avoids creating multiple PersistentClient instances
# to the same path, which causes SQLite lock contention inside ChromaDB.
_chroma_clients: dict[str, chromadb.ClientAPI] = {}
_chroma_lock = threading.Lock()


def get_chroma_collection(
    chroma_path: str,
    name: str = "documents",
) -> Any:
    """Open or create a ChromaDB collection with cosine similarity.

    Supports multiple collections on the same PersistentClient path:
    - "documents" for document-level embeddings
    - "memories" for memory-level embeddings

    Uses a thread-safe module-level singleton for each path to avoid lock contention.
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
