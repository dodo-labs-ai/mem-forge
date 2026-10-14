"""ChromaDB collection management for the Memory vector index."""

from __future__ import annotations

import threading
from typing import Any

import chromadb

__all__ = ["get_chroma_collection"]


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
