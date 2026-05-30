"""Helpers for Chroma vector metadata that depends on stored embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memforge.memory.index_payloads import embedding_vector_hash

__all__ = ["upsert_with_stored_vector_hash"]


def upsert_with_stored_vector_hash(
    collection: Any,
    *,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    metadatas: Sequence[dict[str, Any]],
    documents: Sequence[str] | None = None,
) -> None:
    """Upsert vectors, then stamp metadata with Chroma's stored vector hash.

    Chroma may return embeddings with tiny numeric differences from the Python
    list passed to ``upsert``. Index health compares against Chroma's persisted
    payload, so the metadata hash is derived after the round trip.
    """
    kwargs: dict[str, Any] = {
        "ids": list(ids),
        "embeddings": list(embeddings),
        "metadatas": [_without_vector_hash(metadata) for metadata in metadatas],
    }
    if documents is not None:
        kwargs["documents"] = list(documents)
    collection.upsert(**kwargs)
    _refresh_stored_vector_hashes(collection, list(ids))


def _refresh_stored_vector_hashes(collection: Any, ids: list[str]) -> None:
    if not ids or not hasattr(collection, "get"):
        return

    raw = collection.get(ids=ids, include=["embeddings", "metadatas", "documents"])
    returned_ids = raw.get("ids") or []
    embeddings = raw.get("embeddings")
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents")
    if embeddings is None:
        return

    for index, record_id in enumerate(returned_ids):
        if index >= len(embeddings) or embeddings[index] is None:
            continue
        metadata = dict(metadatas[index] if index < len(metadatas) and metadatas[index] else {})
        metadata["embedding_vector_hash"] = embedding_vector_hash(embeddings[index])
        _update_metadata(collection, record_id, metadata, embeddings[index], _document_at(documents, index))


def _without_vector_hash(metadata: dict[str, Any]) -> dict[str, Any]:
    clean = dict(metadata)
    clean.pop("embedding_vector_hash", None)
    return clean


def _document_at(documents: Any, index: int) -> str | None:
    if documents is None:
        return None
    if index >= len(documents):
        return None
    return documents[index]


def _update_metadata(
    collection: Any,
    record_id: str,
    metadata: dict[str, Any],
    embedding: Sequence[float],
    document: str | None,
) -> None:
    if hasattr(collection, "update"):
        collection.update(ids=[record_id], metadatas=[metadata])
        return

    kwargs: dict[str, Any] = {
        "ids": [record_id],
        "embeddings": [embedding],
        "metadatas": [metadata],
    }
    if document is not None:
        kwargs["documents"] = [document]
    collection.upsert(**kwargs)
