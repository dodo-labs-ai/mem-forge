"""Document vector index ownership helpers."""

from __future__ import annotations

from typing import Any

from memforge.retrieval.vector_metadata import upsert_with_stored_vector_hash

__all__ = ["DocumentVectorIndex"]


def _first(values: Any, default: Any) -> Any:
    if values is None:
        return default
    if len(values) == 0:
        return default
    return values[0]


class DocumentVectorIndex:
    """Own document-vector collection reads, writes, snapshots, and restores."""

    def __init__(self, collection: Any | None) -> None:
        self.collection = collection

    @property
    def enabled(self) -> bool:
        return self.collection is not None

    def is_current(
        self,
        doc_id: str,
        *,
        content_hash: str,
        version: str | None,
    ) -> bool:
        if self.collection is None:
            return True
        raw = self.collection.get(ids=[doc_id], include=["metadatas"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        if not ids or not metadatas:
            return False
        metadata = metadatas[0] or {}
        return (
            metadata.get("content_hash") == content_hash
            and metadata.get("version") == (version or "")
        )

    def snapshot(self, doc_id: str) -> dict[str, Any] | None:
        if self.collection is None:
            return None
        raw = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "documents", "metadatas"],
        )
        ids = raw.get("ids") or []
        if not ids:
            return None
        return {
            "id": ids[0],
            "embedding": _first(raw.get("embeddings"), None),
            "document": _first(raw.get("documents"), None),
            "metadata": _first(raw.get("metadatas"), {}) or {},
        }

    def upsert(
        self,
        *,
        doc_id: str,
        embedding: list[float],
        document: str,
        metadata: dict[str, Any],
    ) -> None:
        if self.collection is None:
            return
        upsert_with_stored_vector_hash(
            self.collection,
            ids=[doc_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )

    def delete(self, doc_id: str) -> None:
        if self.collection is None:
            return
        self.collection.delete(ids=[doc_id])

    def restore(self, doc_id: str, snapshot: dict[str, Any] | None) -> None:
        if self.collection is None:
            return
        if snapshot is None:
            self.delete(doc_id)
            return

        kwargs: dict[str, Any] = {
            "ids": [snapshot["id"]],
            "metadatas": [snapshot.get("metadata") or {}],
        }
        if snapshot.get("embedding") is not None:
            kwargs["embeddings"] = [snapshot["embedding"]]
        if snapshot.get("document") is not None:
            kwargs["documents"] = [snapshot["document"]]
        self.collection.upsert(**kwargs)
