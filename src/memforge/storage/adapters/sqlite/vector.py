"""SqliteVectorStore: wraps the Chroma memories collection.

Owns every distance/score conversion in one place so no caller assumes a
distance metric: similarity() for ranking, within_dedup_threshold() for the
duplicate decision.
"""

from __future__ import annotations

import asyncio
from typing import Any, Sequence

from memforge.retrieval.access_predicate import visible_chroma_where
from memforge.retrieval.vector_metadata import upsert_with_stored_vector_hash
from memforge.storage.adapters.context import AccessScope

__all__ = ["SqliteVectorStore", "SIMILARITY_FLOOR"]

# Cosine distance can exceed 1.0 for opposed vectors, which would yield a
# negative similarity. Scores feed rank fusion, where negatives are
# meaningless, so similarity is floored here.
SIMILARITY_FLOOR = 0.0


class SqliteVectorStore:
    """The embedding channel backed by a single Chroma collection."""

    distance_metric = "cosine"

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    def similarity(self, distance: float) -> float:
        return max(1.0 - distance, SIMILARITY_FLOOR)

    def within_dedup_threshold(self, distance_threshold: float, score: float) -> bool:
        """Whether a returned score is close enough to be a near-duplicate.

        The configured threshold is a distance; the channel returns a score.
        This is the one place the two axes meet, so a caller compares
        duplicates without reconstructing the distance itself.
        """
        return (1.0 - score) < distance_threshold

    async def query(
        self,
        embedding: Sequence[float],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        where = visible_chroma_where(scope, memory_types)
        params: dict[str, Any] = {
            "query_embeddings": [list(embedding)],
            "n_results": limit,
        }
        if where:
            params["where"] = where
        results = await asyncio.to_thread(lambda: self.collection.query(**params))
        if not results or not results.get("ids") or not results["ids"][0]:
            return []
        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        return [(mid, self.similarity(dist)) for mid, dist in zip(ids, distances)]

    async def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        await asyncio.to_thread(
            upsert_with_stored_vector_hash,
            self.collection,
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            metadatas=[dict(m) for m in metadatas],
        )

    async def delete(self, ids: Sequence[str]) -> None:
        await asyncio.to_thread(lambda: self.collection.delete(ids=list(ids)))

    async def get_record(self, memory_id: str) -> dict[str, Any] | None:
        raw = await asyncio.to_thread(
            lambda: self.collection.get(
                ids=[memory_id],
                include=["embeddings", "metadatas"],
            )
        )
        returned = raw.get("ids") or []
        if not returned:
            return None
        # Embeddings can arrive as a NumPy array whose truthiness is ambiguous,
        # so length is the only safe emptiness test here.
        embeddings = raw.get("embeddings")
        metadatas = raw.get("metadatas") or []
        return {
            "id": returned[0],
            "embedding": embeddings[0] if embeddings is not None and len(embeddings) else None,
            "metadata": (metadatas[0] if len(metadatas) else {}) or {},
        }
