"""SqliteVectorStore: wraps the Chroma collection of the Memory vector index.

Owns every distance/score conversion in one place so no caller assumes a
distance metric: similarity() for ranking, within_dedup_threshold() for the
duplicate decision.

A store reaches its collection in one of two ways. Built over an open
collection, it works on that collection. Built with ``of_index_in_use``, it
opens the index in use for every operation, so it follows a rebuilt index the
moment that index is put into use and never keeps a collection that has been
deleted.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, Sequence

from memforge.retrieval.access_predicate import visible_chroma_where
from memforge.retrieval.vector_index import MemoryVectorIndex
from memforge.retrieval.vector_metadata import upsert_with_stored_vector_hash
from memforge.storage.adapters.context import AccessScope
from memforge.vector_index_errors import NoVectorIndex

__all__ = ["SqliteVectorStore", "SIMILARITY_FLOOR"]

IndexOpener = Callable[[], Awaitable[MemoryVectorIndex | None]]

# Cosine distance can exceed 1.0 for opposed vectors, which would yield a
# negative similarity. Scores feed rank fusion, where negatives are
# meaningless, so similarity is floored here.
SIMILARITY_FLOOR = 0.0


class SqliteVectorStore:
    """The embedding channel backed by the Chroma collection of one vector index."""

    distance_metric = "cosine"

    def __init__(self, collection: Any, *, index_fingerprint: str | None = None) -> None:
        """Work on ``collection``, an index built with ``index_fingerprint``.

        A collection given without a fingerprint is bound to no embedding
        model.
        """
        self.collection = collection
        self._index = MemoryVectorIndex(collection=collection, fingerprint=index_fingerprint)
        self._open_index: IndexOpener | None = None

    @classmethod
    def of_index_in_use(cls, open_index: IndexOpener) -> SqliteVectorStore:
        """Work on whichever index ``open_index`` returns at each operation.

        While it returns ``None`` there is nothing to read, and a vector that
        is to be written is refused with ``NoVectorIndex``.
        """
        store = cls(None)
        store._open_index = open_index
        return store

    async def _index_in_use(self) -> MemoryVectorIndex | None:
        if self._open_index is None:
            return self._index
        return await self._open_index()

    async def open_collection(self) -> Any | None:
        """The collection of the index in use, for whoever inspects the index itself."""
        index = await self._index_in_use()
        return None if index is None else index.collection

    async def index_fingerprint(self) -> str | None:
        index = await self._index_in_use()
        return None if index is None else index.fingerprint

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
        return (
            await self.query_many(
                (embedding,),
                (scope,),
                (memory_types,),
                limit,
            )
        )[0]

    async def query_many(
        self,
        embeddings: Sequence[Sequence[float]],
        scopes: Sequence[AccessScope],
        memory_types: Sequence[list[str] | None],
        limit: int,
    ) -> list[list[tuple[str, float]]]:
        if not (len(embeddings) == len(scopes) == len(memory_types)):
            raise ValueError("vector query batch inputs must have equal length")
        if not embeddings:
            return []
        index = await self._index_in_use()
        if index is None:
            return [[] for _ in embeddings]
        collection = index.collection

        groups: dict[str, list[int]] = defaultdict(list)
        where_by_key: dict[str, dict[str, Any] | None] = {}
        for index, (scope, types) in enumerate(zip(scopes, memory_types, strict=True)):
            where = visible_chroma_where(scope, types)
            key = json.dumps(where, sort_keys=True, separators=(",", ":"))
            groups[key].append(index)
            where_by_key[key] = where

        output: list[list[tuple[str, float]]] = [[] for _ in embeddings]
        for key, indices in groups.items():
            params: dict[str, Any] = {
                "query_embeddings": [list(embeddings[index]) for index in indices],
                "n_results": limit,
            }
            where = where_by_key[key]
            if where:
                params["where"] = where
            results = await asyncio.to_thread(lambda: collection.query(**params))
            result_ids = (results or {}).get("ids") or []
            result_distances = (results or {}).get("distances") or []
            for local_index, output_index in enumerate(indices):
                ids = result_ids[local_index] if local_index < len(result_ids) else []
                distances = result_distances[local_index] if local_index < len(result_distances) else [0.0] * len(ids)
                output[output_index] = [(mid, self.similarity(dist)) for mid, dist in zip(ids, distances, strict=True)]
        return output

    async def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        index = await self._index_in_use()
        if index is None:
            raise NoVectorIndex("No Memory vector index is in use, so the vector cannot be written")
        await asyncio.to_thread(
            upsert_with_stored_vector_hash,
            index.collection,
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            metadatas=[dict(m) for m in metadatas],
        )

    async def delete(self, ids: Sequence[str]) -> None:
        index = await self._index_in_use()
        if index is None:
            return
        await asyncio.to_thread(lambda: index.collection.delete(ids=list(ids)))

    async def get_record(self, memory_id: str) -> dict[str, Any] | None:
        index = await self._index_in_use()
        if index is None:
            return None
        raw = await asyncio.to_thread(
            lambda: index.collection.get(
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
