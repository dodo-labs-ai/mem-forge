"""Storage adapter protocols: the three narrow contracts the core binds to.

RelationalStore is the source-of-truth rows, KeywordSearch is the BM25/FTS5
channel, VectorStore is the embedding channel. Enforcement is each
adapter's job, never the caller's.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence, runtime_checkable

from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemorySource,
)
from memforge.storage.adapters.context import AccessScope


@runtime_checkable
class RelationalStore(Protocol):
    """Source-of-truth rows: memories and their provenance, plus the scoped
    relational channels that read those rows.

    Bound to one datastore at construction. Memory-row writes and the
    co-transactional FTS write stay inside the existing Database methods:
    this protocol delegates to those methods rather than relocating their
    SQL, preserving the single-commit atomicity that keeps SQLite and FTS5
    in sync. The read channels (graph, temporal, the post-fusion re-check,
    the ranking fetch, and the source re-check) own the SQL that callers run
    inline today, so a caller never reaches a database connection directly.
    """

    async def insert_memory(self, memory: Memory) -> str: ...
    async def get_memory(self, memory_id: str) -> Memory | None: ...
    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]: ...
    async def get_all_entities(self) -> list[Entity]: ...
    async def get_all_aliases(self) -> list[tuple[str, int]]: ...
    async def filter_visible_ids(
        self, ids: Sequence[str], scope: AccessScope
    ) -> set[str]: ...
    async def filter_ids_supported_by_sources(
        self, ids: Sequence[str], sources: Sequence[str]
    ) -> set[str]: ...
    async def fetch_updated_at(
        self, ids: Sequence[str]
    ) -> dict[str, datetime | None]: ...
    async def graph_search(
        self,
        entity_ids: Sequence[int],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...
    async def temporal_search(
        self,
        after: datetime | None,
        before: datetime | None,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...
    async def add_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        support_kind: str = "extracted",
    ) -> None: ...


@runtime_checkable
class KeywordSearch(Protocol):
    """BM25/FTS5 channel.

    The SQLite implementation is a thin facade: memory-row writes and their
    FTS writes stay inside the existing co-transactional Database methods, so
    this protocol owns only the read-path FTS query and the one standalone
    FTS delete.
    """

    async def remove(self, memory_id: str) -> None: ...
    async def search(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Embedding channel. It owns every distance/score conversion so no
    caller ever assumes cosine: similarity() maps native distance to [0, 1],
    and within_dedup_threshold() decides whether a returned score is close
    enough to be a duplicate against a configured distance threshold.

    distance_metric is a declared label used only by a calibration check to
    assert thresholds match the metric. There is no metric-enum machinery:
    the single in-scope implementation is cosine.
    """

    distance_metric: str

    def similarity(self, distance: float) -> float: ...
    def within_dedup_threshold(
        self, distance_threshold: float, score: float
    ) -> bool: ...
    async def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...
    async def delete(self, ids: Sequence[str]) -> None: ...
    async def query(
        self,
        embedding: Sequence[float],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]: ...
    async def get_record(self, memory_id: str) -> dict[str, Any] | None: ...
