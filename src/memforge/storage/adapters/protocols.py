"""Storage adapter protocols: the three narrow contracts the core binds to.

RelationalStore is the source-of-truth rows, KeywordSearch is the BM25/FTS5
channel, VectorStore is the embedding channel. Enforcement is each
adapter's job, never the caller's.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence, TypedDict, runtime_checkable

from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemoryCurationRun,
    MemoryDerivation,
    MemorySource,
    Project,
)
from memforge.retrieval.filters import MemorySourceFilter
from memforge.storage.adapters.context import AccessScope


class RankingMetadata(TypedDict, total=False):
    """The per-memory inputs the ranker needs alongside RRF scores.

    `updated_at` drives the recency curve; `project_key` drives the
    cross-project affinity penalty. Both come back in one relational read
    so the ranker never makes a second roundtrip.
    """

    updated_at: datetime | None
    project_key: str | None
    repo_identifier: str | None
    memory_level: str | None
    curation_cluster_id: str | None
    covered_memory_count: int


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
    async def upsert_document(self, doc: DocumentRecord) -> None: ...
    async def get_document(self, doc_id: str) -> DocumentRecord | None: ...
    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]: ...
    async def get_all_entities(self) -> list[Entity]: ...
    async def get_all_aliases(self) -> list[tuple[str, int]]: ...
    async def filter_visible_ids(self, ids: Sequence[str], scope: AccessScope) -> set[str]: ...
    async def filter_ids_supported_by_sources(self, ids: Sequence[str], sources: Sequence[str]) -> set[str]: ...
    async def filter_ids_by_source_filter(self, ids: Sequence[str], source_filter: MemorySourceFilter) -> set[str]: ...
    async def fetch_ranking_metadata(self, ids: Sequence[str]) -> Mapping[str, RankingMetadata]: ...
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
        *,
        support_kind: str = "extracted",
        source_observed_at: datetime | None,
    ) -> None: ...
    async def add_memory_derivation(
        self,
        parent_memory_id: str,
        child_memory_id: str,
        *,
        relation: str = "summarizes",
    ) -> None: ...
    async def get_memory_derivation_children(
        self,
        parent_memory_id: str,
    ) -> list[MemoryDerivation]: ...
    async def record_memory_curation_run(
        self,
        run: MemoryCurationRun,
    ) -> None: ...
    async def get_memory_curation_run(
        self,
        run_id: str,
    ) -> MemoryCurationRun | None: ...
    async def promote_to_workspace(
        self,
        memory_id: str,
        *,
        actor_user_id: str,
        reason: str,
    ) -> None:
        """Flip a private memory to workspace visibility.

        The full promotion flow (re-stamping vector metadata in place,
        re-running dedup against the team set) is designed but not yet
        implemented. Implementations must raise NotImplementedError after
        auditing the attempt and after verifying the actor owns the row.
        A non-owner caller must be rejected before any audit emission.
        """
        ...

    async def create_project(self, *, key: str, name: str, is_shared: bool = False) -> Project: ...
    async def get_project(self, project_id: str) -> Project | None: ...
    async def list_projects(self) -> list[Project]: ...
    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        is_shared: bool | None = None,
    ) -> Project | None: ...
    async def list_project_memory_ids(self, project_id: str) -> list[str]:
        """Return the memory ids attached to a project.

        Pairs with `commit_project_deletion`. The handler reads the
        affected ids first, has the owning vector service rewrite their
        embedding metadata to UNSORTED, then commits the relational
        rebucket. Reserved keys (SHARED, UNSORTED) raise `ValueError`;
        an unknown id raises `LookupError`.
        """
        ...

    async def commit_project_deletion(self, project_id: str, affected_ids: Sequence[str]) -> None:
        """Rebucket the named memories to UNSORTED and drop the project
        row, in one transaction.

        `affected_ids` is the same id list the caller already moved on
        the vector side, so the relational rebucket touches exactly the
        rows the vector channel touched. Reserved keys (SHARED,
        UNSORTED) raise `ValueError`.
        """
        ...


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
    def within_dedup_threshold(self, distance_threshold: float, score: float) -> bool: ...
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
