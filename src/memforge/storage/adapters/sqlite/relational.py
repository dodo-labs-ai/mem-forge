"""SqliteRelationalStore: source-of-truth rows and the scoped read channels.

Row writes and their co-transactional FTS writes stay inside the Database
methods, so this store delegates rather than relocating SQL. The graph,
temporal, source, visibility, and ranking reads own the SQL that callers run
inline today, so no caller reaches a connection directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemorySource,
)
from memforge.retrieval.access_predicate import visible_sql
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope

logger = logging.getLogger(__name__)

__all__ = ["SqliteRelationalStore"]

# The IN (...) chunk size for the visibility and ranking readers. It carries
# over the value the inline search loops use today so SQLite's bound-parameter
# limit is never exceeded for a large fused candidate set.
_BATCH_SIZE = 200

# The 1-hop graph expansion only keeps memories sharing at least this many
# entities with a direct hit, matching the existing HAVING clause.
_MIN_SHARED_ENTITIES_FOR_EXPANSION = 2

# The 1-hop expansion contributes at half the weight of a direct entity hit.
_EXPANSION_WEIGHT = 0.5


class SqliteRelationalStore:
    """The row channel backed by the memories table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def insert_memory(self, memory: Memory) -> str:
        return await self._db.insert_memory(memory)

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._db.get_memory(memory_id)

    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]:
        return await self._db.get_memory_sources(memory_id)

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        return await self._db.get_document(doc_id)

    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]:
        return await self._db.get_aliases_for_entity(entity_id)

    async def get_all_entities(self) -> list[Entity]:
        return await self._db.get_all_entities()

    async def get_all_aliases(self) -> list[tuple[str, int]]:
        return await self._db.get_all_aliases()

    async def add_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        support_kind: str = "extracted",
    ) -> None:
        await self._db.add_memory_source(
            memory_id, doc_id, source_type, excerpt, support_kind=support_kind
        )

    async def filter_visible_ids(
        self, ids: Sequence[str], scope: AccessScope
    ) -> set[str]:
        visible: set[str] = set()
        memory_ids = list(ids)
        if not memory_ids:
            return visible
        pred_sql, pred_params = visible_sql(scope, "m")
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            sql = (
                "SELECT m.id FROM memories m "
                f"WHERE m.id IN ({placeholders}) AND {pred_sql}"
            )
            try:
                async with self._db.db.execute(sql, [*batch, *pred_params]) as cursor:
                    async for row in cursor:
                        visible.add(row["id"])
            except Exception:
                logger.exception("Failed to filter visible memory ids")
                return set()
        return visible

    async def filter_ids_supported_by_sources(
        self, ids: Sequence[str], sources: Sequence[str]
    ) -> set[str]:
        memory_ids = list(ids)
        source_list = list(sources)
        if not memory_ids or not source_list:
            return set()
        id_placeholders = ",".join("?" for _ in memory_ids)
        source_placeholders = ",".join("?" for _ in source_list)
        sql = (
            "SELECT DISTINCT ms.memory_id "
            "FROM memory_sources ms "
            "JOIN documents d ON ms.doc_id = d.doc_id "
            f"WHERE ms.memory_id IN ({id_placeholders}) "
            f"AND d.source IN ({source_placeholders})"
        )
        supported: set[str] = set()
        try:
            async with self._db.db.execute(sql, [*memory_ids, *source_list]) as cursor:
                async for row in cursor:
                    supported.add(row[0])
        except Exception:
            logger.exception("Failed to filter ids by supporting sources")
            return set()
        return supported

    async def graph_search(
        self,
        entity_ids: Sequence[int],
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        ids = list(entity_ids)
        if not ids:
            return []

        placeholders = ",".join("?" for _ in ids)
        predicate_sql, predicate_params = visible_sql(scope, "m")
        type_filter = ""
        type_params: list[Any] = []
        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            type_filter = f"AND m.memory_type IN ({type_placeholders})"
            type_params = list(memory_types)

        direct_sql = (
            "SELECT m.id, COUNT(me.entity_id) AS entity_overlap "
            "FROM memories m "
            "JOIN memory_entities me ON m.id = me.memory_id "
            f"WHERE me.entity_id IN ({placeholders}) "
            f"AND {predicate_sql} {type_filter} "
            "GROUP BY m.id "
            "ORDER BY entity_overlap DESC "
            "LIMIT ?"
        )
        direct_params: list[Any] = [*ids, *predicate_params, *type_params, limit]

        direct_results: list[tuple[str, int]] = []
        try:
            async with self._db.db.execute(direct_sql, direct_params) as cursor:
                async for row in cursor:
                    direct_results.append((row[0], int(row[1])))
        except Exception:
            logger.exception("Graph direct search failed")
            return []

        query_entity_count = len(ids)
        scored: list[tuple[str, float]] = [
            (mid, float(overlap) / query_entity_count) for mid, overlap in direct_results
        ]

        if direct_results:
            direct_ids = [mid for mid, _ in direct_results]
            d_placeholders = ",".join("?" for _ in direct_ids)
            expanded_sql = (
                "SELECT m.id, COUNT(DISTINCT me2.entity_id) AS shared_entities "
                "FROM memory_entities me1 "
                "JOIN memory_entities me2 ON me1.entity_id = me2.entity_id "
                "JOIN memories m ON me2.memory_id = m.id "
                f"WHERE me1.memory_id IN ({d_placeholders}) "
                f"AND m.id NOT IN ({d_placeholders}) "
                f"AND {predicate_sql} {type_filter} "
                "GROUP BY m.id "
                f"HAVING shared_entities >= {_MIN_SHARED_ENTITIES_FOR_EXPANSION} "
                "ORDER BY shared_entities DESC "
                "LIMIT ?"
            )
            expanded_params: list[Any] = [
                *direct_ids, *direct_ids, *predicate_params, *type_params, limit,
            ]
            try:
                async with self._db.db.execute(expanded_sql, expanded_params) as cursor:
                    async for row in cursor:
                        shared = int(row[1])
                        scored.append(
                            (row[0], _EXPANSION_WEIGHT * shared / query_entity_count)
                        )
            except Exception:
                logger.exception("Graph 1-hop expansion failed")

        return scored

    async def temporal_search(
        self,
        after: datetime | None,
        before: datetime | None,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        conditions: list[str] = []
        params: list[Any] = []
        if after:
            iso_after = after.isoformat()
            conditions.append("(m.updated_at >= ? OR m.created_at >= ?)")
            params.extend([iso_after, iso_after])
        if before:
            iso_before = before.isoformat()
            conditions.append("(m.updated_at <= ? OR m.created_at <= ?)")
            params.extend([iso_before, iso_before])
        if not conditions:
            return []

        predicate_sql, predicate_params = visible_sql(scope, "m")
        conditions.append(predicate_sql)
        params.extend(predicate_params)
        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        sql = (
            "SELECT m.id FROM memories m "
            "WHERE " + " AND ".join(conditions) + " "
            "ORDER BY m.updated_at DESC "
            "LIMIT ?"
        )
        params.append(limit)

        results: list[tuple[str, float]] = []
        try:
            async with self._db.db.execute(sql, params) as cursor:
                position = 0
                async for row in cursor:
                    # A decreasing score by result order so more-recently-updated
                    # rows rank higher within this channel.
                    position += 1
                    results.append((row[0], 1.0 / position))
        except Exception:
            logger.exception("Temporal search failed")
            return []
        return results

    async def fetch_updated_at(
        self, ids: Sequence[str]
    ) -> dict[str, datetime | None]:
        stamped: dict[str, datetime | None] = {}
        memory_ids = list(ids)
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            try:
                async with self._db.db.execute(
                    f"SELECT id, updated_at FROM memories WHERE id IN ({placeholders})",
                    batch,
                ) as cursor:
                    async for row in cursor:
                        raw = row[1]
                        parsed: datetime | None = None
                        if raw:
                            try:
                                parsed = datetime.fromisoformat(raw)
                            except (ValueError, TypeError):
                                parsed = None
                        stamped[row[0]] = parsed
            except Exception:
                logger.exception("Failed to fetch updated_at for memory ids")
        return stamped
