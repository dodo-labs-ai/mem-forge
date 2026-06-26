"""SqliteRelationalStore: source-of-truth rows and the scoped read channels.

Row writes and their co-transactional FTS writes stay inside the Database
methods, so this store delegates rather than relocating SQL. The graph,
source/date, visibility, and ranking reads own the SQL that callers run inline
today, so no caller reaches a connection directly.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

from memforge.memory.audit import MemoryAuditLogger
from memforge.models import (
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemoryCurationRun,
    MemoryDerivation,
    MemorySource,
    Project,
    Visibility,
)
from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
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

    def __init__(
        self,
        db: Database,
        audit_logger: MemoryAuditLogger | None = None,
    ) -> None:
        self._db = db
        self._audit_logger = audit_logger

    async def insert_memory(self, memory: Memory) -> str:
        return await self._db.insert_memory(memory)

    async def get_memory(self, memory_id: str) -> Memory | None:
        return await self._db.get_memory(memory_id)

    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]:
        return await self._db.get_memory_sources(memory_id)

    async def upsert_document(self, doc: DocumentRecord) -> None:
        await self._db.upsert_document(doc)

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
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> None:
        await self._db.add_memory_source(
            memory_id,
            doc_id,
            source_type,
            excerpt,
            support_kind=support_kind,
            source_updated_at=source_updated_at,
        )

    async def add_memory_derivation(
        self,
        parent_memory_id: str,
        child_memory_id: str,
        *,
        relation: str = "summarizes",
    ) -> None:
        await self._db.add_memory_derivation(
            parent_memory_id,
            child_memory_id,
            relation=relation,
        )

    async def get_memory_derivation_children(
        self,
        parent_memory_id: str,
    ) -> list[MemoryDerivation]:
        return await self._db.get_memory_derivation_children(parent_memory_id)

    async def record_memory_curation_run(
        self,
        run: MemoryCurationRun,
    ) -> None:
        await self._db.record_memory_curation_run(run)

    async def get_memory_curation_run(
        self,
        run_id: str,
    ) -> MemoryCurationRun | None:
        return await self._db.get_memory_curation_run(run_id)

    async def promote_to_workspace(
        self,
        memory_id: str,
        *,
        actor_user_id: str,
        reason: str,
    ) -> None:
        """Flip a private memory to workspace visibility.

        The full flip-and-redo flow (re-stamping vector metadata in place and
        re-running dedup against the team set) is designed but not yet
        implemented. This method locks the contract: it verifies the row
        exists and is private, that the actor owns it, audits the attempt,
        and then refuses with NotImplementedError. A non-owner caller is
        rejected before any audit row is written, so a hostile attempt
        leaves no trail.
        """
        target = await self.get_memory(memory_id)
        if target is None:
            raise LookupError(f"memory {memory_id!r} not found")
        if target.visibility != Visibility.PRIVATE.value:
            raise ValueError(f"memory {memory_id!r} is not private; nothing to promote")
        if target.owner_user_id != actor_user_id:
            raise PermissionError(f"actor {actor_user_id!r} does not own memory {memory_id!r}")
        if self._audit_logger is not None:
            await self._audit_logger.emit(
                "memory_promoted",
                "failed",
                memory_id=memory_id,
                reason="not_implemented",
                payload={
                    "requested_reason": reason,
                    "actor": actor_user_id,
                },
            )
        raise NotImplementedError("promote_to_workspace is not yet implemented")

    async def filter_visible_ids(self, ids: Sequence[str], scope: AccessScope) -> set[str]:
        visible: set[str] = set()
        memory_ids = list(ids)
        if not memory_ids:
            return visible
        pred_sql, pred_params = visible_sql(scope, "m")
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            sql = f"SELECT m.id FROM memories m WHERE m.id IN ({placeholders}) AND {pred_sql}"
            try:
                async with self._db.db.execute(sql, [*batch, *pred_params]) as cursor:
                    async for row in cursor:
                        visible.add(row["id"])
            except Exception:
                logger.exception("Failed to filter visible memory ids")
                return set()
        return visible

    async def filter_ids_by_source_and_time(
        self,
        ids: Sequence[str],
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> set[str]:
        memory_ids = list(ids)
        if not memory_ids:
            return set()
        source_filter = source_filter or MemorySourceFilter()
        has_source_filter = not source_filter.is_empty()
        has_time_filter = time_range is not None and not time_range.is_empty()
        if not has_source_filter and not has_time_filter:
            return set(memory_ids)

        needs_source_join = (
            bool(source_filter.source_types)
            or bool(source_filter.sources)
            or bool(source_filter.clients)
            or (has_time_filter and time_range is not None and time_range.date_type == "source_updated_at")
        )
        needs_document_join = bool(source_filter.sources) or bool(source_filter.clients)

        matched: set[str] = set()
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            id_placeholders = ",".join("?" for _ in batch)
            joins: list[str] = []
            clauses = [f"m.id IN ({id_placeholders})"]
            params: list[Any] = [*batch]

            if needs_source_join:
                joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
            if needs_document_join:
                joins.append("LEFT JOIN documents d ON ms.doc_id = d.doc_id")

            if source_filter.source_types:
                placeholders = ",".join("?" for _ in source_filter.source_types)
                clauses.append(f"ms.source_type IN ({placeholders})")
                params.extend(source_filter.source_types)
            if source_filter.sources:
                placeholders = ",".join("?" for _ in source_filter.sources)
                clauses.append(f"d.source IN ({placeholders})")
                params.extend(source_filter.sources)
            if source_filter.clients:
                placeholders = ",".join("?" for _ in source_filter.clients)
                clauses.append(f"d.client IN ({placeholders})")
                params.extend(source_filter.clients)
            if source_filter.repo_identifiers:
                placeholders = ",".join("?" for _ in source_filter.repo_identifiers)
                clauses.append(f"m.repo_identifier IN ({placeholders})")
                params.extend(source_filter.repo_identifiers)

            if has_time_filter and time_range is not None:
                if time_range.date_type == "source_updated_at":
                    if time_range.after is not None:
                        clauses.append("ms.source_updated_at >= ?")
                        params.append(time_range.after.isoformat())
                    if time_range.before is not None:
                        clauses.append("ms.source_updated_at < ?")
                        params.append(time_range.before.isoformat())
                elif time_range.date_type == "memory_updated_at":
                    if time_range.after is not None:
                        clauses.append("m.updated_at >= ?")
                        params.append(time_range.after.isoformat())
                    if time_range.before is not None:
                        clauses.append("m.updated_at < ?")
                        params.append(time_range.before.isoformat())
                else:
                    raise ValueError(f"Unsupported memory time range date_type: {time_range.date_type}")

            sql = (
                "SELECT DISTINCT m.id "
                "FROM memories m "
                + (" ".join(joins) + " " if joins else "")
                + "WHERE "
                + " AND ".join(clauses)
            )
            try:
                async with self._db.db.execute(sql, params) as cursor:
                    async for row in cursor:
                        matched.add(row[0])
            except Exception:
                logger.exception("Failed to filter ids by structured source/date facets")
                return set()
        return matched

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
                *direct_ids,
                *direct_ids,
                *predicate_params,
                *type_params,
                limit,
            ]
            try:
                async with self._db.db.execute(expanded_sql, expanded_params) as cursor:
                    async for row in cursor:
                        shared = int(row[1])
                        scored.append((row[0], _EXPANSION_WEIGHT * shared / query_entity_count))
            except Exception:
                logger.exception("Graph 1-hop expansion failed")

        return scored

    async def fetch_ranking_metadata(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Return ranking and curation metadata for each id in one read.

        The fields feed recency, project affinity, repo affinity, and
        lineage-aware result shaping, so a single batched ``SELECT`` keeps the
        per-candidate roundtrip count at one regardless of channel count.
        """
        ranked: dict[str, dict[str, Any]] = {}
        memory_ids = list(ids)
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            try:
                async with self._db.db.execute(
                    "SELECT m.id, m.updated_at, m.project_key, "
                    "m.repo_identifier, m.memory_level, m.curation_cluster_id, "
                    "COUNT(md.child_memory_id) AS covered_memory_count "
                    "FROM memories m "
                    "LEFT JOIN memory_derivations md ON md.parent_memory_id = m.id "
                    f"WHERE m.id IN ({placeholders}) "
                    "GROUP BY m.id",
                    batch,
                ) as cursor:
                    async for row in cursor:
                        raw_updated = row[1]
                        parsed: datetime | None = None
                        if raw_updated:
                            try:
                                parsed = datetime.fromisoformat(raw_updated)
                            except (ValueError, TypeError):
                                parsed = None
                        ranked[row[0]] = {
                            "updated_at": parsed,
                            "project_key": row[2],
                            "repo_identifier": row[3],
                            "memory_level": row[4],
                            "curation_cluster_id": row[5],
                            "covered_memory_count": int(row[6] or 0),
                        }
            except Exception:
                logger.exception("Failed to fetch ranking metadata for memory ids")
        return ranked

    async def create_project(self, *, key: str, name: str, is_shared: bool = False) -> Project:
        return await self._db.create_project(key=key, name=name, is_shared=is_shared)

    async def get_project(self, project_id: str) -> Project | None:
        return await self._db.get_project(project_id)

    async def list_projects(self) -> list[Project]:
        return await self._db.list_projects()

    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        is_shared: bool | None = None,
    ) -> Project | None:
        return await self._db.update_project(project_id, name=name, is_shared=is_shared)

    async def list_project_memory_ids(self, project_id: str) -> list[str]:
        return await self._db.list_project_memory_ids(project_id)

    async def commit_project_deletion(self, project_id: str, affected_ids: Sequence[str]) -> None:
        await self._db.commit_project_deletion(project_id, affected_ids)
