"""SqliteRelationalStore: source-of-truth rows and the scoped read channels.

Row writes and their co-transactional FTS writes stay inside the Database
methods, so this store delegates rather than relocating SQL. The graph,
source/date, visibility, and ranking reads own the SQL that callers run inline
today, so no caller reaches a connection directly.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any, Sequence

import aiosqlite

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
    canonicalize_entity_name,
)
from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.protocols import (
    DEFAULT_ENTITY_LINK_LIMIT,
    EntityLinkCandidate,
    EntityLinkResult,
)

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

_MAX_ENTITY_LINK_QUERY_TOKENS = 48
_MAX_ENTITY_LINK_WINDOW_TOKENS = 6
_MAX_ENTITY_LINK_WINDOWS = 128
_ENTITY_LINK_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "when",
        "where",
        "why",
        "with",
    }
)
_ENTITY_LINK_CHANNEL_SCORE = {
    "explicit": 1.0,
    "alias_exact": 0.95,
    "alias_compact": 0.35,
}
_ENTITY_LINK_CHANNEL_ACTIVATES_GRAPH = {
    "explicit": True,
    "alias_exact": True,
    "alias_compact": False,
}


def _entity_link_tokens(query: str) -> list[str]:
    normalized = canonicalize_entity_name(query)
    return [token for token in normalized.split() if token][:_MAX_ENTITY_LINK_QUERY_TOKENS]


def _entity_link_windows(tokens: Sequence[str]) -> dict[str, str]:
    windows: dict[str, str] = {}
    for start in range(len(tokens)):
        for size in range(1, _MAX_ENTITY_LINK_WINDOW_TOKENS + 1):
            if len(windows) >= _MAX_ENTITY_LINK_WINDOWS:
                return windows
            end = start + size
            if end > len(tokens):
                break
            window_tokens = list(tokens[start:end])
            if size == 1:
                token = window_tokens[0]
                if token in _ENTITY_LINK_STOPWORDS or len(token) < 3:
                    continue
            window = " ".join(window_tokens)
            windows.setdefault(window, window)
    return windows


def _entity_link_compact_terms(tokens: Sequence[str]) -> dict[str, str]:
    terms: dict[str, str] = {}
    for window, matched_text in _entity_link_windows(tokens).items():
        compact = window.replace(" ", "")
        if len(compact) < 4:
            continue
        terms.setdefault(compact, matched_text)
        if compact.endswith("s") and len(compact) > 5:
            terms.setdefault(compact[:-1], matched_text)
    return terms


def _explicit_entity_terms(explicit_entities: Sequence[str]) -> dict[str, str]:
    terms: dict[str, str] = {}
    for value in explicit_entities:
        normalized = canonicalize_entity_name(value)
        if normalized:
            terms.setdefault(normalized, value)
    return terms


def _enabled_source_visibility_condition(
    disabled_source_ids: list[str],
) -> tuple[str | None, list[str]]:
    if not disabled_source_ids:
        return None, []
    placeholders = ", ".join("?" for _ in disabled_source_ids)
    return (
        f"""(
            NOT EXISTS (
                SELECT 1
                FROM memory_sources ms_any
                WHERE ms_any.memory_id = m.id
            )
            OR EXISTS (
                SELECT 1
                FROM memory_sources ms_enabled
                WHERE ms_enabled.memory_id = m.id
                  AND (ms_enabled.source_id IS NULL OR ms_enabled.source_id NOT IN ({placeholders}))
            )
        )""",
        list(disabled_source_ids),
    )


def _append_source_time_predicates(
    *,
    source_filter: MemorySourceFilter,
    time_range: MemoryTimeRange | None,
    joins: list[str],
    clauses: list[str],
    params: list[Any],
) -> str:
    """Append canonical source/time predicates and return the deterministic order.

    `source_updated_at` intentionally lives on the same `memory_sources` row as
    exact source facets such as `source_ids`, so a stale Jira provenance row
    cannot match because a different Confluence row was updated recently.
    """

    has_time_filter = time_range is not None and not time_range.is_empty()
    needs_source_join = (
        bool(source_filter.source_ids)
        or bool(source_filter.clients)
        or (has_time_filter and time_range is not None and time_range.date_type == "source_updated_at")
    )
    needs_document_join = bool(source_filter.clients)

    if needs_source_join:
        joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
    if needs_document_join:
        joins.append("LEFT JOIN documents d ON ms.doc_id = d.doc_id")

    if source_filter.source_ids:
        placeholders = ",".join("?" for _ in source_filter.source_ids)
        clauses.append(f"ms.source_id IN ({placeholders})")
        params.extend(source_filter.source_ids)
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

    if has_time_filter and time_range is not None and time_range.date_type == "source_updated_at":
        return "MAX(ms.source_updated_at) DESC, m.id DESC"
    return "m.updated_at DESC, m.id DESC"


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

        matched: set[str] = set()
        for start in range(0, len(memory_ids), _BATCH_SIZE):
            batch = memory_ids[start : start + _BATCH_SIZE]
            id_placeholders = ",".join("?" for _ in batch)
            joins: list[str] = []
            clauses = [f"m.id IN ({id_placeholders})"]
            params: list[Any] = [*batch]

            _append_source_time_predicates(
                source_filter=source_filter,
                time_range=time_range,
                joins=joins,
                clauses=clauses,
                params=params,
            )

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

    async def list_ids_by_source_and_time(
        self,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
        scope: AccessScope,
        *,
        limit: int,
        offset: int,
    ) -> tuple[list[str], int]:
        source_filter = source_filter or MemorySourceFilter()
        has_source_filter = not source_filter.is_empty()
        has_time_filter = time_range is not None and not time_range.is_empty()
        if not has_source_filter and not has_time_filter:
            raise ValueError("list_ids_by_source_and_time requires source_filter or time_range")

        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        params: list[Any] = list(predicate_params)
        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        order_sql = _append_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            joins=joins,
            clauses=clauses,
            params=params,
        )
        has_source_row_join = any("memory_sources ms" in join for join in joins)
        if has_source_row_join and disabled_source_ids:
            placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({placeholders}))")
            params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                params.extend(source_visibility_params)
        join_sql = " ".join(joins)
        where_sql = " AND ".join(clauses)
        group_sql = "GROUP BY m.id" if joins else ""

        count_sql = f"SELECT COUNT(*) FROM (SELECT m.id FROM memories m {join_sql} WHERE {where_sql} {group_sql}) q"
        async with self._db.db.execute(count_sql, params) as cursor:
            row = await cursor.fetchone()
            total = int(row[0]) if row else 0

        page_sql = (
            f"SELECT m.id FROM memories m {join_sql} WHERE {where_sql} "
            f"{group_sql} ORDER BY {order_sql} LIMIT ? OFFSET ?"
        )
        ids: list[str] = []
        async with self._db.db.execute(page_sql, [*params, limit, offset]) as cursor:
            async for row in cursor:
                ids.append(row[0])
        return ids, total

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

    async def link_query_entities(
        self,
        query: str,
        *,
        scope: AccessScope,
        explicit_entities: Sequence[str] = (),
        source_filter: MemorySourceFilter | None = None,
        memory_types: Sequence[str] | None = None,
        limit: int = DEFAULT_ENTITY_LINK_LIMIT,
    ) -> EntityLinkResult:
        max_candidates = max(0, int(limit))
        explicit_terms = _explicit_entity_terms(explicit_entities)
        if max_candidates == 0:
            return EntityLinkResult(unmatched_explicit_entities=tuple(explicit_terms.values()))

        candidates: dict[int, EntityLinkCandidate] = {}
        matched_explicit_terms: set[str] = set()

        async def add_matches(channel: str, terms: dict[str, str]) -> None:
            if not terms:
                return
            rows = await self._lookup_entity_link_rows(
                terms,
                channel=channel,
                scope=scope,
                source_filter=source_filter,
                memory_types=memory_types,
                limit=max(max_candidates * 4, max_candidates),
            )
            for row in rows:
                entity_id = int(row["entity_id"])
                if channel == "explicit":
                    matched_explicit_terms.add(str(row["match_key"]))
                score = _ENTITY_LINK_CHANNEL_SCORE[channel]
                activates_graph = _ENTITY_LINK_CHANNEL_ACTIVATES_GRAPH[channel]
                candidate = EntityLinkCandidate(
                    entity_id=entity_id,
                    canonical_name=str(row["canonical_name"]),
                    matched_alias=str(row["matched_alias"]),
                    channel=channel,
                    contributing_channels=(channel,),
                    score=score,
                    matched_text=terms.get(str(row["match_key"]), str(row["match_key"])),
                    activates_graph=activates_graph,
                )
                existing = candidates.get(entity_id)
                if existing is None:
                    candidates[entity_id] = candidate
                    continue

                contributing_channels = tuple(
                    dict.fromkeys((*existing.contributing_channels, channel))
                )
                if score > existing.score:
                    candidates[entity_id] = EntityLinkCandidate(
                        entity_id=existing.entity_id,
                        canonical_name=candidate.canonical_name,
                        matched_alias=candidate.matched_alias,
                        channel=channel,
                        contributing_channels=contributing_channels,
                        score=score,
                        matched_text=candidate.matched_text,
                        activates_graph=activates_graph,
                    )
                else:
                    candidates[entity_id] = EntityLinkCandidate(
                        entity_id=existing.entity_id,
                        canonical_name=existing.canonical_name,
                        matched_alias=existing.matched_alias,
                        channel=existing.channel,
                        contributing_channels=contributing_channels,
                        score=existing.score,
                        matched_text=existing.matched_text,
                        activates_graph=existing.activates_graph,
                    )

        await add_matches("explicit", explicit_terms)

        tokens = _entity_link_tokens(query)
        await add_matches("alias_exact", _entity_link_windows(tokens))
        await add_matches("alias_compact", _entity_link_compact_terms(tokens))

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (
                -candidate.score,
                candidate.canonical_name,
                candidate.entity_id,
            ),
        )[:max_candidates]
        unmatched_explicit_entities = tuple(
            raw_value
            for normalized, raw_value in explicit_terms.items()
            if normalized not in matched_explicit_terms
        )
        return EntityLinkResult(
            candidates=tuple(ranked),
            unmatched_explicit_entities=unmatched_explicit_entities,
        )

    async def _lookup_entity_link_rows(
        self,
        terms: dict[str, str],
        *,
        channel: str,
        scope: AccessScope,
        source_filter: MemorySourceFilter | None,
        memory_types: Sequence[str] | None,
        limit: int,
    ) -> list[Any]:
        term_values = list(terms)
        if not term_values:
            return []

        placeholders = ",".join("?" for _ in term_values)
        if channel == "alias_compact":
            entity_match = f"REPLACE(e.canonical_name, ' ', '') IN ({placeholders})"
            alias_match = f"REPLACE(ea.alias_normalized, ' ', '') IN ({placeholders})"
            entity_match_key = "REPLACE(e.canonical_name, ' ', '')"
            alias_match_key = "REPLACE(ea.alias_normalized, ' ', '')"
        else:
            entity_match = f"e.canonical_name IN ({placeholders})"
            alias_match = f"ea.alias_normalized IN ({placeholders})"
            entity_match_key = "e.canonical_name"
            alias_match_key = "ea.alias_normalized"

        predicate_sql, predicate_params = visible_sql(scope, "m")
        joins: list[str] = []
        clauses = [predicate_sql]
        params: list[Any] = [*term_values, *term_values, *predicate_params]

        source_filter = source_filter or MemorySourceFilter()
        _append_source_time_predicates(
            source_filter=source_filter,
            time_range=None,
            joins=joins,
            clauses=clauses,
            params=params,
        )

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            clauses.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        has_source_row_join = any("memory_sources ms" in join for join in joins)
        if has_source_row_join and disabled_source_ids:
            disabled_placeholders = ", ".join("?" for _ in disabled_source_ids)
            clauses.append(f"(ms.source_id IS NULL OR ms.source_id NOT IN ({disabled_placeholders}))")
            params.extend(disabled_source_ids)
        else:
            source_visibility_sql, source_visibility_params = _enabled_source_visibility_condition(disabled_source_ids)
            if source_visibility_sql:
                clauses.append(source_visibility_sql)
                params.extend(source_visibility_params)

        join_sql = " ".join(joins)
        where_sql = " AND ".join(clauses)
        sql = (
            "WITH matched_aliases(entity_id, matched_alias, alias_normalized, match_key) AS ("
            f"SELECT e.id, e.canonical_name, e.canonical_name, {entity_match_key} "
            f"FROM entities e WHERE {entity_match} "
            "UNION ALL "
            f"SELECT ea.canonical_id, ea.alias, ea.alias_normalized, {alias_match_key} "
            f"FROM entity_aliases ea WHERE {alias_match}"
            ") "
            "SELECT ma.entity_id, e.canonical_name, ma.matched_alias, "
            "ma.alias_normalized, ma.match_key, COUNT(DISTINCT m.id) AS visible_memory_count "
            "FROM matched_aliases ma "
            "JOIN entities e ON e.id = ma.entity_id "
            "JOIN memory_entities me ON me.entity_id = ma.entity_id "
            "JOIN memories m ON m.id = me.memory_id "
            f"{join_sql} "
            f"WHERE {where_sql} "
            "GROUP BY ma.entity_id, e.canonical_name, ma.matched_alias, ma.alias_normalized, ma.match_key "
            "ORDER BY visible_memory_count DESC, LENGTH(ma.alias_normalized) DESC, e.canonical_name ASC "
            "LIMIT ?"
        )
        try:
            async with self._db.db.execute(sql, [*params, limit]) as cursor:
                return [row async for row in cursor]
        except (aiosqlite.Error, sqlite3.Error):
            logger.exception("SQLite entity linker query failed")
            return []

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
