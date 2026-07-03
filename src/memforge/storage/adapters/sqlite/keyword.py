"""SqliteKeywordSearch: the BM25/FTS5 read channel plus the standalone delete.

Memory-row writes and their FTS writes remain co-transactional inside the
Database methods, so this facade owns only the read-path query and the one
delete that runs outside a row write.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.metadata_text import compact_query_variants, quoted_query_terms
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.protocols import KeywordCandidate, KeywordSourceRef

logger = logging.getLogger(__name__)

__all__ = ["SqliteKeywordSearch"]


def _labels_match_text(labels: Any) -> str:
    if not labels:
        return ""
    if isinstance(labels, str):
        raw_labels = labels.strip()
        if not raw_labels:
            return ""
        try:
            labels = json.loads(raw_labels)
        except json.JSONDecodeError:
            return raw_labels
    if isinstance(labels, list):
        return " ".join(str(label).strip() for label in labels if str(label).strip())
    return str(labels).strip()


def _metadata_match_text(
    *,
    title: Any,
    doc_id: Any,
    source_url: Any,
    space_or_project: Any,
    labels: Any,
    source_name: Any,
) -> str:
    parts = [
        str(title or "").strip(),
        str(doc_id or "").strip(),
        str(space_or_project or "").strip(),
        str(source_name or "").strip(),
    ]
    labels_text = _labels_match_text(labels)
    if labels_text:
        parts.append(labels_text)
    if source_url:
        parts.append(str(source_url).strip())
    return " | ".join(part for part in parts if part)


class SqliteKeywordSearch:
    """The keyword channel backed by the memories_fts FTS5 table."""

    metadata_search_channels = (
        "bm25_metadata_tokens",
        "metadata_alias",
        "metadata_trigram",
    )
    disabled_metadata_search_channels: tuple[str, ...] = ()

    def __init__(self, db: Database) -> None:
        self._db = db

    async def remove(self, memory_id: str) -> None:
        await self._db.db.execute(
            "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.execute(
            "DELETE FROM memory_search_metadata_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.execute(
            "DELETE FROM memory_search_metadata_alias_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.execute(
            "DELETE FROM memory_search_metadata_trigram WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.commit()

    async def search(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        predicate_sql, predicate_params = visible_sql(scope, "m")
        conditions = ["memories_fts MATCH ?"]
        params: list[Any] = [fts_query]
        conditions.append(predicate_sql)
        params.extend(predicate_params)

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        sql = (
            "SELECT f.memory_id, rank "
            "FROM memories_fts f "
            "JOIN memories m ON f.memory_id = m.id "
            "WHERE " + " AND ".join(conditions) + " "
            "ORDER BY rank "
            f"LIMIT {limit}"
        )

        try:
            results: list[tuple[str, float]] = []
            async with self._db.db.execute(sql, params) as cursor:
                async for row in cursor:
                    memory_id = row[0]
                    rank_score = -float(row[1]) if row[1] is not None else 0.0
                    results.append((memory_id, rank_score))
            return results
        except Exception:
            logger.exception("Keyword (FTS5) search failed")
            return []

    async def search_metadata(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[KeywordCandidate]:
        if limit <= 0:
            return []

        try:
            hits: list[KeywordCandidate] = []
            hits.extend(
                await self._search_metadata_fts(
                    fts_query,
                    scope,
                    memory_types,
                    limit,
                    table="memory_search_metadata_fts",
                    channel="bm25_metadata_tokens",
                    matched_field="metadata_any",
                    score_scale=1.0,
                )
            )
            hits.extend(
                await self._search_metadata_fts(
                    fts_query,
                    scope,
                    memory_types,
                    limit,
                    table="memory_search_metadata_alias_fts",
                    channel="metadata_alias",
                    matched_field="metadata_alias",
                    score_scale=0.75,
                )
            )
            hits.extend(
                await self._search_metadata_trigram(
                    fts_query,
                    scope,
                    memory_types,
                    limit,
                )
            )
            return _dedupe_metadata_hits(hits, limit)
        except Exception:
            logger.exception("Metadata keyword search failed")
            return []

    async def _search_metadata_fts(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        table: str,
        channel: str,
        matched_field: str,
        score_scale: float,
    ) -> list[KeywordCandidate]:
        predicate_sql, predicate_params = visible_sql(scope, "m")
        conditions = [f"{table} MATCH ?"]
        params: list[Any] = [fts_query]
        conditions.append(predicate_sql)
        params.extend(predicate_params)

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if disabled_source_ids:
            source_placeholders = ",".join("?" for _ in disabled_source_ids)
            conditions.append(
                f"(f.source_id IS NULL OR f.source_id NOT IN ({source_placeholders}))"
            )
            params.extend(disabled_source_ids)

        where_sql = " AND ".join(conditions)
        top_sql = (
            "SELECT f.memory_id, MIN(rank) AS best_rank "
            f"FROM {table} f "
            "JOIN memories m ON f.memory_id = m.id "
            "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
            "JOIN documents d ON d.doc_id = f.doc_id "
            f"WHERE {where_sql} "
            "GROUP BY f.memory_id "
            "ORDER BY best_rank "
            "LIMIT ?"
        )

        scores: dict[str, float] = {}
        memory_ids: list[str] = []
        async with self._db.db.execute(top_sql, [*params, limit]) as cursor:
            async for row in cursor:
                memory_id = str(row[0])
                memory_ids.append(memory_id)
                scores[memory_id] = (-float(row[1]) if row[1] is not None else 0.0) * score_scale
        if not memory_ids:
            return []

        refs = await self._metadata_refs(
            table=table,
            where_sql=where_sql,
            params=params,
            memory_ids=memory_ids,
            order_by_rank=True,
        )
        return [
            KeywordCandidate(
                memory_id=memory_id,
                score=scores[memory_id],
                channel=channel,
                matched_fields=(matched_field,),
                source_refs=tuple(refs["source_refs"].get(memory_id, ())),
                matched_text=tuple(refs["matched_text"].get(memory_id, ())),
            )
            for memory_id in memory_ids
        ]

    async def _search_metadata_trigram(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[KeywordCandidate]:
        terms = quoted_query_terms(fts_query)
        variant_groups = [compact_query_variants(term) for term in terms]
        variant_groups = [group for group in variant_groups if group]
        if not variant_groups:
            return []

        predicate_sql, predicate_params = visible_sql(scope, "m")
        conditions = [predicate_sql]
        params: list[Any] = [*predicate_params]

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({type_placeholders})")
            params.extend(memory_types)

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if disabled_source_ids:
            source_placeholders = ",".join("?" for _ in disabled_source_ids)
            conditions.append(
                f"(f.source_id IS NULL OR f.source_id NOT IN ({source_placeholders}))"
            )
            params.extend(disabled_source_ids)

        score_parts: list[str] = []
        match_parts: list[str] = []
        score_params: list[Any] = []
        match_params: list[Any] = []
        for group in variant_groups:
            group_match = " OR ".join("INSTR(f.metadata_compact, ?) > 0" for _ in group)
            match_parts.append(f"({group_match})")
            match_params.extend(group)
            score_parts.append(f"CASE WHEN ({group_match}) THEN 1 ELSE 0 END")
            score_params.extend(group)
        conditions.extend(match_parts)
        where_sql = " AND ".join(conditions)
        score_sql = " + ".join(score_parts)
        top_sql = (
            f"SELECT f.memory_id, MAX(({score_sql}) * 0.25) AS score_value "
            "FROM memory_search_metadata_trigram f "
            "JOIN memories m ON f.memory_id = m.id "
            "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
            "JOIN documents d ON d.doc_id = f.doc_id "
            f"WHERE {where_sql} "
            "GROUP BY f.memory_id "
            "ORDER BY score_value DESC "
            "LIMIT ?"
        )
        all_params = [*score_params, *params, *match_params, limit]
        scores: dict[str, float] = {}
        memory_ids: list[str] = []
        async with self._db.db.execute(top_sql, all_params) as cursor:
            async for row in cursor:
                memory_id = str(row[0])
                memory_ids.append(memory_id)
                scores[memory_id] = float(row[1]) if row[1] is not None else 0.0
        if not memory_ids:
            return []

        refs = await self._metadata_refs(
            table="memory_search_metadata_trigram",
            where_sql=where_sql,
            params=[*params, *match_params],
            memory_ids=memory_ids,
            order_by_rank=False,
        )
        return [
            KeywordCandidate(
                memory_id=memory_id,
                score=scores[memory_id],
                channel="metadata_trigram",
                matched_fields=("metadata_trigram",),
                source_refs=tuple(refs["source_refs"].get(memory_id, ())),
                matched_text=tuple(refs["matched_text"].get(memory_id, ())),
            )
            for memory_id in memory_ids
        ]

    async def _metadata_refs(
        self,
        *,
        table: str,
        where_sql: str,
        params: list[Any],
        memory_ids: list[str],
        order_by_rank: bool,
    ) -> dict[str, dict[str, list[Any]]]:
        memory_placeholders = ",".join("?" for _ in memory_ids)
        order_sql = "rank" if order_by_rank else "f.memory_id"
        refs_sql = (
            "SELECT f.memory_id, f.source_id, f.doc_id, f.source_type, "
            "d.title, d.source_url, d.space_or_project, d.labels, s.name AS source_name "
            f"FROM {table} f "
            "JOIN memories m ON f.memory_id = m.id "
            "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
            "JOIN documents d ON d.doc_id = f.doc_id "
            "LEFT JOIN sources s ON s.id = f.source_id "
            f"WHERE {where_sql} AND f.memory_id IN ({memory_placeholders}) "
            f"ORDER BY f.memory_id, {order_sql}"
        )
        source_refs: dict[str, list[KeywordSourceRef]] = {}
        seen_refs: dict[str, set[KeywordSourceRef]] = {}
        matched_text: dict[str, list[str]] = {}
        seen_text: dict[str, set[str]] = {}
        async with self._db.db.execute(refs_sql, [*params, *memory_ids]) as cursor:
            async for row in cursor:
                memory_id = str(row[0])
                ref = KeywordSourceRef(
                    source_id=str(row[1]) if row[1] is not None else None,
                    doc_id=str(row[2]),
                    source_type=str(row[3]),
                )
                if ref not in seen_refs.setdefault(memory_id, set()):
                    seen_refs[memory_id].add(ref)
                    source_refs.setdefault(memory_id, []).append(ref)
                text = _metadata_match_text(
                    title=row[4],
                    doc_id=row[2],
                    source_url=row[5],
                    space_or_project=row[6],
                    labels=row[7],
                    source_name=row[8],
                )
                if text and text not in seen_text.setdefault(memory_id, set()):
                    seen_text[memory_id].add(text)
                    matched_text.setdefault(memory_id, []).append(text)
        return {"source_refs": source_refs, "matched_text": matched_text}


def _dedupe_metadata_hits(hits: list[KeywordCandidate], limit: int) -> list[KeywordCandidate]:
    best: dict[str, KeywordCandidate] = {}
    channel_priority = {
        "bm25_metadata_tokens": 3,
        "metadata_alias": 2,
        "metadata_trigram": 1,
    }
    for hit in hits:
        previous = best.get(hit.memory_id)
        if previous is None:
            best[hit.memory_id] = hit
            continue
        current_key = (channel_priority.get(hit.channel, 0), hit.score)
        previous_key = (channel_priority.get(previous.channel, 0), previous.score)
        if current_key > previous_key:
            best[hit.memory_id] = hit
    return sorted(
        best.values(),
        key=lambda hit: (channel_priority.get(hit.channel, 0), hit.score),
        reverse=True,
    )[:limit]
