"""SqliteKeywordSearch: the BM25/FTS5 read channel plus the standalone delete.

Memory-row writes and their FTS writes remain co-transactional inside the
Database methods, so this facade owns only the read-path query and the one
delete that runs outside a row write.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.metadata_text import compact_query_variants, quoted_query_terms
from memforge.retrieval.query_plan import MetadataLexicalQueryPlan
from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.protocols import KeywordCandidate, KeywordSourceRef

logger = logging.getLogger(__name__)

__all__ = ["SqliteKeywordSearch"]


def _coerce_metadata_query_plan(
    query_plan: MetadataLexicalQueryPlan | str,
) -> MetadataLexicalQueryPlan:
    """Accept legacy adapter calls while callers migrate to the shared plan."""

    if isinstance(query_plan, MetadataLexicalQueryPlan):
        return query_plan
    terms = tuple(dict.fromkeys(quoted_query_terms(query_plan)))
    term_count = len(terms)
    return MetadataLexicalQueryPlan(
        ordinary_terms=terms,
        minimum_should_match=(term_count if term_count <= 2 else math.ceil(0.60 * term_count)),
    )


def _all_term_fts_query(terms: tuple[str, ...]) -> str:
    return " ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _parse_term_ordinals(raw: Any) -> tuple[int, ...]:
    if raw is None:
        return ()
    return tuple(sorted({int(value) for value in str(raw).split(",") if value != ""}))


def _metadata_term_match_ctes(
    *,
    table: str,
    terms: tuple[str, ...],
    base_conditions: list[str],
    base_params: list[Any],
) -> tuple[str, list[Any]]:
    """Build caller-visible per-term membership and DF CTEs."""

    joins = (
        f"FROM {table} f "
        "JOIN memories m ON f.memory_id = m.id "
        "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
        "JOIN documents d ON d.doc_id = f.doc_id "
    )
    base_where = " AND ".join(base_conditions)
    term_selects: list[str] = []
    params: list[Any] = []
    for index, term in enumerate(terms):
        term_selects.append(
            f"SELECT {index} AS term_ord, f.memory_id, f.source_id, "
            f"f.doc_id, f.source_type {joins}"
            f"WHERE {table} MATCH ? AND {base_where}"
        )
        params.extend([_all_term_fts_query((term,)), *base_params])
    params.extend(base_params)
    cte_sql = (
        "WITH term_matches AS ("
        + " UNION ALL ".join(term_selects)
        + "), term_stats AS ("
        "SELECT term_ord, COUNT(DISTINCT memory_id) AS document_frequency "
        "FROM term_matches GROUP BY term_ord"
        "), visible_corpus AS ("
        f"SELECT COUNT(DISTINCT f.memory_id) AS memory_count {joins}"
        f"WHERE {base_where}"
        ")"
    )
    return cte_sql, params


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


def _append_metadata_source_time_predicates(
    *,
    source_filter: MemorySourceFilter,
    time_range: MemoryTimeRange | None,
    conditions: list[str],
    params: list[Any],
) -> None:
    if source_filter.source_ids:
        placeholders = ",".join("?" for _ in source_filter.source_ids)
        conditions.append(f"f.source_id IN ({placeholders})")
        params.extend(source_filter.source_ids)
    if source_filter.clients:
        placeholders = ",".join("?" for _ in source_filter.clients)
        conditions.append(f"d.client IN ({placeholders})")
        params.extend(source_filter.clients)
    if source_filter.repo_identifiers:
        placeholders = ",".join("?" for _ in source_filter.repo_identifiers)
        conditions.append(f"m.repo_identifier IN ({placeholders})")
        params.extend(source_filter.repo_identifiers)

    if time_range is None or time_range.is_empty():
        return
    if time_range.date_type == "source_updated_at":
        if time_range.after is not None:
            conditions.append("ms.source_updated_at >= ?")
            params.append(time_range.after.isoformat())
        if time_range.before is not None:
            conditions.append("ms.source_updated_at < ?")
            params.append(time_range.before.isoformat())
    elif time_range.date_type == "memory_updated_at":
        if time_range.after is not None:
            conditions.append("m.updated_at >= ?")
            params.append(time_range.after.isoformat())
        if time_range.before is not None:
            conditions.append("m.updated_at < ?")
            params.append(time_range.before.isoformat())
    else:
        raise ValueError(f"Unsupported memory time range date_type: {time_range.date_type}")


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
        query_plan: MetadataLexicalQueryPlan | str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
        include_subchannel_hits: bool = False,
    ) -> list[KeywordCandidate]:
        if limit <= 0:
            return []

        plan = _coerce_metadata_query_plan(query_plan)
        if not plan.ordinary_terms and not plan.exact_anchors:
            return []

        try:
            if source_filter is None:
                source_filter = MemorySourceFilter()
            hits: list[KeywordCandidate] = []
            if plan.ordinary_terms:
                hits.extend(
                    await self._search_metadata_fts(
                        plan,
                        scope,
                        memory_types,
                        limit,
                        table="memory_search_metadata_fts",
                        channel="bm25_metadata_tokens",
                        matched_field="metadata_any",
                        score_scale=1.0,
                        source_filter=source_filter,
                        time_range=time_range,
                    )
                )
                hits.extend(
                    await self._search_metadata_fts(
                        plan,
                        scope,
                        memory_types,
                        limit,
                        table="memory_search_metadata_alias_fts",
                        channel="metadata_alias",
                        matched_field="metadata_alias",
                        score_scale=0.75,
                        source_filter=source_filter,
                        time_range=time_range,
                    )
                )
                hits.extend(
                    await self._search_metadata_trigram(
                        plan,
                        scope,
                        memory_types,
                        limit,
                        source_filter,
                        time_range,
                    )
                )
            for anchor in plan.exact_anchors:
                hits.extend(
                    await self._search_metadata_fts(
                        MetadataLexicalQueryPlan(
                            ordinary_terms=(anchor.value,),
                            minimum_should_match=1,
                        ),
                        scope,
                        memory_types,
                        limit,
                        table="memory_search_metadata_fts",
                        channel="bm25_metadata_tokens",
                        matched_field="metadata_exact_anchor",
                        score_scale=100.0,
                        source_filter=source_filter,
                        time_range=time_range,
                    )
                )
            return _dedupe_metadata_hits(
                hits,
                limit,
                include_subchannel_hits=include_subchannel_hits,
            )
        except Exception:
            logger.exception("Metadata keyword search failed")
            return []

    async def _search_metadata_fts(
        self,
        query_plan: MetadataLexicalQueryPlan,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        *,
        table: str,
        channel: str,
        matched_field: str,
        score_scale: float,
        source_filter: MemorySourceFilter,
        time_range: MemoryTimeRange | None,
    ) -> list[KeywordCandidate]:
        predicate_sql, predicate_params = visible_sql(scope, "m")
        base_conditions = [predicate_sql]
        base_params: list[Any] = [*predicate_params]

        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            base_conditions.append(f"m.memory_type IN ({type_placeholders})")
            base_params.extend(memory_types)

        _append_metadata_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            conditions=base_conditions,
            params=base_params,
        )

        disabled_source_ids = await self._db.list_disabled_source_ids_for_user(scope.user_id)
        if disabled_source_ids:
            source_placeholders = ",".join("?" for _ in disabled_source_ids)
            base_conditions.append(
                f"(f.source_id IS NULL OR f.source_id NOT IN ({source_placeholders}))"
            )
            base_params.extend(disabled_source_ids)

        cte_sql, cte_params = _metadata_term_match_ctes(
            table=table,
            terms=query_plan.ordinary_terms,
            base_conditions=base_conditions,
            base_params=base_params,
        )
        top_sql = (
            cte_sql
            + ", qualified_support AS ("
            "SELECT tm.memory_id, tm.source_id, tm.doc_id, tm.source_type, "
            "COUNT(DISTINCT tm.term_ord) AS matched_count, "
            "GROUP_CONCAT(DISTINCT tm.term_ord) AS matched_ordinals, "
            "SUM(ln(1.0 + ((vc.memory_count - ts.document_frequency + 0.5) / "
            "(ts.document_frequency + 0.5)))) AS idf_score "
            "FROM term_matches tm "
            "JOIN term_stats ts ON ts.term_ord = tm.term_ord "
            "CROSS JOIN visible_corpus vc "
            "GROUP BY tm.memory_id, tm.source_id, tm.doc_id, tm.source_type "
            "HAVING COUNT(DISTINCT tm.term_ord) >= ?"
            "), ranked_support AS ("
            "SELECT qs.*, ROW_NUMBER() OVER ("
            "PARTITION BY qs.memory_id "
            "ORDER BY qs.idf_score DESC, qs.matched_count DESC, qs.doc_id"
            ") AS support_rank FROM qualified_support qs"
            ") "
            "SELECT memory_id, idf_score, matched_count, matched_ordinals "
            "FROM ranked_support WHERE support_rank = 1 "
            "ORDER BY idf_score DESC, matched_count DESC, memory_id LIMIT ?"
        )

        scores: dict[str, float] = {}
        matched_terms: dict[str, tuple[str, ...]] = {}
        term_weights: dict[str, tuple[tuple[str, float], ...]] = {}
        memory_ids: list[str] = []
        all_term_weights = await self._caller_visible_term_weights(
            cte_sql=cte_sql,
            cte_params=cte_params,
            terms=query_plan.ordinary_terms,
        )
        async with self._db.db.execute(
            top_sql,
            [*cte_params, query_plan.minimum_should_match, limit],
        ) as cursor:
            async for row in cursor:
                memory_id = str(row[0])
                ordinals = _parse_term_ordinals(row[3])
                terms = tuple(query_plan.ordinary_terms[index] for index in ordinals)
                memory_ids.append(memory_id)
                scores[memory_id] = float(row[1] or 0.0) * score_scale
                matched_terms[memory_id] = terms
                term_weights[memory_id] = tuple(
                    (query_plan.ordinary_terms[index], all_term_weights[index] * score_scale)
                    for index in ordinals
                )
        if not memory_ids:
            return []

        refs = await self._coverage_metadata_refs(
            cte_sql=cte_sql,
            cte_params=cte_params,
            minimum_should_match=query_plan.minimum_should_match,
            memory_ids=memory_ids,
        )
        return [
            KeywordCandidate(
                memory_id=memory_id,
                score=scores[memory_id],
                channel=channel,
                matched_fields=(matched_field,),
                source_refs=tuple(refs["source_refs"].get(memory_id, ())),
                matched_text=tuple(refs["matched_text"].get(memory_id, ())),
                matched_terms=matched_terms[memory_id],
                term_coverage=len(matched_terms[memory_id]) / len(query_plan.ordinary_terms),
                term_weights=term_weights[memory_id],
            )
            for memory_id in memory_ids
        ]

    async def _caller_visible_term_weights(
        self,
        *,
        cte_sql: str,
        cte_params: list[Any],
        terms: tuple[str, ...],
    ) -> dict[int, float]:
        weights = {index: 0.0 for index in range(len(terms))}
        sql = (
            cte_sql
            + " SELECT ts.term_ord, ts.document_frequency, vc.memory_count "
            "FROM term_stats ts CROSS JOIN visible_corpus vc ORDER BY ts.term_ord"
        )
        async with self._db.db.execute(sql, cte_params) as cursor:
            async for row in cursor:
                term_ord = int(row[0])
                document_frequency = int(row[1])
                memory_count = int(row[2])
                weights[term_ord] = math.log(
                    1.0
                    + (memory_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
        return weights

    async def _coverage_metadata_refs(
        self,
        *,
        cte_sql: str,
        cte_params: list[Any],
        minimum_should_match: int,
        memory_ids: list[str],
    ) -> dict[str, dict[str, list[Any]]]:
        placeholders = ",".join("?" for _ in memory_ids)
        sql = (
            cte_sql
            + ", qualified_support AS ("
            "SELECT memory_id, source_id, doc_id, source_type "
            "FROM term_matches "
            "GROUP BY memory_id, source_id, doc_id, source_type "
            "HAVING COUNT(DISTINCT term_ord) >= ?"
            ") "
            "SELECT qs.memory_id, qs.source_id, qs.doc_id, qs.source_type, "
            "d.title, d.source_url, d.space_or_project, d.labels, s.name "
            "FROM qualified_support qs "
            "JOIN documents d ON d.doc_id = qs.doc_id "
            "LEFT JOIN sources s ON s.id = qs.source_id "
            f"WHERE qs.memory_id IN ({placeholders}) "
            "ORDER BY qs.memory_id, qs.doc_id"
        )
        source_refs: dict[str, list[KeywordSourceRef]] = {}
        matched_text: dict[str, list[str]] = {}
        async with self._db.db.execute(
            sql,
            [*cte_params, minimum_should_match, *memory_ids],
        ) as cursor:
            async for row in cursor:
                memory_id = str(row[0])
                source_refs.setdefault(memory_id, []).append(
                    KeywordSourceRef(
                        source_id=str(row[1]) if row[1] is not None else None,
                        doc_id=str(row[2]),
                        source_type=str(row[3]),
                    )
                )
                text = _metadata_match_text(
                    title=row[4],
                    doc_id=row[2],
                    source_url=row[5],
                    space_or_project=row[6],
                    labels=row[7],
                    source_name=row[8],
                )
                if text:
                    matched_text.setdefault(memory_id, []).append(text)
        return {"source_refs": source_refs, "matched_text": matched_text}

    async def _search_metadata_trigram(
        self,
        query_plan: MetadataLexicalQueryPlan,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
        source_filter: MemorySourceFilter,
        time_range: MemoryTimeRange | None,
    ) -> list[KeywordCandidate]:
        terms = list(query_plan.ordinary_terms)
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

        _append_metadata_source_time_predicates(
            source_filter=source_filter,
            time_range=time_range,
            conditions=conditions,
            params=params,
        )

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
        match_count_sql = " + ".join(score_parts)
        conditions.append(f"({match_count_sql}) >= ?")
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
        all_params = [
            *score_params,
            *params,
            *score_params,
            query_plan.minimum_should_match,
            limit,
        ]
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
            params=[*params, *score_params, query_plan.minimum_should_match],
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


def _dedupe_metadata_hits(
    hits: list[KeywordCandidate],
    limit: int,
    *,
    include_subchannel_hits: bool,
) -> list[KeywordCandidate]:
    best: dict[str | tuple[str, str], KeywordCandidate] = {}
    channel_priority = {
        "bm25_metadata_tokens": 3,
        "metadata_alias": 2,
        "metadata_trigram": 1,
    }
    for hit in hits:
        key = (hit.memory_id, hit.channel) if include_subchannel_hits else hit.memory_id
        previous = best.get(key)
        if previous is None:
            best[key] = hit
            continue
        if (
            channel_priority.get(hit.channel, 0),
            hit.score,
        ) > (
            channel_priority.get(previous.channel, 0),
            previous.score,
        ):
            best[key] = hit
    result_limit = limit * len(channel_priority) if include_subchannel_hits else limit
    return sorted(
        best.values(),
        key=lambda hit: (channel_priority.get(hit.channel, 0), hit.score),
        reverse=True,
    )[:result_limit]
