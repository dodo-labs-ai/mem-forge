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

    metadata_search_channels = ("bm25_metadata_tokens",)
    disabled_metadata_search_channels = ("metadata_alias", "metadata_trigram")

    def __init__(self, db: Database) -> None:
        self._db = db

    async def remove(self, memory_id: str) -> None:
        await self._db.db.execute(
            "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.execute(
            "DELETE FROM memory_search_metadata_fts WHERE memory_id = ?", (memory_id,)
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

        predicate_sql, predicate_params = visible_sql(scope, "m")
        conditions = ["memory_search_metadata_fts MATCH ?"]
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
            "FROM memory_search_metadata_fts f "
            "JOIN memories m ON f.memory_id = m.id "
            "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
            "JOIN documents d ON d.doc_id = f.doc_id "
            f"WHERE {where_sql} "
            "GROUP BY f.memory_id "
            "ORDER BY best_rank "
            "LIMIT ?"
        )

        try:
            scores: dict[str, float] = {}
            memory_ids: list[str] = []
            async with self._db.db.execute(top_sql, [*params, limit]) as cursor:
                async for row in cursor:
                    memory_id = str(row[0])
                    memory_ids.append(memory_id)
                    scores[memory_id] = -float(row[1]) if row[1] is not None else 0.0
            if not memory_ids:
                return []

            memory_placeholders = ",".join("?" for _ in memory_ids)
            refs_sql = (
                "SELECT f.memory_id, f.source_id, f.doc_id, f.source_type, "
                "d.title, d.source_url, d.space_or_project, d.labels, s.name AS source_name "
                "FROM memory_search_metadata_fts f "
                "JOIN memories m ON f.memory_id = m.id "
                "JOIN memory_sources ms ON ms.memory_id = f.memory_id AND ms.doc_id = f.doc_id "
                "JOIN documents d ON d.doc_id = f.doc_id "
                "LEFT JOIN sources s ON s.id = f.source_id "
                f"WHERE {where_sql} AND f.memory_id IN ({memory_placeholders}) "
                "ORDER BY f.memory_id, rank"
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
            return [
                KeywordCandidate(
                    memory_id=memory_id,
                    score=scores[memory_id],
                    channel="bm25_metadata_tokens",
                    matched_fields=("metadata_any",),
                    source_refs=tuple(source_refs.get(memory_id, ())),
                    matched_text=tuple(matched_text.get(memory_id, ())),
                )
                for memory_id in memory_ids
            ]
        except Exception:
            logger.exception("Metadata keyword (FTS5) search failed")
            return []
