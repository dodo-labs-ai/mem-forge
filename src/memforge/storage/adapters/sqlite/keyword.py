"""SqliteKeywordSearch: the BM25/FTS5 read channel plus the standalone delete.

Memory-row writes and their FTS writes remain co-transactional inside the
Database methods, so this facade owns only the read-path query and the one
delete that runs outside a row write.
"""

from __future__ import annotations

import logging
from typing import Any

from memforge.storage.database import Database
from memforge.storage.adapters.context import AccessScope

logger = logging.getLogger(__name__)

__all__ = ["SqliteKeywordSearch"]


class SqliteKeywordSearch:
    """The keyword channel backed by the memories_fts FTS5 table."""

    def __init__(self, db: Database) -> None:
        self._db = db

    async def remove(self, memory_id: str) -> None:
        await self._db.db.execute(
            "DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,)
        )
        await self._db.db.commit()

    async def search(
        self,
        fts_query: str,
        scope: AccessScope,
        memory_types: list[str] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        conditions = ["memories_fts MATCH ?"]
        params: list[Any] = [fts_query]

        statuses = scope.allowed_statuses
        status_placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"m.status IN ({status_placeholders})")
        params.extend(statuses)

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
