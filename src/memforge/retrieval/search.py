"""Hybrid search engine for MemForge.

Four retrieval channels run in parallel (vector, BM25/FTS5, entity-graph,
temporal SQL), fused via Reciprocal Rank Fusion (RRF), then ranked with a
recency-weighted final score.  Optional cross-encoder reranking via LLM
can be enabled for higher precision at scale.  Document fallback fills
remaining slots when memory results are sparse.

Architecture reference: docs/architecture.md Section 10.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from memforge.config import RetrievalConfig
from memforge.llm.structured import StructuredLlmError
from memforge.memory.lifecycle import allowed_search_statuses
from memforge.models import Memory, SearchResult
from memforge.provenance import document_content_url, document_pdf_url
from memforge.retrieval.embeddings import EmbeddingCache, embed_texts
from memforge.retrieval.query_analyzer import QueryAnalysis, analyze_query
from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["SearchEngine"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _RankedCandidate:
    """Intermediate result used during fusion and ranking."""
    memory_id: str
    rrf_score: float = 0.0
    final_score: float = 0.0
    updated_at: datetime | None = None


def _age_days(dt: datetime | None) -> float:
    """Return age in fractional days from now (UTC).  Defaults to 0 if None."""
    if dt is None:
        return 0.0
    now = datetime.now(timezone.utc)
    # Ensure timezone-aware comparison
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = now - dt
    return max(delta.total_seconds() / 86400.0, 0.0)


def _recency_score(age_days: float, half_life: float = 90.0) -> float:
    """Exponential decay: exp(-0.693 * age_days / half_life)."""
    return math.exp(-0.693 * age_days / half_life)


def _sanitize_fts_query(text: str) -> str:
    """Escape characters that are special in FTS5 MATCH syntax.

    FTS5 interprets *, ^, (, ), :, " as operators.  We quote each word to
    avoid syntax errors on user input that happens to contain them.
    """
    words = text.split()
    safe: list[str] = []
    for w in words:
        # Strip non-alphanumeric edges (punctuation) but keep the core word
        cleaned = "".join(ch for ch in w if ch.isalnum() or ch in ("-", "_"))
        if cleaned:
            # Quote each token to prevent FTS5 operator interpretation
            safe.append(f'"{cleaned}"')
    return " ".join(safe)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:
    """Hybrid retrieval engine: vector + BM25 + graph + temporal, fused via RRF.

    Parameters
    ----------
    db : Database
        Async SQLite database (must already be connected).
    memory_collection : Any
        ChromaDB collection for memory embeddings (name="memories").
    embed_cfg : dict
        Keys: ``base_url``, ``api_key``, ``model`` — forwarded to
        :func:`embed_texts`.
    config : RetrievalConfig
        Tuning knobs (``rrf_k``, ``recency_half_life_days``, etc.).
    document_collection : Any | None
        Optional ChromaDB collection for document-level embeddings
        (name="documents").  Required for document fallback.
    """

    def __init__(
        self,
        db: Database,
        memory_collection: Any,
        embed_cfg: dict,
        config: RetrievalConfig,
        document_collection: Any | None = None,
        structured_llm_client: Any | None = None,
    ) -> None:
        self._db = db
        self._mem_collection = memory_collection
        self._doc_collection = document_collection
        self._embed_cfg = embed_cfg
        self._config = config
        self._embed_cache = EmbeddingCache(max_size=config.embedding_cache_size)
        self._structured_llm_client = structured_llm_client

    # ==================================================================
    # Public API
    # ==================================================================

    async def search(
        self,
        query: str,
        memory_types: list[str] | None = None,
        sources: list[str] | None = None,
        time_range: dict | None = None,
        entities: list[str] | None = None,
        include_superseded: bool = False,
        top_k: int = 10,
    ) -> dict:
        """Unified search: memories (primary) + documents (fallback).

        Returns
        -------
        dict
            ``query_analysis`` : summary of the analysis step.
            ``results``        : list of :class:`SearchResult`.
            ``total_candidates``: number of unique memories seen across all channels.
            ``retrieval_time_ms``: wall-clock milliseconds for the entire search.
        """
        t0 = time.monotonic()

        # ----- 1. Build known entities dict for query analysis -----
        known_entities = await self._build_known_entities()

        # ----- 2. Analyze query -----
        analysis = await analyze_query(
            query,
            known_entities,
            entity_model=self._config.entity_model,
            entity_timeout_s=self._config.entity_timeout_s,
            structured_llm_client=self._structured_llm_client,
        )

        # ----- 3. Override analysis with explicit params -----
        if entities:
            for ent_name in entities:
                norm = ent_name.strip().lower()
                if norm in known_entities and norm not in analysis.detected_entities:
                    analysis.detected_entities.append(norm)
                    analysis.detected_entity_ids.append(known_entities[norm])
            if analysis.detected_entities:
                analysis.use_graph = True

        if time_range:
            analysis.is_temporal = True
            analysis.use_temporal = True
            if time_range.get("after") and analysis.temporal_start is None:
                try:
                    analysis.temporal_start = datetime.fromisoformat(
                        time_range["after"]
                    ).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass
            if time_range.get("before") and analysis.temporal_end is None:
                try:
                    analysis.temporal_end = datetime.fromisoformat(
                        time_range["before"]
                    ).replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

        # ----- 4. Run active channels in parallel -----
        fetch_k = top_k * 2
        tasks: list[asyncio.Task] = []
        channel_names: list[str] = []

        # Vector search is always on
        tasks.append(asyncio.ensure_future(
            self._vector_search(query, memory_types, sources, include_superseded, fetch_k)
        ))
        channel_names.append("vector")

        # BM25 is always on
        tasks.append(asyncio.ensure_future(
            self._bm25_search(query, analysis, memory_types, sources, include_superseded, fetch_k)
        ))
        channel_names.append("bm25")

        # Graph traversal — only when entities detected
        if analysis.use_graph and analysis.detected_entity_ids:
            tasks.append(asyncio.ensure_future(
                self._graph_search(analysis.detected_entity_ids, memory_types, include_superseded, fetch_k)
            ))
            channel_names.append("graph")

        # Temporal filter — only when temporal intent detected
        if analysis.use_temporal:
            tasks.append(asyncio.ensure_future(
                self._temporal_filter(
                    analysis.temporal_start,
                    analysis.temporal_end,
                    memory_types,
                    include_superseded,
                    fetch_k,
                )
            ))
            channel_names.append("temporal")

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect channel results, logging any errors
        channel_results: list[list[tuple[str, float]]] = []
        for name, result in zip(channel_names, raw_results):
            if isinstance(result, BaseException):
                logger.error("Channel %s failed: %s", name, result, exc_info=result)
                channel_results.append([])
            else:
                channel_results.append(result)

        # ----- 5. Fuse via RRF -----
        fused = self._rrf_fusion(channel_results, k=self._config.rrf_k)
        fused = await self._filter_candidates_by_status(fused, include_superseded)
        total_candidates = len(fused)

        # ----- 6. Apply ranking -----
        ranked = await self._apply_ranking(fused, analysis.is_temporal)

        # ----- 6b. Optional cross-encoder rerank -----
        ranked = await self._rerank_with_llm(query, ranked, top_k)

        # ----- 7. Trim to top_k -----
        ranked = ranked[:top_k]

        # ----- 8. Enrich results -----
        memory_ids = [c.memory_id for c in ranked]
        results = await self._enrich_results(memory_ids, ranked)

        # ----- 9. Document fallback -----
        if len(results) < top_k:
            remaining = top_k - len(results)
            existing_doc_ids = {
                r.source_doc_id for r in results if r.source_doc_id
            }
            fallback = await self._document_fallback(query, remaining, existing_doc_ids)
            results.extend(fallback)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return {
            "query_analysis": {
                "is_temporal": analysis.is_temporal,
                "detected_entities": analysis.detected_entities,
                "strategies_used": channel_names,
            },
            "results": results,
            "total_candidates": total_candidates,
            "retrieval_time_ms": elapsed_ms,
        }

    # ==================================================================
    # Channel implementations
    # ==================================================================

    async def _vector_search(
        self,
        query: str,
        memory_types: list[str] | None,
        sources: list[str] | None,
        include_superseded: bool,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Embed query via cache, then query ChromaDB memories collection."""
        embedding = self._get_or_compute_embedding(query)
        if embedding is None:
            return []

        where_clauses: list[dict] = []
        if memory_types:
            where_clauses.append({"memory_type": {"$in": memory_types}})
        statuses = allowed_search_statuses(include_superseded)
        if len(statuses) == 1:
            where_clauses.append({"status": statuses[0]})
        else:
            where_clauses.append({"status": {"$in": list(statuses)}})
        if sources:
            where_clauses.append({"source": {"$in": sources}})

        where: dict | None = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        try:
            query_params: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": limit,
            }
            if where:
                query_params["where"] = where

            results = self._mem_collection.query(**query_params)
        except Exception:
            logger.exception("ChromaDB vector search failed")
            return []

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        # ChromaDB cosine distance: lower = more similar.
        # Convert to a similarity score for consistent ranking.
        scored: list[tuple[str, float]] = []
        for mid, dist in zip(ids, distances):
            similarity = max(1.0 - dist, 0.0)
            scored.append((mid, similarity))

        return scored

    async def _bm25_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        memory_types: list[str] | None,
        sources: list[str] | None,
        include_superseded: bool,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Query the ``memories_fts`` FTS5 table with optional alias expansion."""
        # Build expanded FTS query
        expanded = await self._expand_query_with_aliases(
            query, analysis.detected_entity_ids
        )
        fts_query = _sanitize_fts_query(expanded)
        if not fts_query:
            return []

        # Build the SQL with joins for source/status filtering
        sql_parts = [
            "SELECT f.memory_id, rank",
            "FROM memories_fts f",
            "JOIN memories m ON f.memory_id = m.id",
        ]
        joins: list[str] = []
        conditions: list[str] = [
            "memories_fts MATCH ?",
        ]
        params: list[Any] = [fts_query]

        statuses = allowed_search_statuses(include_superseded)
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"m.status IN ({placeholders})")
        params.extend(statuses)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({placeholders})")
            params.extend(memory_types)
        if sources:
            joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
            joins.append("JOIN documents d ON ms.doc_id = d.doc_id")
            placeholders = ",".join("?" for _ in sources)
            conditions.append(f"d.source IN ({placeholders})")
            params.extend(sources)

        for j in joins:
            sql_parts.append(j)
        sql_parts.append("WHERE " + " AND ".join(conditions))
        sql_parts.append("ORDER BY rank")
        sql_parts.append(f"LIMIT {limit}")

        full_sql = " ".join(sql_parts)

        try:
            results: list[tuple[str, float]] = []
            async with self._db.db.execute(full_sql, params) as cursor:
                async for row in cursor:
                    # FTS5 rank is negative (more negative = better match).
                    # Invert so higher = better.
                    memory_id = row[0]
                    rank_score = -float(row[1]) if row[1] is not None else 0.0
                    results.append((memory_id, rank_score))
            return results
        except Exception:
            logger.exception("BM25 search failed")
            return []

    async def _graph_search(
        self,
        entity_ids: list[int],
        memory_types: list[str] | None,
        include_superseded: bool,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Entity-graph traversal: direct links + 1-hop co-entity expansion.

        Algorithm from architecture.md Section 14h.
        """
        if not entity_ids:
            return []

        # --- Direct links: memories referencing query entities ---
        placeholders = ",".join("?" for _ in entity_ids)
        statuses = allowed_search_statuses(include_superseded)
        status_placeholders = ",".join("?" for _ in statuses)
        status_filter = f"AND m.status IN ({status_placeholders})"
        type_filter = ""
        type_params: list[Any] = []
        if memory_types:
            type_placeholders = ",".join("?" for _ in memory_types)
            type_filter = f"AND m.memory_type IN ({type_placeholders})"
            type_params = list(memory_types)

        direct_sql = f"""
            SELECT m.id, COUNT(me.entity_id) AS entity_overlap
            FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            WHERE me.entity_id IN ({placeholders})
              {status_filter}
              {type_filter}
            GROUP BY m.id
            ORDER BY entity_overlap DESC
            LIMIT ?
        """
        direct_params: list[Any] = [*entity_ids, *statuses, *type_params, limit]

        direct_results: list[tuple[str, int]] = []
        try:
            async with self._db.db.execute(direct_sql, direct_params) as cursor:
                async for row in cursor:
                    direct_results.append((row[0], int(row[1])))
        except Exception:
            logger.exception("Graph direct search failed")
            return []

        # Score direct results
        num_query_entities = len(entity_ids)
        scored: list[tuple[str, float]] = []
        for mid, overlap in direct_results:
            scored.append((mid, float(overlap) / num_query_entities))

        # --- 1-hop expansion ---
        if direct_results:
            direct_ids = [r[0] for r in direct_results]
            d_placeholders = ",".join("?" for _ in direct_ids)

            expanded_sql = f"""
                SELECT m.id, COUNT(DISTINCT me2.entity_id) AS shared_entities
                FROM memory_entities me1
                JOIN memory_entities me2 ON me1.entity_id = me2.entity_id
                JOIN memories m ON me2.memory_id = m.id
                WHERE me1.memory_id IN ({d_placeholders})
                  AND m.id NOT IN ({d_placeholders})
                  {status_filter}
                  {type_filter}
                GROUP BY m.id
                HAVING shared_entities >= 2
                ORDER BY shared_entities DESC
                LIMIT ?
            """
            expanded_params: list[Any] = [
                *direct_ids, *direct_ids, *statuses, *type_params, limit,
            ]

            try:
                async with self._db.db.execute(expanded_sql, expanded_params) as cursor:
                    async for row in cursor:
                        mid = row[0]
                        shared = int(row[1])
                        scored.append((mid, 0.5 * shared / num_query_entities))
            except Exception:
                logger.exception("Graph 1-hop expansion failed")

        return scored

    async def _temporal_filter(
        self,
        start: datetime | None,
        end: datetime | None,
        memory_types: list[str] | None,
        include_superseded: bool,
        limit: int,
    ) -> list[tuple[str, float]]:
        """SQL date-range filter on memories.created_at / updated_at."""
        conditions: list[str] = []
        params: list[Any] = []

        if start:
            iso_start = start.isoformat()
            conditions.append("(m.updated_at >= ? OR m.created_at >= ?)")
            params.extend([iso_start, iso_start])
        if end:
            iso_end = end.isoformat()
            conditions.append("(m.updated_at <= ? OR m.created_at <= ?)")
            params.extend([iso_end, iso_end])

        if not conditions:
            return []

        statuses = allowed_search_statuses(include_superseded)
        placeholders = ",".join("?" for _ in statuses)
        conditions.append(f"m.status IN ({placeholders})")
        params.extend(statuses)
        if memory_types:
            placeholders = ",".join("?" for _ in memory_types)
            conditions.append(f"m.memory_type IN ({placeholders})")
            params.extend(memory_types)

        where = " AND ".join(conditions)
        sql = f"""
            SELECT m.id FROM memories m
            WHERE {where}
            ORDER BY m.updated_at DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            results: list[tuple[str, float]] = []
            async with self._db.db.execute(sql, params) as cursor:
                idx = 0
                async for row in cursor:
                    # Assign a decreasing score based on result order so that
                    # more-recently-updated memories rank higher within this channel.
                    idx += 1
                    score = 1.0 / idx
                    results.append((row[0], score))
            return results
        except Exception:
            logger.exception("Temporal filter failed")
            return []

    # ==================================================================
    # Fusion
    # ==================================================================

    def _rrf_fusion(
        self,
        channel_results: list[list[tuple[str, float]]],
        k: int = 60,
    ) -> list[_RankedCandidate]:
        """Reciprocal Rank Fusion across all channels.

        For each channel the results are sorted by score descending, then
        each memory receives ``1 / (k + rank)`` where ``rank`` is 1-based.
        Scores are summed across channels.
        """
        scores: dict[str, float] = {}

        for channel in channel_results:
            # Sort channel by score descending to determine per-channel rank
            sorted_channel = sorted(channel, key=lambda x: x[1], reverse=True)
            for rank_0, (memory_id, _score) in enumerate(sorted_channel):
                rank = rank_0 + 1  # 1-based
                scores[memory_id] = scores.get(memory_id, 0.0) + 1.0 / (k + rank)

        # Sort by RRF score descending
        candidates = [
            _RankedCandidate(memory_id=mid, rrf_score=s)
            for mid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return candidates

    async def _filter_candidates_by_status(
        self,
        candidates: list[_RankedCandidate],
        include_superseded: bool,
    ) -> list[_RankedCandidate]:
        """Apply DB-source-of-truth lifecycle visibility after channel fusion."""
        if not candidates:
            return []

        allowed = set(allowed_search_statuses(include_superseded))
        memory_ids = [c.memory_id for c in candidates]
        visible: set[str] = set()
        batch_size = 200

        for i in range(0, len(memory_ids), batch_size):
            batch = memory_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            try:
                async with self._db.db.execute(
                    f"SELECT id, status FROM memories WHERE id IN ({placeholders})",
                    batch,
                ) as cursor:
                    async for row in cursor:
                        if row["status"] in allowed:
                            visible.add(row["id"])
            except Exception:
                logger.exception("Failed to apply lifecycle status filter")
                return []

        return [c for c in candidates if c.memory_id in visible]

    # ==================================================================
    # Ranking
    # ==================================================================

    async def _apply_ranking(
        self,
        candidates: list[_RankedCandidate],
        is_temporal: bool,
    ) -> list[_RankedCandidate]:
        """Apply recency-weighted final ranking.

        ``final_score = w_rrf * rrf_normalized + w_recency * recency``

        Standard:  w_rrf=0.85, w_recency=0.15
        Temporal:  w_rrf=0.70, w_recency=0.30
        """
        if not candidates:
            return candidates

        # Fetch updated_at for each candidate (bulk query)
        id_to_updated: dict[str, datetime | None] = {}
        batch_size = 200
        memory_ids = [c.memory_id for c in candidates]
        for i in range(0, len(memory_ids), batch_size):
            batch = memory_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in batch)
            try:
                async with self._db.db.execute(
                    f"SELECT id, updated_at FROM memories WHERE id IN ({placeholders})",
                    batch,
                ) as cursor:
                    async for row in cursor:
                        dt_str = row[1]
                        dt = None
                        if dt_str:
                            try:
                                dt = datetime.fromisoformat(dt_str)
                            except (ValueError, TypeError):
                                pass
                        id_to_updated[row[0]] = dt
            except Exception:
                logger.exception("Failed to fetch updated_at for ranking batch")

        # Normalize RRF scores to [0, 1]
        max_rrf = max(c.rrf_score for c in candidates) if candidates else 1.0
        if max_rrf == 0:
            max_rrf = 1.0

        half_life = float(self._config.recency_half_life_days)
        w_rrf = 0.70 if is_temporal else 0.85
        w_rec = 0.30 if is_temporal else 0.15

        for c in candidates:
            c.updated_at = id_to_updated.get(c.memory_id)
            rrf_norm = c.rrf_score / max_rrf
            age = _age_days(c.updated_at)
            recency = _recency_score(age, half_life)
            c.final_score = w_rrf * rrf_norm + w_rec * recency

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    # ==================================================================
    # Cross-encoder reranking (config-gated)
    # ==================================================================

    async def _rerank_with_llm(
        self,
        query: str,
        candidates: list[_RankedCandidate],
        top_k: int,
    ) -> list[_RankedCandidate]:
        """Rerank top candidates using an LLM cross-encoder pass.

        Scores each (query, memory) pair independently, resolving RRF
        channel-count bias by evaluating actual relevance regardless of
        which channel found the result.

        Requires ``retrieval.enable_reranking = true`` in config.
        Uses Claude Haiku by default (~200ms, ~$0.001/query).
        """
        if not self._config.enable_reranking:
            return candidates

        rerank_n = min(len(candidates), self._config.rerank_candidates)
        to_rerank = candidates[:rerank_n]
        remainder = candidates[rerank_n:]

        # Fetch memory content for each candidate
        id_to_content: dict[str, str] = {}
        for c in to_rerank:
            try:
                mem = await self._db.get_memory(c.memory_id)
                if mem:
                    id_to_content[c.memory_id] = mem.content
            except Exception:
                pass

        if not id_to_content:
            return candidates

        # Build the reranking prompt
        numbered = []
        idx_to_id: dict[int, str] = {}
        for i, c in enumerate(to_rerank):
            content = id_to_content.get(c.memory_id, "")
            if content:
                numbered.append(f"{i}. {content}")
                idx_to_id[i] = c.memory_id

        if not numbered:
            return candidates

        prompt = (
            f"Rank these memories by relevance to the query. "
            f"Return ONLY a JSON object with a ranking array of memory numbers in order, most relevant first.\n\n"
            f"Query: {query}\n\n"
            f"Memories:\n" + "\n".join(numbered) + "\n\n"
            'Return format: {"ranking": [3, 0, 7, 1]}'
        )

        try:
            if self._structured_llm_client is None:
                return candidates
            response = await self._structured_llm_client.rerank_memories(
                prompt,
                max_tokens=256,
                model=self._config.rerank_model,
            )
            ranking = response.ranking

            # Rebuild candidate list in LLM-ranked order
            id_to_candidate = {c.memory_id: c for c in to_rerank}
            reranked: list[_RankedCandidate] = []
            seen: set[str] = set()

            for idx in ranking:
                if isinstance(idx, int) and idx in idx_to_id:
                    mid = idx_to_id[idx]
                    if mid in id_to_candidate and mid not in seen:
                        c = id_to_candidate[mid]
                        c.final_score = 1.0 - (len(reranked) * 0.01)  # preserve order
                        reranked.append(c)
                        seen.add(mid)

            # Append any candidates the LLM didn't rank
            for c in to_rerank:
                if c.memory_id not in seen:
                    reranked.append(c)

            reranked.extend(remainder)
            return reranked

        except (StructuredLlmError, Exception):
            logger.warning("LLM reranking failed, falling back to RRF ranking", exc_info=True)
            return candidates

    # ==================================================================
    # Result enrichment
    # ==================================================================

    async def _enrich_results(
        self,
        memory_ids: list[str],
        ranked: list[_RankedCandidate],
    ) -> list[SearchResult]:
        """Fetch full Memory objects and primary source for each result."""
        if not memory_ids:
            return []

        # Build a lookup of candidate scores
        score_map = {c.memory_id: c for c in ranked}

        results: list[SearchResult] = []
        for mid in memory_ids:
            candidate = score_map.get(mid)
            if candidate is None:
                continue

            # Fetch memory
            try:
                memory = await self._db.get_memory(mid)
            except Exception:
                logger.exception("Failed to fetch memory %s", mid)
                continue
            if memory is None:
                continue

            # Fetch primary source (most recent)
            source_doc_id = None
            source_doc_title = None
            source_type = None
            content_url = None
            pdf_url = None
            source_url = None

            try:
                mem_sources = await self._db.get_memory_sources(mid)
                if mem_sources:
                    # Sort by added_at descending to get most recent
                    mem_sources.sort(
                        key=lambda s: s.added_at.isoformat() if s.added_at else "",
                        reverse=True,
                    )
                    primary = mem_sources[0]
                    source_doc_id = primary.doc_id
                    source_type = primary.source_type

                    # Fetch document details
                    doc = await self._db.get_document(primary.doc_id)
                    if doc:
                        source_doc_title = doc.title
                        content_url = document_content_url(doc)
                        pdf_url = document_pdf_url(doc)
                        source_url = doc.source_url
            except Exception:
                logger.exception("Failed to fetch sources for memory %s", mid)

            # Determine freshness
            freshness = _compute_freshness(memory, source_url is not None)

            # Contradiction warning
            contradiction_warning = None
            if memory.contradiction_count > 0:
                contradiction_warning = (
                    f"This memory has {memory.contradiction_count} "
                    f"contradiction(s) recorded."
                )

            results.append(SearchResult(
                memory_id=memory.id,
                memory_type=memory.memory_type,
                summary=memory.content,
                confidence=memory.confidence,
                relevance_score=round(candidate.final_score, 4),
                tags=memory.tags,
                source_doc_id=source_doc_id,
                source_doc_title=source_doc_title,
                source_type=source_type,
                content_url=content_url,
                pdf_url=pdf_url,
                source_url=source_url,
                corroborated_by=memory.corroboration_count,
                last_observed_at=(
                    memory.updated_at.isoformat()
                    if memory.updated_at else None
                ),
                freshness=freshness,
                contradiction_warning=contradiction_warning,
                is_document_result=False,
            ))

        return results

    # ==================================================================
    # Document fallback
    # ==================================================================

    async def _document_fallback(
        self,
        query: str,
        remaining_slots: int,
        exclude_doc_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """Basic ChromaDB search on the documents collection.

        Returns ``SearchResult`` objects with ``is_document_result=True``.
        """
        if remaining_slots <= 0 or self._doc_collection is None:
            return []

        embedding = self._get_or_compute_embedding(query)
        if embedding is None:
            return []

        try:
            # Over-fetch so we can exclude docs already represented by memories
            fetch_n = remaining_slots + (len(exclude_doc_ids) if exclude_doc_ids else 0)
            chroma_result = self._doc_collection.query(
                query_embeddings=[embedding],
                n_results=min(fetch_n, 50),
            )
        except Exception:
            logger.exception("Document fallback ChromaDB query failed")
            return []

        if (
            not chroma_result
            or not chroma_result.get("ids")
            or not chroma_result["ids"][0]
        ):
            return []

        doc_ids = chroma_result["ids"][0]
        distances = (
            chroma_result["distances"][0]
            if chroma_result.get("distances")
            else [0.0] * len(doc_ids)
        )

        results: list[SearchResult] = []
        for doc_id, dist in zip(doc_ids, distances):
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue
            if len(results) >= remaining_slots:
                break

            similarity = max(1.0 - dist, 0.0)

            # Fetch document metadata
            title = doc_id
            content_url = None
            pdf_url = None
            source_url = None
            source_type = None

            try:
                doc = await self._db.get_document(doc_id)
                if doc:
                    title = doc.title
                    content_url = document_content_url(doc)
                    pdf_url = document_pdf_url(doc)
                    source_url = doc.source_url
                    source_type = doc.source
            except Exception:
                logger.exception("Failed to fetch document %s for fallback", doc_id)

            results.append(SearchResult(
                memory_id=None,
                memory_type=None,
                summary=f"[Document] {title}",
                confidence=0.0,
                relevance_score=round(similarity, 4),
                tags=[],
                source_doc_id=doc_id,
                source_doc_title=title,
                source_type=source_type,
                content_url=content_url,
                pdf_url=pdf_url,
                source_url=source_url,
                corroborated_by=0,
                last_observed_at=None,
                freshness="unverified",
                contradiction_warning=None,
                is_document_result=True,
            ))

        return results

    # ==================================================================
    # Query expansion
    # ==================================================================

    async def _expand_query_with_aliases(
        self,
        query: str,
        entity_ids: list[int],
    ) -> str:
        """Expand query with known aliases of detected entities for BM25.

        Format: ``"original query terms" ("alias1" OR "alias2" OR "canonical")``
        """
        if not entity_ids:
            return query

        alias_terms: list[str] = []
        for eid in entity_ids:
            try:
                aliases = await self._db.get_aliases_for_entity(eid)
                for a in aliases:
                    norm = a.alias_normalized.strip()
                    if norm and norm.lower() not in query.lower():
                        alias_terms.append(norm)
            except Exception:
                logger.exception("Failed to fetch aliases for entity %d", eid)

        if not alias_terms:
            return query

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in alias_terms:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique.append(t)

        or_clause = " OR ".join(f'"{t}"' for t in unique)
        return f"{query} ({or_clause})"

    # ==================================================================
    # Internal helpers
    # ==================================================================

    async def _build_known_entities(self) -> dict[str, int]:
        """Build a mapping of canonical_name + aliases -> entity_id."""
        entities: dict[str, int] = {}
        try:
            all_entities = await self._db.get_all_entities()
            for ent in all_entities:
                entities[ent.canonical_name] = ent.id
            # Layer in aliases — canonical names take precedence on collision
            all_aliases = await self._db.get_all_aliases()
            for alias_name, canonical_id in all_aliases:
                if alias_name not in entities:
                    entities[alias_name] = canonical_id
        except Exception:
            logger.exception("Failed to build known entities map")
        return entities

    def _get_or_compute_embedding(self, text: str) -> list[float] | None:
        """Return cached embedding or compute via the embedding API.

        Returns ``None`` if the embedding call fails (caller should degrade
        gracefully).
        """
        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached

        try:
            vectors = embed_texts(
                [text],
                base_url=self._embed_cfg.get("base_url", ""),
                api_key=self._embed_cfg.get("api_key", ""),
                model=self._embed_cfg.get("model", ""),
            )
            if vectors:
                self._embed_cache.put(text, vectors[0])
                return vectors[0]
        except Exception:
            logger.exception("Embedding computation failed for query")
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _compute_freshness(memory: Memory, has_source: bool) -> str:
    """Determine the freshness label for a memory.

    - ``current`` : memory updated within the last 7 days, or source accessible.
    - ``stale``   : memory older than 7 days and the source doc was updated
                    more recently (detected via updated_at lag).
    - ``unverified``: no known source or source inaccessible.
    """
    if not has_source:
        return "unverified"

    if memory.status in ("retired", "decayed"):
        return "stale"

    if memory.valid_until:
        now = datetime.now(timezone.utc)
        vu = memory.valid_until
        if vu.tzinfo is None:
            vu = vu.replace(tzinfo=timezone.utc)
        if now > vu:
            return "stale"

    return "current"
