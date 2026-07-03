"""Hybrid search engine for MemForge.

Three retrieval channels run in parallel (vector, BM25/FTS5, entity-graph),
fused via Reciprocal Rank Fusion (RRF), then strictly filtered by source/time
facets and ranked with a recency-weighted final score. Optional cross-encoder
reranking via LLM can be enabled for higher precision at scale.

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
from memforge.models import Memory, SHARED_PROJECT_KEY, SearchResult
from memforge.retrieval.embeddings import EmbeddingCache, embed_texts
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import (
    EntityLinkCandidate,
    KeywordCandidate,
    KeywordSearch,
    RelationalStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CROSS_PROJECT_PENALTY",
    "SearchEngine",
    "W_RECENCY_DEFAULT",
    "W_RRF_DEFAULT",
]


# Ranking weights for the final score. Queries lean on fused recall with a
# small recency contribution.
W_RRF_DEFAULT = 0.85
W_RECENCY_DEFAULT = 0.15

# Cross-project affinity penalty subtracted in `project-first` mode for any
# candidate that is neither the active project nor SHARED. Applied after RRF
# normalization and clamped at zero so a penalized candidate cannot go negative.
CROSS_PROJECT_PENALTY = 0.20
REPO_AFFINITY_BOOST = 0.05
CURATION_CHILD_EXACT_MATCH_MARGIN = 0.15


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
    project_key: str | None = None
    repo_identifier: str | None = None
    memory_level: str | None = None
    curation_cluster_id: str | None = None
    covered_memory_count: int = 0
    retrieval_evidence: dict[str, Any] | None = None


def _affinity_penalty(project_key: str | None, scope: AccessScope) -> float:
    """Cross-project penalty applied after RRF normalization.

    Returns 0.0 when:
      - scope_mode is "workspace" (no project narrowing)
      - the caller did not declare an active_project (legacy callers and
        the per-id readers have no frame of reference, so every project
        is treated equally and existing flat ranking is preserved)
      - project_key is SHARED (the team-wide bucket)
      - project_key equals scope.active_project (the caller's frame)

    Returns CROSS_PROJECT_PENALTY for every other key including UNSORTED,
    so unmapped knowledge degrades like any cross-project hit.
    """
    if scope.scope_mode == "workspace":
        return 0.0
    if scope.active_project is None:
        return 0.0
    if project_key == SHARED_PROJECT_KEY or project_key == scope.active_project:
        return 0.0
    return CROSS_PROJECT_PENALTY


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


def _search_follow_up_for_memory(
    memory: Memory,
    *,
    contradiction_warning: str | None,
) -> dict[str, str] | None:
    """Return a small next-tool hint when summary-only use is likely weak."""
    if contradiction_warning:
        return {
            "suggested_tool": "get_memory",
            "reason": "result_has_contradiction_warning",
        }
    if memory.status != "active":
        return {
            "suggested_tool": "get_memory",
            "reason": "memory_lifecycle_may_matter",
        }
    if memory.memory_type == "procedure":
        return {
            "suggested_tool": "get_memory",
            "reason": "summary_may_omit_operational_steps",
        }
    if memory.memory_type in {"decision", "convention"}:
        return {
            "suggested_tool": "get_memory",
            "reason": "provenance_or_lifecycle_may_matter",
        }
    return None


def _sanitize_fts_query(text: str) -> str:
    """Escape characters that are special in FTS5 MATCH syntax.

    FTS5 interprets ``*``, ``^``, ``(``, ``)``, ``:``, ``"`` as operators.
    Each whitespace-separated token is stripped of punctuation and re-quoted
    as an FTS5 phrase, so the result is always a flat AND of literal phrases.

    This sanitizer is for USER input only. Engine-built FTS5 fragments
    (parenthesized OR groups, quoted phrases produced by the alias expander,
    etc.) MUST NOT be passed through this function: it is structure-blind and
    will demote operators to literal tokens, destroying the query.
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


def _entity_link_candidate_payload(candidate: EntityLinkCandidate) -> dict[str, Any]:
    return {
        "entity_id": candidate.entity_id,
        "canonical_name": candidate.canonical_name,
        "matched_alias": candidate.matched_alias,
        "channel": candidate.channel,
        "contributing_channels": list(candidate.contributing_channels),
        "score": candidate.score,
        "matched_text": candidate.matched_text,
        "activates_graph": candidate.activates_graph,
    }


def _default_access_scope(include_superseded: bool) -> AccessScope:
    """The permissive single-datastore scope: real lifecycle filtering, no
    access narrowing. Carries only the status set the request asked for."""
    return AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        include_private=False,
        allowed_statuses=allowed_search_statuses(include_superseded),
        active_project=None,
        scope_mode="project-first",
    )


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------

class SearchEngine:
    """Hybrid retrieval engine: vector + BM25 + graph, fused via RRF.

    Bound to the storage adapters, never to a database connection or a Chroma
    collection directly. Per-request visibility rides on the ``AccessScope``
    each channel builds; the engine instance carries no caller identity.

    Parameters
    ----------
    relational : RelationalStore
        Source-of-truth rows plus the scoped graph, source/date, and ranking
        reads.
    keyword : KeywordSearch
        The BM25/FTS5 channel.
    vector : VectorStore
        The embedding channel; owns the distance-to-score conversion.
    embed_cfg : dict
        Keys ``base_url``, ``api_key``, ``model`` forwarded to
        :func:`embed_texts` for query embeddings.
    config : RetrievalConfig
        Tuning knobs (``rrf_k``, ``recency_half_life_days``, etc.).
    structured_llm_client : Any | None
        Optional cross-encoder reranking client.
    """

    def __init__(
        self,
        relational: RelationalStore,
        keyword: KeywordSearch,
        vector: VectorStore,
        embed_cfg: dict,
        config: RetrievalConfig,
        structured_llm_client: Any | None = None,
    ) -> None:
        self._relational = relational
        self._keyword = keyword
        self._vector = vector
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
        time_range: MemoryTimeRange | None = None,
        entities: list[str] | None = None,
        include_superseded: bool = False,
        top_k: int = 10,
        *,
        source_filter: MemorySourceFilter | None = None,
        request_scope: AccessScope | None = None,
        offset: int = 0,
    ) -> dict:
        """Unified search: memories (primary) + documents (fallback).

        The keyword-only ``request_scope`` carries the per-request access
        predicate (caller identity, scope mode, and the private-branch
        toggle). Existing positional callers see the permissive single-
        datastore default; surfaces that build a real scope (the admin API,
        the agent-hook channel) opt in by passing one.

        Returns
        -------
        dict
            ``query_analysis`` : summary of the analysis step.
            ``results``        : list of :class:`SearchResult`.
            ``total_candidates``: number of unique memories seen across all channels.
            ``retrieval_time_ms``: wall-clock milliseconds for the entire search.
        """
        t0 = time.monotonic()
        scope = request_scope or _default_access_scope(include_superseded)
        combined_source_filter = source_filter

        if not query.strip():
            if (
                (combined_source_filter is None or combined_source_filter.is_empty())
                and (time_range is None or time_range.is_empty())
            ):
                raise ValueError("queryless search requires source_filter or time_range")
            ids, total_candidates = await self._relational.list_ids_by_source_and_time(
                combined_source_filter,
                time_range,
                scope,
                limit=top_k,
                offset=offset,
            )
            ranked = [
                _RankedCandidate(memory_id=memory_id, rrf_score=float(top_k - index), final_score=float(top_k - index))
                for index, memory_id in enumerate(ids)
            ]
            results = await self._enrich_results(ids, ranked)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return {
                "query_analysis": {
                    "detected_entities": [],
                    "entity_linking": [],
                    "entity_linking_channels": [],
                    "unmatched_explicit_entities": [],
                    "strategies_used": ["source_time_listing"],
                },
                "results": results,
                "total_candidates": total_candidates,
                "total_count": total_candidates,
                "limit": top_k,
                "offset": offset,
                "retrieval_time_ms": elapsed_ms,
            }

        # ----- 1. Link query entities through the scoped relational channel -----
        analysis = QueryAnalysis()
        try:
            link_result = await self._relational.link_query_entities(
                query,
                scope=scope,
                explicit_entities=entities or (),
                source_filter=combined_source_filter,
                memory_types=memory_types,
            )
            analysis.entity_linking = list(link_result.candidates)
            analysis.entity_linking_channels = tuple(
                dict.fromkeys(
                    channel
                    for candidate in link_result.candidates
                    for channel in candidate.contributing_channels
                )
            )
            analysis.unmatched_explicit_entities = link_result.unmatched_explicit_entities
            graph_candidates = [
                candidate for candidate in link_result.candidates if candidate.activates_graph
            ]
            analysis.detected_entities = [candidate.canonical_name for candidate in graph_candidates]
            analysis.detected_entity_ids = [candidate.entity_id for candidate in graph_candidates]
            analysis.use_graph = bool(graph_candidates)
        except Exception:
            logger.exception("Query-time entity linking failed; continuing without graph channel")

        # ----- 2. Run active channels in parallel -----
        fetch_k = max(top_k * 2, top_k + offset)
        tasks: list[asyncio.Task] = []
        channel_names: list[str] = []

        # Vector search is always on
        tasks.append(asyncio.ensure_future(
            self._vector_search(query, memory_types, scope, fetch_k)
        ))
        channel_names.append("vector")

        # BM25 is always on
        tasks.append(asyncio.ensure_future(
            self._bm25_search(query, analysis, memory_types, scope, fetch_k)
        ))
        channel_names.append("bm25")

        tasks.append(asyncio.ensure_future(
            self._bm25_metadata_search(query, memory_types, scope, fetch_k)
        ))
        channel_names.append("bm25_metadata_tokens")

        # Graph traversal — only when entities detected
        if analysis.use_graph and analysis.detected_entity_ids:
            tasks.append(asyncio.ensure_future(
                self._graph_search(analysis.detected_entity_ids, memory_types, scope, fetch_k)
            ))
            channel_names.append("graph")

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect channel results, logging any errors
        channel_results: list[list[tuple[str, float]]] = []
        metadata_evidence: dict[str, dict[str, Any]] = {}
        for name, result in zip(channel_names, raw_results):
            if isinstance(result, BaseException):
                logger.error("Channel %s failed: %s", name, result, exc_info=result)
                channel_results.append([])
            elif name == "bm25_metadata_tokens":
                hits = list(result)
                channel_results.append([(hit.memory_id, hit.score) for hit in hits])
                metadata_evidence.update(
                    {
                        hit.memory_id: _metadata_keyword_evidence(hit)
                        for hit in hits
                    }
                )
            else:
                channel_results.append(result)

        # ----- 5. Fuse via RRF, then apply the source-of-truth re-checks -----
        fused = self._rrf_fusion(channel_results, k=self._config.rrf_k)
        self._attach_retrieval_evidence(fused, metadata_evidence)
        fused = await self._filter_candidates_by_status(fused, scope)
        if (
            (combined_source_filter is not None and not combined_source_filter.is_empty())
            or (time_range is not None and not time_range.is_empty())
        ):
            supported = await self._relational.filter_ids_by_source_and_time(
                [c.memory_id for c in fused],
                combined_source_filter,
                time_range,
            )
            fused = [c for c in fused if c.memory_id in supported]
        total_candidates = len(fused)

        # ----- 6. Apply ranking -----
        ranked = await self._apply_ranking(fused, scope=scope)

        # ----- 6b. Optional cross-encoder rerank -----
        ranked = await self._rerank_with_llm(query, ranked, top_k)

        # ----- 7. Collapse duplicate families and apply the requested page -----
        ranked = self._collapse_curation_families(ranked)
        ranked = ranked[offset: offset + top_k]

        # ----- 8. Enrich results -----
        memory_ids = [c.memory_id for c in ranked]
        results = await self._enrich_results(memory_ids, ranked)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return {
            "query_analysis": {
                "detected_entities": analysis.detected_entities,
                "entity_linking": [
                    _entity_link_candidate_payload(candidate)
                    for candidate in analysis.entity_linking
                ],
                "entity_linking_channels": list(analysis.entity_linking_channels),
                "unmatched_explicit_entities": list(analysis.unmatched_explicit_entities),
                "strategies_used": channel_names,
            },
            "results": results,
            "total_candidates": total_candidates,
            "limit": top_k,
            "offset": offset,
            "retrieval_time_ms": elapsed_ms,
        }

    # ==================================================================
    # Channel implementations
    # ==================================================================

    async def _vector_search(
        self,
        query: str,
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Embed the query via cache, then query the vector channel.

        Source filtering is authoritative on the fused candidate set because a
        memory can have many source rows and vector stores cannot express that
        provenance predicate.
        """
        embedding = self._get_or_compute_embedding(query)
        if embedding is None:
            return []
        return await self._vector.query(embedding, scope, memory_types, limit)

    async def _bm25_search(
        self,
        query: str,
        analysis: QueryAnalysis,
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Query the keyword channel with optional alias expansion.

        Source filtering is applied once on the fused set (Step 8), not per
        channel.
        """
        sanitized_query = _sanitize_fts_query(query)
        if not sanitized_query:
            return []
        alias_clause = await self._build_alias_clause(
            analysis.detected_entity_ids, query
        )
        # When aliases contribute new terms, broaden recall by ORing them
        # against the user's phrase list. The user side is wrapped in parens
        # so its implicit AND binds tighter than the top-level OR; without
        # the parens FTS5 would attach OR only to the last user phrase.
        fts_query = (
            sanitized_query
            if not alias_clause
            else f"({sanitized_query}) OR {alias_clause}"
        )
        return await self._keyword.search(fts_query, scope, memory_types, limit)

    async def _bm25_metadata_search(
        self,
        query: str,
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
    ) -> list[KeywordCandidate]:
        """Query the source-metadata keyword channel."""
        sanitized_query = _sanitize_fts_query(query)
        if not sanitized_query:
            return []
        return await self._keyword.search_metadata(
            sanitized_query,
            scope,
            memory_types,
            limit,
        )

    async def _graph_search(
        self,
        entity_ids: list[int],
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Entity-graph traversal via the relational channel."""
        return await self._relational.graph_search(
            entity_ids, scope, memory_types, limit
        )

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

    @staticmethod
    def _attach_retrieval_evidence(
        candidates: list[_RankedCandidate],
        metadata_evidence: dict[str, dict[str, Any]],
    ) -> None:
        for candidate in candidates:
            evidence = metadata_evidence.get(candidate.memory_id)
            if evidence:
                candidate.retrieval_evidence = {"metadata_lexical": evidence}

    async def _filter_candidates_by_status(
        self,
        candidates: list[_RankedCandidate],
        scope: AccessScope,
    ) -> list[_RankedCandidate]:
        """Apply the source-of-truth visibility re-check after channel fusion."""
        if not candidates:
            return []
        visible = await self._relational.filter_visible_ids(
            [c.memory_id for c in candidates], scope
        )
        return [c for c in candidates if c.memory_id in visible]

    # ==================================================================
    # Ranking
    # ==================================================================

    async def _apply_ranking(
        self,
        candidates: list[_RankedCandidate],
        *,
        scope: AccessScope,
    ) -> list[_RankedCandidate]:
        """Apply recency-weighted final ranking with the cross-project penalty.

        ``final_score = max(0, w_rrf * rrf_normalized + w_recency * recency - penalty)``

        ``W_RRF_DEFAULT`` and ``W_RECENCY_DEFAULT`` are constant regardless of
        whether an explicit date filter was used; date filters only narrow the
        candidate set.
        """
        if not candidates:
            return candidates

        # Single relational read fetches both ranking inputs (updated_at and
        # project_key) so the per-channel ranker never needs a second roundtrip.
        id_to_meta = await self._relational.fetch_ranking_metadata(
            [c.memory_id for c in candidates]
        )

        # Normalize RRF scores to [0, 1]
        max_rrf = max(c.rrf_score for c in candidates) if candidates else 1.0
        if max_rrf == 0:
            max_rrf = 1.0

        half_life = float(self._config.recency_half_life_days)
        w_rrf = W_RRF_DEFAULT
        w_rec = W_RECENCY_DEFAULT

        for c in candidates:
            meta = id_to_meta.get(c.memory_id, {})
            c.updated_at = meta.get("updated_at")
            c.project_key = meta.get("project_key")
            c.repo_identifier = meta.get("repo_identifier")
            c.memory_level = meta.get("memory_level")
            c.curation_cluster_id = meta.get("curation_cluster_id")
            c.covered_memory_count = int(meta.get("covered_memory_count") or 0)
            rrf_norm = c.rrf_score / max_rrf
            age = _age_days(c.updated_at)
            recency = _recency_score(age, half_life)
            penalty = _affinity_penalty(c.project_key, scope)
            repo_boost = (
                REPO_AFFINITY_BOOST
                if scope.active_repo_identifier
                and c.repo_identifier == scope.active_repo_identifier
                else 0.0
            )
            c.final_score = max(
                0.0,
                w_rrf * rrf_norm + w_rec * recency - penalty + repo_boost,
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    @staticmethod
    def _collapse_curation_families(
        candidates: list[_RankedCandidate],
    ) -> list[_RankedCandidate]:
        """Collapse near-duplicate curated families for default search output.

        A consolidated memory represents the cluster by default. A strongly
        higher-scoring atomic child can still surface for exact-error or
        issue-specific queries, where the child likely carries details the
        summary intentionally compressed.
        """
        by_cluster: dict[str, list[_RankedCandidate]] = {}
        unclustered: list[_RankedCandidate] = []
        for candidate in candidates:
            if candidate.curation_cluster_id:
                by_cluster.setdefault(candidate.curation_cluster_id, []).append(candidate)
            else:
                unclustered.append(candidate)

        selected: list[_RankedCandidate] = list(unclustered)
        for members in by_cluster.values():
            consolidated = [
                member for member in members
                if member.memory_level == "consolidated"
            ]
            if not consolidated:
                selected.extend(members)
                continue

            summary = max(consolidated, key=lambda item: item.final_score)
            top = max(members, key=lambda item: item.final_score)
            if (
                top.memory_id != summary.memory_id
                and top.final_score > summary.final_score + CURATION_CHILD_EXACT_MATCH_MARGIN
            ):
                selected.extend([top, summary])
            else:
                selected.append(summary)

        selected.sort(key=lambda item: item.final_score, reverse=True)
        return selected

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
                mem = await self._relational.get_memory(c.memory_id)
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
                numbered.append(_rerank_memory_card(i, content, c.retrieval_evidence))
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
        """Fetch full Memory objects for each result."""
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
                memory = await self._relational.get_memory(mid)
            except Exception:
                logger.exception("Failed to fetch memory %s", mid)
                continue
            if memory is None:
                continue

            try:
                mem_sources = await self._relational.get_memory_sources(mid)
                has_source = bool(mem_sources)
            except Exception:
                logger.exception("Failed to fetch sources for memory %s", mid)
                has_source = False

            # Determine freshness
            freshness = _compute_freshness(memory, has_source)

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
                corroborated_by=memory.corroboration_count,
                last_observed_at=(
                    memory.updated_at.isoformat()
                    if memory.updated_at else None
                ),
                freshness=freshness,
                contradiction_warning=contradiction_warning,
                status=memory.status,
                memory_level=candidate.memory_level or memory.memory_level,
                curation_cluster_id=(
                    candidate.curation_cluster_id or memory.curation_cluster_id
                ),
                covered_memory_count=candidate.covered_memory_count,
                repo_identifier=candidate.repo_identifier or memory.repo_identifier,
                follow_up=_search_follow_up_for_memory(
                    memory,
                    contradiction_warning=contradiction_warning,
                ),
                retrieval_evidence=candidate.retrieval_evidence,
            ))

        return results

    # ==================================================================
    # Query expansion
    # ==================================================================

    async def _build_alias_clause(
        self,
        entity_ids: list[int],
        user_query: str,
    ) -> str:
        """Build an FTS5-valid alias OR group for the detected entities.

        Returns a string of the form
        ``("alias1" OR "alias2" OR "canonical")`` ready to be appended next
        to a separately-sanitized user query, or ``""`` when no aliases
        contribute new terms. Aliases that already appear in ``user_query``
        (case-insensitive substring) are skipped, since they would only
        restate what the user typed.

        The returned fragment is already FTS5-valid: parens, the ``OR``
        operator, and double-quoted phrases are intentional and load-bearing.
        It MUST NOT be passed through :func:`_sanitize_fts_query`, which is
        structure-blind and would demote ``OR`` to a literal token and strip
        the grouping parens.
        """
        if not entity_ids:
            return ""

        alias_terms: list[str] = []
        for eid in entity_ids:
            try:
                aliases = await self._relational.get_aliases_for_entity(eid)
                for a in aliases:
                    norm = a.alias_normalized.strip()
                    if norm and norm.lower() not in user_query.lower():
                        alias_terms.append(norm)
            except Exception:
                logger.exception("Failed to fetch aliases for entity %d", eid)

        if not alias_terms:
            return ""

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in alias_terms:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique.append(t)

        or_clause = " OR ".join(f'"{t}"' for t in unique)
        return f"({or_clause})"

    # ==================================================================
    # Internal helpers
    # ==================================================================

    async def _build_known_entities(self) -> dict[str, int]:
        """Build a mapping of canonical_name + aliases -> entity_id."""
        entities: dict[str, int] = {}
        try:
            all_entities = await self._relational.get_all_entities()
            for ent in all_entities:
                entities[ent.canonical_name] = ent.id
            # Layer in aliases — canonical names take precedence on collision
            all_aliases = await self._relational.get_all_aliases()
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

def _metadata_keyword_evidence(hit: KeywordCandidate) -> dict[str, Any]:
    return {
        "channel": hit.channel,
        "matched_fields": list(hit.matched_fields),
        "matched_text": list(hit.matched_text),
        "source_refs": [
            {
                "source_id": ref.source_id,
                "doc_id": ref.doc_id,
                "source_type": ref.source_type,
            }
            for ref in hit.source_refs
        ],
    }


def _rerank_memory_card(
    index: int,
    content: str,
    retrieval_evidence: dict[str, Any] | None,
) -> str:
    card = f"{index}. Memory: {content}"
    metadata_evidence = (retrieval_evidence or {}).get("metadata_lexical")
    if not metadata_evidence:
        return card
    evidence_lines = []
    matched_texts = metadata_evidence.get("matched_text") or []
    if matched_texts:
        evidence_lines.append("metadata: " + " || ".join(str(text) for text in matched_texts[:3]))
    source_refs = metadata_evidence.get("source_refs") or []
    if source_refs:
        refs = [
            f"{ref.get('source_type')}:{ref.get('doc_id')}"
            for ref in source_refs[:3]
            if isinstance(ref, dict)
        ]
        if refs:
            evidence_lines.append("source_refs: " + ", ".join(refs))
    if evidence_lines:
        return card + "\n   Retrieval evidence: " + " ; ".join(evidence_lines)
    return card


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
        if datetime.now(timezone.utc).date() > memory.valid_until:
            return "stale"

    return "current"
