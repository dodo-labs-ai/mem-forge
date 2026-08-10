"""Hybrid search engine for MemForge.

Vector, content BM25/FTS, metadata lexical, and entity-graph channels run in
parallel, then fuse through intent-specific weighted RRF before the final ranking
and source/time page slicing.

Architecture reference: docs/architecture.md Section 10.
"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Any, Callable

from memforge.config import DEFAULT_RANK_WINDOW_SIZE, DEFAULT_RRF_K, RetrievalConfig
from memforge.llm.structured import StructuredLlmError
from memforge.memory.lifecycle import allowed_search_statuses
from memforge.models import Memory, SHARED_PROJECT_KEY, SearchResult
from memforge.retrieval.embeddings import EmbeddingCache, embed_texts
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.intents import (
    RankedRetrievalIntent,
    fusion_weights,
    resolve_retrieval_intent,
    validate_requested_intent,
)
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.retrieval.query_plan import build_lexical_query_plan
from memforge.retrieval.rank_fusion import (
    RankedChannelItem,
    weighted_reciprocal_rank_fusion,
)
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import (
    EntityLinkCandidate,
    KeywordCandidate,
    KeywordSearch,
    RecentMemoryItem,
    RelationalStore,
    VectorStore,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CROSS_PROJECT_PENALTY",
    "SearchEngine",
    "W_RECENCY_DEFAULT",
    "W_RRF_DEFAULT",
    "sanitize_fts_query",
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
_METADATA_SUBCHANNEL_WEIGHTS = {
    "bm25_metadata_tokens": 0.60,
    "metadata_alias": 0.30,
    "metadata_trigram": 0.10,
}
_METADATA_IDENTIFIER_CHANNELS = {"bm25_metadata_tokens", "metadata_alias"}

_EXTERNAL_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b|#\d+\b|\bINC\d+\b", re.IGNORECASE)
_CODE_SYMBOL_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z0-9]*[./][A-Za-z0-9_./-]+\b)"
    r"|(?:\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b)"
    r"|(?:\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b)"
)
_QUOTED_IDENTITY_RE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
_MAX_QUOTED_IDENTITY_LENGTH = 256


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
    retrieval_evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class _QueryFeatures:
    has_external_id: bool
    has_code_symbol: bool


@dataclass(frozen=True)
class _GraphContribution:
    memory_id: str
    rank: int
    multiplier: float
    entity_id: int


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


def _compute_query_features(
    query: str,
) -> _QueryFeatures:
    return _QueryFeatures(
        has_external_id=bool(_EXTERNAL_ID_RE.search(query)),
        has_code_symbol=bool(_CODE_SYMBOL_RE.search(query)),
    )


def _has_strong_metadata_identity(
    query: str,
    metadata_hits: list[KeywordCandidate],
) -> bool:
    """Require an exact metadata field, without language-specific query parsing."""

    normalized_query = _normalize_identity_text(query)
    if not normalized_query:
        return False
    return any(
        normalized_query == _normalize_identity_text(field)
        for hit in metadata_hits
        if hit.channel in _METADATA_IDENTIFIER_CHANNELS
        for matched_text in hit.matched_text
        for field in matched_text.split("|")
    )


def _normalize_identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def _metadata_lexical_channel(
    metadata_hits: list[KeywordCandidate],
    *,
    k: int,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for channel, weight in _METADATA_SUBCHANNEL_WEIGHTS.items():
        channel_hits = [
            hit for hit in metadata_hits
            if hit.channel == channel
        ]
        channel_hits.sort(key=lambda hit: hit.score, reverse=True)
        for rank_0, hit in enumerate(channel_hits):
            rank = rank_0 + 1
            scores[hit.memory_id] = scores.get(hit.memory_id, 0.0) + weight / (k + rank)
            best_rank[hit.memory_id] = min(best_rank.get(hit.memory_id, rank), rank)
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank.get(item[0], 10**9), item[0]),
    )


def _weighted_rrf_fusion(
    *,
    vector_results: list[tuple[str, float]],
    content_results: list[tuple[str, float]],
    metadata_results: list[tuple[str, float]],
    graph_contributions: dict[str, _GraphContribution],
    intent: RankedRetrievalIntent,
    k: int,
) -> list[_RankedCandidate]:
    weights = fusion_weights(intent)
    fused = weighted_reciprocal_rank_fusion(
        channels={
            "vector": tuple(
                RankedChannelItem(memory_id, score)
                for memory_id, score in vector_results
            ),
            "bm25_content": tuple(
                RankedChannelItem(memory_id, score)
                for memory_id, score in content_results
            ),
            "metadata_lexical": tuple(
                RankedChannelItem(memory_id, score)
                for memory_id, score in metadata_results
            ),
            "graph": tuple(
                RankedChannelItem(
                    memory_id,
                    score=0.0,
                    rank=contribution.rank,
                    multiplier=contribution.multiplier,
                )
                for memory_id, contribution in graph_contributions.items()
            ),
        },
        weights=weights,
        k=k,
    )
    return [
        _RankedCandidate(
            memory_id=item.item_id,
            rrf_score=item.score,
            retrieval_evidence={
                "rank_fusion": {
                    "intent": intent,
                    "rrf_score": item.score,
                    "contributions": [
                        asdict(contribution)
                        for contribution in item.contributions
                    ],
                }
            },
        )
        for item in fused
    ]


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


def sanitize_fts_query(text: str) -> str:
    """Escape characters that are special in FTS5 MATCH syntax.

    FTS5 interprets ``*``, ``^``, ``(``, ``)``, ``:``, ``"`` as operators.
    Each whitespace-separated token is stripped of punctuation and re-quoted
    as an FTS5 phrase, so the result is always a flat AND of literal phrases.

    This sanitizer is for unstructured plain text only. Engine-built FTS5 fragments
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


def _quoted_identity_query(query: str) -> str | None:
    """Return the first bounded identity that the caller explicitly quoted."""

    for match in _QUOTED_IDENTITY_RE.finditer(query):
        identity = " ".join(next(group for group in match.groups() if group).split())
        if identity and len(identity) <= _MAX_QUOTED_IDENTITY_LENGTH:
            return identity
    return None


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
        "visible_memory_count": candidate.visible_memory_count,
        "visible_source_count": candidate.visible_source_count,
        "specificity": candidate.specificity,
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
        embedding_provider: Callable[[str], list[float] | None] | None = None,
    ) -> None:
        self._relational = relational
        self._keyword = keyword
        self._vector = vector
        self._embed_cfg = embed_cfg
        self._config = config
        self._embed_cache = EmbeddingCache(max_size=config.embedding_cache_size)
        self._structured_llm_client = structured_llm_client
        self._embedding_provider = embedding_provider

    # ==================================================================
    # Public API
    # ==================================================================

    async def list_recent_memories(
        self,
        *,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange,
        memory_types: list[str] | None = None,
        page_size: int = 50,
        cursor: str | None = None,
        request_scope: AccessScope | None = None,
    ) -> dict[str, Any]:
        """List current Memories by exact predicates and a request-bound keyset.

        The cursor preserves a fixed UTC upper watermark and the last ordered
        key. It is not a database MVCC snapshot: concurrent or late-arriving
        source changes are intentionally observed by a new listing.
        """
        if time_range.after is None or time_range.before is None:
            raise ValueError("recent-memory listing requires start_at and end_at")
        if time_range.after >= time_range.before:
            raise ValueError("recent-memory listing requires start_at before end_at")
        if page_size < 1 or page_size > 50:
            raise ValueError("page_size must be from 1 to 50")

        scope = request_scope or _default_access_scope(False)
        if scope.allowed_statuses != ("active",):
            scope = replace(scope, allowed_statuses=("active",))
        normalized_filter = source_filter or MemorySourceFilter()
        normalized_types = tuple(dict.fromkeys(memory_types or ()))
        fingerprint = _recent_listing_fingerprint(
            source_filter=normalized_filter,
            time_range=time_range,
            memory_types=normalized_types,
            scope=scope,
        )

        if cursor is None:
            listing_watermark = datetime.now(timezone.utc)
            after = None
        else:
            listing_watermark, after = _decode_recent_listing_cursor(cursor, fingerprint=fingerprint)

        page = await self._relational.list_recent_memory_ids(
            normalized_filter,
            time_range,
            scope,
            memory_types=normalized_types,
            listing_watermark=listing_watermark,
            after=after,
            limit=page_size,
        )
        ids = [item.memory_id for item in page.items]
        ranked = [
            _RankedCandidate(
                memory_id=item.memory_id,
                final_score=float(page_size - index),
                updated_at=item.sort_at,
            )
            for index, item in enumerate(page.items)
        ]
        enriched_results = await self._enrich_results(ids, ranked, scope=scope)
        matched_at_by_memory_id = {
            item.memory_id: item.sort_at.astimezone(timezone.utc).isoformat() for item in page.items
        }
        results: list[dict[str, Any]] = []
        for result in enriched_results:
            payload = asdict(result)
            payload.pop("relevance_score", None)
            payload["matched_at"] = matched_at_by_memory_id[result.memory_id]
            results.append(payload)
        next_cursor = None
        if page.has_more and page.items:
            next_cursor = _encode_recent_listing_cursor(
                fingerprint=fingerprint,
                listing_watermark=listing_watermark,
                after=page.items[-1],
            )
        return {
            "results": results,
            "result_kind": "current_memories",
            "is_changelog": False,
            "time_field": time_range.date_type,
            "resolved_window": {
                "start_at": time_range.after.astimezone(timezone.utc).isoformat(),
                "end_at": time_range.before.astimezone(timezone.utc).isoformat(),
            },
            "listing_watermark": listing_watermark.isoformat(),
            "cursor_kind": "keyset",
            "consistency": "request_bound_watermark_not_mvcc_snapshot",
            "total_candidates": page.total_count,
            "candidate_count_kind": "exact",
            "count_scope": "current_page_read",
            "limit": page_size,
            "has_more": page.has_more,
            "next_cursor": next_cursor,
        }

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
        intent: RankedRetrievalIntent | None = None,
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
        requested_intent = validate_requested_intent(intent)
        t0 = time.monotonic()
        scope = request_scope or _default_access_scope(include_superseded)
        combined_source_filter = source_filter

        if not query.strip():
            if requested_intent is not None:
                raise ValueError("retrieval intent requires a ranked query")
            if (combined_source_filter is None or combined_source_filter.is_empty()) and (
                time_range is None or time_range.is_empty()
            ):
                raise ValueError("queryless search requires source_filter or time_range")
            ids, total_candidates = await self._relational.list_ids_by_source_and_time(
                combined_source_filter,
                time_range,
                scope,
                limit=top_k,
                offset=offset,
                memory_types=tuple(memory_types or ()),
            )
            ranked = [
                _RankedCandidate(memory_id=memory_id, rrf_score=float(top_k - index), final_score=float(top_k - index))
                for index, memory_id in enumerate(ids)
            ]
            results = await self._enrich_results(ids, ranked, scope=scope)
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
                "candidate_count_kind": "exact",
                "ranking_window_size": len(ids),
                "limit": top_k,
                "offset": offset,
                "has_more": offset + len(results) < total_candidates,
                "retrieval_time_ms": elapsed_ms,
            }

        # ----- 1. Link query entities through the scoped relational channel -----
        analysis = QueryAnalysis()
        link_result = await self._relational.link_query_entities(
            query,
            scope=scope,
            explicit_entities=entities or (),
            source_filter=combined_source_filter,
            time_range=time_range,
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

        # ----- 2. Run active channels in parallel -----
        configured_window = max(
            1,
            int(getattr(self._config, "rank_window_size", DEFAULT_RANK_WINDOW_SIZE)),
        )
        fetch_k = max(configured_window, top_k + offset)
        tasks: list[asyncio.Task] = []
        channel_names: list[str] = []

        # Vector search is always on
        tasks.append(asyncio.ensure_future(
            self._vector_search(query, memory_types, scope, fetch_k)
        ))
        channel_names.append("vector")

        # BM25 content is always on
        tasks.append(asyncio.ensure_future(
            self._bm25_search(query, analysis, memory_types, scope, fetch_k)
        ))
        channel_names.append("bm25_content")

        tasks.append(asyncio.ensure_future(
            self._metadata_searches(
                query,
                requested_intent,
                memory_types,
                scope,
                fetch_k,
                source_filter=combined_source_filter,
                time_range=time_range,
            )
        ))
        channel_names.append("bm25_metadata_tokens")

        # Graph traversal — one bounded request per linked entity so each
        # memory's graph rank can be tied to the specificity of the entity that
        # actually contributed it.
        graph_task_entities: list[EntityLinkCandidate] = []
        if analysis.use_graph and graph_candidates:
            tasks.append(asyncio.ensure_future(
                asyncio.gather(
                    *[
                        self._graph_search(
                            [candidate.entity_id],
                            memory_types,
                            scope,
                            fetch_k,
                            source_filter=combined_source_filter,
                            time_range=time_range,
                        )
                        for candidate in graph_candidates
                    ],
                    return_exceptions=True,
                )
            ))
            channel_names.append("graph")
            graph_task_entities = graph_candidates

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect channel results, logging any errors
        vector_results: list[tuple[str, float]] = []
        content_results: list[tuple[str, float]] = []
        metadata_hits: list[KeywordCandidate] = []
        graph_contributions: dict[str, _GraphContribution] = {}
        metadata_evidence: dict[str, dict[str, Any]] = {}
        for name, result in zip(channel_names, raw_results):
            if isinstance(result, BaseException):
                logger.error("Channel %s failed: %s", name, result, exc_info=result)
            elif name == "vector":
                vector_results = list(result)
            elif name == "bm25_content":
                content_results = list(result)
            elif name == "bm25_metadata_tokens":
                hits, query_paths = result
                hits = list(hits)
                metadata_hits = hits
                metadata_evidence.update(_best_metadata_evidence(hits))
                for memory_id, paths in query_paths.items():
                    evidence = metadata_evidence.get(memory_id)
                    if evidence is not None:
                        evidence["query_paths"] = list(paths)
            elif name == "graph":
                per_entity_results = list(result)
                graph_contributions = self._graph_contributions(
                    graph_task_entities,
                    per_entity_results,
                )
            else:
                logger.warning("Unknown retrieval channel result ignored: %s", name)

        features = _compute_query_features(query)
        intent_resolution = resolve_retrieval_intent(
            requested_intent,
            has_external_id=features.has_external_id,
            has_code_symbol=features.has_code_symbol,
            has_strong_metadata_identity=_has_strong_metadata_identity(query, metadata_hits),
            has_traversable_entity=bool(graph_candidates),
        )
        metadata_channel = _metadata_lexical_channel(
            metadata_hits,
            k=getattr(self._config, "rrf_k", DEFAULT_RRF_K),
        )

        # ----- 5. Fuse via intent-specific Weighted RRF, then apply source-of-truth re-checks -----
        fused = _weighted_rrf_fusion(
            vector_results=vector_results,
            content_results=content_results,
            metadata_results=metadata_channel,
            graph_contributions=graph_contributions,
            intent=intent_resolution.resolved_intent,
            k=getattr(self._config, "rrf_k", DEFAULT_RRF_K),
        )
        self._attach_retrieval_evidence(fused, metadata_evidence)
        fused = await self._filter_candidates_by_status(fused, scope)
        if (combined_source_filter is not None and not combined_source_filter.is_empty()) or (
            time_range is not None and not time_range.is_empty()
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
        ranked_count = len(ranked)
        ranked = ranked[offset : offset + top_k]

        # ----- 8. Enrich results -----
        memory_ids = [c.memory_id for c in ranked]
        results = await self._enrich_results(memory_ids, ranked, scope=scope)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return {
            "query_analysis": {
                "detected_entities": analysis.detected_entities,
                "entity_linking": [_entity_link_candidate_payload(candidate) for candidate in analysis.entity_linking],
                "entity_linking_channels": list(analysis.entity_linking_channels),
                "unmatched_explicit_entities": list(analysis.unmatched_explicit_entities),
                "strategies_used": channel_names,
            },
            "retrieval_intent": intent_resolution.to_dict(),
            "results": results,
            "total_candidates": total_candidates,
            "candidate_count_kind": "windowed",
            "ranking_window_size": fetch_k,
            "limit": top_k,
            "offset": offset,
            "has_more": offset + len(results) < ranked_count,
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
        sanitized_query = sanitize_fts_query(query)
        if not sanitized_query:
            return []
        alias_clause = await self._build_alias_clause(analysis.detected_entity_ids, query)
        # When aliases contribute new terms, broaden recall by ORing them
        # against the user's phrase list. The user side is wrapped in parens
        # so its implicit AND binds tighter than the top-level OR; without
        # the parens FTS5 would attach OR only to the last user phrase.
        fts_query = sanitized_query if not alias_clause else f"({sanitized_query}) OR {alias_clause}"
        return await self._keyword.search(fts_query, scope, memory_types, limit)

    async def _bm25_metadata_search(
        self,
        query: str,
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
    ) -> list[KeywordCandidate]:
        """Query the source-metadata keyword channel."""
        query_plan = build_lexical_query_plan(query).metadata
        if not query_plan.ordinary_terms and not query_plan.exact_anchors:
            return []
        return await self._keyword.search_metadata(
            query_plan,
            scope,
            memory_types,
            limit,
            source_filter=source_filter,
            time_range=time_range,
            include_subchannel_hits=True,
        )

    async def _metadata_searches(
        self,
        query: str,
        requested_intent: RankedRetrievalIntent | None,
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
    ) -> tuple[list[KeywordCandidate], dict[str, tuple[str, ...]]]:
        """Run the full metadata query plus one explicitly quoted identity."""

        queries = [("full_query", query)]
        quoted_identity = _quoted_identity_query(query) if requested_intent == "known_item" else None
        if quoted_identity is not None and quoted_identity != query.strip():
            queries.append(("quoted_identity", quoted_identity))

        result_sets = await asyncio.gather(
            *(
                self._bm25_metadata_search(
                    metadata_query,
                    memory_types,
                    scope,
                    limit,
                    source_filter=source_filter,
                    time_range=time_range,
                )
                for _, metadata_query in queries
            )
        )
        best_hits: dict[tuple[str, str], KeywordCandidate] = {}
        best_hit_paths: dict[tuple[str, str], list[str]] = {}
        for (query_path, _), hits in zip(queries, result_sets):
            for hit in hits:
                key = (hit.memory_id, hit.channel)
                previous = best_hits.get(key)
                if previous is None or hit.score > previous.score:
                    best_hits[key] = hit
                    best_hit_paths[key] = [query_path]
                elif hit.score == previous.score:
                    paths = best_hit_paths[key]
                    if query_path not in paths:
                        paths.append(query_path)
        query_paths: dict[str, list[str]] = {}
        for (memory_id, channel), paths in best_hit_paths.items():
            if (memory_id, channel) not in best_hits:
                continue
            combined = query_paths.setdefault(memory_id, [])
            for query_path in paths:
                if query_path not in combined:
                    combined.append(query_path)
        return list(best_hits.values()), {
            memory_id: tuple(paths)
            for memory_id, paths in query_paths.items()
        }

    async def _graph_search(
        self,
        entity_ids: list[int],
        memory_types: list[str] | None,
        scope: AccessScope,
        limit: int,
        *,
        source_filter: MemorySourceFilter | None = None,
        time_range: MemoryTimeRange | None = None,
    ) -> list[tuple[str, float]]:
        """Entity-graph traversal via the relational channel."""
        return await self._relational.graph_search(
            entity_ids,
            scope,
            memory_types,
            limit,
            source_filter=source_filter,
            time_range=time_range,
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
        channels = {
            f"channel_{index}": tuple(
                RankedChannelItem(memory_id, score)
                for memory_id, score in channel
            )
            for index, channel in enumerate(channel_results)
        }
        fused = weighted_reciprocal_rank_fusion(
            channels=channels,
            weights={channel: 1.0 for channel in channels},
            k=k,
        )
        return [
            _RankedCandidate(memory_id=item.item_id, rrf_score=item.score)
            for item in fused
        ]

    @staticmethod
    def _graph_contributions(
        graph_candidates: list[EntityLinkCandidate],
        per_entity_results: list[Any],
    ) -> dict[str, _GraphContribution]:
        contributions: dict[str, _GraphContribution] = {}
        for candidate, result in zip(graph_candidates, per_entity_results):
            if isinstance(result, BaseException):
                logger.error("Graph channel failed for entity %s: %s", candidate.entity_id, result, exc_info=result)
                continue
            multiplier = max(0.0, min(1.0, float(candidate.specificity or 0.0)))
            sorted_results = sorted(list(result), key=lambda item: item[1], reverse=True)
            for rank_0, (memory_id, _score) in enumerate(sorted_results):
                rank = rank_0 + 1
                existing = contributions.get(memory_id)
                if existing is None or (
                    multiplier,
                    -rank,
                    -candidate.entity_id,
                ) > (
                    existing.multiplier,
                    -existing.rank,
                    -existing.entity_id,
                ):
                    contributions[memory_id] = _GraphContribution(
                        memory_id=memory_id,
                        rank=rank,
                        multiplier=multiplier,
                        entity_id=candidate.entity_id,
                    )
        return contributions

    @staticmethod
    def _attach_retrieval_evidence(
        candidates: list[_RankedCandidate],
        metadata_evidence: dict[str, dict[str, Any]],
    ) -> None:
        for candidate in candidates:
            evidence = metadata_evidence.get(candidate.memory_id)
            if evidence:
                candidate.retrieval_evidence = {
                    **(candidate.retrieval_evidence or {}),
                    "metadata_lexical": evidence,
                }

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
        *,
        scope: AccessScope,
    ) -> list[SearchResult]:
        """Fetch full Memory objects for each result."""
        if not memory_ids:
            return []

        # Build a lookup of candidate scores
        score_map = {c.memory_id: c for c in ranked}
        conflict_contexts_by_memory = await self.list_memory_conflict_contexts(
            memory_ids,
            scope=scope,
        )

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
            conflict_contexts = conflict_contexts_by_memory.get(mid, ())
            confirmed_conflicts = sum(item.disposition == "confirmed" for item in conflict_contexts)
            pending_conflicts = sum(item.disposition == "pending" for item in conflict_contexts)
            contradiction_warning = None
            if confirmed_conflicts:
                contradiction_warning = (
                    f"This memory has {confirmed_conflicts} reviewed cross-source conflict(s). See conflict_contexts."
                )
            elif pending_conflicts:
                contradiction_warning = (
                    f"This memory has {pending_conflicts} pending cross-source "
                    f"conflict review(s). See conflict_contexts."
                )
            elif memory.contradiction_count > 0:
                contradiction_warning = f"This memory has {memory.contradiction_count} contradiction(s) recorded."

            results.append(
                SearchResult(
                    memory_id=memory.id,
                    memory_type=memory.memory_type,
                    summary=memory.content,
                    confidence=memory.confidence,
                    relevance_score=round(candidate.final_score, 4),
                    corroborated_by=memory.corroboration_count,
                    last_observed_at=(memory.updated_at.isoformat() if memory.updated_at else None),
                    freshness=freshness,
                    contradiction_warning=contradiction_warning,
                    conflict_contexts=conflict_contexts,
                    status=memory.status,
                    repo_identifier=candidate.repo_identifier or memory.repo_identifier,
                    follow_up=_search_follow_up_for_memory(
                        memory,
                        contradiction_warning=contradiction_warning,
                    ),
                    retrieval_evidence=candidate.retrieval_evidence,
                )
            )

        return results

    async def list_memory_conflict_contexts(
        self,
        memory_ids: list[str] | tuple[str, ...],
        *,
        scope: AccessScope,
    ):
        """Expose the adapter-owned, visibility-safe conflict read model."""

        return await self._relational.list_memory_conflict_contexts(memory_ids, scope)

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
        It MUST NOT be passed through :func:`sanitize_fts_query`, which is
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

    def _get_or_compute_embedding(self, text: str) -> list[float] | None:
        """Return cached embedding or compute via the embedding API.

        Returns ``None`` if the embedding call fails (caller should degrade
        gracefully).
        """
        if self._embedding_provider is not None:
            try:
                return self._embedding_provider(text)
            except Exception:
                logger.exception("Injected embedding provider failed for query")
                return None

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


def _recent_listing_fingerprint(
    *,
    source_filter: MemorySourceFilter,
    time_range: MemoryTimeRange,
    memory_types: tuple[str, ...],
    scope: AccessScope,
) -> str:
    payload = {
        "source_ids": sorted(source_filter.source_ids),
        "clients": sorted(source_filter.clients),
        "repo_identifiers": sorted(source_filter.repo_identifiers),
        "start_at": time_range.after.astimezone(timezone.utc).isoformat() if time_range.after else None,
        "end_at": time_range.before.astimezone(timezone.utc).isoformat() if time_range.before else None,
        "date_type": time_range.date_type,
        "memory_types": sorted(memory_types),
        "scope": {
            "user_id": scope.user_id,
            "include_private": scope.include_private,
            "allowed_statuses": list(scope.allowed_statuses),
            "active_project": scope.active_project,
            "scope_mode": scope.scope_mode,
            "active_repo_identifier": scope.active_repo_identifier,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_recent_listing_cursor(
    *,
    fingerprint: str,
    listing_watermark: datetime,
    after: RecentMemoryItem,
) -> str:
    payload = {
        "v": 1,
        "f": fingerprint,
        "s": listing_watermark.astimezone(timezone.utc).isoformat(),
        "a": after.sort_at.astimezone(timezone.utc).isoformat(),
        "i": after.memory_id,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")


def _decode_recent_listing_cursor(
    cursor: str,
    *,
    fingerprint: str,
) -> tuple[datetime, RecentMemoryItem]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("v") != 1 or payload.get("f") != fingerprint:
            raise ValueError
        listing_watermark = datetime.fromisoformat(str(payload["s"]).replace("Z", "+00:00"))
        sort_at = datetime.fromisoformat(str(payload["a"]).replace("Z", "+00:00"))
        memory_id = str(payload["i"]).strip()
        if (
            listing_watermark.tzinfo is None
            or listing_watermark.utcoffset() is None
            or sort_at.tzinfo is None
            or sort_at.utcoffset() is None
            or not memory_id
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("cursor is invalid or does not match the recent-memory request") from exc
    return listing_watermark.astimezone(timezone.utc), RecentMemoryItem(
        memory_id=memory_id,
        sort_at=sort_at.astimezone(timezone.utc),
    )


def _metadata_keyword_evidence(hit: KeywordCandidate) -> dict[str, Any]:
    return {
        "channel": hit.channel,
        "matched_fields": list(hit.matched_fields),
        "matched_text": list(hit.matched_text),
        "matched_terms": list(hit.matched_terms),
        "term_coverage": hit.term_coverage,
        "term_weights": dict(hit.term_weights),
        "source_refs": [
            {
                "source_id": ref.source_id,
                "doc_id": ref.doc_id,
                "source_type": ref.source_type,
            }
            for ref in hit.source_refs
        ],
    }


def _best_metadata_evidence(hits: list[KeywordCandidate]) -> dict[str, dict[str, Any]]:
    priority = {
        "bm25_metadata_tokens": 3,
        "metadata_alias": 2,
        "metadata_trigram": 1,
    }
    best: dict[str, KeywordCandidate] = {}
    for hit in hits:
        previous = best.get(hit.memory_id)
        if previous is None or (
            priority.get(hit.channel, 0),
            hit.score,
        ) > (
            priority.get(previous.channel, 0),
            previous.score,
        ):
            best[hit.memory_id] = hit
    return {
        memory_id: _metadata_keyword_evidence(hit)
        for memory_id, hit in best.items()
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
