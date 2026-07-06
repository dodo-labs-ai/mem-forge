"""Hybrid search engine for MemForge.

Vector, content BM25/FTS, metadata lexical, and entity-graph channels run in
parallel, then fuse through query-adaptive weighted RRF before the final ranking
and source/time page slicing.

Architecture reference: docs/architecture.md Section 10.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from memforge.config import DEFAULT_RANK_WINDOW_SIZE, RetrievalConfig
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
QUERY_ADAPTIVE_RRF_K = 60

_PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "identifier_lookup": {
        "vector": 0.15,
        "bm25_content": 0.25,
        "metadata_lexical": 0.50,
        "graph": 0.10,
    },
    "semantic_lookup": {
        "vector": 0.45,
        "bm25_content": 0.30,
        "metadata_lexical": 0.15,
        "graph": 0.10,
    },
    "graph_exploration": {
        "vector": 0.25,
        "bm25_content": 0.25,
        "metadata_lexical": 0.20,
        "graph": 0.30,
    },
    "lexical_lookup": {
        "vector": 0.25,
        "bm25_content": 0.35,
        "metadata_lexical": 0.30,
        "graph": 0.10,
    },
}

_METADATA_SUBCHANNEL_WEIGHTS = {
    "bm25_metadata_tokens": 0.60,
    "metadata_alias": 0.30,
    "metadata_trigram": 0.10,
}
_METADATA_IDENTIFIER_CHANNELS = {"bm25_metadata_tokens", "metadata_alias"}

_QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
}
_EXPLANATORY_VERBS = {
    "explain", "understand", "diagnose", "debug", "troubleshoot", "analyze",
    "compare", "cause", "causes", "caused", "reason", "reasons", "affect",
    "affects", "affected",
}
_SENTENCE_CONNECTORS = {
    "the", "a", "an", "to", "for", "of", "in", "on", "with", "after",
    "before", "because", "when", "while", "if", "does", "did", "is", "are",
}
_GRAPH_ACTION_TERMS = {
    "related", "relation", "relationship", "dependency", "dependencies",
    "depends", "impact", "impacts", "similar", "connected", "owner",
    "neighborhood", "upstream", "downstream", "risk", "risks", "affected",
}
_GRAPH_INVENTORY_TERMS = {
    "related", "nearby", "neighbor", "neighbors", "neighborhood", "connected",
    "around", "dependencies", "upstream", "downstream", "affected", "impact",
    "risks", "list", "show", "find",
}
_METADATA_LOOKUP_MODIFIER_TOKENS = {
    "bug", "bugs", "find", "issue", "issues", "jira", "list", "pr", "prs",
    "pull", "request", "requests", "search", "show", "stories", "story",
    "task", "tasks", "ticket", "tickets",
}
_METADATA_IDENTIFIER_COVERAGE_THRESHOLD = 0.75
_EXTERNAL_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b|#\d+\b|\bINC\d+\b", re.IGNORECASE)
_CODE_SYMBOL_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z0-9]*[./][A-Za-z0-9_./-]+\b)"
    r"|(?:\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b)"
)


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


@dataclass(frozen=True)
class _QueryFeatures:
    tokens: tuple[str, ...]
    has_external_id: bool
    has_code_symbol: bool
    code_symbol_terms: tuple[str, ...]
    natural_language_ratio: float
    graph_intent: float


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


def _query_tokens(query: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", query).lower()
    return tuple(token for token in re.split(r"[^0-9a-z]+", normalized) if token)


def _query_code_symbol_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _CODE_SYMBOL_RE.finditer(query):
        for token in _query_tokens(match.group(0)):
            if token not in seen:
                terms.append(token)
                seen.add(token)
    return tuple(terms)


def _compute_query_features(
    query: str,
    *,
    graph_candidates: list[EntityLinkCandidate],
) -> _QueryFeatures:
    tokens = _query_tokens(query)
    code_symbol_terms = _query_code_symbol_terms(query)
    token_set = set(tokens)
    has_question_word = bool(token_set & _QUESTION_WORDS) or "?" in query
    has_explanatory_verb = bool(token_set & _EXPLANATORY_VERBS)
    has_sentence_shape = len(tokens) >= 5 and bool(token_set & _SENTENCE_CONNECTORS)
    natural_language_ratio = min(
        1.0,
        min(0.30, len(tokens) / 25)
        + (0.15 if has_question_word else 0.0)
        + (0.15 if has_explanatory_verb else 0.0)
        + (0.10 if has_sentence_shape else 0.0),
    )

    if graph_candidates:
        graph_action_count = len(token_set & _GRAPH_ACTION_TERMS)
        graph_intent = min(
            1.0,
            0.35 * graph_action_count
            + (0.25 if token_set & _GRAPH_INVENTORY_TERMS else 0.0),
        )
    else:
        graph_intent = 0.0

    return _QueryFeatures(
        tokens=tokens,
        has_external_id=bool(_EXTERNAL_ID_RE.search(query)),
        has_code_symbol=bool(code_symbol_terms),
        code_symbol_terms=code_symbol_terms,
        natural_language_ratio=natural_language_ratio,
        graph_intent=graph_intent,
    )


def _metadata_identifier_core_tokens(tokens: tuple[str, ...]) -> set[str]:
    return {
        token for token in tokens
        if len(token) > 1
        and token not in _METADATA_LOOKUP_MODIFIER_TOKENS
        and token not in _QUESTION_WORDS
        and token not in _SENTENCE_CONNECTORS
    }


def _metadata_identifier_core_query(query: str) -> str:
    tokens = _query_tokens(query)
    core_tokens = [
        token for token in tokens
        if len(token) > 1
        and token not in _METADATA_LOOKUP_MODIFIER_TOKENS
        and token not in _QUESTION_WORDS
        and token not in _SENTENCE_CONNECTORS
    ]
    deduped_core_tokens = tuple(dict.fromkeys(core_tokens))
    if len(deduped_core_tokens) < 2:
        return query
    return " ".join(deduped_core_tokens)


def _metadata_identifier_coverage(
    query_tokens: tuple[str, ...],
    matched_text: tuple[str, ...],
) -> float:
    core_tokens = _metadata_identifier_core_tokens(query_tokens)
    if len(core_tokens) < 2:
        return 0.0
    matched_tokens = set(_query_tokens(" ".join(matched_text)))
    return len(core_tokens & matched_tokens) / len(core_tokens)


def _metadata_supports_code_symbol(
    features: _QueryFeatures,
    matched_text: tuple[str, ...],
) -> bool:
    if not features.code_symbol_terms:
        return False
    matched_tokens = set(_query_tokens(" ".join(matched_text)))
    return set(features.code_symbol_terms).issubset(matched_tokens)


def _select_ranking_profile(
    features: _QueryFeatures,
    *,
    metadata_hits: list[KeywordCandidate],
    graph_candidates: list[EntityLinkCandidate],
) -> str:
    if features.has_external_id:
        return "identifier_lookup"
    if any(hit.channel in _METADATA_IDENTIFIER_CHANNELS for hit in metadata_hits):
        if any(
            _metadata_identifier_coverage(features.tokens, hit.matched_text)
            >= _METADATA_IDENTIFIER_COVERAGE_THRESHOLD
            or (
                features.has_code_symbol
                and _metadata_supports_code_symbol(features, hit.matched_text)
            )
            for hit in metadata_hits
            if hit.channel in _METADATA_IDENTIFIER_CHANNELS and hit.matched_text
        ):
            return "identifier_lookup"
    if (
        graph_candidates
        and features.graph_intent >= 0.65
        and features.graph_intent >= features.natural_language_ratio + 0.15
    ):
        return "graph_exploration"
    if features.natural_language_ratio >= 0.55:
        return "semantic_lookup"
    return "lexical_lookup"


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
    profile: str,
    k: int,
) -> list[_RankedCandidate]:
    weights = _PROFILE_WEIGHTS[profile]
    scores: dict[str, float] = {}

    def add_ranked(channel: str, results: list[tuple[str, float]]) -> None:
        sorted_channel = sorted(results, key=lambda x: x[1], reverse=True)
        for rank_0, (memory_id, _score) in enumerate(sorted_channel):
            rank = rank_0 + 1
            scores[memory_id] = scores.get(memory_id, 0.0) + weights[channel] / (k + rank)

    add_ranked("vector", vector_results)
    add_ranked("bm25_content", content_results)
    add_ranked("metadata_lexical", metadata_results)
    for memory_id, contribution in graph_contributions.items():
        scores[memory_id] = scores.get(memory_id, 0.0) + (
            weights["graph"] / (k + contribution.rank) * contribution.multiplier
        )

    return [
        _RankedCandidate(memory_id=mid, rrf_score=score)
        for mid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
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
            self._bm25_metadata_search(
                query,
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
                hits = list(result)
                metadata_hits = hits
                metadata_evidence.update(_best_metadata_evidence(hits))
            elif name == "graph":
                per_entity_results = list(result)
                graph_contributions = self._graph_contributions(
                    graph_task_entities,
                    per_entity_results,
                )
            else:
                logger.warning("Unknown retrieval channel result ignored: %s", name)

        features = _compute_query_features(query, graph_candidates=graph_candidates)
        ranking_profile = _select_ranking_profile(
            features,
            metadata_hits=metadata_hits,
            graph_candidates=graph_candidates,
        )
        metadata_channel = _metadata_lexical_channel(
            metadata_hits,
            k=getattr(self._config, "rrf_k", QUERY_ADAPTIVE_RRF_K),
        )

        # ----- 5. Fuse via query-adaptive Weighted RRF, then apply source-of-truth re-checks -----
        fused = _weighted_rrf_fusion(
            vector_results=vector_results,
            content_results=content_results,
            metadata_results=metadata_channel,
            graph_contributions=graph_contributions,
            profile=ranking_profile,
            k=getattr(self._config, "rrf_k", QUERY_ADAPTIVE_RRF_K),
        )
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
        ranked_count = len(ranked)
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
                "ranking_profile": ranking_profile,
                "strategies_used": channel_names,
            },
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
        *,
        source_filter: MemorySourceFilter | None,
        time_range: MemoryTimeRange | None,
    ) -> list[KeywordCandidate]:
        """Query the source-metadata keyword channel."""
        sanitized_query = _sanitize_fts_query(_metadata_identifier_core_query(query))
        if not sanitized_query:
            return []
        return await self._keyword.search_metadata(
            sanitized_query,
            scope,
            memory_types,
            limit,
            source_filter=source_filter,
            time_range=time_range,
            include_subchannel_hits=True,
        )

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
