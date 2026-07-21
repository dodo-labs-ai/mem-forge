"""Provider-neutral candidate retrieval for cross-document relation checks."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence

from memforge.memory.evidence import (
    CandidateBucket,
    CandidateBucketResult,
    CandidateMemory,
)
from memforge.models import Memory, MemoryStatus, Visibility
from memforge.retrieval.rank_fusion import (
    FusedRankedItem,
    RankedChannelItem,
    weighted_reciprocal_rank_fusion,
)
from memforge.retrieval.search import sanitize_fts_query
from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID
from memforge.storage.adapters.protocols import KeywordSearch, RelationalStore, VectorStore


_LEXICAL_DISCOVERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class RelationCandidateRetrievalPolicy:
    """Bounded discovery policy; mandatory candidates are outside this budget."""

    initial_budget: int = 32
    expansion_step: int = 32
    max_budget: int = 128
    rank_window_size: int = 128
    lexical_query_term_limit: int = 32
    rrf_k: int = 60
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            CandidateBucket.SHARED_ENTITIES.value: 1.0,
            CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS.value: 1.0,
            CandidateBucket.LEXICAL_BM25.value: 1.0,
        }
    )

    def __post_init__(self) -> None:
        if self.initial_budget < 1:
            raise ValueError("initial discovery budget must be positive")
        if self.expansion_step < 1:
            raise ValueError("discovery expansion step must be positive")
        if self.max_budget < self.initial_budget:
            raise ValueError("max discovery budget must cover the initial budget")
        if self.rank_window_size < self.max_budget:
            raise ValueError("rank window must cover the maximum discovery budget")
        if self.lexical_query_term_limit < 1:
            raise ValueError("lexical discovery query must contain at least one term")


@dataclass(frozen=True, slots=True)
class RetrievedRelationCandidate:
    memory: CandidateMemory
    score: float
    channels: tuple[str, ...]


class StaleCandidateSelectionError(RuntimeError):
    """The selected candidate set no longer satisfies its access contract."""


@dataclass(frozen=True, slots=True)
class CrossDocumentCandidateSelection:
    discovery: tuple[RetrievedRelationCandidate, ...]
    audit: Mapping[str, Any]
    telemetry: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.memory.memory_id for candidate in self.discovery)

    @property
    def snapshot_identity(self) -> str:
        payload = {
            "candidates": [
                {
                    "memory_id": candidate.memory.memory_id,
                    "score": candidate.score,
                    "channels": candidate.channels,
                }
                for candidate in self.discovery
            ]
        }
        return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]

    def bucket_results(self) -> tuple[CandidateBucketResult, ...]:
        results: list[CandidateBucketResult] = []
        if self.discovery:
            results.append(
                CandidateBucketResult(
                    bucket=CandidateBucket.HYBRID_DISCOVERY,
                    bucket_rank=10,
                    complete=True,
                    candidates=tuple(candidate.memory for candidate in self.discovery),
                    scores={candidate.memory.memory_id: candidate.score for candidate in self.discovery},
                    candidate_reasons={
                        candidate.memory.memory_id: ("hybrid_discovery:" + ",".join(candidate.channels))
                        for candidate in self.discovery
                    },
                    reason="bounded hybrid cross-document discovery",
                )
            )
        return tuple(results)

class CrossDocumentCandidateRetriever:
    """Retrieve IDs in parallel, then resolve only lightweight provenance rows."""

    def __init__(
        self,
        *,
        relational: RelationalStore,
        keyword: KeywordSearch,
        vector: VectorStore,
        policy: RelationCandidateRetrievalPolicy | None = None,
    ) -> None:
        self._relational = relational
        self._keyword = keyword
        self._vector = vector
        self._policy = policy or RelationCandidateRetrievalPolicy()

    async def retrieve(
        self,
        *,
        challenger: Memory,
        entity_ids: Sequence[int],
        doc_id: str,
        actor_user_id: str | None,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
    ) -> CrossDocumentCandidateSelection:
        started_at = time.perf_counter()
        scope = _candidate_scope(challenger, actor_user_id=actor_user_id)
        channels, channel_errors = await self._retrieve_channels(
            challenger=challenger,
            entity_ids=entity_ids,
            scope=scope,
        )
        all_ids = tuple(dict.fromkeys(item.item_id for items in channels.values() for item in items))
        provenance_rows = await self._relational.list_active_candidate_memories(all_ids)
        eligible = _eligible_candidate_rows(
            provenance_rows,
            challenger=challenger,
            doc_id=doc_id,
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        eligible_channels = {
            channel: tuple(item for item in items if item.item_id in eligible) for channel, items in channels.items()
        }
        fused = weighted_reciprocal_rank_fusion(
            channels=eligible_channels,
            weights=self._policy.channel_weights,
            k=self._policy.rrf_k,
        )
        discovery_budget = _adaptive_discovery_budget(fused, self._policy)
        selected = fused[:discovery_budget]
        discovery = tuple(
            RetrievedRelationCandidate(
                memory=eligible[item.item_id],
                score=item.score,
                channels=tuple(part.channel for part in item.contributions),
            )
            for item in selected
        )
        return CrossDocumentCandidateSelection(
            discovery=discovery,
            audit={
                "candidate_count_kind": "windowed",
                "rank_window_size": self._policy.rank_window_size,
                "selected_discovery_count": len(discovery),
                "mandatory_candidate_count": 0,
            },
            telemetry={
                "channel_candidate_counts": {channel: len(items) for channel, items in channels.items()},
                "eligible_candidate_count": len(eligible),
                "fused_candidate_count": len(fused),
                "discovery_budget": discovery_budget,
                "channel_errors": list(channel_errors),
                "provenance_rows_loaded": len(provenance_rows),
                "retrieval_latency_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    3,
                ),
            },
        )

    async def load_selected_memories(
        self,
        selection: CrossDocumentCandidateSelection,
        *,
        challenger: Memory,
        doc_id: str,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
    ) -> tuple[CrossDocumentCandidateSelection, Mapping[str, Memory]]:
        """Materialize full rows, then recheck current provenance before use."""

        loaded = await self._relational.list_active_memories(selection.candidate_ids)
        by_id = {memory.id: memory for memory in loaded if _memory_access_compatible(memory, challenger)}
        provenance_rows = await self._relational.list_active_candidate_memories(tuple(by_id))
        eligible = _eligible_candidate_rows(
            provenance_rows,
            challenger=challenger,
            doc_id=doc_id,
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        missing_ids = tuple(
            memory_id for memory_id in selection.candidate_ids if memory_id not in by_id or memory_id not in eligible
        )
        if missing_ids:
            raise StaleCandidateSelectionError("selected candidate could not be materialized with current access")
        materialized_selection = CrossDocumentCandidateSelection(
            discovery=selection.discovery,
            audit=selection.audit,
            telemetry={
                **selection.telemetry,
                "full_memory_rows_loaded": len(selection.candidate_ids),
            },
        )
        return materialized_selection, {memory_id: by_id[memory_id] for memory_id in selection.candidate_ids}

    async def ensure_selection_current(
        self,
        selection: CrossDocumentCandidateSelection,
        *,
        challenger: Memory,
        doc_id: str,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
    ) -> None:
        """Fail closed if access or provenance changed before relation writes."""

        current_challenger = await self._relational.get_memory(challenger.id)
        if (
            current_challenger is None
            or current_challenger.status != MemoryStatus.ACTIVE.value
            or not _memory_access_compatible(current_challenger, challenger)
        ):
            raise StaleCandidateSelectionError("challenger access changed")
        provenance_rows = await self._relational.list_active_candidate_memories(selection.candidate_ids)
        eligible = _eligible_candidate_rows(
            provenance_rows,
            challenger=current_challenger,
            doc_id=doc_id,
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        current_ids = tuple(memory_id for memory_id in selection.candidate_ids if memory_id in eligible)
        if current_ids != selection.candidate_ids:
            raise StaleCandidateSelectionError("candidate access or provenance changed")

    async def _retrieve_channels(
        self,
        *,
        challenger: Memory,
        entity_ids: Sequence[int],
        scope: AccessScope,
    ) -> tuple[dict[str, tuple[RankedChannelItem, ...]], tuple[str, ...]]:
        channel_names = (
            CandidateBucket.SHARED_ENTITIES.value,
            CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS.value,
            CandidateBucket.LEXICAL_BM25.value,
        )
        results = await asyncio.gather(
            self._entity_channel(entity_ids, scope),
            self._vector_channel(challenger.id, scope),
            self._lexical_channel(challenger.content, scope),
            return_exceptions=True,
        )
        channels: dict[str, tuple[RankedChannelItem, ...]] = {}
        errors: list[str] = []
        for channel, result in zip(channel_names, results):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result
                channels[channel] = ()
                errors.append(channel)
            else:
                ranked_items = sorted(
                    (RankedChannelItem(memory_id, score) for memory_id, score in result if memory_id != challenger.id),
                    key=lambda item: (-item.score, item.item_id),
                )
                channels[channel] = tuple(ranked_items[: self._policy.rank_window_size])
        return channels, tuple(errors)

    async def _entity_channel(
        self,
        entity_ids: Sequence[int],
        scope: AccessScope,
    ) -> list[tuple[str, float]]:
        if not entity_ids:
            return []
        return await self._relational.graph_search(
            entity_ids,
            scope,
            None,
            self._policy.rank_window_size,
        )

    async def _vector_channel(
        self,
        challenger_id: str,
        scope: AccessScope,
    ) -> list[tuple[str, float]]:
        record = await self._vector.get_record(challenger_id)
        embedding = record.get("embedding") if record else None
        if embedding is None:
            return []
        return await self._vector.query(
            embedding,
            scope,
            None,
            self._policy.rank_window_size,
        )

    async def _lexical_channel(
        self,
        content: str,
        scope: AccessScope,
    ) -> list[tuple[str, float]]:
        query = _bounded_any_term_fts_query(
            content,
            max_terms=self._policy.lexical_query_term_limit,
        )
        if not query:
            return []
        return await self._keyword.search(
            query,
            scope,
            None,
            self._policy.rank_window_size,
        )


def _bounded_any_term_fts_query(content: str, *, max_terms: int) -> str:
    """Build a safe, bounded BM25 discovery query from Memory content."""

    sanitized = sanitize_fts_query(content)
    if not sanitized:
        return ""
    terms = (term for term in sanitized.split() if term.strip('"').casefold() not in _LEXICAL_DISCOVERY_STOP_WORDS)
    unique_terms = tuple(dict.fromkeys(terms))[:max_terms]
    return " OR ".join(unique_terms)


def _candidate_scope(challenger: Memory, *, actor_user_id: str | None) -> AccessScope:
    private = challenger.visibility == Visibility.PRIVATE.value
    return AccessScope(
        user_id=(challenger.owner_user_id if private else actor_user_id) or LOCAL_DEV_USER_ID,
        include_private=private,
        allowed_statuses=(MemoryStatus.ACTIVE.value,),
        active_project=challenger.project_key,
        scope_mode="project-first",
    )


def _eligible_candidate_rows(
    rows: Sequence[CandidateMemory],
    *,
    challenger: Memory,
    doc_id: str,
    source_id: str,
    excluded_source_ids: frozenset[str],
) -> dict[str, CandidateMemory]:
    eligible: dict[str, CandidateMemory] = {}
    for candidate in rows:
        if candidate.memory_id == challenger.id:
            continue
        if candidate.visibility != challenger.visibility:
            continue
        if candidate.owner_user_id != challenger.owner_user_id:
            continue
        if candidate.repo_identifier != challenger.repo_identifier:
            continue
        if not candidate.doc_id or candidate.doc_id == doc_id:
            continue
        if candidate.source_id is not None and candidate.source_id in excluded_source_ids:
            continue
        existing = eligible.get(candidate.memory_id)
        if existing is None or _candidate_provenance_key(
            candidate,
            challenger_source_id=source_id,
        ) < _candidate_provenance_key(existing, challenger_source_id=source_id):
            eligible[candidate.memory_id] = candidate
    return eligible


def _candidate_provenance_key(
    candidate: CandidateMemory,
    *,
    challenger_source_id: str,
) -> tuple[bool, str, str]:
    """Prefer an independent source while keeping deterministic selection."""

    return (
        candidate.source_id == challenger_source_id,
        candidate.source_id or "",
        candidate.doc_id or "",
    )


def _memory_access_compatible(candidate: Memory, challenger: Memory) -> bool:
    return (
        candidate.visibility == challenger.visibility
        and candidate.owner_user_id == challenger.owner_user_id
        and candidate.repo_identifier == challenger.repo_identifier
    )


def _adaptive_discovery_budget(
    fused: Sequence[FusedRankedItem],
    policy: RelationCandidateRetrievalPolicy,
) -> int:
    if not fused:
        return 0
    upper_bound = min(len(fused), policy.max_budget)
    budget = min(policy.initial_budget, upper_bound)
    required = budget
    for index, candidate in enumerate(fused[:upper_bound]):
        if len(candidate.contributions) > 1:
            required = max(required, index + 1)
    while budget < required:
        budget = min(upper_bound, budget + policy.expansion_step)
    while budget < upper_bound and fused[budget].score == fused[budget - 1].score:
        budget = min(upper_bound, budget + 1)
    return budget
