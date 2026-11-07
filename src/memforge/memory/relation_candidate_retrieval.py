"""Provider-neutral candidate retrieval for relation discovery.

Each of the three ranked lists (shared entities, vector neighbours and BM25)
nominates its first ``per_list_limit`` eligible Memories, and the candidates are
their union. A Memory is eligible only when it was created before the
challenger, so every pair is judged once, from its later Memory.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

from memforge.memory.evidence import (
    CandidateBucket,
    CandidateBucketResult,
    CandidateMemory,
)
from memforge.memory.relation_discovery_contract import resolve_relation_discovery_actor_user_id
from memforge.models import Memory, MemoryStatus, Visibility
from memforge.retrieval.rank_fusion import RankedChannelItem
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
    """How many candidates each ranked list nominates, and how deep each list is read.

    ``per_list_limit`` is the budget: a challenger is compared with at most three
    times that many Memories, whatever the workspace size. Each list is read to
    ``rank_window_size`` so that dropping inaccessible and later Memories usually
    leaves it ``per_list_limit`` deep.
    """

    per_list_limit: int = 5
    rank_window_size: int = 128
    lexical_query_term_limit: int = 32

    def __post_init__(self) -> None:
        if self.per_list_limit < 1:
            raise ValueError("each ranked list must nominate at least one candidate")
        if self.rank_window_size < self.per_list_limit:
            raise ValueError("rank window must cover the per-list limit")
        if self.lexical_query_term_limit < 1:
            raise ValueError("lexical discovery query must contain at least one term")


@dataclass(frozen=True, slots=True)
class Nomination:
    """A Memory that at least one ranked list placed within its first ``per_list_limit`` items.

    The rule's result holds ids only; retrieval pairs it with the candidate's row.
    """

    memory_id: str
    # The best 1-based rank the Memory has in any list that nominated it.
    best_rank: int
    # The lists that nominated it, in the order the lists were given.
    channels: tuple[str, ...]


def nominate_candidates(
    channels: Mapping[str, Sequence[RankedChannelItem]],
    *,
    per_list_limit: int,
) -> tuple[Nomination, ...]:
    """Each list nominates its first ``per_list_limit`` Memories, and the candidates are their union.

    A Memory a list repeats takes one place, at its first rank. Candidates are
    ordered by best rank, then Memory id, so the order does not depend on which
    list is read first.
    """

    best_rank: dict[str, int] = {}
    nominated_by: dict[str, list[str]] = {}
    for channel, items in channels.items():
        memory_ids = list(dict.fromkeys(item.item_id for item in items))
        for rank, memory_id in enumerate(memory_ids[:per_list_limit], start=1):
            best_rank[memory_id] = min(rank, best_rank.get(memory_id, rank))
            nominated_by.setdefault(memory_id, []).append(channel)
    return tuple(
        Nomination(memory_id, best_rank[memory_id], tuple(nominated_by[memory_id]))
        for memory_id in sorted(best_rank, key=lambda memory_id: (best_rank[memory_id], memory_id))
    )


@dataclass(frozen=True, slots=True)
class RetrievedRelationCandidate:
    memory: CandidateMemory
    best_rank: int
    channels: tuple[str, ...]


class StaleCandidateSelectionError(RuntimeError):
    """The selected candidate set no longer satisfies its access contract."""


@dataclass(frozen=True, slots=True)
class RelationCandidateSelection:
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
                    "best_rank": candidate.best_rank,
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
                    candidate_reasons={
                        candidate.memory.memory_id: ("hybrid_discovery:" + ",".join(candidate.channels))
                        for candidate in self.discovery
                    },
                    reason="bounded hybrid relation discovery",
                )
            )
        return tuple(results)


class RelationCandidateRetriever:
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
        actor_user_id: str | None,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
    ) -> RelationCandidateSelection:
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
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        eligible_channels = {
            channel: tuple(item for item in items if item.item_id in eligible) for channel, items in channels.items()
        }
        nominations = nominate_candidates(eligible_channels, per_list_limit=self._policy.per_list_limit)
        discovery = tuple(
            RetrievedRelationCandidate(
                memory=eligible[nomination.memory_id],
                best_rank=nomination.best_rank,
                channels=nomination.channels,
            )
            for nomination in nominations
        )
        return RelationCandidateSelection(
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
                "per_list_limit": self._policy.per_list_limit,
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
        selection: RelationCandidateSelection,
        *,
        challenger: Memory,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
    ) -> tuple[RelationCandidateSelection, Mapping[str, Memory]]:
        """Materialize full rows, then recheck current provenance before use."""

        loaded = await self._relational.list_active_memories(selection.candidate_ids)
        by_id = {memory.id: memory for memory in loaded if _memory_access_compatible(memory, challenger)}
        provenance_rows = await self._relational.list_active_candidate_memories(tuple(by_id))
        eligible = _eligible_candidate_rows(
            provenance_rows,
            challenger=challenger,
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        missing_ids = tuple(
            memory_id for memory_id in selection.candidate_ids if memory_id not in by_id or memory_id not in eligible
        )
        if missing_ids:
            raise StaleCandidateSelectionError("selected candidate could not be materialized with current access")
        materialized_selection = RelationCandidateSelection(
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
        selection: RelationCandidateSelection,
        *,
        challenger: Memory,
        source_id: str,
        excluded_source_ids: Sequence[str] = (),
        expected_support_set_hashes: Mapping[str, str] | None = None,
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
            source_id=source_id,
            excluded_source_ids=frozenset(excluded_source_ids),
        )
        current_ids = tuple(memory_id for memory_id in selection.candidate_ids if memory_id in eligible)
        if current_ids != selection.candidate_ids:
            raise StaleCandidateSelectionError("candidate access or provenance changed")
        if expected_support_set_hashes is not None:
            expected = dict(expected_support_set_hashes)
            if set(expected) != set(selection.candidate_ids):
                raise StaleCandidateSelectionError("candidate Support fence is incomplete")
            current_support = await self._relational.get_active_memory_support_states(
                selection.candidate_ids
            )
            if any(
                current_support[memory_id].current_support_set_hash != expected[memory_id]
                for memory_id in selection.candidate_ids
            ):
                raise StaleCandidateSelectionError("candidate current Support changed")

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
        user_id=resolve_relation_discovery_actor_user_id(
            visibility=challenger.visibility,
            owner_user_id=challenger.owner_user_id,
            requested_actor_user_id=actor_user_id,
        )
        or LOCAL_DEV_USER_ID,
        include_private=private,
        allowed_statuses=(MemoryStatus.ACTIVE.value,),
        active_project=challenger.project_key,
        scope_mode="project-first",
    )


def _eligible_candidate_rows(
    rows: Sequence[CandidateMemory],
    *,
    challenger: Memory,
    source_id: str,
    excluded_source_ids: frozenset[str],
) -> dict[str, CandidateMemory]:
    """Active Memories created before the challenger that it may be related to, its own Source Unit's included."""
    eligible: dict[str, CandidateMemory] = {}
    challenger_order = _creation_order(challenger.created_at, challenger.id)
    for candidate in rows:
        # The challenger itself orders equal, so it is left out with every later Memory.
        if _creation_order(candidate.created_at, candidate.memory_id) >= challenger_order:
            continue
        if candidate.visibility != challenger.visibility:
            continue
        if candidate.owner_user_id != challenger.owner_user_id:
            continue
        if candidate.repo_identifier != challenger.repo_identifier:
            continue
        if not candidate.doc_id:
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


def _creation_order(created_at: datetime | None, memory_id: str) -> tuple[datetime, str]:
    """When a Memory was created, as UTC, with its id breaking ties within one commit.

    A time without an offset is UTC. Every store records the time a Memory is
    created, so a missing one means a store did not read it, and discovery fails
    rather than judge the pair from both sides.
    """

    if created_at is None:
        raise ValueError(f"relation discovery needs the creation time of Memory {memory_id}")
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc), memory_id
    return created_at.astimezone(timezone.utc), memory_id
