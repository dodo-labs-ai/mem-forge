"""Resolve canonical Memory identity without owning lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Mapping

from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassificationError,
    MemoryPairClassifier,
    MemoryRelationType,
)
from memforge.models import Memory

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class IdentityResolutionRequest:
    challenger: Memory
    doc_id: str
    entity_ids: tuple[int, ...] = ()
    excluded_memory_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class IdentityCandidateQuery:
    memory: Memory
    doc_id: str
    entity_ids: tuple[int, ...]
    excluded_memory_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class IdentityResolutionPolicy:
    max_requests_per_batch: int = 32

    def __post_init__(self) -> None:
        if self.max_requests_per_batch < 1:
            raise ValueError("max_requests_per_batch must be positive")


@dataclass(frozen=True, slots=True)
class IdentityPairDecision:
    """A classified candidate snapshot retained after its workset completes."""

    candidate_memory_id: str
    candidate_content_hash: str
    candidate_visibility: str
    candidate_owner_user_id: str | None
    candidate_project_key: str | None
    candidate_repo_identifier: str | None
    relation_type: MemoryRelationType
    direction: RelationDirection
    reason: str


@dataclass(frozen=True, slots=True)
class IdentityResolution:
    challenger: Memory
    target: Memory | None
    equivalence_proof: Mapping[str, object] | None
    classified_pairs: tuple[IdentityPairDecision, ...]
    classification_complete: bool = True
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IdentityResolutionMetrics:
    pair_count: int
    llm_calls: int
    prompt_chars: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class IdentityResolutionBatch:
    resolutions: tuple[IdentityResolution, ...]
    metrics: IdentityResolutionMetrics


class IdentityResolver:
    """Resolve one reconciliation scope through exact and batched semantic proof."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        pair_classifier: MemoryPairClassifier | None,
        llm_model: str,
        policy: IdentityResolutionPolicy | None = None,
    ) -> None:
        self._memory_store = memory_store
        self._pair_classifier = pair_classifier
        self._llm_model = llm_model
        self._policy = policy or IdentityResolutionPolicy()

    async def resolve(
        self,
        requests: tuple[IdentityResolutionRequest, ...],
    ) -> IdentityResolutionBatch:
        """Resolve requests in order with a bounded candidate and pair workset."""

        started = perf_counter()
        resolutions: list[IdentityResolution | None] = [None] * len(requests)
        pair_count = llm_calls = prompt_chars = 0
        for start in range(0, len(requests), self._policy.max_requests_per_batch):
            request_batch = requests[start : start + self._policy.max_requests_per_batch]
            batch_resolutions, batch_pair_count, batch_llm_calls, batch_prompt_chars = (
                await self._resolve_batch(request_batch)
            )
            resolutions[start : start + len(request_batch)] = batch_resolutions
            pair_count += batch_pair_count
            llm_calls += batch_llm_calls
            prompt_chars += batch_prompt_chars

        if any(resolution is None for resolution in resolutions):
            raise RuntimeError("identity resolution did not cover every request")
        return IdentityResolutionBatch(
            resolutions=tuple(
                resolution
                for resolution in resolutions
                if resolution is not None
            ),
            metrics=IdentityResolutionMetrics(
                pair_count=pair_count,
                llm_calls=llm_calls,
                prompt_chars=prompt_chars,
                elapsed_ms=max(0, round((perf_counter() - started) * 1000)),
            ),
        )

    async def _resolve_batch(
        self,
        requests: tuple[IdentityResolutionRequest, ...],
    ) -> tuple[list[IdentityResolution], int, int, int]:
        pending: dict[int, tuple[MemoryPair, ...]] = {}
        resolved: dict[int, IdentityResolution] = {}
        all_pairs: list[MemoryPair] = []
        unresolved: list[tuple[int, IdentityResolutionRequest]] = []
        exact_candidates = await self._memory_store.find_access_compatible_exact_candidates_batch(requests)
        if len(exact_candidates) != len(requests):
            raise RuntimeError("exact identity batch did not cover every request")
        for index, (request, exact) in enumerate(zip(requests, exact_candidates, strict=True)):
            challenger = request.challenger
            if (
                exact is not None
                and exact.content_hash == challenger.content_hash
                and exact.content.strip() == challenger.content.strip()
            ):
                resolved[index] = IdentityResolution(
                    challenger=challenger,
                    target=exact,
                    equivalence_proof={
                        "method": "exact_content",
                        "candidate_content_hash": challenger.content_hash,
                        "incumbent_content_hash": exact.content_hash,
                    },
                    classified_pairs=(),
                )
                continue
            unresolved.append((index, request))

        candidate_batches = await self._memory_store.find_access_compatible_equivalence_candidates_batch(
            tuple(
                IdentityCandidateQuery(
                    memory=request.challenger,
                    doc_id=request.doc_id,
                    entity_ids=request.entity_ids,
                    excluded_memory_ids=request.excluded_memory_ids,
                )
                for _, request in unresolved
            )
        )
        if len(candidate_batches) != len(unresolved):
            raise RuntimeError(
                "identity candidate batch coverage invalid: "
                f"expected_count={len(unresolved)}, actual_count={len(candidate_batches)}"
            )
        for (index, request), candidates in zip(
            unresolved,
            candidate_batches,
            strict=True,
        ):
            challenger = request.challenger
            exact_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.content_hash == challenger.content_hash
                    and candidate.content.strip() == challenger.content.strip()
                ),
                None,
            )
            if exact_candidate is not None:
                resolved[index] = IdentityResolution(
                    challenger=challenger,
                    target=exact_candidate,
                    equivalence_proof={
                        "method": "exact_content",
                        "candidate_content_hash": challenger.content_hash,
                        "incumbent_content_hash": exact_candidate.content_hash,
                    },
                    classified_pairs=(),
                )
                continue
            pairs = tuple(MemoryPair(challenger=challenger, candidate=candidate) for candidate in candidates)
            if not pairs:
                resolved[index] = IdentityResolution(
                    challenger=challenger,
                    target=None,
                    equivalence_proof=None,
                    classified_pairs=(),
                )
                continue
            pending[index] = pairs
            all_pairs.extend(pairs)

        llm_calls = prompt_chars = 0
        try:
            if all_pairs and self._pair_classifier is None:
                raise MemoryPairClassificationError("semantic classifier unavailable")
            classification = (
                await self._pair_classifier.classify(tuple(all_pairs))
                if all_pairs and self._pair_classifier is not None
                else None
            )
            if classification is not None:
                llm_calls = classification.llm_calls
                prompt_chars = classification.prompt_chars
            decisions = classification.decisions if classification is not None else ()
        except MemoryPairClassificationError as error:
            llm_calls = error.llm_calls
            prompt_chars = error.prompt_chars
            for index, pairs in pending.items():
                resolved[index] = IdentityResolution(
                    challenger=pairs[0].challenger,
                    target=None,
                    equivalence_proof=None,
                    classified_pairs=(),
                    classification_complete=False,
                    failure_reason=str(error),
                )
        else:
            decisions_by_key = {decision.pair.key: decision for decision in decisions}
            for index, pairs in pending.items():
                pair_decisions = tuple(decisions_by_key[pair.key] for pair in pairs)
                equivalent = next(
                    (
                        decision
                        for decision in pair_decisions
                        if decision.relation_type is MemoryRelationType.EQUIVALENT
                    ),
                    None,
                )
                target = equivalent.pair.candidate if equivalent is not None else None
                resolved[index] = IdentityResolution(
                    challenger=pairs[0].challenger,
                    target=target,
                    equivalence_proof=(
                        {
                            "method": "structured_relation_classifier",
                            "model": self._llm_model,
                            "reason": equivalent.reason,
                            "candidate_content_hash": pairs[0].challenger.content_hash,
                            "incumbent_content_hash": target.content_hash,
                        }
                        if equivalent is not None and target is not None
                        else None
                    ),
                    classified_pairs=tuple(
                        IdentityPairDecision(
                            candidate_memory_id=decision.pair.candidate.id,
                            candidate_content_hash=decision.pair.candidate.content_hash,
                            candidate_visibility=decision.pair.candidate.visibility,
                            candidate_owner_user_id=decision.pair.candidate.owner_user_id,
                            candidate_project_key=decision.pair.candidate.project_key,
                            candidate_repo_identifier=decision.pair.candidate.repo_identifier,
                            relation_type=decision.relation_type,
                            direction=decision.direction,
                            reason=decision.reason,
                        )
                        for decision in pair_decisions
                    ),
                )
        return (
            [resolved[index] for index in range(len(requests))],
            len(all_pairs),
            llm_calls,
            prompt_chars,
        )
