"""Resolve canonical Memory identity without owning lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassificationError,
    MemoryPairClassifier,
    MemoryPairDecision,
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
class IdentityResolution:
    challenger: Memory
    target: Memory | None
    equivalence_proof: Mapping[str, object] | None
    classified_pairs: tuple[MemoryPairDecision, ...]
    classification_complete: bool = True
    failure_reason: str | None = None


class IdentityResolver:
    """Resolve one reconciliation scope through exact and batched semantic proof."""

    def __init__(
        self,
        *,
        memory_store: MemoryStore,
        pair_classifier: MemoryPairClassifier | None,
        llm_model: str,
    ) -> None:
        self._memory_store = memory_store
        self._pair_classifier = pair_classifier
        self._llm_model = llm_model

    async def resolve(
        self,
        requests: tuple[IdentityResolutionRequest, ...],
    ) -> tuple[IdentityResolution, ...]:
        """Resolve requests in order while coalescing all semantic pairs."""

        pending: dict[int, tuple[MemoryPair, ...]] = {}
        resolved: dict[int, IdentityResolution] = {}
        all_pairs: list[MemoryPair] = []

        for index, request in enumerate(requests):
            challenger = request.challenger
            exact = await self._memory_store.find_access_compatible_exact_candidate(
                challenger,
                excluded_memory_ids=request.excluded_memory_ids,
            )
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

            candidates = await self._memory_store.find_access_compatible_equivalence_candidates(
                challenger,
                excluded_memory_ids=request.excluded_memory_ids,
                doc_id=request.doc_id,
                entity_ids=request.entity_ids,
            )
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

        try:
            if all_pairs and self._pair_classifier is None:
                raise MemoryPairClassificationError("semantic classifier unavailable")
            classification = (
                await self._pair_classifier.classify(tuple(all_pairs))
                if all_pairs and self._pair_classifier is not None
                else None
            )
            decisions = classification.decisions if classification is not None else ()
        except MemoryPairClassificationError as error:
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
                    classified_pairs=pair_decisions,
                )

        return tuple(resolved[index] for index in range(len(requests)))
