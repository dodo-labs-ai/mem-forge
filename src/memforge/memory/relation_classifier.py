"""Provider-neutral semantic classification for exact Memory pairs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from memforge.memory.evidence import RelationDirection
from memforge.models import Memory


class MemoryPairClassificationError(RuntimeError):
    """The semantic pair ledger could not be completed safely."""

    def __init__(
        self,
        message: str,
        *,
        pair_count: int = 0,
        llm_calls: int = 0,
        prompt_chars: int = 0,
    ) -> None:
        super().__init__(message)
        self.pair_count = pair_count
        self.llm_calls = llm_calls
        self.prompt_chars = prompt_chars


class MemoryRelationType(str, Enum):
    EQUIVALENT = "equivalent"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    UNRELATED = "unrelated"


MEMORY_PAIR_CLASSIFIER_VERSION = "memory-relation-v1"


@dataclass(frozen=True, slots=True)
class MemoryPair:
    challenger: Memory
    candidate: Memory

    @property
    def key(self) -> tuple[str, str]:
        return self.challenger.id, self.candidate.id


@dataclass(frozen=True, slots=True)
class MemoryPairDecision:
    pair: MemoryPair
    relation_type: MemoryRelationType
    direction: RelationDirection
    reason: str

    def __post_init__(self) -> None:
        directional = self.relation_type is MemoryRelationType.REFINES
        if directional == (self.direction is RelationDirection.SYMMETRIC):
            raise ValueError("REFINES must be directional and other relations symmetric")


@dataclass(frozen=True, slots=True)
class MemoryPairClassification:
    decisions: tuple[MemoryPairDecision, ...]
    llm_calls: int
    prompt_chars: int


@dataclass(frozen=True, slots=True)
class MemoryPairClassificationPlan:
    pair_count: int
    llm_calls: int
    prompt_chars: int


class MemoryPairClassifier(Protocol):
    def plan(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> MemoryPairClassificationPlan: ...

    async def classify(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> MemoryPairClassification: ...


@dataclass(frozen=True, slots=True)
class MemoryPairClassificationPolicy:
    max_pairs_per_call: int = 64
    max_prompt_chars: int = 120_000
    max_output_tokens: int = 32_768
    max_memory_content_chars: int = 4_000

    def __post_init__(self) -> None:
        if self.max_pairs_per_call < 1:
            raise ValueError("max_pairs_per_call must be positive")
        if self.max_prompt_chars < 1:
            raise ValueError("max_prompt_chars must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.max_memory_content_chars < 1:
            raise ValueError("max_memory_content_chars must be positive")


MEMORY_RELATION_PROMPT = """Classify the semantic relationship of every exact Memory pair.

Use these definitions strictly:
- EQUIVALENT: the claims express one durable proposition with exactly the same truth conditions.
- REFINES: one claim is compatible with the other but narrows it, adds a condition, or adds a material detail.
- CONTRADICTS: the claims make mutually incompatible assertions about the same subject.
- UNRELATED: none of the relationships above applies.

Equivalence is symmetric and must be false when either claim narrows, broadens,
conditions, updates, contradicts, or adds any material fact. For REFINES, set
direction to challenger_to_candidate when the challenger is more specific, or
candidate_to_challenger when the candidate is more specific. For every other
classification direction must be symmetric. Labels never imply authority,
recency, preference, or permission to mutate either Memory.

Compare the proposition rather than presentation alone, but preserve material
modality. A normative requirement and a descriptive state have different truth
conditions and must not be EQUIVALENT, even when their subject, action, and value
match. The same applies to plans versus completed actions, recommendations versus
requirements, and predictions versus observed facts. Attribution, document
framing, labels, and examples are non-material only when they do not change
authority, subject, action or value, scope, polarity, conditions, time, or modality.
Treat "a document, case, or record states that P" and a direct statement of P as
equivalent only when P is the durable knowledge and neither claim is about the
recording act, its completeness, or its authority.

<memory_pair_groups>
{groups_json}
</memory_pair_groups>

Return exactly one decision for every pair_index and no other pair_index.
"""


def _prompt_memory(memory: Memory, *, max_content_chars: int) -> dict[str, str]:
    content = memory.content
    if len(content) > max_content_chars:
        content = content[:max_content_chars] + "\n[truncated]"
    return {
        "id": memory.id,
        "content": content,
        "type": memory.memory_type,
    }


def _grouped_pair_payload(
    indexed_pairs: tuple[tuple[int, MemoryPair], ...],
    *,
    max_content_chars: int,
) -> str:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for pair_index, pair in indexed_pairs:
        challenger_id = pair.challenger.id
        if challenger_id not in groups:
            groups[challenger_id] = {
                "challenger": _prompt_memory(
                    pair.challenger,
                    max_content_chars=max_content_chars,
                ),
                "candidates": [],
            }
            order.append(challenger_id)
        groups[challenger_id]["candidates"].append(
            {
                "pair_index": pair_index,
                "candidate": _prompt_memory(
                    pair.candidate,
                    max_content_chars=max_content_chars,
                ),
            }
        )
    return json.dumps([groups[challenger_id] for challenger_id in order], ensure_ascii=False)


class StructuredMemoryPairClassifier:
    """Classify exact pairs and reject any incomplete structured ledger."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        policy: MemoryPairClassificationPolicy | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._policy = policy or MemoryPairClassificationPolicy()

    async def classify(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> MemoryPairClassification:
        if not pairs:
            return MemoryPairClassification(decisions=(), llm_calls=0, prompt_chars=0)
        attempted_pairs = llm_calls = prompt_chars = 0
        try:
            decisions_by_index: dict[int, MemoryPairDecision] = {}
            batches = self._batches(pairs)
            for indexed_pairs, prompt in batches:
                attempted_pairs += len(indexed_pairs)
                llm_calls += 1
                prompt_chars += len(prompt)
                response = await self._client.classify_memory_relations(
                    prompt,
                    max_tokens=self._policy.max_output_tokens,
                    model=self._model,
                )
                raw_decisions = tuple(response.decisions)
                batch_indices = tuple(index for index, _ in indexed_pairs)
                self._validate_coverage(raw_decisions, expected_indices=batch_indices)
                by_index = {int(decision.pair_index): decision for decision in raw_decisions}
                for pair_index, pair in indexed_pairs:
                    decision = by_index[pair_index]
                    decisions_by_index[pair_index] = MemoryPairDecision(
                        pair=pair,
                        relation_type=MemoryRelationType(decision.classification),
                        direction=RelationDirection(decision.direction),
                        reason=str(decision.reason or ""),
                    )
            return MemoryPairClassification(
                decisions=tuple(decisions_by_index[index] for index in range(len(pairs))),
                llm_calls=llm_calls,
                prompt_chars=prompt_chars,
            )
        except MemoryPairClassificationError as error:
            if not (attempted_pairs or llm_calls or prompt_chars):
                raise
            raise MemoryPairClassificationError(
                str(error),
                pair_count=attempted_pairs,
                llm_calls=llm_calls,
                prompt_chars=prompt_chars,
            ) from error
        except Exception as error:
            raise MemoryPairClassificationError(
                f"memory relation classification failed: {error}",
                pair_count=attempted_pairs,
                llm_calls=llm_calls,
                prompt_chars=prompt_chars,
            ) from error

    def plan(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> MemoryPairClassificationPlan:
        batches = self._batches(pairs)
        return MemoryPairClassificationPlan(
            pair_count=len(pairs),
            llm_calls=len(batches),
            prompt_chars=sum(len(prompt) for _, prompt in batches),
        )

    def _batches(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> tuple[tuple[tuple[tuple[int, MemoryPair], ...], str], ...]:
        batches: list[tuple[tuple[tuple[int, MemoryPair], ...], str]] = []
        start = 0
        while start < len(pairs):
            end = min(len(pairs), start + self._policy.max_pairs_per_call)
            while True:
                indexed_pairs = tuple((index, pairs[index]) for index in range(start, end))
                prompt = MEMORY_RELATION_PROMPT.format(
                    groups_json=_grouped_pair_payload(
                        indexed_pairs,
                        max_content_chars=self._policy.max_memory_content_chars,
                    )
                )
                if len(prompt) <= self._policy.max_prompt_chars:
                    batches.append((indexed_pairs, prompt))
                    start = end
                    break
                if end - start == 1:
                    raise MemoryPairClassificationError("one Memory pair exceeds the configured prompt budget")
                end = start + max(1, (end - start) // 2)
        return tuple(batches)

    @staticmethod
    def _validate_coverage(
        decisions: tuple[Any, ...],
        *,
        expected_indices: tuple[int, ...],
    ) -> None:
        expected_index_set = set(expected_indices)
        actual_indices = [int(decision.pair_index) for decision in decisions]
        counts = Counter(actual_indices)
        duplicate_indices = {index for index, count in counts.items() if count > 1}
        actual_index_set = set(actual_indices)
        missing_indices = expected_index_set - actual_index_set
        unexpected_indices = actual_index_set - expected_index_set
        if len(decisions) != len(expected_indices) or duplicate_indices or missing_indices or unexpected_indices:
            raise MemoryPairClassificationError(
                "memory relation decision coverage invalid: "
                f"expected_count={len(expected_indices)}, "
                f"actual_count={len(decisions)}, "
                f"missing_count={len(missing_indices)}, "
                f"duplicate_count={len(duplicate_indices)}, "
                f"unexpected_count={len(unexpected_indices)}"
            )
