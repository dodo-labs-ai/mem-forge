"""Provider-neutral relationship rules and exact Memory-pair classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from memforge.llm.batch_runner import BatchStats, ItemFailure, ItemTask, LlmBatchRunner, LlmRequest, RejectedRow
from memforge.llm.structured import MemoryRelationResponse, StructuredLlmError
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
        terminal_category: str | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.pair_count = pair_count
        self.llm_calls = llm_calls
        self.prompt_chars = prompt_chars
        self.terminal_category = terminal_category
        self.error_code = error_code


class MemoryRelationType(str, Enum):
    EQUIVALENT = "equivalent"
    REFINES = "refines"
    CONTRADICTS = "contradicts"
    UNRELATED = "unrelated"


MEMORY_PAIR_CLASSIFIER_VERSION = "memory-relation-v3"


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
class UnjudgedPair:
    """A pair the classifier could not judge even alone: it exceeds capacity or its output stays invalid."""

    pair: MemoryPair
    failure: ItemFailure


@dataclass(frozen=True, slots=True)
class MemoryPairClassification:
    decisions: tuple[MemoryPairDecision, ...]
    llm_calls: int
    prompt_chars: int
    unjudged: tuple[UnjudgedPair, ...] = ()


class MemoryPairClassifier(Protocol):
    async def classify(
        self,
        pairs: tuple[MemoryPair, ...],
    ) -> MemoryPairClassification: ...


@dataclass(frozen=True, slots=True)
class MemoryPairClassificationPolicy:
    """Task shape of a relation request; capacity packing belongs to the batch runner.

    Every Memory is shown in full, and ``max_output_tokens`` caps the output one
    request asks for.
    """

    max_output_tokens: int = 32_768

    def __post_init__(self) -> None:
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")


MEMORY_RELATION_RULES = """Use these definitions strictly:
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

Before returning CONTRADICTS, prove that the claims concern the same subject and
an overlapping operational scope, including system, environment, repository,
project, time, and modality when those facts are present. State the overlap
explicitly: a universal rule includes its subsets, so an incompatible exception
within that subset can contradict the universal rule. REFINES requires compatible
assertions; a narrower scope alone cannot make incompatible values compatible.
Different document lineage alone does not prove different operational subjects.
Set
same_subject_and_scope=true only after that proof and state the two mutually
incompatible assertions in incompatible_assertions. If the claims concern
different systems, environments, templates, examples, time periods, or disjoint
scopes, return UNRELATED or REFINES as appropriate. For every non-CONTRADICTS decision,
incompatible_assertions must be an empty string.

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
"""

MEMORY_RELATION_PROMPT = "Classify the semantic relationship of every exact Memory pair.\n\n" + MEMORY_RELATION_RULES + """
<memory_pair_groups>
{groups_json}
</memory_pair_groups>

Return exactly one decision for every pair_index and no other pair_index.
"""


# Requested output per relation request: a response envelope plus one decision per pair.
_RELATION_OUTPUT_BASE_TOKENS = 512
_RELATION_OUTPUT_TOKENS_PER_PAIR = 768


def relation_output_tokens(policy: MemoryPairClassificationPolicy, pair_count: int) -> int:
    """The output a relation request asks for; the runner bounds it by the route."""

    return min(policy.max_output_tokens, _RELATION_OUTPUT_BASE_TOKENS + _RELATION_OUTPUT_TOKENS_PER_PAIR * pair_count)


async def _pair_outcomes(
    runner: LlmBatchRunner, task: ItemTask, *, pair_count: int, label: str,
) -> dict[str, tuple[Any, ...] | ItemFailure]:
    """Run the pair items; an error raised by the task itself becomes a classification error."""

    try:
        return await runner.run_items(task)
    except Exception as error:
        raise MemoryPairClassificationError(
            f"{label} failed: {error}", pair_count=pair_count,
            llm_calls=runner.stats.calls, prompt_chars=runner.stats.prompt_chars,
        ) from error


async def run_pair_items(runner: LlmBatchRunner, task: ItemTask, *, pair_count: int, label: str) -> list[Any]:
    """Return one result per pair, or raise for the first pair left without one."""

    results = []
    for outcome in (await _pair_outcomes(runner, task, pair_count=pair_count, label=label)).values():
        if isinstance(outcome, ItemFailure):
            raise _classification_error(outcome, pair_count=pair_count, stats=runner.stats)
        results.append(outcome[0])
    return results


async def judge_pair_items(
    runner: LlmBatchRunner, task: ItemTask, pairs: tuple[MemoryPair, ...], *, label: str,
) -> tuple[list[Any], tuple[UnjudgedPair, ...]]:
    """Return each judged pair's result and the pairs that cannot be judged even alone.

    A transient failure raises: sending the pair again may succeed.
    """

    results = []
    unjudged = []
    for item_id, outcome in (await _pair_outcomes(runner, task, pair_count=len(pairs), label=label)).items():
        if not isinstance(outcome, ItemFailure):
            results.append(outcome[0])
        elif outcome.unjudgeable:
            unjudged.append(UnjudgedPair(pairs[int(item_id)], outcome))
        else:
            raise _classification_error(outcome, pair_count=len(pairs), stats=runner.stats)
    return results, tuple(unjudged)


def _classification_error(
    failure: ItemFailure, *, pair_count: int, stats: BatchStats,
) -> MemoryPairClassificationError:
    """Report one unfinished relation item with the usage spent on the whole ledger."""

    error = failure.error
    detail = f": {error}" if error is not None else ""
    provider = error if isinstance(error, StructuredLlmError) else None
    return MemoryPairClassificationError(
        f"memory relation classification failed ({failure.error_code}){detail}",
        pair_count=pair_count,
        llm_calls=stats.calls,
        prompt_chars=stats.prompt_chars,
        terminal_category=provider.terminal_category if provider else None,
        error_code=failure.error_code,
    )


def _auditable_relation_reason(decision: Any) -> str:
    reason = str(getattr(decision, "reason", "") or "").strip()
    if str(decision.classification) != MemoryRelationType.CONTRADICTS.value:
        return reason
    incompatible = str(getattr(decision, "incompatible_assertions", "") or "").strip()
    scope_proof = bool(getattr(decision, "same_subject_and_scope", False))
    proof = f"same_subject_and_scope={str(scope_proof).lower()}; incompatible_assertions={incompatible}"
    return f"{reason} [{proof}]" if reason else proof


def _prompt_memory(memory: Memory) -> dict[str, object]:
    return {
        "id": memory.id,
        "content": memory.content,
        "type": memory.memory_type,
        "visibility": memory.visibility,
        "project_key": memory.project_key,
        "repo_identifier": memory.repo_identifier,
        "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
        "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
        "created_at": memory.created_at.isoformat() if memory.created_at else None,
        "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
    }


def _grouped_pair_payload(indexed_pairs: tuple[tuple[int, MemoryPair], ...]) -> str:
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for pair_index, pair in indexed_pairs:
        challenger_id = pair.challenger.id
        if challenger_id not in groups:
            groups[challenger_id] = {
                "challenger": _prompt_memory(pair.challenger),
                "candidates": [],
            }
            order.append(challenger_id)
        groups[challenger_id]["candidates"].append(
            {
                "pair_index": pair_index,
                "candidate": _prompt_memory(pair.candidate),
            }
        )
    return json.dumps([groups[challenger_id] for challenger_id in order], ensure_ascii=False)


class StructuredMemoryPairClassifier:
    """Classify exact pairs; a pair that cannot be judged even alone is returned as unjudged."""

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
        runner = LlmBatchRunner(self._client, model=self._model)

        def render(item_ids: tuple[str, ...], _context: tuple) -> LlmRequest:
            indexed_pairs = tuple((int(item_id), pairs[int(item_id)]) for item_id in item_ids)
            prompt = MEMORY_RELATION_PROMPT.format(groups_json=_grouped_pair_payload(indexed_pairs))
            return LlmRequest(prompt, MemoryRelationResponse, relation_output_tokens(self._policy, len(item_ids)))

        def decode(response: MemoryRelationResponse, _item_ids: tuple[str, ...], _context: tuple):
            """Each decision is validated alone by its own rule; the runner rejects an unrequested pair_index."""
            for decision in response.decisions:
                pair_index = int(decision.pair_index)
                if not 0 <= pair_index < len(pairs):
                    yield str(pair_index), RejectedRow(f"pair_index {pair_index} was not requested")
                elif (error := decision.row_error()) is not None:
                    yield str(pair_index), RejectedRow(f"pair_index {pair_index}: {error}")
                else:
                    yield str(pair_index), MemoryPairDecision(
                        pair=pairs[pair_index],
                        relation_type=MemoryRelationType(decision.classification),
                        direction=RelationDirection(decision.direction),
                        reason=_auditable_relation_reason(decision),
                    )

        decisions, unjudged = await judge_pair_items(runner, ItemTask(
            item_ids=tuple(str(index) for index in range(len(pairs))), render=render, decode=decode,
            call=self._client.classify_memory_relations, label=lambda item_id: f"pair_index {item_id}",
        ), pairs, label="memory relation classification")
        return MemoryPairClassification(
            decisions=tuple(decisions), llm_calls=runner.stats.calls, prompt_chars=runner.stats.prompt_chars,
            unjudged=unjudged,
        )
