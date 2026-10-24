"""Refinement verification: before one Candidate revises a kept old Memory, check the pair alone.

Sparse Relation proposes a revision with a proof that is one field among many
in a response judged across the whole catalog, and its "preserves the old
claim" answer tends to follow a more detailed Candidate. Verification is a
Decision asked of that one pair in its own request: does the Candidate state
every fact of the old claim? ``preserves`` lets the revision proceed; ``drops``,
the safe answer, counts as an ineligible proof, so the old Memory is kept and
the Candidate added. A pair that cannot be judged even alone leaves the
refinement uncertain. A transient failure raises like any lifecycle model call.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from memforge.llm.batch_runner import ItemTask, LlmBatchRunner, LlmRequest, RejectedRow
from memforge.llm.decision_model import DecisionTask, decision_task_model
from memforge.llm.structured import RefinementVerificationResponse
from memforge.memory.relation_classifier import MemoryPair, judge_pair_items
from memforge.models import Memory

REFINEMENT_VERIFICATION_CONTRACT = "refinement-verification-v1"
REFINEMENT_VERIFICATION_TASK = DecisionTask("refinement_verification", REFINEMENT_VERIFICATION_CONTRACT)

REFINEMENT_VERIFICATION_PROMPT = """A stored claim (old_claim) may be replaced by a new claim (new_claim). The current \
source still states old_claim, so a replacement must not lose any of it.

Answer "preserves" only when new_claim states every fact old_claim states: the same subject and population, \
every condition and exception, the same scope and environment, the same modality (a requirement stays a \
requirement, a plan stays a plan), the same attribution and effective time, the same polarity, and every value, \
name, identifier, reason and consequence old_claim gives. Additional detail in new_claim is fine. A reworded fact \
counts as stated when its meaning is unchanged.

Answer "drops" when any fact of old_claim is missing from new_claim, is narrowed to a subset, is kept only as an \
example, or is replaced by a different subject, scope or instance. Also answer "drops" when you are not sure.

<pair>
{pair_json}
</pair>

Return exactly one decision, for pair_index {pair_index}.
"""


@dataclass(frozen=True, slots=True)
class RefinementVerification:
    """Each verified pair by its answer, keyed by (Candidate index, old Memory id); unjudged pairs have none."""

    preserved: frozenset[tuple[int, str]]
    dropped: frozenset[tuple[int, str]]
    unjudged: frozenset[tuple[int, str]]
    prompt_chars: int


def _claim(memory: Memory) -> dict[str, object]:
    return {
        "text": memory.content,
        "type": memory.memory_type,
        "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
        "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
    }


async def verify_refinements(
    pairs: tuple[tuple[int, MemoryPair], ...], *, client: Any, model: str,
) -> RefinementVerification:
    """Ask each pair alone whether its Candidate (``pair.challenger``) states every fact of ``pair.candidate``.

    Each item is ``(candidate_index, MemoryPair(challenger=Candidate, candidate=old Memory))``.
    """

    task_model = decision_task_model(client, REFINEMENT_VERIFICATION_TASK, model)

    async def verify(pair: MemoryPair):
        runner = LlmBatchRunner(client, model=task_model)

        def render(_item_ids: tuple[str, ...], _context: tuple) -> LlmRequest:
            payload = {"pair_index": 0, "old_claim": _claim(pair.candidate), "new_claim": _claim(pair.challenger)}
            return LlmRequest(
                REFINEMENT_VERIFICATION_PROMPT.format(
                    pair_json=json.dumps(payload, ensure_ascii=False), pair_index=0,
                ),
                RefinementVerificationResponse,
            )

        def decode(response: RefinementVerificationResponse, _item_ids: tuple[str, ...], _context: tuple):
            for decision in response.decisions:
                if decision.pair_index != 0:
                    yield str(decision.pair_index), RejectedRow(f"pair_index {decision.pair_index} was not requested")
                else:
                    yield "0", decision.answer

        answers, unjudged = await judge_pair_items(runner, ItemTask(
            item_ids=("0",), render=render, decode=decode, call=client.verify_refinements,
            label=lambda item_id: f"pair_index {item_id}",
        ), (pair,), label="refinement verification")
        return answers, unjudged, runner.stats

    tasks = [asyncio.create_task(verify(pair)) for _, pair in pairs]
    try:
        outcomes = await asyncio.gather(*tasks)
    except BaseException:
        # A failed pair fails the revision: stop the pairs still asking, then raise its own error.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    preserved, dropped, unjudged = set(), set(), set()
    for (candidate_index, pair), (answers, pair_unjudged, _stats) in zip(pairs, outcomes, strict=True):
        key = (candidate_index, pair.candidate.id)
        if pair_unjudged:
            unjudged.add(key)
        elif answers == ["preserves"]:
            preserved.add(key)
        else:
            dropped.add(key)
    return RefinementVerification(
        preserved=frozenset(preserved),
        dropped=frozenset(dropped),
        unjudged=frozenset(unjudged),
        prompt_chars=sum(stats.prompt_chars for _, _, stats in outcomes),
    )
