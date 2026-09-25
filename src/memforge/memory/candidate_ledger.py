"""Uniqueness selection for one extracted Source Unit revision."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Sequence

from memforge.llm.batch_runner import OUTPUT_INVALID, ItemFailure, ItemTask, LlmBatchRunner, LlmRequest
from memforge.llm.structured import CandidateLedgerResponse
from memforge.models import RawMemory

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateLedgerError",
    "CandidateLedgerDrop",
    "CandidateLedgerResult",
    "select_unique_memory_candidates",
]


_CANDIDATE_LEDGER_PROMPT = """Select the non-redundant durable Memory candidates in one
admission batch extracted from a Source Unit revision.

Return a decisions array with exactly one judgment per candidate. Set each
judgment's candidate_index to the index of the candidate it judges.

For each mapped candidate:
- KEEP when the candidate has any material truth condition not fully captured by another visible kept
  candidate, or when durable value is uncertain, partially overlapping, or conflicting.
- DROP_REDUNDANT only when a visible lower-index candidate fully entails this candidate. Set
  canonical_index to that visible lower index. Lower indices are deterministic canonical precedence;
  never point forward or outside this batch.
- DROP_LOW_VALUE only when the candidate is merely instance output or source-recoverable detail and
  preserves no reusable decision, rule, invariant, conclusion, or procedure. Do not set canonical_index.

Candidates are ordered by deterministic specificity precedence. Different wording, evidence events, or
Observation ids do not make claims distinct. Keep candidates that only partially overlap, add a condition,
record a different outcome, or preserve a distinct durable fact. Do not rewrite or merge candidate content.

<candidates>
{candidates_json}
</candidates>

Return only the decisions object required by the response schema."""

# Requested output: one decision with a reason of up to 1000 characters per
# candidate, with a floor for the envelope. The runner bounds it by the route.
_LEDGER_DECISION_OUTPUT_TOKENS = 320
_LEDGER_MIN_OUTPUT_TOKENS = 1024


@dataclass(frozen=True)
class CandidateLedgerDrop:
    """Transient audit detail for one candidate removed by admission."""

    candidate: RawMemory
    canonical_candidate: RawMemory | None
    method: Literal["exact_content", "structured_ledger", "structured_quality"]
    reason: str


@dataclass(frozen=True)
class _IndexedLedgerDecision:
    """One validated judgment for the candidate at ``index``."""

    index: int
    action: Literal["KEEP", "DROP_REDUNDANT", "DROP_LOW_VALUE"]
    canonical_index: int | None
    reason: str


@dataclass(frozen=True)
class CandidateLedgerResult:
    """Selected original candidates and ledger accounting."""

    candidates: tuple[RawMemory, ...]
    input_count: int
    semantic_input_count: int
    dropped_exact_count: int
    dropped_redundant_count: int
    dropped_low_value_count: int
    structured_llm_calls: int
    structured_llm_elapsed_ms: int
    validation_retries: int
    fallback_batch_count: int
    fallback_candidate_count: int
    prompt_chars: int
    drops: tuple[CandidateLedgerDrop, ...]


class CandidateLedgerError(RuntimeError):
    """A uniqueness ledger could not safely authorize candidate persistence."""

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        input_count: int,
        semantic_input_count: int,
        structured_llm_calls: int = 0,
        structured_llm_elapsed_ms: int = 0,
        validation_retries: int = 0,
        prompt_chars: int = 0,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.input_count = input_count
        self.semantic_input_count = semantic_input_count
        self.structured_llm_calls = structured_llm_calls
        self.structured_llm_elapsed_ms = structured_llm_elapsed_ms
        self.validation_retries = validation_retries
        self.prompt_chars = prompt_chars


async def select_unique_memory_candidates(
    candidates: Sequence[RawMemory],
    *,
    structured_llm_client,
    llm_model: str | None,
) -> CandidateLedgerResult:
    """Return original candidates selected by capacity-packed admission requests."""

    original = tuple(candidates)
    exact_unique, exact_drops = _collapse_exact_duplicates(original)
    dropped_exact_count = len(exact_drops)
    semantic_count = len(exact_unique)

    if semantic_count <= 1:
        return CandidateLedgerResult(
            candidates=exact_unique,
            input_count=len(original),
            semantic_input_count=semantic_count,
            dropped_exact_count=dropped_exact_count,
            dropped_redundant_count=0,
            dropped_low_value_count=0,
            structured_llm_calls=0,
            structured_llm_elapsed_ms=0,
            validation_retries=0,
            fallback_batch_count=0,
            fallback_candidate_count=0,
            prompt_chars=0,
            drops=exact_drops,
        )

    ordered_candidates = tuple(
        candidate
        for _, candidate in sorted(
            enumerate(exact_unique),
            key=lambda item: (
                -len(re.sub(r"\s+", " ", item[1].content.strip())),
                item[1].memory_type,
                item[1].content,
                item[0],
            ),
        )
    )
    selector = getattr(structured_llm_client, "select_memory_candidates", None)
    if selector is None:
        raise CandidateLedgerError(
            "structured_client_unavailable",
            "complete candidate ledger requires a structured LLM client",
            input_count=len(original),
            semantic_input_count=semantic_count,
        )

    def render(item_ids: tuple[str, ...], _context: tuple) -> LlmRequest:
        payload = [
            {
                "index": index,
                "memory_type": ordered_candidates[index].memory_type,
                "content": ordered_candidates[index].content,
                "source_observation_id": ordered_candidates[index].source_observation_id,
            }
            for index in map(int, item_ids)
        ]
        prompt = _CANDIDATE_LEDGER_PROMPT.format(
            candidates_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
        max_tokens = max(_LEDGER_MIN_OUTPUT_TOKENS, _LEDGER_DECISION_OUTPUT_TOKENS * len(item_ids))
        return LlmRequest(prompt, CandidateLedgerResponse, max_tokens)

    def decode(response: CandidateLedgerResponse, item_ids: tuple[str, ...], _context: tuple):
        visible = frozenset(map(int, item_ids))
        return ((str(decision.index), decision) for decision in _ledger_decisions(response, visible))

    runner = LlmBatchRunner(structured_llm_client, model=llm_model)
    indices = tuple(range(semantic_count))
    item_ids = tuple(map(str, indices))
    started = perf_counter()
    try:
        outcomes = await runner.run_items(ItemTask(item_ids=item_ids, render=render, decode=decode, call=selector))
        fallback_batch_count = runner.stats.failed_requests
    except ValueError as error:
        # Admission is best effort: a route whose request budget cannot be
        # resolved admits every candidate.
        logger.warning("candidate_ledger_unavailable candidates=%d", semantic_count, exc_info=True)
        unavailable = ItemFailure("provider_error", type(error).__name__, error)
        outcomes = dict.fromkeys(item_ids, unavailable)
        fallback_batch_count = 1
    structured_llm_elapsed_ms = max(0, round((perf_counter() - started) * 1000))

    decisions_by_index: dict[int, _IndexedLedgerDecision] = {}
    fallback_candidate_count = 0
    for index in indices:
        outcome = outcomes[str(index)]
        if isinstance(outcome, ItemFailure) and outcome.category == "capacity_exceeded":
            raise CandidateLedgerError(
                "budget_exceeded",
                "one candidate ledger request exceeds the route's input capacity",
                input_count=len(original),
                semantic_input_count=semantic_count,
                structured_llm_calls=runner.stats.calls,
                structured_llm_elapsed_ms=structured_llm_elapsed_ms,
                validation_retries=runner.stats.corrections,
                prompt_chars=runner.stats.prompt_chars,
            )
        if isinstance(outcome, ItemFailure):
            fallback_candidate_count += 1
            decisions_by_index[index] = _IndexedLedgerDecision(
                index=index, action="KEEP", canonical_index=None, reason=_fallback_reason(outcome),
            )
        else:
            decisions_by_index[index] = outcome[0]

    decisions_by_index = _normalize_ledger_canonicals(decisions_by_index)
    _validate_complete_ledger(
        tuple(decisions_by_index.values()),
        candidate_count=semantic_count,
    )

    selected_ids = {
        id(candidate)
        for index, candidate in enumerate(ordered_candidates)
        if decisions_by_index[index].action == "KEEP"
    }
    selected = tuple(candidate for candidate in exact_unique if id(candidate) in selected_ids)
    redundant_drops = tuple(
        CandidateLedgerDrop(
            candidate=ordered_candidates[index],
            canonical_candidate=ordered_candidates[decision.canonical_index],
            method="structured_ledger",
            reason=decision.reason,
        )
        for index, decision in decisions_by_index.items()
        if decision.action == "DROP_REDUNDANT" and decision.canonical_index is not None
    )
    low_value_drops = tuple(
        CandidateLedgerDrop(
            candidate=ordered_candidates[index],
            canonical_candidate=None,
            method="structured_quality",
            reason="low_value_admission",
        )
        for index, decision in decisions_by_index.items()
        if decision.action == "DROP_LOW_VALUE"
    )
    dropped_redundant_count = sum(decision.action == "DROP_REDUNDANT" for decision in decisions_by_index.values())
    dropped_low_value_count = sum(decision.action == "DROP_LOW_VALUE" for decision in decisions_by_index.values())
    return CandidateLedgerResult(
        candidates=selected,
        input_count=len(original),
        semantic_input_count=semantic_count,
        dropped_exact_count=dropped_exact_count,
        dropped_redundant_count=dropped_redundant_count,
        dropped_low_value_count=dropped_low_value_count,
        structured_llm_calls=runner.stats.calls,
        structured_llm_elapsed_ms=structured_llm_elapsed_ms,
        validation_retries=runner.stats.corrections,
        fallback_batch_count=fallback_batch_count,
        fallback_candidate_count=fallback_candidate_count,
        prompt_chars=runner.stats.prompt_chars,
        drops=exact_drops + redundant_drops + low_value_drops,
    )


def _collapse_exact_duplicates(
    candidates: tuple[RawMemory, ...],
) -> tuple[tuple[RawMemory, ...], tuple[CandidateLedgerDrop, ...]]:
    canonical_by_content: dict[str, RawMemory] = {}
    unique: list[RawMemory] = []
    drops: list[CandidateLedgerDrop] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate.content.strip())
        canonical = canonical_by_content.get(normalized)
        if canonical is not None:
            drops.append(
                CandidateLedgerDrop(
                    candidate=candidate,
                    canonical_candidate=canonical,
                    method="exact_content",
                    reason="normalized content is identical",
                )
            )
            continue
        canonical_by_content[normalized] = candidate
        unique.append(candidate)
    return tuple(unique), tuple(drops)


def _fallback_reason(failure: ItemFailure) -> str:
    """Why a candidate was admitted without a ledger judgment."""

    if failure.error_code == OUTPUT_INVALID:
        return "structured_admission_invalid"
    return "structured_admission_unavailable"


def _validate_complete_ledger(
    decisions: Sequence[_IndexedLedgerDecision],
    *,
    candidate_count: int,
) -> dict[int, _IndexedLedgerDecision]:
    by_index: dict[int, _IndexedLedgerDecision] = {}
    for decision in decisions:
        index = decision.index
        if index in by_index:
            raise ValueError(f"duplicate decision for candidate index {index}")
        if index >= candidate_count:
            raise ValueError(f"unknown candidate index {index}")
        by_index[index] = decision

    expected = set(range(candidate_count))
    missing = sorted(expected - set(by_index))
    if missing:
        raise ValueError(f"missing candidate indices {missing}")

    kept_indices = {index for index, decision in by_index.items() if decision.action == "KEEP"}

    for index, decision in by_index.items():
        canonical_index = decision.canonical_index
        if decision.action in {"KEEP", "DROP_LOW_VALUE"}:
            if canonical_index is not None:
                raise ValueError(f"{decision.action} index {index} must not name a canonical index")
            continue
        if canonical_index is None:
            raise ValueError(f"DROP_REDUNDANT index {index} requires canonical_index")
        if canonical_index == index:
            raise ValueError(f"candidate index {index} cannot be canonical for itself")
        if canonical_index not in kept_indices:
            raise ValueError(f"DROP_REDUNDANT index {index} must target a KEEP decision")

    return by_index


def _ledger_decisions(response: CandidateLedgerResponse, visible: frozenset[int]):
    """Yield each judgment; a redundant drop must name a lower index visible in its request."""

    for judgment in response.decisions:
        index = judgment.candidate_index
        redundant = judgment.action == "DROP_REDUNDANT"
        if redundant and (judgment.canonical_index is None or judgment.canonical_index >= index):
            raise ValueError(f"DROP_REDUNDANT index {index} must target a lower index")
        if redundant and judgment.canonical_index not in visible:
            raise ValueError(f"DROP_REDUNDANT index {index} must target a visible lower index")
        yield _IndexedLedgerDecision(
            index=index,
            action=judgment.action,
            canonical_index=judgment.canonical_index if redundant else None,
            reason=judgment.reason,
        )


def _normalize_ledger_canonicals(
    decisions: dict[int, _IndexedLedgerDecision],
) -> dict[int, _IndexedLedgerDecision]:
    normalized = dict(decisions)
    for index, decision in decisions.items():
        if decision.action != "DROP_REDUNDANT":
            continue
        canonical_index = decision.canonical_index
        while canonical_index is not None:
            canonical = decisions[canonical_index]
            if canonical.action == "KEEP":
                break
            canonical_index = canonical.canonical_index
        normalized[index] = _IndexedLedgerDecision(
            index=index,
            action=decision.action,
            canonical_index=canonical_index,
            reason=decision.reason,
        )
    return normalized
