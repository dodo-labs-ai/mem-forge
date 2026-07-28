"""Bounded uniqueness selection for one extracted Source Unit revision."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Sequence

from memforge.llm.structured import CandidateLedgerDecision
from memforge.models import RawMemory

__all__ = [
    "CandidateLedgerError",
    "CandidateLedgerDrop",
    "CandidateLedgerResult",
    "select_unique_memory_candidates",
]


_CANDIDATE_LEDGER_PROMPT = """Select the non-redundant durable Memory candidates in one bounded
admission batch extracted from a Source Unit revision.

Return exactly one judgment in every named field in <decision_slots>:
- when a field maps to a candidate index, return that candidate's judgment in the field;
- when a field maps to null, return null;
- do not return the candidate index inside the judgment. The datastore owns candidate identity.

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

<decision_slots>
{decision_slots_json}
</decision_slots>

Return only the fixed-slot JSON object required by the response schema."""

_VALIDATION_ATTEMPTS = 2
_DEFAULT_MAX_CONTEXT_CHARS = 100_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_CANDIDATE_LEDGER_DECISION_BATCH_SIZE = 24


@dataclass(frozen=True)
class CandidateLedgerDrop:
    """Transient audit detail for one candidate removed by admission."""

    candidate: RawMemory
    canonical_candidate: RawMemory | None
    method: Literal["exact_content", "structured_ledger", "structured_quality"]
    reason: str


@dataclass(frozen=True)
class _IndexedLedgerDecision:
    """Datastore-bound judgment after a response slot receives its index."""

    index: int
    action: Literal["KEEP", "DROP_REDUNDANT", "DROP_LOW_VALUE"]
    canonical_index: int | None
    reason: str


@dataclass(frozen=True)
class CandidateLedgerResult:
    """Selected original candidates and bounded ledger accounting."""

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
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> CandidateLedgerResult:
    """Return original candidates selected by bounded admission batches."""

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

    decisions_by_index: dict[int, _IndexedLedgerDecision] = {}
    structured_llm_calls = 0
    structured_llm_elapsed_ms = 0
    validation_retries = 0
    fallback_batch_count = 0
    fallback_candidate_count = 0
    prompt_chars = 0
    offset = 0
    while offset < semantic_count:
        stop = min(
            semantic_count,
            offset + _CANDIDATE_LEDGER_DECISION_BATCH_SIZE,
        )
        prompt: str | None = None
        expected_indices: tuple[int, ...] = ()
        while stop > offset:
            expected_indices = tuple(range(offset, stop))
            payload = [
                {
                    "index": index,
                    "memory_type": ordered_candidates[index].memory_type,
                    "content": ordered_candidates[index].content,
                    "source_observation_id": ordered_candidates[index].source_observation_id,
                }
                for index in expected_indices
            ]
            slot_map = {
                f"slot_{slot_index:02d}": (expected_indices[slot_index] if slot_index < len(expected_indices) else None)
                for slot_index in range(_CANDIDATE_LEDGER_DECISION_BATCH_SIZE)
            }
            candidate_prompt = _CANDIDATE_LEDGER_PROMPT.format(
                candidates_json=json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                decision_slots_json=json.dumps(slot_map, separators=(",", ":")),
            )
            if len(candidate_prompt) <= max_context_chars:
                prompt = candidate_prompt
                break
            stop -= 1
        if prompt is None:
            raise CandidateLedgerError(
                "budget_exceeded",
                (f"one candidate ledger request exceeds the context budget of {max_context_chars} chars"),
                input_count=len(original),
                semantic_input_count=semantic_count,
                structured_llm_calls=structured_llm_calls,
                structured_llm_elapsed_ms=structured_llm_elapsed_ms,
                validation_retries=validation_retries,
                prompt_chars=prompt_chars,
            )
        validation_error: ValueError | None = None
        batch_decisions: dict[int, _IndexedLedgerDecision] | None = None
        for attempt in range(_VALIDATION_ATTEMPTS):
            prompt_chars += len(prompt)
            call_started = perf_counter()
            try:
                structured_llm_calls += 1
                response = await selector(
                    prompt,
                    max_tokens=_DEFAULT_MAX_OUTPUT_TOKENS,
                    model=llm_model,
                )
            except Exception:
                structured_llm_elapsed_ms += max(0, round((perf_counter() - call_started) * 1000))
                batch_decisions = {
                    index: _IndexedLedgerDecision(
                        index=index,
                        action="KEEP",
                        canonical_index=None,
                        reason="structured_admission_unavailable",
                    )
                    for index in expected_indices
                }
                fallback_batch_count += 1
                fallback_candidate_count += len(expected_indices)
                break
            structured_llm_elapsed_ms += max(0, round((perf_counter() - call_started) * 1000))
            try:
                batch_decisions = _validate_ledger_batch(
                    response.ordered_slots(),
                    expected_indices=expected_indices,
                )
                break
            except ValueError as exc:
                validation_error = exc
                if attempt + 1 >= _VALIDATION_ATTEMPTS:
                    break
                validation_retries += 1
                prompt = (
                    f"{prompt}\n\n<validation_feedback>\n"
                    f"The previous response was rejected: {exc}. Return exactly one valid "
                    "judgment in every mapped slot and null in every unused slot.\n"
                    "</validation_feedback>"
                )
        if batch_decisions is None:
            raise CandidateLedgerError(
                "invalid_ledger",
                f"complete candidate ledger validation failed: {validation_error}",
                input_count=len(original),
                semantic_input_count=semantic_count,
                structured_llm_calls=structured_llm_calls,
                structured_llm_elapsed_ms=structured_llm_elapsed_ms,
                validation_retries=validation_retries,
                prompt_chars=prompt_chars,
            )
        decisions_by_index.update(batch_decisions)
        offset = stop

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
        structured_llm_calls=structured_llm_calls,
        structured_llm_elapsed_ms=structured_llm_elapsed_ms,
        validation_retries=validation_retries,
        fallback_batch_count=fallback_batch_count,
        fallback_candidate_count=fallback_candidate_count,
        prompt_chars=prompt_chars,
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


def _validate_ledger_batch(
    slots: Sequence[CandidateLedgerDecision | None],
    *,
    expected_indices: tuple[int, ...],
) -> dict[int, _IndexedLedgerDecision]:
    if len(slots) != _CANDIDATE_LEDGER_DECISION_BATCH_SIZE:
        raise ValueError("candidate ledger response does not contain the fixed protocol slots")
    by_index: dict[int, _IndexedLedgerDecision] = {}
    visible_indices = set(expected_indices)
    for slot_index, judgment in enumerate(slots):
        if slot_index >= len(expected_indices):
            if judgment is not None:
                raise ValueError(f"unused decision slot {slot_index} must be null")
            continue
        index = expected_indices[slot_index]
        if judgment is None:
            raise ValueError(f"decision slot {slot_index} for candidate index {index} must not be null")
        if judgment.action == "DROP_REDUNDANT" and (
            judgment.canonical_index is None or judgment.canonical_index >= index
        ):
            raise ValueError(f"DROP_REDUNDANT index {index} must target a lower index")
        if judgment.action == "DROP_REDUNDANT" and judgment.canonical_index not in visible_indices:
            raise ValueError(f"DROP_REDUNDANT index {index} must target a visible lower index")
        by_index[index] = _IndexedLedgerDecision(
            index=index,
            action=judgment.action,
            canonical_index=(
                judgment.canonical_index
                if judgment.action == "DROP_REDUNDANT"
                else None
            ),
            reason=judgment.reason,
        )
    return by_index


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
