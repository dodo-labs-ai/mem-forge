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


_CANDIDATE_LEDGER_PROMPT = """Select the non-redundant durable Memory candidates extracted from one
Source Unit revision.

Return exactly one decision for every index listed in <decision_indices>:
- KEEP when the candidate has any material truth condition not fully captured by another kept candidate.
- DROP_REDUNDANT only when a lower-index candidate fully entails this candidate. Set canonical_index to
  that lower index. Lower indices are deterministic canonical precedence; never point forward.

Candidates are ordered by deterministic specificity precedence. Different wording, evidence events, or
Observation ids do not make claims distinct. Keep candidates that only partially overlap, add a condition,
record a different outcome, or preserve a distinct durable fact. Do not rewrite or merge candidate content.

<candidates>
{candidates_json}
</candidates>

<decision_indices>
{decision_indices_json}
</decision_indices>

Return only a JSON object with a decisions array."""

_VALIDATION_ATTEMPTS = 2
_DEFAULT_MAX_CANDIDATES = 200
_DEFAULT_MAX_CONTEXT_CHARS = 100_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8192
_CANDIDATE_LEDGER_DECISION_BATCH_SIZE = 24


@dataclass(frozen=True)
class CandidateLedgerDrop:
    """Transient audit detail for one candidate removed as redundant."""

    candidate: RawMemory
    canonical_candidate: RawMemory
    method: Literal["exact_content", "structured_ledger"]
    reason: str


@dataclass(frozen=True)
class CandidateLedgerResult:
    """Selected original candidates and bounded ledger accounting."""

    candidates: tuple[RawMemory, ...]
    input_count: int
    semantic_input_count: int
    dropped_exact_count: int
    dropped_redundant_count: int
    structured_llm_calls: int
    structured_llm_elapsed_ms: int
    validation_retries: int
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
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
) -> CandidateLedgerResult:
    """Return original candidates selected by a complete, bounded ledger."""

    original = tuple(candidates)
    exact_unique, exact_drops = _collapse_exact_duplicates(original)
    dropped_exact_count = len(exact_drops)
    semantic_count = len(exact_unique)

    if semantic_count > max_candidates:
        raise CandidateLedgerError(
            "budget_exceeded",
            (f"candidate count {semantic_count} exceeds the complete-ledger budget of {max_candidates}"),
            input_count=len(original),
            semantic_input_count=semantic_count,
        )
    if semantic_count <= 1:
        return CandidateLedgerResult(
            candidates=exact_unique,
            input_count=len(original),
            semantic_input_count=semantic_count,
            dropped_exact_count=dropped_exact_count,
            dropped_redundant_count=0,
            structured_llm_calls=0,
            structured_llm_elapsed_ms=0,
            validation_retries=0,
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
    payload = [
        {
            "index": index,
            "memory_type": candidate.memory_type,
            "content": candidate.content,
            "source_observation_id": candidate.source_observation_id,
        }
        for index, candidate in enumerate(ordered_candidates)
    ]
    candidates_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    selector = getattr(structured_llm_client, "select_memory_candidates", None)
    if selector is None:
        raise CandidateLedgerError(
            "structured_client_unavailable",
            "complete candidate ledger requires a structured LLM client",
            input_count=len(original),
            semantic_input_count=semantic_count,
        )

    decisions_by_index: dict[int, CandidateLedgerDecision] = {}
    structured_llm_calls = 0
    structured_llm_elapsed_ms = 0
    validation_retries = 0
    prompt_chars = 0
    for offset in range(
        0,
        semantic_count,
        _CANDIDATE_LEDGER_DECISION_BATCH_SIZE,
    ):
        expected_indices = tuple(
            range(
                offset,
                min(
                    semantic_count,
                    offset + _CANDIDATE_LEDGER_DECISION_BATCH_SIZE,
                ),
            )
        )
        prompt = _CANDIDATE_LEDGER_PROMPT.format(
            candidates_json=candidates_json,
            decision_indices_json=json.dumps(expected_indices),
        )
        if len(prompt) > max_context_chars:
            raise CandidateLedgerError(
                "budget_exceeded",
                (f"candidate ledger context {len(prompt)} chars exceeds the budget of {max_context_chars}"),
                input_count=len(original),
                semantic_input_count=semantic_count,
            )
        validation_error: ValueError | None = None
        batch_decisions: dict[int, CandidateLedgerDecision] | None = None
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
            except Exception as exc:
                structured_llm_elapsed_ms += max(0, round((perf_counter() - call_started) * 1000))
                raise CandidateLedgerError(
                    "structured_llm_error",
                    f"candidate ledger structured call failed: {exc}",
                    input_count=len(original),
                    semantic_input_count=semantic_count,
                    structured_llm_calls=structured_llm_calls,
                    structured_llm_elapsed_ms=structured_llm_elapsed_ms,
                    validation_retries=validation_retries,
                    prompt_chars=prompt_chars,
                ) from exc
            structured_llm_elapsed_ms += max(0, round((perf_counter() - call_started) * 1000))
            try:
                batch_decisions = _validate_ledger_batch(
                    response.decisions,
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
                    "decision for every requested index.\n"
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
    semantic_drops = tuple(
        CandidateLedgerDrop(
            candidate=ordered_candidates[index],
            canonical_candidate=ordered_candidates[decision.canonical_index],
            method="structured_ledger",
            reason=decision.reason,
        )
        for index, decision in decisions_by_index.items()
        if decision.action == "DROP_REDUNDANT" and decision.canonical_index is not None
    )
    return CandidateLedgerResult(
        candidates=selected,
        input_count=len(original),
        semantic_input_count=semantic_count,
        dropped_exact_count=dropped_exact_count,
        dropped_redundant_count=semantic_count - len(selected),
        structured_llm_calls=structured_llm_calls,
        structured_llm_elapsed_ms=structured_llm_elapsed_ms,
        validation_retries=validation_retries,
        prompt_chars=prompt_chars,
        drops=exact_drops + semantic_drops,
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
    decisions: Sequence[CandidateLedgerDecision],
    *,
    candidate_count: int,
) -> dict[int, CandidateLedgerDecision]:
    by_index: dict[int, CandidateLedgerDecision] = {}
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
    if not kept_indices:
        raise ValueError("at least one candidate must be kept")

    for index, decision in by_index.items():
        canonical_index = decision.canonical_index
        if decision.action == "KEEP":
            if canonical_index is not None:
                raise ValueError(f"KEEP index {index} must not name a canonical index")
            continue
        if canonical_index is None:
            raise ValueError(f"DROP_REDUNDANT index {index} requires canonical_index")
        if canonical_index == index:
            raise ValueError(f"candidate index {index} cannot be canonical for itself")
        if canonical_index not in kept_indices:
            raise ValueError(f"DROP_REDUNDANT index {index} must target a KEEP decision")

    return by_index


def _validate_ledger_batch(
    decisions: Sequence[CandidateLedgerDecision],
    *,
    expected_indices: tuple[int, ...],
) -> dict[int, CandidateLedgerDecision]:
    expected = set(expected_indices)
    by_index: dict[int, CandidateLedgerDecision] = {}
    for decision in decisions:
        if decision.index in by_index:
            raise ValueError(f"duplicate decision for candidate index {decision.index}")
        if decision.index not in expected:
            raise ValueError(f"unrequested candidate index {decision.index}")
        if decision.action == "DROP_REDUNDANT" and (
            decision.canonical_index is None or decision.canonical_index >= decision.index
        ):
            raise ValueError(f"DROP_REDUNDANT index {decision.index} must target a lower index")
        by_index[decision.index] = decision
    missing = sorted(expected - set(by_index))
    if missing:
        raise ValueError(f"missing candidate indices {missing}")
    return by_index


def _normalize_ledger_canonicals(
    decisions: dict[int, CandidateLedgerDecision],
) -> dict[int, CandidateLedgerDecision]:
    normalized = dict(decisions)
    for index, decision in decisions.items():
        if decision.action == "KEEP":
            continue
        canonical_index = decision.canonical_index
        while canonical_index is not None:
            canonical = decisions[canonical_index]
            if canonical.action == "KEEP":
                break
            canonical_index = canonical.canonical_index
        normalized[index] = decision.model_copy(update={"canonical_index": canonical_index})
    return normalized
