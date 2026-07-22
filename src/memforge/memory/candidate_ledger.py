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

Return exactly one decision for every candidate index:
- KEEP when the candidate has any material truth condition not fully captured by another kept candidate.
- DROP_REDUNDANT only when one kept candidate fully entails this candidate. Set canonical_index to that KEEP.

Prefer the most specific self-contained candidate as canonical. Different wording, evidence events, or
Observation ids do not make claims distinct. Keep candidates that only partially overlap, add a condition,
record a different outcome, or preserve a distinct durable fact. Do not rewrite or merge candidate content.

<candidates>
{candidates_json}
</candidates>

Return only a JSON object with a decisions array."""

_VALIDATION_ATTEMPTS = 2
_DEFAULT_MAX_CANDIDATES = 200
_DEFAULT_MAX_CONTEXT_CHARS = 100_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8192


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
            (
                f"candidate count {semantic_count} exceeds the complete-ledger "
                f"budget of {max_candidates}"
            ),
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

    payload = [
        {
            "index": index,
            "memory_type": candidate.memory_type,
            "content": candidate.content,
            "source_observation_id": candidate.source_observation_id,
        }
        for index, candidate in enumerate(exact_unique)
    ]
    prompt = _CANDIDATE_LEDGER_PROMPT.format(
        candidates_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if len(prompt) > max_context_chars:
        raise CandidateLedgerError(
            "budget_exceeded",
            (
                f"candidate ledger context {len(prompt)} chars exceeds the "
                f"budget of {max_context_chars}"
            ),
            input_count=len(original),
            semantic_input_count=semantic_count,
        )

    selector = getattr(structured_llm_client, "select_memory_candidates", None)
    if selector is None:
        raise CandidateLedgerError(
            "structured_client_unavailable",
            "complete candidate ledger requires a structured LLM client",
            input_count=len(original),
            semantic_input_count=semantic_count,
        )

    decisions_by_index: dict[int, CandidateLedgerDecision] | None = None
    validation_error: ValueError | None = None
    structured_llm_calls = 0
    structured_llm_elapsed_ms = 0
    validation_retries = 0
    prompt_chars = 0
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
            structured_llm_elapsed_ms += max(
                0, round((perf_counter() - call_started) * 1000)
            )
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
        structured_llm_elapsed_ms += max(
            0, round((perf_counter() - call_started) * 1000)
        )
        try:
            decisions_by_index = _validate_complete_ledger(
                response.decisions,
                candidate_count=semantic_count,
            )
            break
        except ValueError as exc:
            validation_error = exc
            if attempt + 1 >= _VALIDATION_ATTEMPTS:
                break
            validation_retries += 1
            prompt = (
                f"{prompt}\n\n<validation_feedback>\n"
                f"The previous response was rejected: {exc}. Return a complete candidate ledger "
                "with one valid decision per index.\n"
                "</validation_feedback>"
            )

    if decisions_by_index is None:
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

    selected = tuple(
        candidate
        for index, candidate in enumerate(exact_unique)
        if decisions_by_index[index].action == "KEEP"
    )
    semantic_drops = tuple(
        CandidateLedgerDrop(
            candidate=exact_unique[index],
            canonical_candidate=exact_unique[decision.canonical_index],
            method="structured_ledger",
            reason=decision.reason,
        )
        for index, decision in decisions_by_index.items()
        if decision.action == "DROP_REDUNDANT"
        and decision.canonical_index is not None
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

    kept_indices = {
        index for index, decision in by_index.items() if decision.action == "KEEP"
    }
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
            raise ValueError(
                f"DROP_REDUNDANT index {index} must target a KEEP decision"
            )

    return by_index
