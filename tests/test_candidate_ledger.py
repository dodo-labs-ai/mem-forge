from __future__ import annotations

import pytest

from memforge.llm.structured import CandidateLedgerDecision, CandidateLedgerResponse
from memforge.memory.candidate_ledger import (
    CandidateLedgerError,
    select_unique_memory_candidates,
)
from memforge.models import RawMemory


def _candidate(
    content: str,
    *,
    observation_id: str,
    memory_type: str = "fact",
) -> RawMemory:
    return RawMemory(
        content=content,
        memory_type=memory_type,
        confidence=0.9,
        source_observation_id=observation_id,
        evidence_quote=content,
    )


def _ledger_response(
    *decisions: CandidateLedgerDecision | None,
) -> CandidateLedgerResponse:
    return CandidateLedgerResponse(
        **{f"slot_{index:02d}": (decisions[index] if index < len(decisions) else None) for index in range(24)}
    )


class _LedgerClient:
    def __init__(self, *responses: CandidateLedgerResponse) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def select_memory_candidates(self, prompt: str, **kwargs) -> CandidateLedgerResponse:
        del kwargs
        self.prompts.append(prompt)
        return self.responses.pop(0)


class _IntermittentLedgerClient:
    def __init__(
        self,
        *outcomes: CandidateLedgerResponse | Exception,
    ) -> None:
        self.outcomes = list(outcomes)
        self.prompts: list[str] = []

    async def select_memory_candidates(self, prompt: str, **kwargs) -> CandidateLedgerResponse:
        del kwargs
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_candidate_ledger_retries_once_when_decision_coverage_is_incomplete():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    second = _candidate("The trigger was not processed.", observation_id="obs-2")
    client = _LedgerClient(
        _ledger_response(CandidateLedgerDecision(action="KEEP")),
        _ledger_response(
            CandidateLedgerDecision(action="KEEP"),
            CandidateLedgerDecision(action="KEEP"),
        ),
    )

    result = await select_unique_memory_candidates(
        [first, second],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (first, second)
    assert len(client.prompts) == 2
    assert "<validation_feedback>" in client.prompts[1]
    assert result.structured_llm_calls == 2
    assert result.validation_retries == 1
    assert result.prompt_chars == sum(len(prompt) for prompt in client.prompts)
    assert result.structured_llm_elapsed_ms >= 0


@pytest.mark.asyncio
async def test_candidate_ledger_keeps_batch_after_second_invalid_ledger():
    candidates = [
        _candidate("The trigger remained OPEN.", observation_id="obs-1"),
        _candidate("The trigger was not processed.", observation_id="obs-2"),
    ]
    incomplete = _ledger_response(CandidateLedgerDecision(action="KEEP"))
    client = _LedgerClient(incomplete, incomplete)

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 2
    assert result.validation_retries == 1
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 2
    assert result.prompt_chars == sum(len(prompt) for prompt in client.prompts)
    assert len(client.prompts) == 2


@pytest.mark.asyncio
async def test_candidate_ledger_collapses_exact_duplicates_without_an_llm_call():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    duplicate = _candidate("  The   trigger remained OPEN. ", observation_id="obs-2")
    client = _LedgerClient()

    result = await select_unique_memory_candidates(
        [first, duplicate],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (first,)
    assert result.dropped_exact_count == 1
    assert result.dropped_redundant_count == 0
    assert result.structured_llm_calls == 0
    assert client.prompts == []


@pytest.mark.asyncio
async def test_candidate_ledger_does_not_exact_collapse_case_sensitive_identifiers():
    upper = _candidate("Read configuration from FOO.", observation_id="obs-1")
    lower = _candidate("Read configuration from foo.", observation_id="obs-2")

    with pytest.raises(CandidateLedgerError) as exc_info:
        await select_unique_memory_candidates(
            [upper, lower],
            structured_llm_client=None,
            llm_model=None,
        )

    assert exc_info.value.error_type == "structured_client_unavailable"


@pytest.mark.asyncio
async def test_candidate_ledger_bounds_calls_independently_from_source_unit_cardinality():
    candidates = [
        _candidate(
            f"Durable candidate {index:03d} with distinct content.",
            observation_id=f"obs-{index}",
        )
        for index in range(205)
    ]
    client = _LedgerClient(
        *(
            _ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in range(start, stop)))
            for start, stop in (
                (0, 24),
                (24, 48),
                (48, 72),
                (72, 96),
                (96, 120),
                (120, 144),
                (144, 168),
                (168, 192),
                (192, 205),
            )
        )
    )

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 9
    assert len(client.prompts) == 9
    assert '"slot_00":0' in client.prompts[0]
    assert '"slot_00":192' in client.prompts[-1]
    assert '"index":204' not in client.prompts[0]
    assert '"index":204' in client.prompts[-1]


@pytest.mark.asyncio
async def test_candidate_ledger_rejects_oversized_context_before_calling_llm():
    candidates = [
        _candidate("A" * 200, observation_id="obs-1"),
        _candidate("B" * 200, observation_id="obs-2"),
    ]
    client = _LedgerClient()

    with pytest.raises(CandidateLedgerError, match="context") as exc_info:
        await select_unique_memory_candidates(
            candidates,
            structured_llm_client=client,
            llm_model=None,
            max_context_chars=200,
        )

    assert exc_info.value.error_type == "budget_exceeded"
    assert client.prompts == []


@pytest.mark.asyncio
async def test_candidate_ledger_shrinks_request_batch_to_context_budget():
    first = _candidate("A" * 800, observation_id="obs-1")
    second = _candidate("B" * 800, observation_id="obs-2")
    client = _LedgerClient(
        _ledger_response(CandidateLedgerDecision(action="KEEP")),
        _ledger_response(CandidateLedgerDecision(action="KEEP")),
    )

    result = await select_unique_memory_candidates(
        [first, second],
        structured_llm_client=client,
        llm_model=None,
        max_context_chars=3_000,
    )

    assert result.candidates == (first, second)
    assert len(client.prompts) == 2
    assert all(len(prompt) <= 3_000 for prompt in client.prompts)
    assert '"index":0' in client.prompts[0]
    assert '"index":1' not in client.prompts[0]
    assert '"index":1' in client.prompts[1]


@pytest.mark.asyncio
async def test_candidate_ledger_composes_bounded_decision_batches():
    candidates = [
        _candidate(
            f"Durable candidate {index:02d} with distinct content.",
            observation_id=f"obs-{index}",
        )
        for index in range(55)
    ]
    client = _LedgerClient(
        *(
            _ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in range(start, stop)))
            for start, stop in ((0, 24), (24, 48), (48, 55))
        )
    )

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 3
    assert len(client.prompts) == 3
    assert '"slot_00":0' in client.prompts[0]
    assert '"slot_00":24' in client.prompts[1]
    assert '"slot_00":48' in client.prompts[2]
    assert '"slot_07":null' in client.prompts[2]
    assert '"index":54' not in client.prompts[0]
    assert '"index":54' in client.prompts[2]


@pytest.mark.asyncio
async def test_candidate_ledger_keeps_failed_admission_batch_and_continues():
    candidates = [
        _candidate(
            f"Durable candidate {index:02d} with distinct content.",
            observation_id=f"obs-{index}",
        )
        for index in range(55)
    ]
    client = _IntermittentLedgerClient(
        _ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in range(24))),
        RuntimeError("provider returned invalid structured output"),
        _ledger_response(*(CandidateLedgerDecision(action="KEEP") for _ in range(48, 55))),
    )

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 3
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 24
    assert len(client.prompts) == 3
    assert '"slot_00":48' in client.prompts[-1]


@pytest.mark.asyncio
async def test_candidate_ledger_normalizes_lower_index_canonical_chains():
    longest = _candidate("A specific durable fact with details.", observation_id="obs-1")
    middle = _candidate("A durable fact with details.", observation_id="obs-2")
    shortest = _candidate("A durable fact.", observation_id="obs-3")
    client = _LedgerClient(
        _ledger_response(
            CandidateLedgerDecision(action="KEEP"),
            CandidateLedgerDecision(
                action="DROP_REDUNDANT",
                canonical_index=0,
            ),
            CandidateLedgerDecision(
                action="DROP_REDUNDANT",
                canonical_index=1,
            ),
        )
    )

    result = await select_unique_memory_candidates(
        [shortest, longest, middle],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (longest,)
    assert {drop.canonical_candidate.content for drop in result.drops} == {longest.content}


@pytest.mark.asyncio
async def test_candidate_ledger_drops_only_explicit_low_value_admission_decisions():
    durable = _candidate(
        "Enable the reduction toggle only after the compatibility suite passes.",
        observation_id="obs-1",
        memory_type="procedure",
    )
    instance_output = _candidate(
        "Test case 17 returned 204 rows in this run.",
        observation_id="obs-2",
    )
    client = _LedgerClient(
        _ledger_response(
            CandidateLedgerDecision(action="KEEP"),
            CandidateLedgerDecision(action="DROP_LOW_VALUE"),
        )
    )

    result = await select_unique_memory_candidates(
        [durable, instance_output],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (durable,)
    assert result.dropped_low_value_count == 1
    [drop] = result.drops
    assert drop.candidate is instance_output
    assert drop.canonical_candidate is None
    assert drop.method == "structured_quality"


@pytest.mark.asyncio
async def test_candidate_ledger_ignores_canonical_index_outside_redundant_action():
    first = _candidate("A durable fact with details.", observation_id="obs-1")
    second = _candidate("A different durable fact.", observation_id="obs-2")
    client = _LedgerClient(
        _ledger_response(
            CandidateLedgerDecision(action="KEEP"),
            CandidateLedgerDecision(action="KEEP", canonical_index=0),
        )
    )

    result = await select_unique_memory_candidates(
        [first, second],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (first, second)
    assert result.structured_llm_calls == 1
    assert result.validation_retries == 0


@pytest.mark.asyncio
async def test_candidate_ledger_keeps_batch_when_canonical_target_stays_outside_visible_batch():
    candidates = [_candidate("X" * (100 - index), observation_id=f"obs-{index}") for index in range(26)]
    first_batch = [CandidateLedgerDecision(action="KEEP") for _ in range(24)]
    first_batch[1] = CandidateLedgerDecision(
        action="DROP_REDUNDANT",
        canonical_index=0,
    )
    client = _LedgerClient(
        _ledger_response(*first_batch),
        _ledger_response(
            CandidateLedgerDecision(
                action="DROP_REDUNDANT",
                canonical_index=1,
            ),
            CandidateLedgerDecision(action="KEEP"),
        ),
        _ledger_response(
            CandidateLedgerDecision(
                action="DROP_REDUNDANT",
                canonical_index=1,
            ),
            CandidateLedgerDecision(action="KEEP"),
        ),
    )

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == tuple(candidate for index, candidate in enumerate(candidates) if index != 1)
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 2
    assert len(client.prompts) == 3
    assert "<validation_feedback>" in client.prompts[-1]
