from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from memforge.llm.structured import (
    OUTPUT_TRUNCATED,
    CandidateLedgerDecision,
    CandidateLedgerResponse,
    StructuredLlmError,
)
from memforge.memory.candidate_ledger import (
    _CANDIDATE_LEDGER_PROMPT,
    CandidateLedgerError,
    select_unique_memory_candidates,
)
from memforge.models import RawMemory
from tests.llm_fixture import FixtureBudgetClient

# Each "Durable candidate NN with distinct content." adds five prompt words.
_WORDS_PER_CANDIDATE = 5


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


def _distinct_candidates(count: int) -> list[RawMemory]:
    return [
        _candidate(f"Durable candidate {index:03d} with distinct content.", observation_id=f"obs-{index}")
        for index in range(count)
    ]


def _capacity(candidates_per_request: int) -> int:
    return len(_CANDIDATE_LEDGER_PROMPT.split()) + _WORDS_PER_CANDIDATE * candidates_per_request


def _ledger_response(
    *decisions: CandidateLedgerDecision,
) -> CandidateLedgerResponse:
    return CandidateLedgerResponse(decisions=list(decisions))


def _indices(prompt: str) -> list[int]:
    candidates = json.loads(prompt.split("<candidates>\n", 1)[1].split("\n</candidates>", 1)[0])
    return [candidate["index"] for candidate in candidates]


def _keep(index: int) -> CandidateLedgerDecision:
    return CandidateLedgerDecision(candidate_index=index, action="KEEP")


def _keep_all(prompt: str) -> CandidateLedgerResponse:
    return _ledger_response(*(_keep(index) for index in _indices(prompt)))


@dataclass
class _LedgerClient(FixtureBudgetClient):
    async def select_memory_candidates(self, prompt: str, *, max_tokens: int, model=None):
        return await self.call(prompt, max_tokens=max_tokens, model=model)


def _queued(*responses: CandidateLedgerResponse) -> _LedgerClient:
    pending = list(responses)
    return _LedgerClient(respond=lambda _prompt: pending.pop(0))


@pytest.mark.asyncio
async def test_candidate_ledger_packs_requests_by_route_capacity_with_bounded_concurrency():
    candidates = _distinct_candidates(55)
    client = _LedgerClient(respond=_keep_all, input_tokens=_capacity(20), max_concurrent=2)

    result = await select_unique_memory_candidates(candidates, structured_llm_client=client, llm_model=None)

    assert result.candidates == tuple(candidates)
    assert [_indices(prompt) for prompt in client.prompts] == [
        list(range(0, 20)), list(range(20, 40)), list(range(40, 55)),
    ]
    assert result.structured_llm_calls == 3
    assert client.peak_in_flight == 2


@pytest.mark.asyncio
async def test_candidate_ledger_has_no_fixed_item_cap_when_the_route_has_room():
    candidates = _distinct_candidates(205)
    client = _LedgerClient(respond=_keep_all)

    result = await select_unique_memory_candidates(candidates, structured_llm_client=client, llm_model=None)

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 1
    assert _indices(client.prompts[0]) == list(range(205))


@pytest.mark.asyncio
async def test_candidate_ledger_halves_a_truncated_request_until_every_candidate_is_judged():
    candidates = _distinct_candidates(8)

    def respond(prompt: str) -> CandidateLedgerResponse:
        if len(_indices(prompt)) > 2:
            raise StructuredLlmError("truncated", terminal_category="invalid_response", error_code=OUTPUT_TRUNCATED)
        return _keep_all(prompt)

    client = _LedgerClient(respond=respond)

    result = await select_unique_memory_candidates(candidates, structured_llm_client=client, llm_model=None)

    assert result.candidates == tuple(candidates)
    assert result.fallback_batch_count == 0
    assert sorted(index for prompt in client.prompts if len(_indices(prompt)) <= 2 for index in _indices(prompt)) == list(range(8))


@pytest.mark.asyncio
async def test_candidate_ledger_retries_once_when_decision_coverage_is_incomplete():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    second = _candidate("The trigger was not processed.", observation_id="obs-2")
    client = _queued(_ledger_response(_keep(0)), _ledger_response(_keep(0), _keep(1)))

    result = await select_unique_memory_candidates(
        [first, second],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (first, second)
    assert len(client.prompts) == 2
    assert "<correction>" in client.prompts[1]
    assert result.structured_llm_calls == 2
    assert result.validation_retries == 1
    assert result.prompt_chars == sum(len(prompt) for prompt in client.prompts)
    assert result.structured_llm_elapsed_ms >= 0


@pytest.mark.asyncio
async def test_candidate_ledger_retries_once_when_decision_coverage_is_excessive():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    second = _candidate("The trigger was not processed.", observation_id="obs-2")
    client = _queued(
        _ledger_response(_keep(0), _keep(1), _keep(2)),
        _ledger_response(_keep(0), _keep(1)),
    )

    result = await select_unique_memory_candidates(
        [first, second],
        structured_llm_client=client,
        llm_model=None,
    )

    assert result.candidates == (first, second)
    assert len(client.prompts) == 2
    assert "not requested" in client.prompts[1]
    assert result.structured_llm_calls == 2
    assert result.validation_retries == 1


@pytest.mark.asyncio
async def test_candidate_ledger_keeps_batch_after_second_invalid_ledger():
    candidates = [
        _candidate("The trigger remained OPEN.", observation_id="obs-1"),
        _candidate("The trigger was not processed.", observation_id="obs-2"),
    ]
    incomplete = _ledger_response(_keep(0))
    client = _queued(incomplete, incomplete)

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
    client = _queued()

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
async def test_candidate_ledger_fails_closed_when_one_candidate_exceeds_route_capacity():
    candidates = [
        _candidate(" ".join(["A"] * 200), observation_id="obs-1"),
        _candidate(" ".join(["B"] * 200), observation_id="obs-2"),
    ]
    client = _LedgerClient(respond=_keep_all, input_tokens=_capacity(20))

    with pytest.raises(CandidateLedgerError, match="capacity") as exc_info:
        await select_unique_memory_candidates(
            candidates,
            structured_llm_client=client,
            llm_model=None,
        )

    assert exc_info.value.error_type == "budget_exceeded"
    assert client.prompts == []


@pytest.mark.asyncio
async def test_candidate_ledger_keeps_failed_admission_request_and_continues():
    candidates = _distinct_candidates(55)

    def respond(prompt: str) -> CandidateLedgerResponse:
        if 20 in _indices(prompt):
            raise StructuredLlmError("provider unavailable", terminal_category="provider_error", error_code="provider_error")
        return _keep_all(prompt)

    client = _LedgerClient(respond=respond, input_tokens=_capacity(20))

    result = await select_unique_memory_candidates(candidates, structured_llm_client=client, llm_model=None)

    assert result.candidates == tuple(candidates)
    assert result.structured_llm_calls == 3
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 20
    assert len(client.prompts) == 3


@dataclass
class _UnbudgetedLedgerClient(_LedgerClient):
    def request_budget(self, model=None):
        raise ValueError("route has no resolvable request budget")


@pytest.mark.asyncio
async def test_candidate_ledger_admits_every_candidate_when_the_route_has_no_request_budget():
    candidates = _distinct_candidates(3)
    client = _UnbudgetedLedgerClient(respond=_keep_all)

    result = await select_unique_memory_candidates(candidates, structured_llm_client=client, llm_model=None)

    assert result.candidates == tuple(candidates)
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 3
    assert client.prompts == []


@pytest.mark.asyncio
async def test_candidate_ledger_does_not_hide_a_client_programming_error():
    def respond(_prompt: str) -> CandidateLedgerResponse:
        raise RuntimeError("client transport broke")

    with pytest.raises(RuntimeError, match="transport broke"):
        await select_unique_memory_candidates(
            _distinct_candidates(3), structured_llm_client=_LedgerClient(respond=respond), llm_model=None,
        )


@pytest.mark.asyncio
async def test_candidate_ledger_binds_decisions_by_index_not_position():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    second = _candidate("The trigger was not processed.", observation_id="obs-2")
    low_value = CandidateLedgerDecision(candidate_index=1, action="DROP_LOW_VALUE")
    client = _queued(
        # Omits index 0 and repeats index 1: the same row count, so only IDs expose it.
        _ledger_response(low_value, low_value),
        _ledger_response(low_value, _keep(0)),
    )

    result = await select_unique_memory_candidates([first, second], structured_llm_client=client, llm_model=None)

    assert "more than once" in client.prompts[1]
    assert result.validation_retries == 1
    [drop] = result.drops
    judged = json.loads(client.prompts[0].split("<candidates>\n", 1)[1].split("\n</candidates>", 1)[0])
    assert drop.candidate.content == judged[1]["content"]


@pytest.mark.asyncio
async def test_candidate_ledger_normalizes_lower_index_canonical_chains():
    longest = _candidate("A specific durable fact with details.", observation_id="obs-1")
    middle = _candidate("A durable fact with details.", observation_id="obs-2")
    shortest = _candidate("A durable fact.", observation_id="obs-3")
    client = _queued(
        _ledger_response(
            _keep(0),
            CandidateLedgerDecision(candidate_index=1, action="DROP_REDUNDANT", canonical_index=0),
            CandidateLedgerDecision(candidate_index=2, action="DROP_REDUNDANT", canonical_index=1),
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
    client = _queued(
        _ledger_response(
            _keep(0),
            CandidateLedgerDecision(candidate_index=1, action="DROP_LOW_VALUE"),
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
    client = _queued(
        _ledger_response(
            _keep(0),
            CandidateLedgerDecision(candidate_index=1, action="KEEP", canonical_index=0),
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
    candidates = _distinct_candidates(26)

    def respond(prompt: str) -> CandidateLedgerResponse:
        indices = _indices(prompt)
        decisions = [_keep(index) for index in indices]
        if indices[0] == 0:
            decisions[1] = CandidateLedgerDecision(candidate_index=1, action="DROP_REDUNDANT", canonical_index=0)
        else:
            decisions[0] = CandidateLedgerDecision(candidate_index=indices[0], action="DROP_REDUNDANT", canonical_index=1)
        return _ledger_response(*decisions)

    client = _LedgerClient(respond=respond, input_tokens=_capacity(24))

    result = await select_unique_memory_candidates(
        candidates,
        structured_llm_client=client,
        llm_model=None,
    )

    assert [_indices(prompt) for prompt in client.prompts] == [list(range(24)), [24, 25], [24, 25]]
    assert result.candidates == tuple(candidate for index, candidate in enumerate(candidates) if index != 1)
    assert result.fallback_batch_count == 1
    assert result.fallback_candidate_count == 2
    assert "<correction>" in client.prompts[-1]
