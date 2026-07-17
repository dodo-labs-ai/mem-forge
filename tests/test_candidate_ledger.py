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


class _LedgerClient:
    def __init__(self, *responses: CandidateLedgerResponse) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def select_memory_candidates(self, prompt: str, **kwargs) -> CandidateLedgerResponse:
        del kwargs
        self.prompts.append(prompt)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_candidate_ledger_retries_once_when_decision_coverage_is_incomplete():
    first = _candidate("The trigger remained OPEN.", observation_id="obs-1")
    second = _candidate("The trigger was not processed.", observation_id="obs-2")
    client = _LedgerClient(
        CandidateLedgerResponse(
            decisions=[CandidateLedgerDecision(index=0, action="KEEP")]
        ),
        CandidateLedgerResponse(
            decisions=[
                CandidateLedgerDecision(index=0, action="KEEP"),
                CandidateLedgerDecision(index=1, action="KEEP"),
            ]
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


@pytest.mark.asyncio
async def test_candidate_ledger_fails_closed_after_second_incomplete_ledger():
    candidates = [
        _candidate("The trigger remained OPEN.", observation_id="obs-1"),
        _candidate("The trigger was not processed.", observation_id="obs-2"),
    ]
    incomplete = CandidateLedgerResponse(
        decisions=[CandidateLedgerDecision(index=0, action="KEEP")]
    )
    client = _LedgerClient(incomplete, incomplete)

    with pytest.raises(CandidateLedgerError, match="complete candidate ledger") as exc_info:
        await select_unique_memory_candidates(
            candidates,
            structured_llm_client=client,
            llm_model=None,
        )

    assert exc_info.value.error_type == "invalid_ledger"
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
async def test_candidate_ledger_rejects_oversized_semantic_input_before_calling_llm():
    candidates = [
        _candidate(f"Durable fact number {index}.", observation_id=f"obs-{index}")
        for index in range(3)
    ]
    client = _LedgerClient()

    with pytest.raises(CandidateLedgerError, match="candidate count") as exc_info:
        await select_unique_memory_candidates(
            candidates,
            structured_llm_client=client,
            llm_model=None,
            max_candidates=2,
        )

    assert exc_info.value.error_type == "budget_exceeded"
    assert client.prompts == []


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
