"""Contracts for complete Source Unit incumbent classification."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.llm.structured import (
    ReconciliationDecision,
    ReconciliationResponse,
    StructuredLlmError,
)
from memforge.models import Memory, RawMemory, ReconcileAction, content_hash
from memforge.pipeline.reconciler import _parse_decisions, reconcile_memories


def _memory(mem_id: str, content: str, *, corroboration_count: int = 1) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        confidence=0.9,
        corroboration_count=corroboration_count,
        created_at=now,
        updated_at=now,
        status="active",
    )


def test_parse_decisions_preserves_flag_for_review() -> None:
    raw = RawMemory(content="PostgreSQL version is 16", memory_type="fact")
    existing = [_memory("mem-old0001", "PostgreSQL version is 14", corroboration_count=3)]

    [operation] = _parse_decisions(
        [
            {
                "index": 0,
                "action": "SUPERSEDE",
                "memory_id": existing[0].id,
                "reason": "Version changed",
                "flag_for_review": True,
            }
        ],
        [raw],
        existing,
    )

    assert operation.action is ReconcileAction.SUPERSEDE
    assert operation.flag_for_review is True


@pytest.mark.parametrize("index", [None])
def test_parse_decisions_can_remove_an_incumbent_without_a_new_candidate(index) -> None:
    existing = [_memory("mem-old0001", "PostgreSQL version is 14")]

    [operation] = _parse_decisions(
        [
            {
                "index": index,
                "action": "DELETE",
                "memory_id": existing[0].id,
                "reason": "The Source Unit no longer supports this claim",
            }
        ],
        [],
        existing,
    )

    assert operation.action is ReconcileAction.DELETE
    assert operation.memory_id == existing[0].id
    assert operation.memory is None


@pytest.mark.asyncio
async def test_prompt_defines_canonical_replacement_and_destructive_evidence_rules() -> None:
    incumbent = _memory(
        "mem-old0001",
        "Option A should depend on OD assignment validation.",
    )

    class Client:
        prompt = ""

        async def reconcile_memories(self, prompt: str, **kwargs):
            del kwargs
            self.prompt = prompt
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="still supported",
                    )
                ]
            )

    client = Client()
    await reconcile_memories(
        new_extractions=[
            RawMemory(
                content="Option A is standalone and uses the prospective slot builder.",
                memory_type="decision",
            )
        ],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=client,
        updated_document="### Option A: Reuse Prospective Slot Building",
        update_mode="diff_guided",
        changed_hunks="-dependent on validation\n+standalone",
        update_plan_stats={"reason": "small_diff"},
    )

    assert "replacement memory content must state the current durable fact" in client.prompt
    assert "Do not write replacement content as edit history" in client.prompt
    assert "DELETE or SUPERSEDE an existing memory only when <changed_hunks>" in client.prompt
    assert "Do not DELETE solely because support is absent from unrelated context" in client.prompt
    assert "exactly one explicit decision for every existing memory ID" in client.prompt


@pytest.mark.asyncio
async def test_classifier_failure_with_incumbents_fails_closed() -> None:
    class FailingClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            raise StructuredLlmError("structured unavailable")

    result = await reconcile_memories(
        new_extractions=[RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")],
        existing_memories=[_memory("mem-existing", "Service uses PostgreSQL 15.")],
        doc_type="design",
        structured_llm_client=FailingClient(),
        updated_document="# Design\n\nService uses PostgreSQL 16.",
        include_metadata=True,
    )

    assert result.operations == []
    assert result.failure is not None
    assert result.failure.error_type == "structured_llm_error"


@pytest.mark.asyncio
async def test_more_than_thirty_incumbents_use_bounded_batches_and_close_one_ledger() -> None:
    incumbents = [_memory(f"mem-{index:08d}", f"Stable claim {index}") for index in range(65)]

    class CompleteBatchClient:
        def __init__(self) -> None:
            self.offset = 0
            self.batch_sizes: list[int] = []

        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            batch = incumbents[self.offset : self.offset + 30]
            self.offset += len(batch)
            self.batch_sizes.append(len(batch))
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=None,
                        action="NOOP",
                        memory_id=memory.id,
                        reason="still supported",
                    )
                    for memory in batch
                ]
            )

    client = CompleteBatchClient()
    operations = await reconcile_memories(
        new_extractions=[],
        existing_memories=incumbents,
        doc_type="design",
        structured_llm_client=client,
        updated_document="# Current design",
    )

    assert client.batch_sizes == [30, 30, 5]
    assert {operation.memory_id for operation in operations} == {
        memory.id for memory in incumbents
    }
    assert all(operation.action is ReconcileAction.NOOP for operation in operations)


@pytest.mark.asyncio
async def test_any_incomplete_batch_invalidates_the_entire_ledger() -> None:
    incumbents = [_memory(f"mem-{index:08d}", f"Stable claim {index}") for index in range(31)]

    class MissingDecisionClient:
        def __init__(self) -> None:
            self.offset = 0

        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            batch = incumbents[self.offset : self.offset + 30]
            self.offset += len(batch)
            if len(batch) == 1:
                batch = []
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=None,
                        action="NOOP",
                        memory_id=memory.id,
                        reason="still supported",
                    )
                    for memory in batch
                ]
            )

    result = await reconcile_memories(
        new_extractions=[],
        existing_memories=incumbents,
        doc_type="design",
        structured_llm_client=MissingDecisionClient(),
        updated_document="# Current design",
        include_metadata=True,
    )

    assert result.operations == []
    assert result.failure is not None
    assert "missing incumbent decisions" in result.failure.error
