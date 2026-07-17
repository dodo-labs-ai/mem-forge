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


def test_parse_update_preserves_exact_source_observation_lineage() -> None:
    raw = RawMemory(
        content="The Jira discussion settled on option A.",
        memory_type="decision",
        evidence_quote="Proper message shows as expected",
        evidence_anchor="projection_batch",
        source_observation_id="obs-comment-42",
    )
    existing = [_memory("mem-old0001", "The Jira discussion preferred option B.")]

    [operation] = _parse_decisions(
        [
            {
                "index": 0,
                "action": "UPDATE",
                "memory_id": existing[0].id,
                "updated_content": "The Jira discussion settled on option A.",
                "reason": "The decision changed",
            }
        ],
        [raw],
        existing,
    )

    assert operation.memory is not None
    assert operation.memory.evidence_quote == raw.evidence_quote
    assert operation.memory.evidence_anchor == raw.evidence_anchor
    assert operation.memory.source_observation_id == raw.source_observation_id


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


@pytest.mark.asyncio
async def test_multiple_new_extractions_each_produce_one_merged_operation() -> None:
    incumbent = _memory("mem-existing", "Service uses PostgreSQL 15.")

    class CompleteClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="SUPERSEDE",
                        memory_id=incumbent.id,
                        reason="Version changed",
                    ),
                    ReconciliationDecision(index=1, action="ADD", reason="New backup policy"),
                ]
            )

    result = await reconcile_memories(
        new_extractions=[
            RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact"),
            RawMemory(content="Backups run daily.", memory_type="procedure"),
        ],
        existing_memories=[incumbent],
        doc_type="ticket",
        structured_llm_client=CompleteClient(),
        include_metadata=True,
    )

    assert result.failure is None
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.SUPERSEDE,
        ReconcileAction.ADD,
    ]


@pytest.mark.asyncio
async def test_compatible_duplicate_noops_normalize_to_one_incumbent_keep() -> None:
    incumbent = _memory("mem-existing", "Retries use exponential backoff.")

    class DuplicateNoopClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="Already captured",
                    ),
                    ReconciliationDecision(
                        index=1,
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="Same durable rule",
                    ),
                    ReconciliationDecision(
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="Still supported",
                    ),
                ]
            )

    result = await reconcile_memories(
        new_extractions=[
            RawMemory(content="Retries back off exponentially.", memory_type="procedure"),
            RawMemory(content="Retry delays increase after failures.", memory_type="fact"),
        ],
        existing_memories=[incumbent],
        doc_type="ticket",
        structured_llm_client=DuplicateNoopClient(),
        include_metadata=True,
    )

    assert result.failure is None
    incumbent_operations = [
        operation for operation in result.operations if operation.memory_id == incumbent.id
    ]
    assert len(incumbent_operations) == 1
    assert incumbent_operations[0].action is ReconcileAction.NOOP
    assert len(result.operations) == 2
    assert result.operations[1].action is ReconcileAction.NOOP
    assert result.operations[1].memory_id is None


@pytest.mark.asyncio
async def test_conflicting_duplicate_incumbent_decisions_fail_closed() -> None:
    incumbent = _memory("mem-existing", "Service uses PostgreSQL 15.")

    class ConflictingClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="SUPERSEDE",
                        memory_id=incumbent.id,
                        reason="Version changed",
                    ),
                    ReconciliationDecision(
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="Still supported",
                    ),
                ]
            )

    result = await reconcile_memories(
        new_extractions=[RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")],
        existing_memories=[incumbent],
        doc_type="ticket",
        structured_llm_client=ConflictingClient(),
        include_metadata=True,
    )

    assert result.operations == []
    assert result.failure is not None
    assert "conflicting incumbent decisions" in result.failure.error


@pytest.mark.asyncio
async def test_missing_new_extraction_decision_invalidates_batch() -> None:
    incumbent = _memory("mem-existing", "Stable claim")

    class MissingCandidateClient:
        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        action="NOOP",
                        memory_id=incumbent.id,
                        reason="Still supported",
                    )
                ]
            )

    result = await reconcile_memories(
        new_extractions=[RawMemory(content="New claim", memory_type="fact")],
        existing_memories=[incumbent],
        doc_type="ticket",
        structured_llm_client=MissingCandidateClient(),
        include_metadata=True,
    )

    assert result.operations == []
    assert result.failure is not None
    assert "missing new extraction decisions" in result.failure.error


@pytest.mark.asyncio
async def test_invalid_replacement_without_candidate_index_retries_only_reconciliation_batch() -> None:
    incumbent = _memory("mem-existing", "Service uses PostgreSQL 15.")

    class CorrectingClient:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        async def reconcile_memories(self, prompt: str, **kwargs):
            del kwargs
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                return ReconciliationResponse(
                    decisions=[
                        ReconciliationDecision(
                            index=0,
                            action="ADD",
                            reason="New version claim",
                        ),
                        ReconciliationDecision(
                            action="SUPERSEDE",
                            memory_id=incumbent.id,
                            reason="Version changed",
                        ),
                    ]
                )
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="SUPERSEDE",
                        memory_id=incumbent.id,
                        reason="Version changed",
                    )
                ]
            )

    client = CorrectingClient()
    result = await reconcile_memories(
        new_extractions=[
            RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")
        ],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=client,
        updated_document="# Design\n\nService uses PostgreSQL 16.",
        include_metadata=True,
    )

    assert result.failure is None
    assert client.calls == 2
    assert "replacement decision for incumbent mem-existing requires a new extraction index" in client.prompts[1]
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.SUPERSEDE
    ]


@pytest.mark.asyncio
async def test_persistent_replacement_without_candidate_is_deferred_to_review() -> None:
    incumbent = _memory("mem-existing", "Service uses PostgreSQL 15.")

    class PersistentlyInvalidClient:
        def __init__(self) -> None:
            self.calls = 0

        async def reconcile_memories(self, prompt: str, **kwargs):
            del prompt, kwargs
            self.calls += 1
            return ReconciliationResponse(
                decisions=[
                    ReconciliationDecision(
                        index=0,
                        action="ADD",
                        reason="New version claim",
                    ),
                    ReconciliationDecision(
                        action="SUPERSEDE",
                        memory_id=incumbent.id,
                        reason="Version changed but no candidate was selected",
                    ),
                ]
            )

    client = PersistentlyInvalidClient()
    result = await reconcile_memories(
        new_extractions=[
            RawMemory(content="Service uses PostgreSQL 16.", memory_type="fact")
        ],
        existing_memories=[incumbent],
        doc_type="design",
        structured_llm_client=client,
        updated_document="# Design\n\nService uses PostgreSQL 16.",
        include_metadata=True,
    )

    assert result.failure is None
    assert client.calls == 2
    assert [operation.action for operation in result.operations] == [
        ReconcileAction.ADD,
        ReconcileAction.DELETE,
    ]
    review_operation = result.operations[1]
    assert review_operation.memory_id == incumbent.id
    assert review_operation.flag_for_review is True
    assert "unresolved replacement without a candidate" in review_operation.reason
