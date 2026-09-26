from __future__ import annotations

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.models import Memory


class _IncumbentStore:
    def __init__(self, memory_ids: tuple[str, ...]) -> None:
        self.memory_ids = memory_ids
        self.memory_batches: list[tuple[str, ...]] = []

    async def get_source_unit_support_unit_ids(self, source_unit_id: str):
        assert source_unit_id == "unit-1"
        return {memory_id: (f"eu-{memory_id}",) for memory_id in self.memory_ids}

    async def list_active_memories(self, memory_ids):
        requested = tuple(memory_ids)
        self.memory_batches.append(requested)
        return [
            Memory(
                id=memory_id,
                memory_type="fact",
                content=memory_id,
                content_hash=f"hash-{memory_id}",
            )
            for memory_id in requested
        ]

    async def get_memories_by_source_doc(self, doc_id: str, *, support_kind: str):
        assert doc_id == "doc-1"
        assert support_kind == "extracted"
        return []

    async def get_memory(self, memory_id: str):
        raise AssertionError(f"incumbents must not be loaded one at a time: {memory_id}")


@pytest.mark.asyncio
async def test_projected_incumbent_loading_keeps_complete_large_coverage_batched() -> None:
    memory_ids = tuple(f"mem-{index:03d}" for index in range(501))
    store = _IncumbentStore(memory_ids)
    engine = MemoryEngine(
        cross_document_candidates=object(),
        db=store,
        memory_store=object(),
        structured_llm_client=None,
    )

    incumbents, unit_support = await engine._active_projected_incumbents(
        doc_id="doc-1",
        source_unit_id="unit-1",
    )

    assert [memory.id for memory in incumbents] == list(memory_ids)
    assert store.memory_batches == [memory_ids]
    assert set(unit_support) == set(memory_ids)
