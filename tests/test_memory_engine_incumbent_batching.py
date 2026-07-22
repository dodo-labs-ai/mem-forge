from __future__ import annotations

from types import SimpleNamespace

import pytest

from memforge.memory.engine import MemoryEngine
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole
from memforge.models import Memory
from memforge.source_projection import (
    AnchorKind,
    ImpactResult,
    ProjectionCoverage,
    RevisionDelta,
    SourceAnchor,
)


class _IncumbentStore:
    def __init__(self, memory_ids: tuple[str, ...]) -> None:
        self.memory_ids = memory_ids
        self.memory_batches: list[tuple[str, ...]] = []
        self.evidence_batches: list[tuple[tuple[str, ...], str | None]] = []

    async def get_source_unit_support_reference_ids(self, source_unit_id: str):
        assert source_unit_id == "unit-1"
        return {memory_id: (f"ref-{memory_id}",) for memory_id in self.memory_ids}

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

    async def get_active_memory_support_evidence_many(self, memory_ids, *, source_id=None):
        requested = tuple(memory_ids)
        self.evidence_batches.append((requested, source_id))
        return {
            memory_id: (
                ActiveSupportEvidence(
                    memory_id=memory_id,
                    source_id="src-1",
                    reference_id=f"ref-{memory_id}",
                    evidence_unit_id=f"eu-{memory_id}",
                    role=EvidenceRole.PRIMARY,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id="obs-1",
                        observation_revision_id="obsrev-1",
                    ),
                    excerpt=None,
                ),
            )
            for memory_id in requested
        }

    async def get_active_memory_support_evidence(self, memory_id: str, *, source_id=None):
        raise AssertionError(f"support evidence must not be loaded one at a time: {memory_id}")


@pytest.mark.asyncio
async def test_projected_incumbent_loading_and_impact_keep_complete_large_coverage_batched() -> None:
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
    impacts = await engine._projected_incumbent_impacts(
        projection=SimpleNamespace(
            source_id="src-1",
            deltas=(
                RevisionDelta(
                    source_unit_id="unit-1",
                    previous_unit_revision_id="unitrev-0",
                    current_unit_revision_id="unitrev-1",
                    axes=frozenset(),
                    coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
                ),
            ),
        ),
        incumbent_ids=frozenset(memory.id for memory in incumbents),
        unit_support=unit_support,
    )

    assert [memory.id for memory in incumbents] == list(memory_ids)
    assert store.memory_batches == [memory_ids]
    assert store.evidence_batches == [(memory_ids, "src-1")]
    assert impacts == {memory_id: ImpactResult.DISJOINT for memory_id in memory_ids}
