"""Sparse identity and non-destructive relationship discovery."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from memforge.llm.batch_runner import ItemTask, LlmBatchRunner, LlmRequest
from memforge.llm.relation_catalog import RelationCoverage, RequestCatalog
from memforge.llm.structured import MemoryRelationCatalogResponse
from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MEMORY_RELATION_RULES, MemoryPair, MemoryPairClassification,
    MemoryPairDecision, MemoryRelationType,
    MemoryPairClassificationPolicy, MemoryPairContext,
    _auditable_relation_reason, _prompt_memory, relation_output_tokens, run_pair_items,
)
from memforge.models import Memory

SPARSE_MEMORY_CLASSIFIER_VERSION = "memory-relation-v4-sparse"


def _catalog_request(pairs: tuple[MemoryPair, ...]):
    new: RequestCatalog[tuple[Memory, MemoryPairContext | None]] = RequestCatalog("NEW")
    old: RequestCatalog[tuple[Memory, MemoryPairContext | None]] = RequestCatalog("MEM")
    allowed: dict[str, set[str]] = {}
    pair_by_refs: dict[tuple[str, str], MemoryPair] = {}
    for pair in pairs:
        new_ref = new.add(pair.challenger.id, (pair.challenger, pair.challenger_context))
        old_ref = old.add(pair.candidate.id, (pair.candidate, pair.candidate_context))
        if (new_ref, old_ref) in pair_by_refs:
            raise ValueError("duplicate allowed Memory pair")
        allowed.setdefault(new_ref, set()).add(old_ref)
        pair_by_refs[new_ref, old_ref] = pair

    def present(ref: str, record: tuple[Memory, MemoryPairContext | None]) -> dict[str, object]:
        value = _prompt_memory(record[0], context=record[1])
        value["id"] = ref
        return value

    payload = dict(
        new_claims=[present(ref, value) for ref, value in new.records.items()],
        existing_claims=[present(ref, value) for ref, value in old.records.items()],
        allowed_existing_ids={ref: sorted(ids) for ref, ids in allowed.items()},
    )
    prompt = MEMORY_RELATION_RULES + """
<memory_relation_catalog>
""" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + """
</memory_relation_catalog>
Return exactly one results row for every NEW candidate_id, including candidates
with no discovered relationships. Check only that candidate's allowed_existing_ids.
Return only equivalent, refines, or contradicts edges. Never return unrelated.
An empty relations array completes the candidate without asserting that omitted
pairs are unrelated. Do not infer lifecycle authority from a relationship.
"""
    return prompt, RelationCoverage({ref: frozenset(ids) for ref, ids in allowed.items()}), pair_by_refs


class SparseMemoryRelationClassifier:
    """Deduplicate input catalogs and validate every completion before emitting edges.

    Each allowed pair is one runner item, so a catalog that outgrows the route is
    split between pairs and every allowed pair stays represented exactly once.
    """

    def __init__(self, *, client: Any, model: str, policy: MemoryPairClassificationPolicy | None = None):
        self._client = client
        self._model = model
        self._policy = policy or MemoryPairClassificationPolicy()

    async def classify(self, pairs: tuple[MemoryPair, ...]) -> MemoryPairClassification:
        if not pairs:
            return MemoryPairClassification((), 0, 0)
        runner = LlmBatchRunner(self._client, model=self._model)

        catalogs: dict[tuple[str, ...], tuple] = {}

        def catalog(item_ids: tuple[str, ...]):
            """Build each request's catalog once; decode reads the one render built."""

            if item_ids not in catalogs:
                catalogs[item_ids] = _catalog_request(tuple(pairs[int(item_id)] for item_id in item_ids))
            return catalogs[item_ids]

        def render(item_ids: tuple[str, ...], _context: tuple) -> LlmRequest:
            prompt, _coverage, _by_refs = catalog(item_ids)
            return LlmRequest(prompt, MemoryRelationCatalogResponse, relation_output_tokens(self._policy, len(item_ids)))

        def decode(raw: Any, item_ids: tuple[str, ...], _context: tuple):
            response = MemoryRelationCatalogResponse.model_validate(
                raw.model_dump() if isinstance(raw, BaseModel) else raw
            )
            _prompt, coverage, by_refs = catalog(item_ids)
            coverage.validate(response.results)
            discovered = {
                by_refs[row.candidate_id, edge.existing_id].key: MemoryPairDecision(
                    pair=by_refs[row.candidate_id, edge.existing_id],
                    relation_type=MemoryRelationType(edge.classification),
                    direction=RelationDirection(edge.direction),
                    reason=_auditable_relation_reason(edge),
                )
                for row in response.results for edge in row.relations
            }
            # None completes a pair without asserting that it is unrelated.
            return ((item_id, discovered.get(pairs[int(item_id)].key)) for item_id in item_ids)

        results = await run_pair_items(runner, ItemTask(
            item_ids=tuple(str(index) for index in range(len(pairs))), render=render, decode=decode,
            call=self._client.discover_memory_relations,
        ), pair_count=len(pairs), label="memory relation catalog")
        decisions = tuple(decision for decision in results if decision is not None)
        return MemoryPairClassification(decisions, runner.stats.calls, runner.stats.prompt_chars)
