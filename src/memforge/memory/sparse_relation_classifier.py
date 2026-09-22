"""Sparse identity and non-destructive relationship discovery."""

from __future__ import annotations

import json
from typing import Any

from memforge.llm.relation_catalog import RelationCoverage, RequestCatalog
from memforge.llm.structured import (
    MemoryRelationCatalogResponse, StructuredLlmError, structured_llm_max_concurrent,
)
from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MEMORY_RELATION_RULES, MemoryPair, MemoryPairClassification,
    MemoryPairClassificationError, MemoryPairClassificationPlan,
    MemoryPairDecision, MemoryRelationType, MemoryPairClassificationPolicy,
    MemoryPairContext,
    _auditable_relation_reason, _prompt_memory,
)
from memforge.pipeline.bounded_work import collect_bounded
from memforge.models import Memory

SPARSE_MEMORY_CLASSIFIER_VERSION = "memory-relation-v4-sparse"


def _catalog_request(pairs: tuple[MemoryPair, ...], max_content_chars: int):
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
        value = _prompt_memory(record[0], context=record[1], max_content_chars=max_content_chars)
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
    """Deduplicate input catalogs and validate every completion before emitting edges."""

    def __init__(self, *, client: Any, model: str, policy: MemoryPairClassificationPolicy | None = None):
        self._client = client
        self._model = model
        self._policy = policy or MemoryPairClassificationPolicy()

    def _output_tokens(self, count: int) -> int:
        return self._client.request_budget(self._model).output_reserve(
            min(self._policy.max_output_tokens, 512 + 768 * count)
        )

    def _request_fits(self, prompt: str, count: int, *, reserve_correction: bool = True) -> bool:
        return self._client.request_fits(
            prompt, response_format=MemoryRelationCatalogResponse, model=self._model,
            max_tokens=self._output_tokens(count), reserve_correction=reserve_correction,
        )

    def _catalogs(self, pairs: tuple[MemoryPair, ...]):
        if not pairs:
            return ()
        prompt, coverage, by_refs = _catalog_request(pairs, self._policy.max_memory_content_chars)
        if len(prompt) <= self._policy.max_prompt_chars and self._request_fits(prompt, len(pairs)):
            return ((pairs, prompt, coverage, by_refs),)
        if len(pairs) == 1:
            raise MemoryPairClassificationError("one complete relation catalog exceeds configured capacity")
        # Capacity is a transport boundary; every allowed pair remains represented.
        middle = len(pairs) // 2
        return self._catalogs(pairs[:middle]) + self._catalogs(pairs[middle:])

    def plan(self, pairs: tuple[MemoryPair, ...]) -> MemoryPairClassificationPlan:
        try:
            catalogs = self._catalogs(pairs)
        except ValueError as error:
            raise MemoryPairClassificationError(str(error), pair_count=len(pairs)) from error
        return MemoryPairClassificationPlan(len(pairs), len(catalogs), sum(len(c[1]) for c in catalogs))

    async def classify(self, pairs: tuple[MemoryPair, ...]) -> MemoryPairClassification:
        calls = chars = 0
        try:
            catalogs = self._catalogs(pairs)

            async def run(catalog: Any):
                nonlocal calls, chars
                workset, prompt, coverage, by_refs = catalog
                request_prompt = prompt
                for attempt in range(2):
                    if not self._request_fits(request_prompt, len(workset), reserve_correction=not attempt):
                        raise MemoryPairClassificationError("relation catalog correction exceeds capacity")
                    calls += 1
                    chars += len(request_prompt)
                    raw = await self._client.discover_memory_relations(
                        request_prompt, max_tokens=self._output_tokens(len(workset)), model=self._model,
                    )
                    response = MemoryRelationCatalogResponse.model_validate(
                        raw.model_dump() if isinstance(raw, MemoryRelationCatalogResponse) else raw
                    )
                    try:
                        coverage.validate(response.results)
                    except ValueError as error:
                        if attempt:
                            raise
                        request_prompt = prompt + "\n<coverage_correction>" + str(error) + (
                            ". Regenerate all candidate completion rows exactly once and only allowed "
                            "incumbent references.</coverage_correction>"
                        )
                        continue
                    return tuple(
                        MemoryPairDecision(
                            pair=by_refs[row.candidate_id, edge.existing_id],
                            relation_type=MemoryRelationType(edge.classification),
                            direction=RelationDirection(edge.direction),
                            reason=_auditable_relation_reason(edge),
                        )
                        for row in response.results for edge in row.relations
                    )
                raise AssertionError("unreachable coverage state")

            outcomes = await collect_bounded(catalogs, run,
                                            max_concurrent=structured_llm_max_concurrent(self._client))
            by_key = {decision.pair.key: decision for outcome in outcomes for decision in outcome}
            return MemoryPairClassification(
                tuple(by_key[pair.key] for pair in pairs if pair.key in by_key), calls, chars,
            )
        except Exception as error:
            raise MemoryPairClassificationError(
                f"memory relation catalog failed: {error}", pair_count=len(pairs),
                llm_calls=calls, prompt_chars=chars,
                terminal_category=error.terminal_category if isinstance(error, StructuredLlmError) else None,
                error_code=getattr(error, "error_code", None),
            ) from error
