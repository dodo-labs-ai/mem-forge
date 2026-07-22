from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from memforge.memory.identity_resolver import (
    IdentityResolutionRequest,
    IdentityResolver,
)
from memforge.memory.evidence import RelationDirection
from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassification,
    MemoryPairClassificationError,
    MemoryPairClassificationPlan,
    MemoryPairDecision,
    MemoryRelationType,
    MemoryPairClassificationPolicy,
    MEMORY_RELATION_PROMPT,
    StructuredMemoryPairClassifier,
)
from memforge.models import Memory, content_hash


def _memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="decision",
        content=content,
        content_hash=content_hash(content),
    )


def test_equivalent_identity_prompt_preserves_normative_modality() -> None:
    prompt = " ".join(MEMORY_RELATION_PROMPT.split())
    assert "A requirement and a configured state may be equivalent" not in prompt
    assert "normative requirement" in prompt
    assert "descriptive state" in prompt
    assert "different truth conditions" in prompt


@pytest.mark.asyncio
async def test_structured_classifier_reports_usage_when_a_later_batch_fails() -> None:
    class FailingSecondBatchClient:
        def __init__(self) -> None:
            self.calls = 0

        async def classify_memory_relations(self, _prompt, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("provider timeout")
            return SimpleNamespace(
                decisions=[
                    SimpleNamespace(
                        pair_index=0,
                        classification="unrelated",
                        direction="symmetric",
                        reason="fixture",
                    )
                ]
            )

    challenger = _memory("challenger", "Current claim")
    classifier = StructuredMemoryPairClassifier(
        client=FailingSecondBatchClient(),
        model="test-model",
        policy=MemoryPairClassificationPolicy(max_pairs_per_call=1),
    )
    pairs = (
        MemoryPair(challenger, _memory("candidate-1", "First candidate")),
        MemoryPair(challenger, _memory("candidate-2", "Second candidate")),
    )

    with pytest.raises(MemoryPairClassificationError, match="provider timeout") as caught:
        await classifier.classify(pairs)

    assert caught.value.pair_count == 2
    assert caught.value.llm_calls == 2
    assert caught.value.prompt_chars > 0


@dataclass
class _CandidateStore:
    exact_by_challenger: dict[str, Memory]
    semantic_by_challenger: dict[str, tuple[Memory, ...]]

    async def find_access_compatible_exact_candidate(self, challenger, **_kwargs):
        return self.exact_by_challenger.get(challenger.id)

    async def find_access_compatible_equivalence_candidates(self, challenger, **_kwargs):
        return self.semantic_by_challenger.get(challenger.id, ())


class _PairClassifier:
    def __init__(self, relation_by_pair: dict[tuple[str, str], MemoryRelationType]) -> None:
        self._relation_by_pair = relation_by_pair
        self.calls: list[tuple[MemoryPair, ...]] = []

    def plan(self, pairs: tuple[MemoryPair, ...]):
        return MemoryPairClassificationPlan(
            pair_count=len(pairs),
            llm_calls=1 if pairs else 0,
            prompt_chars=0,
        )

    async def classify(self, pairs: tuple[MemoryPair, ...]):
        self.calls.append(pairs)
        return MemoryPairClassification(
            decisions=tuple(
                MemoryPairDecision(
                    pair=pair,
                    relation_type=self._relation_by_pair[
                        (
                            pair.challenger.id,
                            pair.candidate.id,
                        )
                    ],
                    direction=(
                        RelationDirection.CHALLENGER_TO_CANDIDATE
                        if self._relation_by_pair[
                            (
                                pair.challenger.id,
                                pair.candidate.id,
                            )
                        ]
                        is MemoryRelationType.REFINES
                        else RelationDirection.SYMMETRIC
                    ),
                    reason="contract fixture",
                )
                for pair in pairs
            ),
            llm_calls=1,
            prompt_chars=0,
        )


@pytest.mark.asyncio
async def test_identity_resolver_batches_scope_and_reuses_only_equivalent_memory() -> None:
    exact_challenger = _memory("mem-new-exact", "Deployments require approval.")
    exact_incumbent = _memory("mem-exact", exact_challenger.content)
    equivalent_challenger = _memory(
        "mem-new-equivalent",
        "Production deployment requires approval.",
    )
    equivalent_incumbent = _memory(
        "mem-equivalent",
        "Approval is mandatory before production deployment.",
    )
    refinement_candidate = _memory(
        "mem-refinement-candidate",
        "Production deployment requires Security Owner approval.",
    )
    refining_challenger = _memory(
        "mem-new-refining",
        "Production deployment requires Security Owner approval and a change ticket.",
    )
    refined_incumbent = _memory(
        "mem-refined",
        "Production deployment requires approval.",
    )
    store = _CandidateStore(
        exact_by_challenger={exact_challenger.id: exact_incumbent},
        semantic_by_challenger={
            equivalent_challenger.id: (
                refinement_candidate,
                equivalent_incumbent,
            ),
            refining_challenger.id: (refined_incumbent,),
        },
    )
    classifier = _PairClassifier(
        {
            (equivalent_challenger.id, refinement_candidate.id): MemoryRelationType.REFINES,
            (equivalent_challenger.id, equivalent_incumbent.id): MemoryRelationType.EQUIVALENT,
            (refining_challenger.id, refined_incumbent.id): MemoryRelationType.REFINES,
        }
    )
    resolver = IdentityResolver(
        memory_store=store,
        pair_classifier=classifier,
        llm_model="test-model",
    )

    batch = await resolver.resolve(
        (
            IdentityResolutionRequest(exact_challenger, "doc-a"),
            IdentityResolutionRequest(equivalent_challenger, "doc-b"),
            IdentityResolutionRequest(refining_challenger, "doc-c"),
        )
    )
    results = batch.resolutions

    assert [result.target for result in results] == [
        exact_incumbent,
        equivalent_incumbent,
        None,
    ]
    assert results[0].classified_pairs == ()
    assert [decision.relation_type for decision in results[1].classified_pairs] == [
        MemoryRelationType.REFINES,
        MemoryRelationType.EQUIVALENT,
    ]
    assert [decision.relation_type for decision in results[2].classified_pairs] == [MemoryRelationType.REFINES]
    assert len(classifier.calls) == 1
    assert batch.metrics.pair_count == 3
    assert batch.metrics.llm_calls == 1
    assert batch.metrics.prompt_chars == 0
    assert [(pair.challenger.id, pair.candidate.id) for pair in classifier.calls[0]] == [
        (equivalent_challenger.id, refinement_candidate.id),
        (equivalent_challenger.id, equivalent_incumbent.id),
        (refining_challenger.id, refined_incumbent.id),
    ]


class _IncompleteStructuredClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    async def classify_memory_relations(
        self,
        prompt: str,
        *,
        max_tokens: int,
        model: str | None = None,
    ):
        self.calls.append((prompt, max_tokens, model))
        return SimpleNamespace(
            decisions=[
                SimpleNamespace(
                    pair_index=0,
                    classification="equivalent",
                    direction="symmetric",
                    reason="fixture",
                )
            ]
        )


@pytest.mark.asyncio
async def test_identity_resolver_fails_closed_for_incomplete_structured_pair_ledger() -> None:
    challenger = _memory("mem-new", "Production deployment requires approval.")
    first = _memory("mem-first", "Approval is mandatory before production deployment.")
    second = _memory("mem-second", "Production deployment requires Security approval.")
    client = _IncompleteStructuredClient()
    resolver = IdentityResolver(
        memory_store=_CandidateStore(
            exact_by_challenger={},
            semantic_by_challenger={challenger.id: (first, second)},
        ),
        pair_classifier=StructuredMemoryPairClassifier(
            client=client,
            model="test-model",
        ),
        llm_model="test-model",
    )

    batch = await resolver.resolve((IdentityResolutionRequest(challenger, "doc-a"),))
    result = batch.resolutions[0]

    assert result.target is None
    assert result.equivalence_proof is None
    assert result.classified_pairs == ()
    assert result.classification_complete is False
    assert result.failure_reason == (
        "memory relation decision coverage invalid: "
        "expected_count=2, actual_count=1, missing_count=1, "
        "duplicate_count=0, unexpected_count=0"
    )
    assert len(client.calls) == 1
    assert batch.metrics.pair_count == 2
    assert batch.metrics.llm_calls == 1
    assert batch.metrics.prompt_chars > 0


class _CompleteStructuredClient:
    def __init__(self) -> None:
        self.payloads: list[list[dict[str, object]]] = []

    async def classify_memory_relations(self, prompt: str, **_kwargs):
        payload_text = prompt.split("<memory_pair_groups>\n", 1)[1].split(
            "\n</memory_pair_groups>",
            1,
        )[0]
        payload = json.loads(payload_text)
        self.payloads.append(payload)
        decisions = []
        for group in payload:
            for item in group["candidates"]:
                candidate_id = item["candidate"]["id"]
                equivalent = candidate_id == "mem-equivalent"
                decisions.append(
                    SimpleNamespace(
                        pair_index=item["pair_index"],
                        classification="equivalent" if equivalent else "refines",
                        direction="symmetric" if equivalent else "challenger_to_candidate",
                        reason="fixture",
                    )
                )
        return SimpleNamespace(decisions=decisions)


@pytest.mark.asyncio
async def test_identity_resolver_preserves_pair_attribution_across_classifier_batches() -> None:
    first_challenger = _memory("mem-new-a", "Production deployment requires approval.")
    first_candidate = _memory("mem-refines", "Production deployment requires Security approval.")
    equivalent = _memory("mem-equivalent", "Approval is required for production deployment.")
    second_challenger = _memory("mem-new-b", "Backups run daily.")
    second_candidate = _memory("mem-daily", "Daily backups run at midnight.")
    client = _CompleteStructuredClient()
    resolver = IdentityResolver(
        memory_store=_CandidateStore(
            exact_by_challenger={},
            semantic_by_challenger={
                first_challenger.id: (first_candidate, equivalent),
                second_challenger.id: (second_candidate,),
            },
        ),
        pair_classifier=StructuredMemoryPairClassifier(
            client=client,
            model="test-model",
            policy=MemoryPairClassificationPolicy(max_pairs_per_call=2),
        ),
        llm_model="test-model",
    )

    batch = await resolver.resolve(
        (
            IdentityResolutionRequest(first_challenger, "doc-a"),
            IdentityResolutionRequest(second_challenger, "doc-b"),
        )
    )
    results = batch.resolutions

    assert [result.target for result in results] == [equivalent, None]
    assert batch.metrics.pair_count == 3
    assert batch.metrics.llm_calls == 2
    assert batch.metrics.prompt_chars > 0
    assert len(client.payloads) == 2
    assert [item["pair_index"] for payload in client.payloads for group in payload for item in group["candidates"]] == [
        0,
        1,
        2,
    ]
    assert [group["challenger"]["id"] for payload in client.payloads for group in payload] == [
        first_challenger.id,
        second_challenger.id,
    ]
