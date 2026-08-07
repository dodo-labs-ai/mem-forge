"""Bounded, deterministic selection evidence for ranked intent policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from memforge.retrieval.intents import RankedRetrievalIntent
from memforge.retrieval.rank_fusion import (
    RankedChannelItem,
    weighted_reciprocal_rank_fusion,
)


ChannelResults = Mapping[str, tuple[RankedChannelItem, ...]]
FusionPolicy = Mapping[str, float]


@dataclass(frozen=True)
class PolicySweepScenario:
    """One family-specific ranking contrast with a hard target rank."""

    family: str
    intent: RankedRetrievalIntent
    target_id: str
    max_rank: int
    qrels: Mapping[str, int]
    channels: ChannelResults


@dataclass(frozen=True)
class PolicyMetrics:
    """Aggregated metrics and hard-gate state for one query family."""

    family: str
    case_count: int
    mrr: float
    ndcg_at_5: float
    hit_at_5: float
    hard_pass: bool


@dataclass(frozen=True)
class PolicySweepResult:
    """Selected fixed policies and the complete bounded comparison evidence."""

    selected: Mapping[RankedRetrievalIntent, Mapping[str, float]]
    metrics: Mapping[RankedRetrievalIntent, Mapping[str, PolicyMetrics]]


_SELECTED_POLICIES: dict[RankedRetrievalIntent, dict[str, float]] = {
    "general_hybrid": {
        "vector": 0.45,
        "bm25_content": 0.30,
        "metadata_lexical": 0.15,
        "graph": 0.10,
    },
    "known_item": {
        "vector": 0.15,
        "bm25_content": 0.25,
        "metadata_lexical": 0.50,
        "graph": 0.10,
    },
    "relationship": {
        "vector": 0.25,
        "bm25_content": 0.25,
        "metadata_lexical": 0.20,
        "graph": 0.30,
    },
}

_CANDIDATES: dict[RankedRetrievalIntent, dict[str, FusionPolicy]] = {
    "general_hybrid": {
        "vector_primary": _SELECTED_POLICIES["general_hybrid"],
        "lexical_heavy": {
            "vector": 0.25,
            "bm25_content": 0.35,
            "metadata_lexical": 0.30,
            "graph": 0.10,
        },
        "identity_primary": _SELECTED_POLICIES["known_item"],
    },
    "known_item": {
        "identity_primary": _SELECTED_POLICIES["known_item"],
        "balanced": _SELECTED_POLICIES["general_hybrid"],
        "graph_primary": _SELECTED_POLICIES["relationship"],
    },
    "relationship": {
        "graph_primary": _SELECTED_POLICIES["relationship"],
        "balanced": _SELECTED_POLICIES["general_hybrid"],
        "identity_primary": _SELECTED_POLICIES["known_item"],
    },
}

_SCENARIOS: tuple[PolicySweepScenario, ...] = (
    PolicySweepScenario(
        family="multilingual_vector_recall",
        intent="general_hybrid",
        target_id="target",
        max_rank=1,
        qrels={"target": 3, "lexical-0": 1},
        channels={
            "vector": (RankedChannelItem("target", 1.0),),
            "bm25_content": tuple(
                RankedChannelItem(f"lexical-{index}", 10.0 - index)
                for index in range(6)
            ),
            "metadata_lexical": (),
            "graph": (),
        },
    ),
    PolicySweepScenario(
        family="multilingual_vector_recall",
        intent="general_hybrid",
        target_id="target-variant",
        max_rank=1,
        qrels={"target-variant": 3, "metadata-0": 1},
        channels={
            "vector": (RankedChannelItem("target-variant", 1.0),),
            "bm25_content": (),
            "metadata_lexical": tuple(
                RankedChannelItem(f"metadata-{index}", 10.0 - index)
                for index in range(6)
            ),
            "graph": (),
        },
    ),
    PolicySweepScenario(
        family="known_item_identity",
        intent="known_item",
        target_id="target",
        max_rank=1,
        qrels={"target": 3, "semantic-competitor": 1},
        channels={
            "vector": (RankedChannelItem("semantic-competitor", 1.0),),
            "bm25_content": (RankedChannelItem("semantic-competitor", 1.0),),
            "metadata_lexical": (RankedChannelItem("target", 1.0),),
            "graph": (),
        },
    ),
    PolicySweepScenario(
        family="known_item_identity",
        intent="known_item",
        target_id="target-variant",
        max_rank=1,
        qrels={"target-variant": 3, "semantic-competitor-variant": 1},
        channels={
            "vector": (RankedChannelItem("semantic-competitor-variant", 1.0),),
            "bm25_content": (
                RankedChannelItem("semantic-competitor-variant", 1.0),
                RankedChannelItem("target-variant", 0.5),
            ),
            "metadata_lexical": (RankedChannelItem("target-variant", 1.0),),
            "graph": (),
        },
    ),
    PolicySweepScenario(
        family="relationship_graph",
        intent="relationship",
        target_id="target",
        max_rank=1,
        qrels={"target": 3, "metadata-competitor": 1},
        channels={
            "vector": (),
            "bm25_content": (),
            "metadata_lexical": (RankedChannelItem("metadata-competitor", 1.0),),
            "graph": (RankedChannelItem("target", 1.0),),
        },
    ),
    PolicySweepScenario(
        family="relationship_graph",
        intent="relationship",
        target_id="target-variant",
        max_rank=1,
        qrels={"target-variant": 3, "content-competitor": 1},
        channels={
            "vector": (),
            "bm25_content": (RankedChannelItem("content-competitor", 1.0),),
            "metadata_lexical": (),
            "graph": (RankedChannelItem("target-variant", 1.0),),
        },
    ),
)


def run_bounded_intent_policy_sweep() -> PolicySweepResult:
    """Evaluate the finite candidate set and select the only hard-safe policy."""

    selected: dict[RankedRetrievalIntent, Mapping[str, float]] = {}
    all_metrics: dict[RankedRetrievalIntent, dict[str, PolicyMetrics]] = {}
    for intent in _CANDIDATES:
        scenarios = tuple(scenario for scenario in _SCENARIOS if scenario.intent == intent)
        candidate_metrics: dict[str, PolicyMetrics] = {}
        passing: list[tuple[str, FusionPolicy, PolicyMetrics]] = []
        for name, weights in _CANDIDATES[intent].items():
            case_metrics = tuple(
                _evaluate_scenario(scenario, weights)
                for scenario in scenarios
            )
            metrics = PolicyMetrics(
                family=scenarios[0].family,
                case_count=len(case_metrics),
                mrr=sum(metric[0] for metric in case_metrics) / len(case_metrics),
                ndcg_at_5=sum(metric[1] for metric in case_metrics) / len(case_metrics),
                hit_at_5=sum(metric[2] for metric in case_metrics) / len(case_metrics),
                hard_pass=all(metric[3] for metric in case_metrics),
            )
            candidate_metrics[name] = metrics
            if metrics.hard_pass:
                passing.append((name, weights, metrics))
        if not passing:
            raise RuntimeError(f"no hard-safe policy for {intent}")
        passing.sort(
            key=lambda item: (
                -item[2].mrr,
                -item[2].ndcg_at_5,
                -item[2].hit_at_5,
                item[0],
            )
        )
        selected[intent] = dict(passing[0][1])
        all_metrics[intent] = candidate_metrics
    return PolicySweepResult(selected=selected, metrics=all_metrics)


def _evaluate_scenario(
    scenario: PolicySweepScenario,
    weights: FusionPolicy,
) -> tuple[float, float, float, bool]:
    ranked = weighted_reciprocal_rank_fusion(
        channels=scenario.channels,
        weights=weights,
        k=60,
    )
    ranked_ids = tuple(item.item_id for item in ranked)
    target_rank = ranked_ids.index(scenario.target_id) + 1
    first_relevant_rank = next(
        rank
        for rank, memory_id in enumerate(ranked_ids, start=1)
        if scenario.qrels.get(memory_id, 0) > 0
    )
    return (
        1.0 / first_relevant_rank,
        _ndcg_at_5(ranked_ids, scenario.qrels),
        1.0 if target_rank <= 5 else 0.0,
        target_rank <= scenario.max_rank,
    )


def _ndcg_at_5(ranked_ids: tuple[str, ...], qrels: Mapping[str, int]) -> float:
    def discounted_gain(grades: tuple[int, ...]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades[:5], start=1)
            if grade > 0
        )

    observed = tuple(qrels.get(memory_id, 0) for memory_id in ranked_ids)
    ideal = tuple(sorted(qrels.values(), reverse=True))
    ideal_gain = discounted_gain(ideal)
    return discounted_gain(observed) / ideal_gain if ideal_gain else 0.0


__all__ = [
    "PolicyMetrics",
    "PolicySweepResult",
    "run_bounded_intent_policy_sweep",
]
