from __future__ import annotations

import pytest


def test_default_rrf_damping_preserves_cross_language_vector_recall_against_consensus_distractors() -> None:
    """A top vector-only hit must remain usable when lexical channels cannot cross languages."""

    from memforge.config import DEFAULT_RRF_K, RetrievalConfig
    from memforge.retrieval.intents import fusion_weights
    from memforge.retrieval.rank_fusion import (
        RankedChannelItem,
        weighted_reciprocal_rank_fusion,
    )

    vector_ranks = (1, 6, 5, 15, 9, 7, 16)
    content_ranks = (16, 33, 15, 28, 44, 21)
    distractor_ids = tuple(f"same-language-consensus-{index}" for index in range(6))
    channels = {
        "vector": (
            RankedChannelItem("cross-language-target", 0.0, rank=vector_ranks[0]),
            *(
                RankedChannelItem(memory_id, 0.0, rank=rank)
                for memory_id, rank in zip(distractor_ids, vector_ranks[1:])
            ),
        ),
        "bm25_content": tuple(
            RankedChannelItem(memory_id, 0.0, rank=rank)
            for memory_id, rank in zip(distractor_ids, content_ranks)
        ),
        "metadata_lexical": (),
        "graph": (),
    }

    passing_damping_constants = []
    for damping_constant in (10, 20, 30, 60):
        ranked = weighted_reciprocal_rank_fusion(
            channels=channels,
            weights=fusion_weights("general_hybrid"),
            k=damping_constant,
        )
        ranked_ids = tuple(item.item_id for item in ranked)
        if ranked_ids.index("cross-language-target") + 1 <= 5:
            passing_damping_constants.append(damping_constant)

    assert RetrievalConfig().rrf_k == DEFAULT_RRF_K
    assert passing_damping_constants == [10, 20]
    assert DEFAULT_RRF_K == max(passing_damping_constants)


def test_bounded_sweep_selects_production_policy_for_every_ranked_intent() -> None:
    from memforge.evals.retrieval.policy_sweep import run_bounded_intent_policy_sweep
    from memforge.retrieval.intents import RANKED_RETRIEVAL_INTENTS, fusion_weights

    result = run_bounded_intent_policy_sweep()

    assert set(result.selected) == set(RANKED_RETRIEVAL_INTENTS)
    assert {
        intent: dict(policy)
        for intent, policy in result.selected.items()
    } == {
        intent: fusion_weights(intent)
        for intent in RANKED_RETRIEVAL_INTENTS
    }
    assert {
        intent: {metrics.family for metrics in candidates.values()}
        for intent, candidates in result.metrics.items()
    } == {
        "general_hybrid": {"multilingual_vector_recall"},
        "known_item": {"known_item_identity"},
        "relationship": {"relationship_graph"},
    }
    assert all(
        any(metrics.hard_pass for metrics in candidates.values())
        for candidates in result.metrics.values()
    )
    selected_names = {
        "general_hybrid": "vector_primary",
        "known_item": "identity_primary",
        "relationship": "graph_primary",
    }
    for intent, selected_name in selected_names.items():
        metrics = result.metrics[intent][selected_name]
        assert metrics.case_count == 2
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.ndcg_at_5 == pytest.approx(1.0)
        assert metrics.hit_at_5 == pytest.approx(1.0)
        assert metrics.hard_pass is True


@pytest.mark.asyncio
async def test_full_golden_eval_remains_a_hard_gate_after_policy_selection(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    report = await run_sqlite_case_set(
        load_case_set("retrieval-core-v1"),
        db_path=tmp_path / "retrieval-eval.db",
    )

    assert report.hard_failures == ()
