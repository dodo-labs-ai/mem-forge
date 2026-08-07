from __future__ import annotations

import pytest


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
