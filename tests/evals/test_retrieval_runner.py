from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from memforge.config import DEFAULT_RRF_K


@pytest.mark.asyncio
async def test_sqlite_runner_executes_core_hard_cases(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    report = await run_sqlite_case_set(
        load_case_set("retrieval-core-v1"),
        db_path=tmp_path / "retrieval-eval.db",
    )

    assert report.case_count == 12
    assert report.hard_failures == ()

    assert report.case_results["exact_external_id_lookup"].ranked_ids[0] == "mem-blocker-hint"
    assert report.case_results["metadata_title_exact"].rank("mem-access-review") <= 3
    wrapped_title = report.case_results["quoted_title_in_self_contained_query"]
    assert wrapped_title.rank("mem-access-review") <= 3
    assert wrapped_title.evidence_by_memory["mem-access-review"]["metadata_lexical"][
        "query_paths"
    ] == ["full_query", "quoted_identity"]
    assert report.case_results["compact_trigram_metadata_recall"].rank("mem-blocker-hint") <= 10
    cluster_case = report.case_results["metadata_three_of_five_source_cluster_recall"]
    assert {
        f"mem-process-tree-{index:02d}" for index in range(1, 7)
    }.issubset(cluster_case.ranked_ids[:10])
    assert report.case_results["queryless_source_listing"].total_candidates == 23

    assert report.qrels["metadata_title_exact"] == {"mem-access-review": 3}
    assert report.run["exact_external_id_lookup"]["mem-blocker-hint"] > 0
    assert report.run["exact_external_id_lookup"]["mem-blocker-hint"] == 1.0
    assert report.to_json()["summary"] == {
        "case_count": 12,
        "hard_failures": 0,
    }
    assert list(tmp_path.glob("retrieval-eval-*.db")) == []


@pytest.mark.asyncio
async def test_sqlite_runner_injects_ranked_channels_and_reports_vector_contribution(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    report = await run_sqlite_case_set(
        load_case_set("retrieval-core-v1"),
        db_path=tmp_path / "retrieval-eval.db",
    )

    case = report.case_results["vector_target_with_lexical_distractor"]
    assert case.rank("mem-cross-language-lock") <= 5
    assert case.evidence_by_memory["mem-cross-language-lock"]["rank_fusion"] == {
        "intent": "general_hybrid",
        "rrf_score": pytest.approx(0.45 / (DEFAULT_RRF_K + 1)),
        "contributions": [
            {
                "channel": "vector",
                "rank": 1,
                "channel_weight": 0.45,
                "multiplier": 1.0,
                "weighted_score": pytest.approx(0.45 / (DEFAULT_RRF_K + 1)),
            }
        ],
    }
    assert report.to_json()["case_results"]["vector_target_with_lexical_distractor"][
        "evidence_by_memory"
    ]["mem-cross-language-lock"]["rank_fusion"]["contributions"][0]["channel"] == "vector"


@pytest.mark.asyncio
async def test_sqlite_runner_reports_requested_and_fallback_intent_resolution(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    report = await run_sqlite_case_set(
        load_case_set("retrieval-core-v1"),
        db_path=tmp_path / "retrieval-eval.db",
    )

    assert report.case_results[
        "requested_general_hybrid_overrides_identity"
    ].retrieval_intent == {
        "requested_intent": "general_hybrid",
        "resolved_intent": "general_hybrid",
        "intent_source": "requested",
        "fallback_reason": None,
    }
    assert report.case_results["requested_known_item_is_honored"].retrieval_intent[
        "resolved_intent"
    ] == "known_item"
    assert report.case_results[
        "requested_relationship_falls_back"
    ].retrieval_intent == {
        "requested_intent": "relationship",
        "resolved_intent": "general_hybrid",
        "intent_source": "fallback",
        "fallback_reason": "no_traversable_entity",
    }
    assert report.case_results[
        "requested_relationship_is_honored"
    ].retrieval_intent == {
        "requested_intent": "relationship",
        "resolved_intent": "relationship",
        "intent_source": "requested",
        "fallback_reason": None,
    }


def test_search_engine_embedding_provider_degrades_on_exception() -> None:
    from memforge.config import RetrievalConfig
    from memforge.retrieval.search import SearchEngine

    def broken_provider(_query: str) -> list[float]:
        raise RuntimeError("boom")

    engine = SearchEngine(
        relational=object(),
        keyword=object(),
        vector=object(),
        embed_cfg={},
        config=RetrievalConfig(),
        embedding_provider=broken_provider,
    )

    assert engine._get_or_compute_embedding("query") is None


@pytest.mark.asyncio
async def test_sqlite_runner_reports_required_channel_failure(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    case_set = load_case_set("retrieval-core-v1").replace_case(
        "metadata_title_exact",
        expected=load_case_set("retrieval-core-v1")
        .get_case("metadata_title_exact")
        .expected.with_required_channels(
            "mem-access-review",
            ("graph",),
        ),
    )

    report = await run_sqlite_case_set(case_set, db_path=tmp_path / "retrieval-eval.db")

    assert len(report.hard_failures) == 1
    assert report.hard_failures[0].case_id == "metadata_title_exact"
    assert "graph" in report.hard_failures[0].message


@pytest.mark.asyncio
async def test_sqlite_runner_requires_all_declared_channels(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    case_set = load_case_set("retrieval-core-v1")
    target_case = case_set.get_case("metadata_title_exact")
    case_set = case_set.replace_case(
        "metadata_title_exact",
        expected=target_case.expected.with_required_channels(
            "mem-access-review",
            ("bm25_metadata_tokens", "graph"),
        ),
    )

    report = await run_sqlite_case_set(case_set, db_path=tmp_path / "retrieval-eval.db")

    assert len(report.hard_failures) == 1
    assert "graph" in report.hard_failures[0].message


@pytest.mark.asyncio
async def test_sqlite_runner_applies_time_range(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    case_set = load_case_set("retrieval-core-v1")
    listing_case = case_set.get_case("queryless_source_listing")
    case_set = case_set.replace_case(
        "queryless_source_listing",
        time_range={
            "after": "2026-01-02T00:00:00+00:00",
            "date_type": "source_updated_at",
        },
        expected=replace(
            listing_case.expected,
            relevant={},
            total_candidates=0,
        ),
    )

    report = await run_sqlite_case_set(case_set, db_path=tmp_path / "retrieval-eval.db")

    assert report.hard_failures == ()
    assert report.case_results["queryless_source_listing"].total_candidates == 0
    assert report.case_results["queryless_source_listing"].ranked_ids == ()


@pytest.mark.asyncio
async def test_sqlite_fixture_preserves_visibility_repo_and_zero_confidence(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.fixtures.corpus import seed_sqlite_fixture

    fixture = deepcopy(load_case_set("retrieval-core-v1").manifest.fixtures["default"])
    fixture["memories"].append(
        {
            "id": "mem-private-low-confidence",
            "content": "Private fixture memory.",
            "confidence": 0.0,
                "visibility": "private",
                "owner_user_id": "eval-user",
                "repo_identifier": "repo/example",
            }
    )

    db = await seed_sqlite_fixture(
        db_path=tmp_path / "retrieval-eval.db",
        fixture=fixture,
    )
    try:
        memory = await db.get_memory("mem-private-low-confidence")
    finally:
        await db.close()

    assert memory is not None
    assert memory.confidence == 0.0
    assert memory.visibility == "private"
    assert memory.owner_user_id == "eval-user"
    assert memory.repo_identifier == "repo/example"
