from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest


@pytest.mark.asyncio
async def test_sqlite_runner_executes_core_hard_cases(tmp_path) -> None:
    from memforge.evals.retrieval import load_case_set
    from memforge.evals.retrieval.runner import run_sqlite_case_set

    report = await run_sqlite_case_set(
        load_case_set("retrieval-core-v1"),
        db_path=tmp_path / "retrieval-eval.db",
    )

    assert report.case_count == 4
    assert report.hard_failures == ()

    assert report.case_results["exact_external_id_lookup"].ranked_ids[0] == "mem-blocker-hint"
    assert report.case_results["metadata_title_exact"].rank("mem-access-review") <= 3
    assert report.case_results["compact_trigram_metadata_recall"].rank("mem-blocker-hint") <= 10
    assert report.case_results["queryless_source_listing"].total_candidates == 17

    assert report.qrels["metadata_title_exact"] == {"mem-access-review": 3}
    assert report.run["exact_external_id_lookup"]["mem-blocker-hint"] > 0
    assert report.run["exact_external_id_lookup"]["mem-blocker-hint"] == 1.0
    assert report.to_json()["summary"] == {
        "case_count": 4,
        "hard_failures": 0,
    }
    assert list(tmp_path.glob("retrieval-eval-*.db")) == []


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
