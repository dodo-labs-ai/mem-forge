from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.config import RetrievalConfig
from memforge.retrieval.search import SearchEngine, _RankedCandidate
from memforge.storage.adapters.context import AccessScope


class _RankingRelational:
    def __init__(self, metadata):
        self._metadata = metadata

    async def fetch_ranking_metadata(self, ids):
        return {memory_id: self._metadata[memory_id] for memory_id in ids}


def _engine(metadata) -> SearchEngine:
    return SearchEngine(
        relational=_RankingRelational(metadata),
        keyword=None,
        vector=None,
        embed_cfg={},
        config=RetrievalConfig(),
    )


def _scope(repo: str | None = None) -> AccessScope:
    return AccessScope(
        user_id="dev",
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
        active_repo_identifier=repo,
    )


@pytest.mark.asyncio
async def test_repo_affinity_boosts_same_repo_candidate():
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    engine = _engine({
        "same-repo": {
            "updated_at": now,
            "project_key": "UNSORTED",
            "repo_identifier": "github.tools.sap/hcm/memforge-cloud",
            "memory_level": "atomic",
            "curation_cluster_id": None,
            "covered_memory_count": 0,
        },
        "other-repo": {
            "updated_at": now,
            "project_key": "UNSORTED",
            "repo_identifier": "github.tools.sap/hcm/other",
            "memory_level": "atomic",
            "curation_cluster_id": None,
            "covered_memory_count": 0,
        },
    })

    ranked = await engine._apply_ranking(
        [
            _RankedCandidate("other-repo", rrf_score=1.0),
            _RankedCandidate("same-repo", rrf_score=1.0),
        ],
        is_temporal=False,
        scope=_scope("github.tools.sap/hcm/memforge-cloud"),
    )

    assert [candidate.memory_id for candidate in ranked] == [
        "same-repo",
        "other-repo",
    ]
    assert ranked[0].repo_identifier == "github.tools.sap/hcm/memforge-cloud"


def test_curation_family_collapses_child_when_consolidated_is_close():
    ranked = [
        _RankedCandidate(
            "child",
            final_score=0.91,
            memory_level="atomic",
            curation_cluster_id="cluster-1",
        ),
        _RankedCandidate(
            "summary",
            final_score=0.88,
            memory_level="consolidated",
            curation_cluster_id="cluster-1",
            covered_memory_count=2,
        ),
        _RankedCandidate("unrelated", final_score=0.80),
    ]

    collapsed = SearchEngine._collapse_curation_families(ranked)

    assert [candidate.memory_id for candidate in collapsed] == ["summary", "unrelated"]


def test_curation_family_keeps_exact_child_when_it_strongly_outranks_summary():
    ranked = [
        _RankedCandidate(
            "child",
            final_score=0.99,
            memory_level="atomic",
            curation_cluster_id="cluster-1",
        ),
        _RankedCandidate(
            "summary",
            final_score=0.70,
            memory_level="consolidated",
            curation_cluster_id="cluster-1",
            covered_memory_count=2,
        ),
    ]

    collapsed = SearchEngine._collapse_curation_families(ranked)

    assert [candidate.memory_id for candidate in collapsed] == ["child", "summary"]
