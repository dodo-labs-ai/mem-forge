import asyncio
from types import SimpleNamespace

import pytest

from memforge.memory.evidence import (
    CandidateBucket,
    CandidateMemory,
    build_candidate_universe,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateSelection,
    CrossDocumentCandidateRetriever,
    RelationCandidateRetrievalPolicy,
)


def _candidate(
    memory_id: str,
    *,
    doc_id: str = "doc-other",
    source_id: str = "src-enabled",
    visibility: str = "workspace",
    owner_user_id: str | None = None,
    repo_identifier: str | None = "repo-a",
) -> CandidateMemory:
    return CandidateMemory(
        memory_id=memory_id,
        source_id=source_id,
        doc_id=doc_id,
        source_lineage_id=doc_id,
        visibility=visibility,
        owner_user_id=owner_user_id,
        repo_identifier=repo_identifier,
    )


class _Relational:
    def __init__(self) -> None:
        self.provenance_ids: tuple[str, ...] = ()
        self.full_memory_ids: tuple[str, ...] = ()

    async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
        assert entity_ids == [7]
        assert memory_types is None
        assert limit == 4
        return [("mem-a", 0.9), ("mem-b", 0.8), ("mem-same-doc", 0.7)]

    async def list_active_candidate_memories(self, memory_ids):
        self.provenance_ids = tuple(memory_ids)
        return [
            _candidate("mem-a"),
            _candidate("mem-b"),
            _candidate("mem-c"),
            _candidate("mem-d"),
            _candidate("mem-same-doc", doc_id="doc-new"),
        ]

    async def list_active_memories(self, memory_ids):
        self.full_memory_ids = tuple(memory_ids)
        return [
            SimpleNamespace(
                id=memory_id,
                visibility="workspace",
                owner_user_id=None,
                repo_identifier="repo-a",
            )
            for memory_id in memory_ids
        ]


class _Keyword:
    async def search(self, query, scope, memory_types, limit):
        assert query
        assert " OR " in query
        assert memory_types is None
        assert limit == 4
        return [("mem-c", 8.0), ("mem-d", 7.0)]


class _Vector:
    distance_metric = "cosine"

    async def get_record(self, memory_id):
        assert memory_id == "mem-new"
        return {"embedding": [0.1, 0.2]}

    async def query(self, embedding, scope, memory_types, limit):
        assert embedding == [0.1, 0.2]
        assert memory_types is None
        assert limit == 4
        return [("mem-b", 0.95), ("mem-c", 0.85)]


@pytest.mark.asyncio
async def test_hybrid_discovery_reuses_lightweight_candidates_before_full_row_loading() -> None:
    relational = _Relational()
    retriever = CrossDocumentCandidateRetriever(
        relational=relational,
        keyword=_Keyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            initial_budget=1,
            expansion_step=1,
            max_budget=4,
            rank_window_size=4,
        ),
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content="The service uses a bounded queue.",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        doc_id="doc-new",
        actor_user_id="user-a",
    )

    assert selection.candidate_ids == ("mem-b", "mem-c")
    assert relational.provenance_ids == (
        "mem-a",
        "mem-b",
        "mem-same-doc",
        "mem-c",
        "mem-d",
    )
    assert selection.audit["selected_discovery_count"] == 2
    assert selection.telemetry["discovery_budget"] == 2
    assert selection.audit["candidate_count_kind"] == "windowed"
    assert selection.telemetry["channel_errors"] == []

    materialized, loaded = await retriever.load_selected_memories(
        selection,
        challenger=challenger,
        doc_id="doc-new",
    )
    assert relational.full_memory_ids == selection.candidate_ids
    assert materialized.candidate_ids == selection.candidate_ids
    assert tuple(loaded) == selection.candidate_ids
    assert materialized.audit["selected_discovery_count"] == 2
    assert "selected_candidate_count" not in materialized.audit
    assert materialized.telemetry["full_memory_rows_loaded"] == 2

    universe = build_candidate_universe(
        relation_run_id="relrun-1",
        evidence_unit_id="eu-new",
        bucket_results=selection.bucket_results(),
        recall_candidate_cap=4,
    )
    assert universe.incomplete_mandatory_buckets == ()
    assert universe.mandatory_candidate_count == 0
    assert [candidate.memory_id for candidate in universe.candidates] == [
        "mem-b",
        "mem-c",
    ]
    assert universe.candidates[0].bucket is CandidateBucket.HYBRID_DISCOVERY
    assert universe.candidates[0].reason == (
        "hybrid_discovery:shared_entities,semantic_vector_neighbors"
    )


@pytest.mark.asyncio
async def test_worker_cancellation_is_not_downgraded_to_channel_failure() -> None:
    class CancelledRelational(_Relational):
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            raise asyncio.CancelledError

    retriever = CrossDocumentCandidateRetriever(
        relational=CancelledRelational(),
        keyword=_Keyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            initial_budget=1,
            expansion_step=1,
            max_budget=4,
            rank_window_size=4,
        ),
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content="The service uses a bounded queue.",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
    )

    with pytest.raises(asyncio.CancelledError):
        await retriever.retrieve(
            challenger=challenger,
            entity_ids=[7],
            doc_id="doc-new",
            actor_user_id="user-a",
        )
@pytest.mark.asyncio
async def test_failed_discovery_channel_is_audited_without_making_it_mandatory() -> None:
    class BrokenKeyword(_Keyword):
        async def search(self, query, scope, memory_types, limit):
            raise RuntimeError("keyword unavailable")

    retriever = CrossDocumentCandidateRetriever(
        relational=_Relational(),
        keyword=BrokenKeyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            initial_budget=2,
            expansion_step=1,
            max_budget=4,
            rank_window_size=4,
        ),
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content="The service uses a bounded queue.",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        doc_id="doc-new",
        actor_user_id="user-a",
    )

    assert selection.telemetry["channel_errors"] == [CandidateBucket.LEXICAL_BM25.value]
    retry_selection = CrossDocumentCandidateSelection(
        discovery=selection.discovery,
        audit=selection.audit,
        telemetry={**selection.telemetry, "channel_errors": []},
    )
    assert retry_selection.snapshot_identity == selection.snapshot_identity
    universe = build_candidate_universe(
        relation_run_id="relrun-2",
        evidence_unit_id="eu-new",
        bucket_results=selection.bucket_results(),
        recall_candidate_cap=4,
    )
    assert universe.incomplete_mandatory_buckets == ()


@pytest.mark.asyncio
async def test_exact_postfilter_preserves_only_access_compatible_provenance() -> None:
    class ExactRelational:
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            return [(memory_id, 1.0) for memory_id in (
                "eligible",
                "same-doc",
                "wrong-visibility",
                "wrong-owner",
                "wrong-repo",
                "disabled-only",
                "mixed-source",
            )]

        async def list_active_candidate_memories(self, memory_ids):
            del memory_ids
            return [
                _candidate("eligible"),
                _candidate("same-doc", doc_id="doc-new"),
                _candidate("wrong-visibility", visibility="private"),
                _candidate("wrong-owner", owner_user_id="other"),
                _candidate("wrong-repo", repo_identifier="repo-b"),
                _candidate("disabled-only", source_id="src-disabled"),
                _candidate("mixed-source", source_id="src-disabled"),
                _candidate("mixed-source", source_id="src-enabled"),
            ]

    class EmptyKeyword:
        async def search(self, query, scope, memory_types, limit):
            del query, scope, memory_types, limit
            return []

    class EmptyVector:
        distance_metric = "cosine"

        async def get_record(self, memory_id):
            del memory_id
            return None

    retriever = CrossDocumentCandidateRetriever(
        relational=ExactRelational(),
        keyword=EmptyKeyword(),
        vector=EmptyVector(),
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content="Exact access boundary",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        doc_id="doc-new",
        actor_user_id="user-a",
        excluded_source_ids=("src-disabled",),
    )

    assert set(selection.candidate_ids) == {"eligible", "mixed-source"}


@pytest.mark.asyncio
async def test_large_memory_corpus_stays_behind_bounded_channel_windows() -> None:
    """Corpus size belongs to the indexes, not to the detector process."""

    class LargeRelational:
        def __init__(self) -> None:
            self.graph_limit = 0
            self.provenance_ids: tuple[str, ...] = ()

        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, kwargs
            self.graph_limit = limit
            return [
                (f"mem-{index:05d}", float(10_000 - index))
                for index in range(10_000)
            ]

        async def list_active_candidate_memories(self, memory_ids):
            self.provenance_ids = tuple(memory_ids)
            return [_candidate(memory_id) for memory_id in memory_ids]

    class LargeKeyword:
        def __init__(self) -> None:
            self.limit = 0
            self.query = ""

        async def search(self, query, scope, memory_types, limit):
            del scope, memory_types
            self.limit = limit
            self.query = query
            return [
                (f"mem-{index:05d}", float(10_000 - index))
                for index in range(64, 10_064)
            ]

    class LargeVector:
        distance_metric = "cosine"

        def __init__(self) -> None:
            self.limit = 0

        async def get_record(self, memory_id):
            del memory_id
            return {"embedding": [0.1]}

        async def query(self, embedding, scope, memory_types, limit):
            del embedding, scope, memory_types
            self.limit = limit
            return [
                (f"mem-{index:05d}", float(10_000 - index))
                for index in range(32, 10_032)
            ]

    relational = LargeRelational()
    keyword = LargeKeyword()
    vector = LargeVector()
    retriever = CrossDocumentCandidateRetriever(
        relational=relational,
        keyword=keyword,
        vector=vector,
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content=" ".join(f"term{index}" for index in range(100)),
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        doc_id="doc-new",
        actor_user_id="user-a",
    )

    assert relational.graph_limit == 128
    assert keyword.limit == 128
    assert len(keyword.query.split(" OR ")) == 32
    assert vector.limit == 128
    assert len(relational.provenance_ids) == 192
    assert len(selection.candidate_ids) <= 128
    assert selection.telemetry["provenance_rows_loaded"] == 192
    assert selection.audit["candidate_count_kind"] == "windowed"
