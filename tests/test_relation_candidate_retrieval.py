import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from memforge.memory.evidence import (
    CandidateBucket,
    CandidateMemory,
    build_candidate_universe,
)
from memforge.memory.relation_candidate_retrieval import (
    Nomination,
    RelationCandidateSelection,
    RelationCandidateRetriever,
    RelationCandidateRetrievalPolicy,
    RetrievedRelationCandidate,
    StaleCandidateSelectionError,
    nominate_candidates,
)
from memforge.retrieval.rank_fusion import RankedChannelItem
from memforge.storage.adapters.protocols import (
    ActiveMemorySupportState,
)

EARLIER = datetime(2026, 1, 1, tzinfo=timezone.utc)
CHALLENGER_CREATED = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _candidate(
    memory_id: str,
    *,
    doc_id: str | None = "doc-other",
    source_id: str = "src-enabled",
    visibility: str = "workspace",
    owner_user_id: str | None = None,
    repo_identifier: str | None = "repo-a",
    created_at: datetime | None = EARLIER,
) -> CandidateMemory:
    return CandidateMemory(
        memory_id=memory_id,
        source_id=source_id,
        doc_id=doc_id,
        source_lineage_id=doc_id,
        visibility=visibility,
        owner_user_id=owner_user_id,
        repo_identifier=repo_identifier,
        created_at=created_at,
    )


class _Relational:
    def __init__(self) -> None:
        self.provenance_ids: tuple[str, ...] = ()
        self.full_memory_ids: tuple[str, ...] = ()

    async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
        assert entity_ids == [7]
        assert memory_types is None
        assert limit == 4
        return [("mem-a", 0.9), ("mem-b", 0.8), ("mem-e", 0.7)]

    async def list_active_candidate_memories(self, memory_ids):
        self.provenance_ids = tuple(memory_ids)
        return [
            _candidate("mem-a"),
            _candidate("mem-b"),
            _candidate("mem-c"),
            _candidate("mem-d"),
            _candidate("mem-e"),
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
    retriever = RelationCandidateRetriever(
        relational=relational,
        keyword=_Keyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            per_list_limit=1,
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
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
    )

    assert selection.candidate_ids == ("mem-a", "mem-b", "mem-c")
    assert relational.provenance_ids == (
        "mem-a",
        "mem-b",
        "mem-e",
        "mem-c",
        "mem-d",
    )
    assert selection.audit["selected_discovery_count"] == 3
    assert selection.telemetry["per_list_limit"] == 1
    assert selection.audit["candidate_count_kind"] == "windowed"
    assert selection.telemetry["channel_errors"] == []

    materialized, loaded = await retriever.load_selected_memories(
        selection,
        challenger=challenger,
        source_id="src-new",
    )
    assert relational.full_memory_ids == selection.candidate_ids
    assert materialized.candidate_ids == selection.candidate_ids
    assert tuple(loaded) == selection.candidate_ids
    assert materialized.audit["selected_discovery_count"] == 3
    assert "selected_candidate_count" not in materialized.audit
    assert materialized.telemetry["full_memory_rows_loaded"] == 3

    universe = build_candidate_universe(
        relation_run_id="relrun-1",
        evidence_unit_id="eu-new",
        bucket_results=selection.bucket_results(),
        recall_candidate_cap=4,
    )
    assert universe.incomplete_mandatory_buckets == ()
    assert universe.mandatory_candidate_count == 0
    assert [candidate.memory_id for candidate in universe.candidates] == [
        "mem-a",
        "mem-b",
        "mem-c",
    ]
    assert universe.candidates[0].bucket is CandidateBucket.HYBRID_DISCOVERY
    assert [candidate.reason for candidate in universe.candidates] == [
        "hybrid_discovery:shared_entities",
        "hybrid_discovery:semantic_vector_neighbors",
        "hybrid_discovery:lexical_bm25",
    ]


@pytest.mark.asyncio
async def test_selected_ledger_fails_closed_when_one_full_row_disappears() -> None:
    class MissingFullRowRelational(_Relational):
        async def list_active_memories(self, memory_ids):
            loaded = await super().list_active_memories(memory_ids)
            return loaded[:-1]

    relational = MissingFullRowRelational()
    retriever = RelationCandidateRetriever(
        relational=relational,
        keyword=_Keyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            per_list_limit=1,
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
        created_at=CHALLENGER_CREATED,
    )
    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
    )

    with pytest.raises(StaleCandidateSelectionError, match="could not be materialized"):
        await retriever.load_selected_memories(
            selection,
            challenger=challenger,
            source_id="src-new",
        )


@pytest.mark.asyncio
async def test_selection_revalidation_fails_when_current_support_changes() -> None:
    candidate = _candidate("mem-candidate")
    challenger = SimpleNamespace(
        id="mem-new",
        status="active",
        content="Current claim",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
        created_at=CHALLENGER_CREATED,
    )

    class SupportChangingRelational:
        support_hash = "current-support"

        async def get_memory(self, memory_id):
            assert memory_id == challenger.id
            return challenger

        async def list_active_candidate_memories(self, memory_ids):
            assert tuple(memory_ids) == (candidate.memory_id,)
            return [candidate]

        async def get_active_memory_support_states(self, memory_ids):
            return {
                memory_id: ActiveMemorySupportState(
                    unit_ids=(),
                    support_set_hash=self.support_hash,
                    current_unit_ids=(),
                    current_support_set_hash=self.support_hash,
                )
                for memory_id in memory_ids
            }

    relational = SupportChangingRelational()
    retriever = RelationCandidateRetriever(
        relational=relational,  # type: ignore[arg-type]
        keyword=SimpleNamespace(),
        vector=SimpleNamespace(),
    )
    selection = RelationCandidateSelection(
        discovery=(
            RetrievedRelationCandidate(
                memory=candidate,
                best_rank=1,
                channels=("lexical_bm25",),
            ),
        ),
        audit={},
    )
    expected = {candidate.memory_id: relational.support_hash}

    await retriever.ensure_selection_current(
        selection,
        challenger=challenger,
        source_id="src-new",
        expected_support_set_hashes=expected,
    )
    relational.support_hash = "changed-support"

    with pytest.raises(StaleCandidateSelectionError, match="current Support changed"):
        await retriever.ensure_selection_current(
            selection,
            challenger=challenger,
            source_id="src-new",
            expected_support_set_hashes=expected,
        )


@pytest.mark.asyncio
async def test_worker_cancellation_is_not_downgraded_to_channel_failure() -> None:
    class CancelledRelational(_Relational):
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            raise asyncio.CancelledError

    retriever = RelationCandidateRetriever(
        relational=CancelledRelational(),
        keyword=_Keyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            per_list_limit=1,
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
        created_at=CHALLENGER_CREATED,
    )

    with pytest.raises(asyncio.CancelledError):
        await retriever.retrieve(
            challenger=challenger,
            entity_ids=[7],
            actor_user_id="user-a",
            source_id="src-new",
        )


@pytest.mark.asyncio
async def test_failed_discovery_channel_is_audited_without_making_it_mandatory() -> None:
    class BrokenKeyword(_Keyword):
        async def search(self, query, scope, memory_types, limit):
            raise RuntimeError("keyword unavailable")

    retriever = RelationCandidateRetriever(
        relational=_Relational(),
        keyword=BrokenKeyword(),
        vector=_Vector(),
        policy=RelationCandidateRetrievalPolicy(
            per_list_limit=1,
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
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
    )

    assert selection.telemetry["channel_errors"] == [CandidateBucket.LEXICAL_BM25.value]
    retry_selection = RelationCandidateSelection(
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
            return [
                (memory_id, 1.0)
                for memory_id in (
                    "eligible",
                    "wrong-visibility",
                    "wrong-owner",
                    "wrong-repo",
                    "disabled-only",
                    "mixed-source",
                    "same-and-cross-source",
                )
            ]

        async def list_active_candidate_memories(self, memory_ids):
            del memory_ids
            return [
                _candidate("eligible"),
                _candidate("wrong-visibility", visibility="private"),
                _candidate("wrong-owner", owner_user_id="other"),
                _candidate("wrong-repo", repo_identifier="repo-b"),
                _candidate("disabled-only", source_id="src-disabled"),
                _candidate("mixed-source", source_id="src-disabled"),
                _candidate("mixed-source", source_id="src-enabled"),
                _candidate("same-and-cross-source", source_id="src-new"),
                _candidate("same-and-cross-source", source_id="src-other"),
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

    retriever = RelationCandidateRetriever(
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
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
        excluded_source_ids=("src-disabled",),
    )

    assert set(selection.candidate_ids) == {
        "eligible",
        "mixed-source",
        "same-and-cross-source",
    }
    selected = {item.memory.memory_id: item.memory for item in selection.discovery}
    assert selected["same-and-cross-source"].source_id == "src-other"


@pytest.mark.asyncio
async def test_memories_of_the_challengers_own_document_are_eligible_candidates() -> None:
    class SameDocumentRelational:
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            return [(memory_id, 1.0) for memory_id in ("mem-new", "same-doc", "other-doc", "no-doc")]

        async def list_active_candidate_memories(self, memory_ids):
            del memory_ids
            return [
                _candidate("mem-new", doc_id="doc-new", source_id="src-new"),
                _candidate("same-doc", doc_id="doc-new", source_id="src-new"),
                _candidate("other-doc"),
                _candidate("no-doc", doc_id=None),
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

    retriever = RelationCandidateRetriever(
        relational=SameDocumentRelational(),
        keyword=EmptyKeyword(),
        vector=EmptyVector(),
    )
    challenger = SimpleNamespace(
        id="mem-new",
        content="Payroll runs weekly.",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
    )

    assert set(selection.candidate_ids) == {"same-doc", "other-doc"}


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
            return [(f"mem-{index:05d}", float(10_000 - index)) for index in range(10_000)]

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
            return [(f"mem-{index:05d}", float(10_000 - index)) for index in range(64, 10_064)]

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
            return [(f"mem-{index:05d}", float(10_000 - index)) for index in range(32, 10_032)]

    relational = LargeRelational()
    keyword = LargeKeyword()
    vector = LargeVector()
    retriever = RelationCandidateRetriever(
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
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(
        challenger=challenger,
        entity_ids=[7],
        actor_user_id="user-a",
        source_id="src-new",
    )

    assert relational.graph_limit == 128
    assert keyword.limit == 128
    assert len(keyword.query.split(" OR ")) == 32
    assert vector.limit == 128
    assert len(relational.provenance_ids) == 192
    assert len(selection.candidate_ids) == 15
    assert selection.telemetry["provenance_rows_loaded"] == 192
    assert selection.audit["candidate_count_kind"] == "windowed"


def test_each_list_nominates_its_first_items_and_the_candidates_are_their_union() -> None:
    channels = {
        "shared_entities": [RankedChannelItem("mem-x", 3.0), RankedChannelItem("mem-y", 2.0), RankedChannelItem("mem-z", 1.0)],
        "lexical_bm25": [RankedChannelItem("mem-y", 9.0), RankedChannelItem("mem-w", 8.0)],
    }

    nominations = nominate_candidates(channels, per_list_limit=2)

    assert nominations == (
        Nomination("mem-x", 1, ("shared_entities",)),
        Nomination("mem-y", 1, ("shared_entities", "lexical_bm25")),
        Nomination("mem-w", 2, ("lexical_bm25",)),
    )


@pytest.mark.asyncio
async def test_only_memories_created_before_the_challenger_are_candidates() -> None:
    rows = [
        _candidate("mem-earlier"),
        _candidate("mem-later", created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)),
        _candidate("mem-a-same-commit", created_at=CHALLENGER_CREATED),
        _candidate("mem-z-same-commit", created_at=CHALLENGER_CREATED),
        _candidate("mem-naive-earlier", created_at=datetime(2026, 1, 15)),
    ]

    class OrderedRelational:
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            return [(row.memory_id, 1.0) for row in rows]

        async def list_active_candidate_memories(self, memory_ids):
            del memory_ids
            return rows

    class EmptyKeyword:
        async def search(self, query, scope, memory_types, limit):
            del query, scope, memory_types, limit
            return []

    class EmptyVector:
        async def get_record(self, memory_id):
            del memory_id
            return None

    retriever = RelationCandidateRetriever(relational=OrderedRelational(), keyword=EmptyKeyword(), vector=EmptyVector())
    challenger = SimpleNamespace(
        id="mem-m-same-commit",
        content="Payroll runs weekly.",
        memory_type="fact",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-a",
        project_key="PAY",
        created_at=CHALLENGER_CREATED,
    )

    selection = await retriever.retrieve(challenger=challenger, entity_ids=[7], actor_user_id="user-a", source_id="src-new")

    assert set(selection.candidate_ids) == {"mem-earlier", "mem-a-same-commit", "mem-naive-earlier"}


@pytest.mark.asyncio
async def test_every_pair_is_judged_from_exactly_one_side() -> None:
    created = {
        "mem-1": EARLIER,
        "mem-2": CHALLENGER_CREATED,
        "mem-3": CHALLENGER_CREATED,
        "mem-4": datetime(2026, 3, 1, tzinfo=timezone.utc),
    }
    rows = [_candidate(memory_id, created_at=moment) for memory_id, moment in created.items()]

    class AllRelational:
        async def graph_search(self, entity_ids, scope, memory_types, limit, **kwargs):
            del entity_ids, scope, memory_types, limit, kwargs
            return [(memory_id, 1.0) for memory_id in created]

        async def list_active_candidate_memories(self, memory_ids):
            del memory_ids
            return rows

    class EmptyKeyword:
        async def search(self, query, scope, memory_types, limit):
            del query, scope, memory_types, limit
            return []

    class EmptyVector:
        async def get_record(self, memory_id):
            del memory_id
            return None

    retriever = RelationCandidateRetriever(relational=AllRelational(), keyword=EmptyKeyword(), vector=EmptyVector())
    judged: list[frozenset[str]] = []
    for memory_id, moment in created.items():
        challenger = SimpleNamespace(
            id=memory_id,
            content="Payroll runs weekly.",
            memory_type="fact",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier="repo-a",
            project_key="PAY",
            created_at=moment,
        )
        selection = await retriever.retrieve(challenger=challenger, entity_ids=[7], actor_user_id="user-a", source_id="src-new")
        judged += [frozenset((memory_id, candidate)) for candidate in selection.candidate_ids]

    assert sorted(judged, key=sorted) == sorted(
        {frozenset((first, second)) for first in created for second in created if first != second}, key=sorted
    )
